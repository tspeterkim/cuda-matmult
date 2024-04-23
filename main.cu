#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <vector>

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
                     cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void CudaDeviceInfo() {
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device ID: %d\n\
        Name: %s\n\
        Compute Capability: %d.%d\n\
        memoryBusWidth: %d\n\
        maxThreadsPerBlock: %d\n\
        maxThreadsPerMultiProcessor: %d\n\
        maxRegsPerBlock: %d\n\
        maxRegsPerMultiProcessor: %d\n\
        totalGlobalMem: %zuMB\n\
        sharedMemPerBlock: %zuKB\n\
        sharedMemPerMultiprocessor: %zuKB\n\
        totalConstMem: %zuKB\n\
        multiProcessorCount: %d\n\
        Warp Size: %d\n",
            deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
            props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
            props.regsPerBlock, props.regsPerMultiprocessor,
            props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
            props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
            props.multiProcessorCount, props.warpSize);
};

void run_cublas_fp32(cublasHandle_t handle, int M, int N, int K, float alpha,
                                     float *A, float *B, float beta, float *C) {
    // cuBLAS uses column-major order. So we change the order of our row-major A &
    // B, since (B^T*A^T)^T = (A*B)
    // This runs cuBLAS in full fp32 mode
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
                    N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // if statement is necessary to make things work under tile quantization
    if ((row < K) && (col < K)) {
        float tmp = 0.0;
        for (int k = 0; k < K; ++k) {
            tmp += A[row*K + k] * B[k*K + col];
        }
        C[row*K + col] = tmp;
    }
}

void run_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                                         float beta, float *C) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

template <const int TILE_WIDTH>
__global__ void sgemm_tiled(int M, int N, int K, float alpha,
                            const float *A, const float *B,
                            float beta, float *C) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;

    int row = TILE_WIDTH * by + ty;
    int col = TILE_WIDTH * bx + tx;

    float tmp = 0.0;
    for (int ti = 0; ti < K / TILE_WIDTH; ti++) { // ti = tile_index
        As[ty][tx] = A[(row * K) + (ti * TILE_WIDTH + tx)];
        Bs[ty][tx] = B[(ti * TILE_WIDTH + ty) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            tmp += As[ty][k] * Bs[k][tx];   
        }
        __syncthreads();
    }
    C[row * M + col] = tmp;
}

void run_sgemm_tiled(int M, int N, int K, float alpha, float *A,
                        float *B, float beta, float *C) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
    // out all of L1 to SMEM. This doesn't currently make a difference, since
    // occupancy is limited by reg and thread count, but it's good to do anyway.
    cudaFuncSetAttribute(sgemm_tiled<32>,
                            cudaFuncAttributePreferredSharedMemoryCarveout,
                            cudaSharedmemCarveoutMaxShared);
    sgemm_tiled<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

template <const int TILE_WIDTH, const int COARSE_FACTOR>
__global__ void sgemm_tiled_coarsed_pmpp(int M, int N, int K, float alpha,
                                            const float *A, const float *B,
                                            float beta, float *C) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    // Initialize PValue for all output elements
    float PValue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        PValue[c] = 0.0f;
    }

    // Loop over the M and N tiles required to compute P element
    for (int ph = 0; ph < K / TILE_WIDTH; ++ph) {
        // Collaborative loading of M tile into shared memory
        Mds[ty][tx] = A[row * K + ph * TILE_WIDTH + tx];

        // Collaborative loading of N tile into shared memory
        for (int c = 0; c < COARSE_FACTOR; ++c) {
                
            int col = colStart + c * TILE_WIDTH;
            
            Nds[ty][tx] = B[(ph * TILE_WIDTH + ty) * K + col];
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k) {
                    PValue[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c * TILE_WIDTH;
        C[row * K + col] = PValue[c];
    }
}

void run_sgemm_tiled_coarsed_pmpp(int M, int N, int K, float alpha, float *A,
                                    float *B, float beta, float *C) {
    if (CEIL_DIV(M,32) < 8) {
        // Reduce coarse_factor from 8 to 4 for K=128. Otherwise invalid gridDim
        const int coarse_factor = 4;
        dim3 gridDim(CEIL_DIV(M, 32)/coarse_factor, CEIL_DIV(N, 32));
        dim3 blockDim(32, 32);
        sgemm_tiled_coarsed_pmpp<32, coarse_factor>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    } else {
        const int coarse_factor = 8;
        dim3 gridDim(CEIL_DIV(M, 32)/coarse_factor, CEIL_DIV(N, 32));
        dim3 blockDim(32, 32);
        sgemm_tiled_coarsed_pmpp<32, coarse_factor>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }
}

template <const int TILE_WIDTH, const int COARSE_FACTOR>
__global__ void sgemm_tiled_coarsed_blog(int M, int N, int K, float alpha,
                                            const float *A, const float *B,
                                            float beta, float *C) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;

    int row = TILE_WIDTH * by + COARSE_FACTOR * ty;
    int col = TILE_WIDTH * bx + tx;

    float tmp[COARSE_FACTOR] = {0.0};
    for (int ti = 0; ti < K / TILE_WIDTH; ti++) {
        for (int c = 0; c < COARSE_FACTOR; c++) {
            As[ty*COARSE_FACTOR + c][tx] = A[(row+c) * K + ti*TILE_WIDTH + tx];
            Bs[ty*COARSE_FACTOR + c][tx] = B[(ti*TILE_WIDTH + ty*COARSE_FACTOR + c) * K + col];
        }
        __syncthreads();

        for (int c = 0; c < COARSE_FACTOR; c++) {
            for (int k = 0; k < TILE_WIDTH; k++) {
                tmp[c] += As[ty*COARSE_FACTOR + c][k] * Bs[k][tx];  
            }
        }
        __syncthreads();
    }
    for (int c = 0; c < COARSE_FACTOR; c++) {
        C[(row+c) * K + col] = tmp[c];
    }
}

void run_sgemm_tiled_coarsed_blog(int M, int N, int K, float alpha, float *A,
                                    float *B, float beta, float *C) {
    const int coarse_factor = 8;
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32 / coarse_factor);
    sgemm_tiled_coarsed_blog<32, coarse_factor>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void randomize_matrix(float *mat, int N) {
    // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
    // precision is too low and the same random number is generated.
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

bool verify_matrix(float *matRef, float *matOut, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        diff = std::fabs(matRef[i] - matOut[i]);
        if (diff > 0.01) {
            printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
                    matRef[i], matOut[i], diff, i);
            return false;
        }
    }
    return true;
}

void print_matrix(const float *A, int M, int N) {
    int i;
    std::cout << std::setprecision(2)
        << std::fixed; // Set floating-point precision and fixed notation
    std::cout << "[";
    for (i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0)
            std::cout << std::setw(5) << A[i]; // Set field width and write the value
        else
            std::cout << std::setw(5) << A[i] << ", ";
        if ((i + 1) % N == 0) {
            if (i + 1 < M * N)
                std::cout << ";\n";
        }
    }
    std::cout << "]\n";
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
    switch (kernel_num) {
        case 0:
            run_cublas_fp32(handle, M, N, K, alpha, A, B, beta, C);
            break;
        case 1:
            run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
            break;
        case 2:
            run_sgemm_tiled(M, N, K, alpha, A, B, beta, C);
            break;
        case 3:
            run_sgemm_tiled_coarsed_pmpp(M, N, K, alpha, A, B, beta, C);
            break;
        case 4:
            run_sgemm_tiled_coarsed_blog(M, N, K, alpha, A, B, beta, C);
            break;
        default:
            throw std::invalid_argument("Unknown kernel number");
    }
}

int main(int argc, char **argv) {
    // print some device info
    // CudaDeviceInfo();

    // Declare the handle, create the handle, cublasCreate will return a value of
    // type cublasStatus_t to determine whether the handle was created
    // successfully (the value is 0)
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    }

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    
    std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};
    long m, n, k, max_size;
    max_size = SIZE[SIZE.size() - 1];
    std::cout << "Max size: " << max_size << std::endl;

    float alpha = 1.0, beta = 0.0; // GEMM input parameters, C=α*AB+β*C

    float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

    A = (float *)malloc(sizeof(float) * max_size * max_size);
    B = (float *)malloc(sizeof(float) * max_size * max_size);
    C = (float *)malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *)malloc(sizeof(float) * max_size * max_size);
    
    randomize_matrix(A, max_size*max_size);
    randomize_matrix(B, max_size*max_size);
    
    cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));

    for (int kernel_num = 1; kernel_num <= 4; kernel_num++) {
        std::cout << "=== Profiling kernel_num: " << kernel_num << " ===" << std::endl;
        for (int size : SIZE) {
            m = n = k = size; // Assuming sqaure matrices
            std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
                << ", beta: " << beta << std::endl;
            
            // cuBLAS reference for sanity check
            // run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);
            // cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

            // Run matmult kernel
            run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);

            // Copy results from GPU
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
            cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

            // Sanity check
            // if (!verify_matrix(C_ref, C, m*n)) {
            //     std::cout << "Failed to pass the correctness verification against NVIDIA "
            //            "cuBLAS." << std::endl;
            //     exit(EXIT_FAILURE);
            // }

            // For manual GFLOPS measurement. We let Nsight Compute do this.
            // cudaEventRecord(beg);
            // int repeat_times = 50;
            // for (int j = 0; j < repeat_times; j++) {
            //     // We don't reset dC between runs to save time
            //     run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
            // }
            // cudaEventRecord(end);
            // cudaEventSynchronize(beg);
            // cudaEventSynchronize(end);
            // cudaEventElapsedTime(&elapsed_time, beg, end);
            // elapsed_time /= 1000.; // Convert to seconds

            // long flops = 2 * m * n * k;
            // printf(
            //     "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
            //     "(%ld).\n",
            //     elapsed_time / repeat_times,
            //     (repeat_times * flops * 1e-9) / elapsed_time, m);
            // fflush(stdout);
            // // make dC and dC_ref equal again (we modified dC while calling our kernel
            // // for benchmarking)
            // cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n, 
            //                         cudaMemcpyDeviceToDevice));
        }
    }
    return 0;
}
