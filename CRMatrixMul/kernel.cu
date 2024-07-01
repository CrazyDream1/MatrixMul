#include "kernel.cuh"

const char* matrixToString(const float* A, int m, int n)
{
    std::ostringstream oss;
    oss << "Matrix:\n";
    oss << std::fixed << std::setprecision(5);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float value = A[i + m * j];
            oss << value << " ";
        }
        oss << "\n";
    }
    std::string result = oss.str();
    return result.c_str();
}

void matrixMultiplyGPU(const float* A, const float* B, float* C, int m, int n, int k) {
    cudaError_t cudaStatus;
    cublasStatus_t cublasStat;
    float *d_A, *d_B, *d_C;
    cublasHandle_t handle;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!\n");
        goto Error;
    }

    cublasStat = cublasCreate(&handle);
    if (cublasStat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_A, m * k * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_B, k * n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_C, m * n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    cublasStat = cublasSetMatrix(m, k, sizeof(float), A, m, d_A, m);
    if (cublasStat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS set matrix failed\n");
        goto Error;
    }
    cublasStat = cublasSetMatrix(k, n, sizeof(float), B, k, d_B, k);
    if (cublasStat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS set matrix failed\n");
        goto Error;
    }

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    if (cublasStat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS matrix multiplication failed\n");
        goto Error;
    }

    cublasStat = cublasGetMatrix(m, n, sizeof(float), d_C, m, C, m);
    if (cublasStat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS get matrix failed\n");
        goto Error;
    }
    cudaDeviceSynchronize();

Error:
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

void matrixMultiplyCPU(const float* A, const float* B, float* C, int m, int n, int k)
{
    for (int i = 0; i < m * n; ++i) {
        C[i] = 0.0f;
    }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            for (int p = 0; p < k; ++p) {
                C[i + j * m] += A[i + p * m] * B[p + j * k];
            }
        }
    }
}
