#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

extern "C" {
	__declspec(dllexport) const char* matrixToString(const float* A, int m, int n);

	__declspec(dllexport) void matrixMultiplyGPU(const float* A, const float* B, float* C, int m, int n, int k);

	__declspec(dllexport) void matrixMultiplyCPU(const float* A, const float* B, float* C, int m, int n, int k);
}
