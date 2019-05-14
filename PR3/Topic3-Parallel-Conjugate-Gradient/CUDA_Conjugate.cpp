/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <sys/time.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h> // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>	  // helper function CUDA error checking and initialization

const char *sSDKname = "conjugateGradient";

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
	I[0] = 0, J[0] = 0, J[1] = 1;
	val[0] = (float)rand() / RAND_MAX + 10.0f;
	val[1] = (float)rand() / RAND_MAX;
	int start;

	for (int i = 1; i < N; i++)
	{
		if (i > 1)
		{
			I[i] = I[i - 1] + 3;
		}
		else
		{
			I[1] = 2;
		}

		start = (i - 1) * 3 + 2;
		J[start] = i - 1;
		J[start + 1] = i;

		if (i < N - 1)
		{
			J[start + 2] = i + 1;
		}

		val[start] = val[start - 1];
		val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

		if (i < N - 1)
		{
			val[start + 2] = (float)rand() / RAND_MAX;
		}
	}

	I[N] = nz;
}

int main(int argc, char **argv)
{
	int nz = 0, *I = NULL, *J = NULL;
	float *val = NULL;
	const float tol = 1e-16f;
	const int max_iter = 10000;
	float *x;
	float *rhs;
	float a, b, na, r0, r1;
	int *d_col, *d_row;
	float *d_val, *d_x, dot;
	float *d_r, *d_p, *d_Ax;
	int k;
	float alpha, beta, alpham1;
	int ret_code;
	int M = atoi(argv[1]), N = M; // Get N & M from args1
	double elapsedTime, elapsedTime2;
	timeval start, end, end2;
	FILE *f;

	// This will pick the best possible CUDA capable device
	cudaDeviceProp deviceProp;
	//int devID = findCudaDevice(argc, (const char **)argv);
	int devID = 1;

	if (devID < 0)
	{
		printf("exiting...\n");
		exit(EXIT_SUCCESS);
	}

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	// Statistics about the GPU device
	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
		   deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	int version = (deviceProp.major * 0x10 + deviceProp.minor);

	printf("version %d\n", version);

	if (version < 0x11)
	{
		printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);

		// cudaDeviceReset causes the driver to clean up all state. While
		// not mandatory in normal operation, it is good practice.  It is also
		// needed to ensure correct operation when the application is being
		// profiled. Calling cudaDeviceReset causes all profile data to be
		// flushed before the application exits
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
	/* Start get input from file */

	// double B[M][N];
	// double C[M];
	double **B, *C;
	if ((B = (double **)malloc(sizeof(double *) * M)) == NULL)
	{
		std::cout << "Error matrix B[M]" << std::endl;
	}
	for (int i = 0; i < M; i++)
	{
		if ((B[i] = (double *)malloc(sizeof(double) * N)) == NULL)
		{
			std::cout << "Error matrix B[N]" << std::endl;
		}
	}
	if ((C = (double *)malloc(sizeof(double) * M)) == NULL)
	{
		std::cout << "Error matrix C[N]" << std::endl;
	}

	std::ifstream matrixfile("matrix");
	if (!(matrixfile.is_open()))
	{
		std::cout << "Error: file not found" << std::endl;
		return 0;
	}
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			matrixfile >> B[i][j];
		}
	}
	for (int i = 0; i < M; i++)
	{
		matrixfile >> C[i];
	}
	matrixfile.close();
	/* End get file matix */
	/* Start Dense to CSR */
	double *d_B;
	cudaMallocManaged(&d_B, M * N * sizeof(double));
	cudaMemcpy(d_B, B, M * N * sizeof(double), cudaMemcpyDefault);

	int *RowNonzero;
	int *CsrRowPtr;
	int *CsrColInd;
	double *CsrVal;
	int totalNnz;

	cudaMallocManaged(&RowNonzero, N * sizeof(int));
	cudaMallocManaged(&CsrRowPtr, (N + 1) * sizeof(int));

	cusparseHandle_t handle;
	cusparseCreate(&handle);

	cusparseMatDescr_t descr;
	cusparseCreateMatDescr(&descr);

	cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descr, d_B, N, RowNonzero, &totalNnz);
	cudaMallocManaged(&CsrColInd, totalNnz * sizeof(int));
	cudaMallocManaged(&CsrVal, totalNnz * sizeof(double));

	cusparseDdense2csr(handle, M, N, descr, d_B, N, RowNonzero, CsrVal, CsrRowPtr, CsrColInd);
	cudaDeviceSynchronize();

	nz = totalNnz;
	I = (int *)malloc(sizeof(int) * (N + 1));
	J = (int *)malloc(sizeof(int) * nz);
	val = (float *)malloc(sizeof(float) * nz);

	// std::cout << "totalNnz : " << totalNnz << std::endl;
	std::cout << "Copy Val:    ";
	for (int i = 0; i < nz; ++i)
	{
		// std::cout << std::setw(4) << CsrVal[i];
		val[i] = CsrVal[i];
	}
	std::cout << "\n Copy ColInd to J: ";
	for (int i = 0; i < nz; ++i)
	{
		// std::cout << std::setw(4) << CsrColInd[i];
		J[i] = CsrColInd[i];
	}
	std::cout << "\nCopy RowPtr to I: ";
	for (int i = 0; i < N + 1; ++i)
	{
		// std::cout << std::setw(4) << CsrRowPtr[i];
		I[i] = CsrRowPtr[i];
	}
	std::cout << std::endl;

	
	cusparseDestroyMatDescr(descr);
	cusparseDestroy(handle);
	cudaFree(d_B);
	cudaFree(CsrColInd);
	cudaFree(RowNonzero);
	cudaFree(CsrRowPtr);
	cudaFree(CsrVal);

	cudaDeviceReset();
	/* End Dense to CSR */
	/* Generate a random tridiagonal symmetric matrix in CSR format */
	//M = N = 1048576;
	// nz = (N - 2) * 3 + 4;
	// I = (int *)malloc(sizeof(int) * (N + 1));
	// J = (int *)malloc(sizeof(int) * nz);
	// val = (float *)malloc(sizeof(float) * nz);
	// genTridiag(I, J, val, N, nz);

	// std::cout << "totalNnz : " << nz << std::endl;
	// std::cout << " Val:    ";
	// for (int i = 0; i < nz; ++i)
	// {
	// 	 std::cout << std::setw(4) << val[i] <<" ";

	// }
	// std::cout << "\n  value J: ";
	// for (int i = 0; i < nz; ++i)
	// {
	// 	std::cout << std::setw(4) <<  J[i];

	// }
	// std::cout << "\nVallue I: ";
	// for (int i = 0; i < N + 1; ++i)
	// {
	// 	std::cout << std::setw(4) << I[i];

	// }
	// std::cout << std::endl;

	x = (float *)malloc(sizeof(float) * N);
	rhs = (float *)malloc(sizeof(float) * N);

	for (int i = 0; i < N; i++)
	{
		rhs[i] = C[i];
		x[i] = 0.0;
	}

	printf("create rhs & x\n");

	// /* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);

	printf("Get handle to the CUBLAS context\n");

	checkCudaErrors(cublasStatus);

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);

	printf("Get handle to the CUSPARSE context\n");

	checkCudaErrors(cusparseStatus);

	// cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);

	checkCudaErrors(cusparseStatus);

	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	printf("CUSPARSE config\n");

	checkCudaErrors(cudaMalloc((void **)&d_col, nz * sizeof(int)));
	printf("cuda malloc d_col\n");
	//checkCudaErrors(cudaMalloc((void **)&d_row, (N + 1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, nz * sizeof(int)));
	printf("cuda malloc d_row\n");
	checkCudaErrors(cudaMalloc((void **)&d_val, nz * sizeof(float)));
	printf("cuda malloc d_val\n");
	checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(float)));
	printf("cuda malloc d_x\n");
	checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(float)));
	printf("cuda malloc d_r\n");
	checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(float)));
	printf("cuda malloc d_p\n");
	checkCudaErrors(cudaMalloc((void **)&d_Ax, N * sizeof(float)));
	printf("cuda malloc d_Ax\n");

	cudaMemcpy(d_col, J, nz * sizeof(int), cudaMemcpyHostToDevice);
	printf("cuda memcpy d_col J");
	//cudaMemcpy(d_row, I, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, nz * sizeof(int), cudaMemcpyHostToDevice);
	printf("cuda memcpy d_row I");
	cudaMemcpy(d_val, val, nz * sizeof(float), cudaMemcpyHostToDevice);
	printf("cuda memcpy d_val val");
	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	printf("cuda memcpy d_x x");
	cudaMemcpy(d_r, rhs, N * sizeof(float), cudaMemcpyHostToDevice);
	printf("cuda memcpy d_r rhs");

	alpha = 1.0;
	alpham1 = -1.0;
	beta = 0.0;
	r0 = 0.;

	cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);

	cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
	cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

	k = 1;

	printf("cuda cg preparation\n");
	gettimeofday(&start, NULL); //start count time
	while (r1 > tol * tol && k <= max_iter)
	{
		printf("ITER %d", k);
		if (k > 1)
		{
			b = r1 / r0;
			cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
			cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
		}
		else
		{
			cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
		}

		cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
		cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
		a = r1 / dot;

		cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
		na = -a;
		cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

		r0 = r1;
		cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
		cudaThreadSynchronize();
		printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
		k++;
	}
	gettimeofday(&end, NULL); //finish count time
	cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

	float rsum, diff, err = 0.0;

	for (int i = 0; i < N; i++)
	{
		rsum = 0.0;

		for (int j = I[i]; j < I[i + 1]; j++)
		{
			rsum += val[j] * x[J[j]];
		}

		diff = fabs(rsum - rhs[i]);

		if (diff > err)
		{
			err = diff;
		}
	}

	/* start output */
	//Output time and iterations
	elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
	elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
	std::cout << "Time: " << elapsedTime << " ms." << std::endl
			  << std::endl;

	// print Output the matrixes for debug

	for (int i = 0; i < N; i++)
	{
		std::cout << x[i] << std::endl;
	}

	//Generate files for debug purpouse

	// ofstream Af;
	// Af.open("gpu");
	// for (int i = 0; i < N; i++)
	// {
	// 	Af << x[i] << std::endl;
	// }
	// Af.close();

	/* end output */

	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);
	for (int i = 0; i < M; i++)
	{
		free(B[i]);
	}
	free(B);
	free(C);
	free(I);
	free(J);
	free(val);
	free(x);
	free(rhs);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ax);

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	printf("Test Summary:  Error amount = %f\n", err);
	exit((k <= max_iter) ? 0 : 1);
}
