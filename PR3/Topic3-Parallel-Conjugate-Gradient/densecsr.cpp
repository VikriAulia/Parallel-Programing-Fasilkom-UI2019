    #include<cuda_runtime.h>
    #include<cusparse_v2.h>

    #include <iostream>
    #include <iomanip>

    const int N = 6; 
    const int M = 6;

     int main() {

       double B[M][N] = {
         { 10.0,  0.0,  0.0,  0.0, -2.0,  0.0 },
         {  3.0,  9.0,  0.0,  0.0,  0.0,  3.0 },
         {  0.0,  7.0,  8.0,  7.0,  0.0,  0.0 },
         {  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 },
         {  0.0,  8.0,  0.0,  9.0,  9.0, 13.0 },
         {  0.0,  4.0,  0.0,  0.0,  2.0, -1.0 }
       }; 

      double *d_B;

      cudaMallocManaged( &d_B, M*N*sizeof(double));
      cudaMemcpy(d_B, B, M*N*sizeof(double), cudaMemcpyDefault);

      int *RowNonzero; 
      int *CsrRowPtr; 
      int *CsrColInd; 
      double *CsrVal; 
      int totalNnz; 

      cudaMallocManaged(&RowNonzero, N*sizeof(int));
      cudaMallocManaged(&CsrRowPtr, (N+1)*sizeof(int));

      cusparseHandle_t handle;
      cusparseCreate(&handle);

      cusparseMatDescr_t descr;
      cusparseCreateMatDescr(&descr);

      cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descr, d_B, N, RowNonzero, &totalNnz);
      cudaMallocManaged(&CsrColInd, totalNnz*sizeof(int));
      cudaMallocManaged(&CsrVal, totalNnz*sizeof(double));

      cusparseDdense2csr(handle, M, N, descr, d_B, N, RowNonzero, CsrVal, CsrRowPtr, CsrColInd); 
      cudaDeviceSynchronize();

      std::cout << "totalNnz : " << totalNnz << std::endl;
      std::cout << "Val:    ";
      for ( int i = 0; i < totalNnz; ++i ) {
        std::cout << std::setw(4) << CsrVal[i];
      }
      std::cout << "\nColInd: ";
      for ( int i = 0; i < totalNnz; ++i ) {
        std::cout << std::setw(4) << CsrColInd[i];
      }
      std::cout << "\nRowPtr: ";
      for ( int i = 0; i < N+1; ++i ) {
        std::cout << std::setw(4) << CsrRowPtr[i];
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
    }