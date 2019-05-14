#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void increase(int *c, int N){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(tid < N)
     c[tid] = tid;
   }

__global__ void kernel0( int *a )
{
int idx = blockIdx.x*blockDim.x + threadIdx.x;
a[idx] = 7;
}

__global__ void kernel1( int *a )
{
int idx = blockIdx.x*blockDim.x + threadIdx.x;
a[idx] = blockIdx.x;
}

__global__ void kernel2( int *a )
{
int idx = blockIdx.x*blockDim.x + threadIdx.x;
a[idx] = threadIdx.x;
}

void printarray0(int *c, int n)
{
    for (int i = 0; i < n; i++) printf("c[%d] = %d \n ", i, c[i]);
}

void printarray1(int *a, int n)
{
    int i = 0;
    for (i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Args kurang. usage: blockgrid [data array length] [threadsPerBlock] [numBlocks]");
        return EXIT_SUCCESS;
    }
    int N = atoi(argv[1]);
    int N2 = N+5;
    int isize = N*sizeof(int);
    dim3 threadsPerBlock(atoi(argv[2]));
    dim3 numBlocks(atoi(argv[3]));

 int c[N2];
 int *dev_c;

// fungsi index incement
 printf("=== index thread ke isi array ===\n");
 cudaMalloc( (void**)&dev_c, N*sizeof(int) );
 for(int i=0; i< N; ++i)/* Untuk isi array dengan -1 */
 {
  c[i] = -1;
 }
 cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice);
 increase<<<numBlocks, threadsPerBlock>>>(dev_c,N);
 cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost );
 printarray0(c, N);
 cudaFree( dev_c );

 // function kernel0
 printf("=== program kernel0 (7) ===\n");
 int *a_h, *a_d;
 a_h = (int*) malloc(isize);
 cudaMalloc((void**)&a_d, isize);
 // printarray("before", a_h, N);
 kernel0<<<numBlocks, threadsPerBlock>>>(a_d);
 cudaMemcpy(a_h, a_d, isize, cudaMemcpyDeviceToHost);
 printarray1(a_h, N);
 cudaFree(a_d);
 free(a_h);

 // function kernel1
 printf("=== program kernel1 blockIdx.x ===\n");
 int *b_h, *b_d;
 b_h = (int*) malloc(isize);
 cudaMalloc((void**)&b_d, isize);
 // printarray("before", b_h, N);
 kernel1<<<numBlocks, threadsPerBlock>>>(b_d);
 cudaMemcpy(b_h, b_d, isize, cudaMemcpyDeviceToHost);
 printarray1(b_h, N);
 cudaFree(b_d);
 free(b_h);

 // function kernel2
 printf("=== program kernel2 threadIdx.x ===\n");
 int *c_h, *c_d;
 c_h = (int*) malloc(isize);
 cudaMalloc((void**)&c_d, isize);
 // printarray("before", c_h, N);
 kernel2<<<numBlocks, threadsPerBlock>>>(c_d);
 cudaMemcpy(c_h, c_d, isize, cudaMemcpyDeviceToHost);
 printarray1(c_h, N);
 cudaFree(c_d);
 free(c_h);

 return EXIT_SUCCESS;
}
