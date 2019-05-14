// incrementArray.cu

#include <stdio.h>
#include <assert.h>
#include <cuda.h>

void incrementArrayOnHost(float *a, int N)
{
    int i;
    for (i=0; i < N; i++) a[i] = a[i]+1.f;
}

__global__ void incrementArrayOnDevice(float *a, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx<N) a[idx] = a[idx]+1.f;
}

void printarray(float *a, int n)
{
    int i = 0;
    for (i = 0; i < n; i++) printf("%f ", a[i]);
    printf("\n");
}

int main(int argc, char** argv)
{
    /* init program args & global variable */
    if(argc < 2){
        printf("usage: incrementArray [threadPerBlock] [repetitions] [startSizeArray]\n");
        return EXIT_SUCCESS;
    }
    int threadPerBlock = atoi(argv[1]);
    int repetitions = atoi(argv[2]);
    int N = atoi(argv[3]);
    float *a_h, *b_h; // pointers to host memory
    float *a_d; // pointer to device memory
    int i,rep;
    
    int totalSuccess = 0;
    /* end init */

    /* looping from 0 to repettions */
    for (rep=0; rep<=repetitions;rep++){
    /* Start Looping */
    size_t size = N*sizeof(float);
    // allocate arrays on host
    a_h = (float *)malloc(size);
    b_h = (float *)malloc(size);
    // allocate array on device
    cudaMalloc((void **) &a_d, size);
    // initialization of host data
    for (i=0; i<N; i++) a_h[i] = (float)i;
    // copy data from host to device
    cudaMemcpy(a_d, a_h, sizeof(float)*N, cudaMemcpyHostToDevice);
    // do calculation on host
    incrementArrayOnHost(a_h, N);
    //printarray(a_h, N);
    // do calculation on device:
    // Part 1 of 2. Compute execution configuration
    int numBlocks = N/threadPerBlock + (N%threadPerBlock == 0?0:1);//data/threadPerBlock = JumlahBlock
    /* init nBlocks and blockSize */
    // dim3 threadPerBlock(threadPerBlock);
    // dim3 numBlocks(numBlocks);
    /* end init block */
    // Part 2 of 2. Call incrementArrayOnDevice kernel
    incrementArrayOnDevice <<<numBlocks,threadPerBlock>>> (a_d, N);
    // Retrieve result from device and store in b_h
    cudaMemcpy(b_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
    // print results
    //printarray(a_h, N);

    //for (i=0; i<N; i++) assert(a_h[i] == b_h[i]);
    /* start cek result */
    int success = 1;
        for (i=0; i<N; i++) {
            if (a_h[i] != b_h[i]) {
                success = 0;
                break;
            }
        }
    /* end end result */
    printf("rep %d a[%d] = %s >> numBlocks = %d ,threadPerBlock = %d\n", rep, N, (success == 1) ? "true" : "false", numBlocks, threadPerBlock);
        if (success == 1) totalSuccess += 1;
        N= N*2;;// double N size
	threadPerBlock = threadPerBlock * 2; //increse thread number
    }/* end looping */
    printf("\nsuccess rate: %f%%\n", totalSuccess / ((float)repetitions) * 100.0);
    // cleanup
    free(a_h); free(b_h); cudaFree(a_d);
    return EXIT_SUCCESS;
}

