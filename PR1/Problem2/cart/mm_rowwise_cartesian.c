
#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

#define MASTER 0       /* id of the first process */
#define FROM_MASTER 1  /* setting a message type */
#define FROM_WORKER 2  /* setting a message type */

MPI_Status status;

// void printmatrix(int row, int col, int **matrix)
void printmatrix(int row, int col, int matrix[row][col])
{
    int i, j = 0;
    for(i = 0; i < row; i++)
    {
        for(j = 0; j < col; j++)
        {
            printf("%d\t",matrix[i][j]);
        }
        printf("\n");
    }
}

void multiply_two_arrays(int NRA, int NCA, int NCB) {

    int source,irank,        /* process id of message source */
    dest,           /* process id of message destination */
    nbytes,         /* number of bytes in message */
    mtype,          /* message type */
    rows,           /* rows of A sent to each worker */
    averow, extra, offset,
    i, j, k, count;

    int a[NRA][NCA],   /* matrix A to be multiplied */
    b[NCA][NCB],   /* matrix B to be multiplied */
    c[NRA][NCB];   /* result matrix C */
    // int **a, **b, **c;
    double texec, tcomm = 0;
    int numprocs,   /* number of processes in partition */
    procid,         /* a process identifier */
    numworkers;     /* number of worker processes */
    int dims[2];      /* Dimensions of the grid */
    int coords[2];    /* Coordinates in the grid */
    int neighbors[4]; /* Neighbors in 2D grid */
    int period[2] = {1, 1};
    double exec_time = 0;
    MPI_Comm comm2d;


    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    numworkers = numprocs-1;
    exec_time -= MPI_Wtime();
/*check dimension */
    if (numprocs < 16) {
        dims[0] = 2;
    } else if (numprocs >= 16 && numprocs < 64) {
        dims[0] = 4;
    } else if (numprocs >= 64 && numprocs < 256) {
        dims[0] = 8;
    } else {
        dims[0] = 16;
    }
    dims[1] = numprocs / dims[0];

    if (dims[0] * dims[1] != numprocs) {
        fprintf(stderr, "Incompatible dimensions: %i x %i != %i\n",
                dims[0], dims[1], numprocs);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }


/* end check dimension
    dims[0] = 4;
    dims[1] = numprocs / dims[0];*/
    /* Create the 2D Cartesian communicator */
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 1, &comm2d); 
    /* Find out and store the ranks with which to perform halo exchange */
    MPI_Cart_shift(comm2d, 0, 1, &neighbors[0], &neighbors[1]);
    MPI_Cart_shift(comm2d, 1, 1, &neighbors[2], &neighbors[3]); 
    /* Find out and store also the Cartesian coordinates of a rank */
    MPI_Cart_coords(comm2d, procid, 2, coords);

    /******* master process ***********/
    if (procid == MASTER) {

        // inits
        for (i=0; i<NRA; i++)
            for (j=0; j<NCA; j++)
                a[i][j]= 1;
        // printmatrix(NRA, NCA, a);

        for (i=0; i<NCA; i++)
            for (j=0; j<NCB; j++)
                b[i][j]= 2;
        // printmatrix(NCA, NCB, b);

        /* send matrix data to the worker processes */
        averow = NRA/numworkers;
        extra = NRA%numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        texec -= MPI_Wtime();
        for (dest=1; dest<=numworkers; dest++) {
            rows = (dest <= extra) ? averow+1 : averow;
            tcomm -= MPI_Wtime();
            MPI_Send(&offset,1,MPI_INT,dest,mtype,comm2d);
            MPI_Send(&rows,1,MPI_INT,dest,mtype,comm2d);
            count = rows*NCA;
            MPI_Send(&a[offset][0],count,MPI_INT,dest,mtype,comm2d);
            count = NCA*NCB;
            MPI_Send(&b,count,MPI_INT,dest,mtype,comm2d);
            tcomm += MPI_Wtime();
            offset = offset + rows;
        }

        /* wait for results from all worker processes */
        mtype = FROM_WORKER;
        for (i=1; i<=numworkers; i++) {
            source = i;
            tcomm -= MPI_Wtime();
            MPI_Recv(&offset,1,MPI_INT,source,mtype,comm2d, &status);
            MPI_Recv(&rows,1,MPI_INT,source,mtype,comm2d, &status);
            count = rows*NCB;
            MPI_Recv(&c[offset][0],count,MPI_INT,source,mtype,comm2d, &status);
            tcomm += MPI_Wtime();
        }

        // printmatrix(NRA, NCB, c);

        texec += MPI_Wtime();
	printf("\nNP------Data----Comm Time------Process Time\n");
        printf("%d\t%d\t%lf\t%lf\n", numworkers+1, NRA, tcomm, texec);

    } /* end of master */

    /************ worker process *************/
    if (procid > MASTER) {

        mtype = FROM_MASTER;
        source = MASTER;

        MPI_Recv(&offset,1,MPI_INT,source,mtype,comm2d,&status);
        MPI_Recv(&rows,1,MPI_INT,source,mtype,comm2d,&status);
        count = rows*NCA;
        MPI_Recv(&a,count,MPI_INT,source,mtype,comm2d,&status);
        count = NCA*NCB;
        MPI_Recv(&b,count,MPI_INT,source,mtype,comm2d,&status);

        for (k=0; k<NCB; k++) {       /* multiply our part */
            for (i=0; i<rows; i++) {
                c[i][k] = 0.0;
                for (j=0; j<NCA; j++){
                    c[i][k] = c[i][k] + a[i][j] * b[j][k];
                }
            }
        }

        mtype = FROM_WORKER;
        MPI_Send(&offset,1,MPI_INT,MASTER,mtype,comm2d);
        MPI_Send(&rows,1,MPI_INT,MASTER,mtype,comm2d);
        MPI_Send(&c,rows*NCB,MPI_INT,MASTER,mtype,comm2d);

    }

    MPI_Barrier(comm2d);
    MPI_Finalize();
   for (irank = 0; irank < numprocs; irank++) {
        if (procid == irank) {
            printf("Rank =%3i => Coordinate (%2i,%2i) neighbors =%3i %3i %3i %3i\n",
                  procid, coords[0], coords[1], neighbors[0], neighbors[1],
                   neighbors[2], neighbors[3]);
        }}
/*    if(procid == 0){
        exec_time += MPI_Wtime();
        printf("np :%d time : %f", numprocs, exec_time);
    }
*/
}

int main(int argc, char **argv) {

    int NRA, NCA, NCB;
    NRA = atoi(argv[1]);
    NCA = atoi(argv[2]);
    NCB = atoi(argv[3]);
    MPI_Init(&argc, &argv);
    multiply_two_arrays(NRA, NCA, NCB);

    return 0;
}
