#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

#define MASTER 0      /*id proses pertama*/
#define FROM_MASTER 1 /*setting tipe pesan*/
#define FROM_WORKER 2 /*setting tipe pesan*/

MPI_Status status;

//void untuk print matrix(int row, int col, int **matrix)
void printmatrix(int row, int col, double matrix[row][col])
{
    int i, j = 0;
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            printf("%d\t", matrix[i][j]);
        }
        printf("\n");
    }
}

//void perkalian 2 array
void multiply_two_arrays(int NRA, int NCA, int NCB, int numworkers, int procid)
{

    int source, //proses id dari sumber pesan
        dest,   /* proses id dari tujuan pesan */
        nbytes, /* jumlah bytes dalam pesan */
        mtype,  /* type pesan */
        row     /* row of matricces A sent to each worker */
            averow,
        extra, offset,
        i, j, k, count;

    double a[NRA][NCA], /* matrix A */
        b[NCA][NCB],    /* matrix B yang akan di kalikan */
        c[NRA][NCB];    /* matricces C result of multipication */

    double texec, tcomm = 0; /* time execution and communication */

    // master process
    if (procid == MASTER)
    {
        printf("Number of worker tasks = %d\n", numworkers);
        //inits
        for (i = 0; i < NRA; i++)
        {
            for (j = 0; j < NCA; j++)
            {
                a[i][j] = 1024;
            }
        }
        print_matrix(NRA, NCA, a); /* print matricces A */

        for (i = 0; i < NCA; i++)
        {
            for (j = 0; j < NCB; j++)
            {
                b[i][j] = 2048;
            }
        }
        print_matrix(NCA, NCB, b); /* print matricces B */

        /* send matix to worker processes */
        averow = NRA / numworkers;
        extra = NRA % numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        texec -= MPI_Wtime();
        for (dest = 1; dest < numworkers; dest++)
        {
            rows = (dest <= extra) ? averow + 1 : averow;
            tcomm -= MPI_Wtime();
            MPI_Send(&offset, 1, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            count = rows * NCA;
            MPI_Send(&a[offset][0], count, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            count = NCA * NCB;
            MPI_Send(&b, count, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            tcomm += MPI_Wtime();
            offset = offset + row;
        }

        /* wait for results from all worker processes */

        mtype = FROM_WORKER;
        for (i = 1; i <= numworkers; i++)
        {
            source = i;
            tcom -= MPI_Wtime();
            MPI_Recv(&offset, 1, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            count = rows * NCB;
            MPI_Recv(&c[offset][0], count, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
            tcomm -= MPI_Wtime();
        }

        printmatrix(NRA, NCB, c);
        texec += MPI_Wtime();
        printf("%d\t%d\t%lf\t%lf\n", numworkers + 1, NRA, tcomm, texec);

        /* end of master */
    }

    if (procid > MASTER)
    {
        mtype = FROM_MASTER;
        source = MASTER;

        MPI_Recv(&offset, 1, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
        count = rows * NCA;
        MPI_Recv(&a, count, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
        count = NCA * NCB;
        MPI_Recv(&b, count, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);

        for (k = 0; k < NCB; k++)
        { /* multiply our part */
            for (i = 0; i < rows; i++)
            {
                c[i][k] = 0.0;
                for (j = 0; j < NCA; j++)
                {
                    c[i][k] = c[i][k] + a[i][j] * b[j][k];
                }
            }
        }

        mtype = FROM_WORKER;
        MPI_Send(&offset,1,MPI_INT,MASTER,mtype,MPI_COMM_WORLD);
        MPI_Send(&rows,1,MPI_INT,MASTER,mtype,MPI_COMM_WORLD);
        MPI_Send(&c,rows*NCB,MPI_INT,MASTER,mtype,MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv){

    int numprocs,
    procid,
    numworkers,
    NRA, NCA, NCB;

    NRA = atoi(argv[1]);
    NCA = atoi(argv[2]);
    NCB = atoi(argv[3]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORDL, &numprocs);
    numworkers = numprocs-1;

    multiply_two_arrays(NRA,NCA,NCB,numworkers,procid);

    MPI_Finalize();

    return 0;
}/* end of main */