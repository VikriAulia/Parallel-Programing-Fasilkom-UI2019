#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main (int argc, char * argv[]){
	int i,j,size,rank;
	char hostname[1024];
	MPI_Status status;
	hostname[1023] = '\0';
	
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	gethostname(hostname, 1023);
	printf("Hostname : %s\n", hostname);

MPI_Finalize();
return EXIT_SUCCESS;
}
