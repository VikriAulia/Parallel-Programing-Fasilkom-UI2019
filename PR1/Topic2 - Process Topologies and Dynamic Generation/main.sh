#!/bin/bash
name="main1"
prog="$name.c"
exe="$name.o"
out="$name.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP\tData\tComm Time\tProcess Time" >> $out &&
mpirun -hostfile mpi_hostfile -np 5 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 10 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 25 $exe 360 >> $out 
#mpirun -hostfile mpi_hostfile -np 5 $exe 1152 >> $out &&
#mpirun -hostfile mpi_hostfile -np 10 $exe 1152 >> $out &&
#mpirun -hostfile mpi_hostfile -np 25 $exe 1152 >> $out 


