#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
name="rowwise"
prog="$name.c"
exe="$name.o"
out="$name-040319.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP	Data	Comm Time	Proccess Time" > $out &&
mpirun -hostfile mpi_hostfile -np 10 $exe 500 500 500 >> $out &&
mpirun -hostfile mpi_hostfile -np 16 $exe 500 500 500 >> $out &&
mpirun -hostfile mpi_hostfile -np 24 $exe 500 500 500 >> $out &&
mpirun -hostfile mpi_hostfile -np 10 $exe 800 800 800 >>  $out &&
mpirun -hostfile mpi_hostfile -np 16 $exe 800 800 800 >> $out &&
mpirun -hostfile mpi_hostfile -np 24 $exe 800 800 800 >> $out  
