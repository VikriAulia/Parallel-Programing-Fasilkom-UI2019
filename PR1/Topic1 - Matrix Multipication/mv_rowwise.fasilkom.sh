#!/bin/bash
name="mv_rowwise"
prog="$name.c"
exe="$name.o"
out="$name.fasilkom.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP\tData\tComm Time\tProcess Time" > $out &&
mpirun -np 3 --hostfile mpi_hostfile $exe 1152 >> $out &&
mpirun -np 3 --hostfile mpi_hostfile $exe 1152 >> $out &&
mpirun -np 3 --hostfile mpi_hostfile $exe 1152 >> $out &&
mpirun -np 3 --hostfile mpi_hostfile $exe 1152 >> $out &&
mpirun -np 3 --hostfile mpi_hostfile $exe 1152 >> $out &&
mpirun -np 5 --hostfile mpi_hostfile $exe 1152 >> $out &&
mpirun -np 5 --hostfile mpi_hostfile $exe 1152 >> $out &&
mpirun -np 5 --hostfile mpi_hostfile $exe 1152 >> $out &&
mpirun -np 5 --hostfile mpi_hostfile $exe 1152 >> $out &&
mpirun -np 5 --hostfile mpi_hostfile $exe 1152 >> $out
