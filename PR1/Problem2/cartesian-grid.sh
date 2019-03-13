#!/bin/bash
name="cartesian-grid"
prog="$name.c"
exe="$name.o"
out="$name.result"
clear && mpicc -o $exe $prog -lm &&
mpirun -hostfile mpi_hostfile -np 2 $exe >> $out &&
mpirun -hostfile mpi_hostfile -np 4 $exe >> $out &&
mpirun -hostfile mpi_hostfile -np 8 $exe >> $out &&
mpirun -hostfile mpi_hostfile -np 16 $exe >> $out &&
mpirun -hostfile mpi_hostfile -np 32 $exe >> $out  
