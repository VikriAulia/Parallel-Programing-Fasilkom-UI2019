#!/bin/bash
name="mm_rowwise"
prog="$name.c"
exe="$name.o"
proc=16
out="$name.$proc.result"
echo "NP------Data----Comm Time------Process Time" > $out &&
clear && mpicc -o $exe $prog -lm &&
mpirun -hostfile mpi_hostfile -np $proc $exe 320 320 320 >> $out &&
mpirun -hostfile mpi_hostfile -np $proc $exe 576 576 576 >> $out &&
mpirun -hostfile mpi_hostfile -np $proc $exe 800 800 800 >> $out
