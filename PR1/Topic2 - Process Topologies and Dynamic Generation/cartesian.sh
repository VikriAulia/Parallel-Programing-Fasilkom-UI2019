#!/bin/bash
name="cartesian"
prog="$name.c"
exe="$name.o"
out="$name.result"
clear && mpicc -o $exe $prog -lm &&
mpirun -hostfile mpi_hostfile -np 4 $exe >> $out 
