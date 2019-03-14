#!/bin/bash
name="columnwise"
prog="$name.c"
exe="$name.o"
out="$name-040319.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP	Data	Comm Time	Process Time" >> $out &&
mpirun -hostfile mpi_hostfile -np 10 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 16 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 24 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 32 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 10 $exe 1152 >> $out &&
mpirun -hostfile mpi_hostfile -np 16 $exe 1152 >> $out &&
mpirun -hostfile mpi_hostfile -np 24 $exe 1152 >> $out &&
mpirun -hostfile mpi_hostfile -np 32 $exe 1152 >> $out


