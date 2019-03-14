#!/bin/bash
#
#$ -cwd $ -j y $ -S /bin/bash
#
name="cartesian-cannon"
prog="$name.c"
exe="$name.o" 
out="$name.result"
clear && mpicc -o $exe $prog -lm &&
echo "NP	Data	Comm Time	Process Time" > $out &&
mpirun -hostfile mpi_hostfile -np 2 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 2 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 2 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 2 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 2 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 4 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 4 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 4 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 4 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 4 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 8 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 8 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 8 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 8 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 8 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 16 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 16 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 16 $exe 360 >> $out &&
mpirun -hostfile mpi_hostfile -np 16 $exe 360 >> $out
