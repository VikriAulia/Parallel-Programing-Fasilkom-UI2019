#!/bin/bash
name="mm_cannon"
prog="$name.c"
exe="$name.o"
out="$name.hasil.fasilkom"
clear && mpicc -o $exe $prog -lm &&
echo "NP\tData\tComm Time\tProcess Time" > $out &&
mpirun -np 2 $exe 512 >> $out &&
mpirun -np 2 $exe 512 >> $out &&
mpirun -np 4 $exe 512 >> $out &&
mpirun -np 4 $exe 512 >> $out &&
mpirun -np 8 $exe 512 >> $out &&
mpirun -np 8 $exe 512 >> $out &&
mpirun -np 16 $exe 512 >> $out &&
mpirun -np 16 $exe 512 >> $out &&
mpirun -np 24 $exe 512 >> $out &&
mpirun -np 24 $exe 512 >> $out &&
mpirun -np 32 $exe 512 >> $out &&
mpirun -np 32 $exe 512 >> $out &&
mpirun -np 2 $exe 1024 >> $out &&
mpirun -np 2 $exe 1024 >> $out &&
mpirun -np 4 $exe 1024 >> $out &&
mpirun -np 4 $exe 1024 >> $out &&
mpirun -np 8 $exe 1024 >> $out &&
mpirun -np 8 $exe 1024 >> $out &&
mpirun -np 16 $exe 1024 >> $out &&
mpirun -np 16 $exe 1024 >> $out &&
mpirun -np 24 $exe 1024 >> $out &&
mpirun -np 24 $exe 1024 >> $out &&
mpirun -np 32 $exe 1024 >> $out &&
mpirun -np 32 $exe 1024 >> $out
