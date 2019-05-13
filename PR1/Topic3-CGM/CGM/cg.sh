#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
exe="mpi_conjugate_gradient"
for np in {16..2..2}
do
n=512
n2=1024
n3=2048
data="data$n"
data2="data$n2"
data3="data$n3"
out="pcg.$np.$n.test"
out2="pcg.$np.$n.test2"
out3="pcg.$np.$n.test3"
clear &&
#mpirun -hostfile ~/mpi_hostfile -np 2 $exe 512 0.000001 1000 < $data >> $out &&
#mpirun -hostfile ~/mpi_hostfile -np 3 $exe 512 0.000001 1000 < $data >> $out &&
#mpirun -hostfile ~/mpi_hostfile -np 4 $exe 512 0.000001 1000 < $data >> $out &&
#mpirun -hostfile ~/mpi_hostfile -np 5 $exe 512 0.000001 1000 < $data >> $out &&
#mpirun -hostfile ~/mpi_hostfile -np 6 $exe 512 0.000001 1000 < $data >> $out &&
#mpirun -hostfile ~/mpi_hostfile -np 7 $exe 512 0.000001 1000 < $data >> $out &&
#mpirun -hostfile mpi_hostfile -np $np $exe $n 0.0000000000000001 1000 < $data >> $out
mpirun -hostfile mpi_hostfile -np $np $exe $n 0.0000000000000001 1000 < $data >> $out &&
mpirun -hostfile mpi_hostfile -np $np $exe $n 0.0000000000000001 1000 < $data >> $out2 &&
mpirun -hostfile mpi_hostfile -np $np $exe $n 0.0000000000000001 1000 < $data >> $out3 &&
mpirun -hostfile mpi_hostfile -np $np $exe $n 0.0000000000000001 1000 < $data2 >> $out &&
mpirun -hostfile mpi_hostfile -np $np $exe $n 0.0000000000000001 1000 < $data2 >> $out2 &&
mpirun -hostfile mpi_hostfile -np $np $exe $n 0.0000000000000001 1000 < $data2 >> $out3 &&
mpirun -hostfile mpi_hostfile -np $np $exe $n 0.0000000000000001 1000 < $data3 >> $out &&
mpirun -hostfile mpi_hostfile -np $np $exe $n 0.0000000000000001 1000 < $data3 >> $out2 &&
mpirun -hostfile mpi_hostfile -np $np $exe $n 0.0000000000000001 1000 < $data3 >> $out3
done
