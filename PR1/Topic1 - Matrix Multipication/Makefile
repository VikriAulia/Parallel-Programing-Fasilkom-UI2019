.PHONY: all build clean run scp-to-kawung

all: clean build run scp-to-kawung

clean:
	rm -rf row_wise
	rm -rf checker_board 
	rm -rf canon 
	rm -rf fox
	rm -rf column_wise

build:
	mpicc -o row_wise row_wise.c
	mpicc -o checker_board checker_board.c 
	mpicc -o canon canon.c
	mpicc -o fox fox.c
	mpicc -o column_wise column_wise.c

run:
	mpirun ./row_wise 360 360 360
scp-to-kawung:
	sshpass -p d1d13ks4y4d1? scp ../assignment-1.zip didi.yudha@kawung.cs.ui.ac.id:/home/fasilkom/mahasiswa/d/didi.yudha/parallel-programming/assignment-1