#!/bin/bash
#SBATCH --job-name=CS453_WEIGHTED #the name of your job

#change to your NAUID and file path
#SBATCH --output=/scratch/ap2974/"CS 453"/Generator/wng.out #this is the file for stdout 
#SBATCH --error=/scratch/ap2974/"CS 453"/Generator/wng.err #this is the file for stderr

#SBATCH --time=00:01:00		#Job timelimit is 3 minutes
#SBATCH --mem=5000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C v100 # k80 #GPU Model: k80, p100, v100, a100
#SBATCH --account=cs453-spr24			

#compute capability
CC=70 #K80: 37, P100: 60, V100: 70, A100: 80

#make sure not to redefine MODE, N, etc. in the source file

if [[ $CC -eq 37 ]] 
then
	module load cuda/11.7
else
	module load cuda
fi

nvcc -O3 -arch=compute_$CC -code=sm_$CC -lcuda -lineinfo -Xcompiler -fopenmp weighted_nodes_word_generator.cu -o weighted_nodes

srun ./weighted_nodes