#!/bin/bash
#SBATCH --job-name=CS453_a6_NNI  #the name of your job

#change to your NAUID and file path
#SBATCH --output=/scratch/nrs285/CS453_a6_NNI.out #this is the file for stdout 
#SBATCH --error=/scratch/nrs285/CS453_a6_NNI.err #this is the file for stderr

#SBATCH --time=00:10:00		#Job timelimit is 3 minutes
#SBATCH --mem=10000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C a100 # k80 #GPU Model: k80, p100, v100, a100
#SBATCH --account=cs453-spr24	
##SBATCH --reservation=cs453-spr24-res					

#compute capability
CC=80 #K80: 37, P100: 60, V100: 70, A100: 80

# source file + executable name
NAME=CS453_a6_NNI

# program defined constants

#N=7490
SEQLENGTH=5
NUMLAYERS=2
HIDDENSIZE=512
MINIBATCH=5


#make sure not to redefine MODE, N, etc. in the source file

if [[ $CC -eq 37 ]] 
then
	module load cuda/11.7
else
	module load cuda
fi

module load cudnn/8.9.3

# Delete previous .exe so if compilation fails the job ends
#\rm $NAME.exe

for MODE in 1 # 2 3 4 5 6 7 8 
do
	for WIDTH in 90
	do
		for HEIGHT in 107
		do
			nvcc -O3 -arch=compute_$CC -code=sm_$CC -lcuda -lcudnn -lineinfo -Wno-deprecated-gpu-targets $NAME.cu -o $NAME.exe
		
			#3 time trials
			for i in 1 #2 3
			do
				#echo "Mode: $MODE, Blocksize: $BLOCKSIZE, Trial: $i"
				srun ./$NAME.exe $SEQLENGTH $NUMLAYERS $HIDDENSIZE $MINIBATCH
			
				#declare timeTrial$i=$(srun ./$NAME.exe $N $DIM $EPSILON $FILENAME | sed -n 's/Total time: //p')
			done
			#echo "Optimization set (same as MODE): $MODE"
			#echo $timeTrial1 $timeTrial2 $timeTrial3 | awk '{printf "Average time: %.5f\n", ($1 + $2 + $3)/3}'
			echo "------------------------------"
		done
	done
done
