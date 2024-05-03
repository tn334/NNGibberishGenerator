#!/bin/bash
#SBATCH --job-name=CS453_a6_KNN  #the name of your job

#change to your NAUID and file path
#SBATCH --output=/scratch/nrs285/CS453_a6_KNN.out #this is the file for stdout 
#SBATCH --error=/scratch/nrs285/CS453_a6_KNN.err #this is the file for stderr

#SBATCH --time=00:10:00		#Job timelimit is 3 minutes
#SBATCH --mem=5000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C v100 # k80 #GPU Model: k80, p100, v100, a100
#SBATCH --account=cs453-spr24	
##SBATCH --reservation=cs453-spr24-res					

#compute capability
CC=70 #K80: 37, P100: 60, V100: 70, A100: 80

# source file + executable name
NAME=CS453_a6_KNN

# program defined constants

#N=7490
DIM=39 #9
FILENAME=20_letter_frequency_list_padded_comma_sep.txt #5_letter_frequency_list_padded_comma_sep.txt
#NUMNEIGHBORS=7
N=282378 #19599
BLOCKSIZE=64
WORDSTOGENERATE=100


#make sure not to redefine MODE, N, etc. in the source file

if [[ $CC -eq 37 ]] 
then
	module load cuda/11.7
else
	module load cuda
fi

# Delete previous .exe so if compilation fails the job ends
#\rm $NAME.exe

for MODE in 0 # 2 3 4 5 6 7 8 
do
	for WIDTH in 90
	do
		for NUMNEIGHBORS in 1 2 3 4 5
		do
			nvcc -O3 -arch=compute_$CC -code=sm_$CC -DBLOCKSIZE=$BLOCKSIZE -DMODE=$MODE -DWORDSTOGENERATE=$WORDSTOGENERATE -lcuda -lineinfo -diag-suppress 186 -Wno-deprecated-gpu-targets -Xcompiler -fopenmp $NAME.cu -o $NAME.exe
		
			#3 time trials
			for i in 1 #2 3
			do
				echo "Mode: $MODE, Number of Neighbors: $BLOCKSIZE, Trial: $i"
				srun ./$NAME.exe $N $DIM $NUMNEIGHBORS $FILENAME
			
				#declare timeTrial$i=$(srun ./$NAME.exe $N $DIM $EPSILON $FILENAME | sed -n 's/Total time: //p')
			done
			#echo "Optimization set (same as MODE): $MODE"
			#echo $timeTrial1 $timeTrial2 $timeTrial3 | awk '{printf "Average time: %.5f\n", ($1 + $2 + $3)/3}'
			echo "------------------------------"
		done
	done
done
