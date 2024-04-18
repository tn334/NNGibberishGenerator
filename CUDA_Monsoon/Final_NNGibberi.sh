#!/bin/bash
#SBATCH --job-name=CS453_FINAL_NN  #the name of your job

# ////////////////////////////////////////////////////////////////////
# Change the path to your specific directory                        //
# ////////////////////////////////////////////////////////////////////
#SBATCH --output=/scratch/<directory_path>/NN_<NAU_ID>.out
#SBATCH --error=/scratch/<directory_path>/NN_<NAU_ID>.err

#SBATCH --time=00:05:00            #Job timelimit is N minutes
#SBATCH --mem=0                    #memory requested in MiB (request all with 0)
#SBATCH -G 4                       #resource requirement (all GPUs)
#SBATCH -C a100                    #GPU Model: v100, a100
#SBATCH --account=cs453-spr24                                

#load cuda library/resources before running
module load cuda # load module a100/v100

#compute capability
CC=80  # a100 code
# CC=70  # v100 code

# ////////////////////////////////////////////////////////////////////
# Set a file name for the .cu file as well as the output file       //
# ////////////////////////////////////////////////////////////////////
nvcc -O3 -arch=compute_$CC -code=sm_$CC -lcuda -lineinfo -Xcompiler -fopenmp \
                                         <cuda_filename>.cu -o <output_filename>

# #3 time trials
# for MODE in 2 #6 1 2 3 4 5 6
# do
#     for i in 1 #1 2 3
#     do
#         for BLOCKSIZE in 128
#         do
#         nvcc -O3 -DMODE=$MODE -DBLOCKSIZE=$BLOCKSIZE -arch=compute_$CC -code=sm_$CC -lcuda \
#             -lineinfo -Xcompiler -fopenmp CS453_a5_baseline_ap2974.cu -o a5_ap2974

#         # biggest_number_solution.cu -o biggest_number_solution

#         echo "Mode: $MODE, Trial: $i, Blocksize: $BLOCKSIZE"

#         # srun ncu -f -o similarity_profiler --clock-control=none --set full 
#         srun ncu -f --clock-control=none --set full -o similarity_profiler ./a5_ap2974 $N $d $e $fileName # compute-sanitizer --tool=memcheck

#         done
#     done
# done