
// libraries
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

// function declarations
__device__ float generateRandomFloat(int tid);
__device__ char generateRandomLetter(int tid);
__device__ int char_to_index(char letter)


/*  
################################################################
#                       MAIN FUNCTION                          #
################################################################
*/
int main(int argc, char const *argv[])
{
    return 0;
}


/*  
################################################################
#                       HELPER FUNCTIONS                       #
################################################################
*/

// kernel function to generate random floats between 0 and 1
__device__ float generateRandomFloat(int tid)
{
    // create random number generator
    curandState state;
    curand_init(clock64(), tid, 0, &state);

    // generate random number between 0 and 1
    return curand_uniform(&state);
}

// kernel function to generate a random lowercase letter
__device__ char generateRandomLetter(int tid)
{
    // create random number generator
    curandState state;
    curand_init(clock64(), tid, 0, &state);

    // generate random int between 0 and 25
    int randomIdx = curand(&state) % 26;

    // convert ascii value to corresponding lowercase letter
    return 'a' + randomIdx;
}

// kernel function to turn a letter into an index
__device__ int char_to_index(char letter)
{
    if (islower(letter)) 
    {
        return letter - 'a';
    }

    return letter - 'A';
}

