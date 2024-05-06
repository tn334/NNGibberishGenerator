//example of running the program: ./A5_similarity_search_starter 7490 5 10 bee_dataset_1D_feature_vectors.txt

#include <curand.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>
#include <math.h>

//Error checking GPU calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//Define any constants here
//Feel free to change BLOCKSIZE
//#define BLOCKSIZE 128
#define DEFAULT_FILL 50.0
#define DEFAULT_MIN 1000000000.0

using namespace std;

//function prototypes
//Some of these are for debugging so I did not remove them from the starter file

void printDataset();

void warmUpGPU();

void checkParams(unsigned int N, unsigned int DIM);

void importDataset(char * fname, unsigned int N, unsigned int DIM, float * dataset);

void generateWordsCPU(float * dataset, unsigned int N, unsigned int DIM, const int NEARESTNEIGHBORS);

void calcKNN(float * distanceArray, const int NEARESTNEIGHBORS, unsigned int N, int *KNNSet, const int DIM );

bool selectRandomNeighbor(float * dataset, float * newWordArray, int * KNNSet, const int freqLocation, const int NEARESTNEIGHBORS, const int DIM);

void decodeWord(float * newWordArray, int DIM);

void outputDistanceMatrixToFile(float * distanceMatrix, unsigned int N);

void calcDistanceArrayCPU(float * dataset, float * distanceArray, float * newWordArray, const unsigned int N, const unsigned int DIM, const int wordIndex);

int translateLetter(char letterToTranslate);

int translateLetterSequel(char letterToTranslate);

//Part 1: Computing the distance matrix 

//Baseline kernel --- one thread per point/feature vector
__device__ float generateRandomNumbers(int startIdx, int endIdx);

__global__ void generateWordsBaseline(float * dataset, float * distanceArray, const unsigned int N, const unsigned int DIM);

__device__ calcKNNGPU(float * distanceArray, const int NEARESTNEIGHBORS, unsigned int N, int *KNNSet, const int DIM);
__device__ selectRandNeighborGPU(float * dataset, float * newWordArray, int * KNNSet, const int freqLocation, const int NEARESTNEIGHBORS, const int DIM);
__device__ decodeWordGPU(float * newWordArray, int DIM);
__device__ translateLetterGPU(char letterToTranslate);
__device__ calcDistanceArrayGPU(float * dataset, float * distanceArray, float * newWordArray, const unsigned int N, const unsigned int DIM, const int wordIndex);
//Part 2: querying the distance matrix
//__global__ void queryDistanceMatrixBaseline(float * distanceMatrix, const unsigned int N, const unsigned int DIM, const float epsilon, unsigned int * resultSet);


int main(int argc, char *argv[])
{
  printf("\nMODE: %d", MODE);
  warmUpGPU(); 
  //srand(time(0));
  srand(42);



  char inputFname[500];
  unsigned int N=0;
  unsigned int DIM=0;
  unsigned int NUMNEIGHBORS = 0;
  //float epsilon=0;

  if (argc != 5) {
    fprintf(stderr,"Please provide the following on the command line: N (number of lines in the file), dimensionality (number of coordinates per point), number of neighbors, dataset filename.\n");
    exit(0);
  }

  sscanf(argv[1],"%d",&N);
  sscanf(argv[2],"%d",&DIM);
  sscanf(argv[3],"%d",&NUMNEIGHBORS);
  printf("N: %d\n", N);
  printf("DIM: %d\n", DIM);
  printf("Number of neighbors: %d\n", NUMNEIGHBORS);
  strcpy(inputFname,argv[4]);

  checkParams(N, DIM);

  printf("\nAllocating the following amount of memory for the dataset: %f GiB", (sizeof(float)*N*DIM)/(1024*1024*1024.0));
  //printf("\nAllocating the following amount of memory for the distance matrix: %f GiB", (sizeof(float)*N*N)/(1024*1024*1024.0));
  

  float * dataset=(float*)malloc(sizeof(float*)*DIM*N);
  importDataset(inputFname, N, DIM, dataset);



  //CPU-only mode
  //It only computes the distance matrix but does not query the distance matrix
  if(MODE==0){
    // generate 100 words
    double tstart = omp_get_wtime();
    for (int index = 0; index < WORDSTOGENERATE; index++)
    {
      generateWordsCPU( dataset, N, DIM, NUMNEIGHBORS);
    }
    double tend = omp_get_wtime();
    printf("\nTime to compute distance matrix on the CPU: %f", tend - tstart);
    printf("\nReturning after computing on the CPU");
    return(0);
  }
  

  double tstart=omp_get_wtime();

  //Allocate memory for the dataset
  float * dev_dataset;
  gpuErrchk(cudaMalloc((float**)&dev_dataset, sizeof(float)*DIM*N));
  gpuErrchk(cudaMemcpy(dev_dataset, dataset, sizeof(float)*DIM*N, cudaMemcpyHostToDevice));

  //For part 1 that computes the distance Array
  float * dev_distanceArray;
  gpuErrchk(cudaMalloc((float**)&dev_distanceArray, sizeof(float)*N));
  

  //For part 2 for querying the distance matrix
  unsigned int * resultSet = (float *)calloc(WORDSTOGENERATE*N, sizeof(float));
  unsigned int * dev_resultSet;
  gpuErrchk(cudaMalloc((float**)&dev_resultSet, sizeof(float)*N));
  gpuErrchk(cudaMemcpy(dev_resultSet, resultSet, sizeof(float)*N, cudaMemcpyHostToDevice));

  
  //Baseline kernels
  if(MODE==1){
  // Optimization set 1: baseline of both kernels
  unsigned int BLOCKDIM = BLOCKSIZE; 
  unsigned int NBLOCKS = NUMWORDSTOGENERATE*1.0;
  //Part 1: Compute distance matrix
  generateWordsBaseline<<<NBLOCKS, BLOCKDIM>>>(dev_dataset, dev_distanceArray, N, DIM);
  }


  //Note to reader: you can move querying the distance matrix outside of the mode
  //Part 2: Query distance matrix
  //queryDistanceMatrixBaseline<<<NBLOCKS,BLOCKDIM>>>(dev_distanceMatrix, N, DIM, epsilon, dev_resultSet);
  
  //Copy result set from the GPU
  gpuErrchk(cudaMemcpy(resultSet, dev_resultSet, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost));

  //Compute the sum of the result set array
  unsigned int totalWithinEpsilon=0;

  //Write code here
  for (unsigned int i = 0; i < N; i++)
  {
    totalWithinEpsilon += resultSet[i];
  }
  
  printf("\nTotal number of points within epsilon: %u", totalWithinEpsilon);

  double tend=omp_get_wtime();

  printf("\n[MODE: %d, N: %d]\nTotal time: %f", MODE, N, tend-tstart);

  
  //For outputing the distance matrix for post processing (not needed for assignment --- feel free to remove)
  //float * distanceMatrix = (float*)calloc(N*N, sizeof(float));
  //gpuErrchk(cudaMemcpy(distanceMatrix, dev_distanceMatrix, sizeof(float)*N*N, cudaMemcpyDeviceToHost));
  //outputDistanceMatrixToFile(distanceMatrix, N);
 

  //Free memory here
  gpuErrchk(cudaFree(dev_dataset));
  gpuErrchk(cudaFree(dev_distanceArray));
  gpuErrchk(cudaFree(dev_resultSet));
  
  free(dataset);

  printf("\n\n");
  return 0;
}


//prints the dataset that is stored in one 1-D array
void printDataset(unsigned int N, unsigned int DIM, float * dataset)
{
    for (int i=0; i<N; i++){
      for (int j=0; j<DIM; j++){
        if(j!=(DIM-1)){
          printf("%.0f,", dataset[i*DIM+j]);
        }
        else{
          printf("%.0f\n", dataset[i*DIM+j]);
        }
      }
      
    }  
}




//Import dataset as one 1-D array with N*DIM elements
//N can be made smaller for testing purposes
//DIM must be equal to the data dimensionality of the input dataset
void importDataset(char * fname, unsigned int N, unsigned int DIM, float * dataset)
{
    
    FILE *fp = fopen(fname, "r");

    if (!fp) {
        fprintf(stderr, "Unable to open file\n");
        fprintf(stderr, "Error: dataset was not imported. Returning.");
        exit(0);
    }

    unsigned int bufferSize = DIM*10;

    char buf[bufferSize];
    unsigned int rowCnt = 0;
    unsigned int colCnt = 0;
    while (fgets(buf, bufferSize, fp) && rowCnt<N) {
        // after fgets, buf should be a single line of the form a,1833,p,77,p,15,l,3,e,1
        colCnt = 0;

        char *field = strtok(buf, ",");
        char charTemp;
        int numTemp;

        sscanf(field,"%c",&charTemp);
        dataset[rowCnt*DIM+colCnt]=(float)translateLetter(charTemp);
        colCnt++;
        field = strtok(NULL, ",");
        sscanf(field,"%d",&numTemp);
        dataset[rowCnt*DIM+colCnt]=(float)numTemp;

        
        while (field) {

          colCnt++;
          field = strtok(NULL, ",");
          
          if (field!=NULL)
          {
            sscanf(field,"%c",&charTemp);
            dataset[rowCnt*DIM+colCnt]=(float)translateLetter(charTemp);
            colCnt++;
            field = strtok(NULL, ",");
            sscanf(field,"%d",&numTemp);
            dataset[rowCnt*DIM+colCnt-2] /= (float)numTemp;
            dataset[rowCnt*DIM+colCnt]=(float)numTemp;
          }   

        }
        /*
        printf("\n");
        for (int i =0; i < DIM; i++)
        {
          printf("%f ", dataset[rowCnt*DIM+i]);
        }
        printf("\n");*/

        rowCnt++;
    }

    fclose(fp);

}


void warmUpGPU(){
printf("\nWarming up GPU for time trialing...\n");
cudaDeviceSynchronize();
return;
}


void checkParams(unsigned int N, unsigned int DIM)
{
  if( N <= 0 || DIM <= 0 ){
    fprintf(stderr, "\n Invalid parameters: Error, N: %u, DIM: %u", N, DIM);
    fprintf(stderr, "\nReturning");
    exit(0); 
  }
}

// Dependencies: translateLetter(), calcDistanceArrayCPU, calcKNN, selectRandomNeighbor
void generateWordsCPU(float * dataset, unsigned int N, unsigned int DIM, const int NUMNEIGHBORS)
{
  // distance array holds location values for comparison between our word and all stored words AKA N
  float * distanceArray = (float*)malloc(sizeof(float)*N);
  float * newWordArray = (float*)malloc(sizeof(float)*DIM); // random letter selected from a-z
  int newWordIdx = 1;
  float encodedLetter;
  //float nextEncodedLetter;
  double tstart = omp_get_wtime();
  int KNNSet[NUMNEIGHBORS];

 // loop through all values of KNN
  for (int i = 0; i < NUMNEIGHBORS; i++)
  {
    KNNSet[i] = 0;
  }

  // input random letter to 0 index 
  // grab a random first letter from the dataset to match
  // the frequency of letters to begin words
  // rand % N gives some value to indicate our row, *DIM to jump to the correct row, + 0 to incidate first letter
  encodedLetter =  dataset[(rand() % N)*DIM + 0];
  //char randomLetter = (rand() % 26);
  
  //translate randomLetter to int
  //encodedLetter = (float)translateLetter(randomLetter);

  // loop through new word creation
  for(int index = 0; index < DIM; index++)
  {
    //check if at beginning of word, add random letter
    if(index == 0)
    {
      newWordArray[index] = encodedLetter;
    }
    // check if at even index to add letter 
    else if(index % 2 == 0)
    {
      //newWordArray[index] = (float)(rand() % 26); 
      newWordArray[index] = (float)(rand() % 20);
    }
    // process frequencies
    else
    {
      // append '-1' to all other slots
      //newWordArray[index] = DEFAULT_FILL;
      newWordArray[index] = (float)(rand() % 10 + index);
    }
  }

  //Write code here
  bool wordEnd = false;
  for (newWordIdx = 1; newWordIdx < DIM && !wordEnd; newWordIdx++)
  {
    // LOOP THROUGH NEW WORD GEN use 
    //printf("Attempt to enter calcDistanceArrayCPU\n\n");
    calcDistanceArrayCPU(dataset, distanceArray, newWordArray, N, DIM, newWordIdx);

    // KNN function thingy takes distArr, nnearestneighbors, N
    //printf("Made it past calcDistanceArrayCPU\n\n");
    calcKNN(distanceArray, NUMNEIGHBORS, N, KNNSet, DIM);

    // print out words that correspond to the KNN idx Match
    /*for(int word = 0; word < NUMNEIGHBORS; word++)
    {
      int wordRowIdx = KNNSet[word];
      float neighbor[DIM];
      for (int index = 0; index < DIM; index++)
      {
        neighbor[index] = dataset[wordRowIdx + index];
      }
      decodeWord(neighbor, DIM);
    }*/

    // Call function to select a random neighbor from the set of K closest neighbors
    // If the random neighbor selected is _, then fill up the rest of the word with _ and end the word
    wordEnd = selectRandomNeighbor(dataset, newWordArray, KNNSet, newWordIdx, NUMNEIGHBORS, DIM);

  }


  decodeWord(newWordArray, DIM);

  free(newWordArray);
  free(distanceArray);
}

void calcKNN(float * distanceArray, const int NEARESTNEIGHBORS, unsigned int N, int *KNNSet, const int DIM )
{
  float currentMin;
  // loop through distanceArray for NEARESTNEIGHBORS iterations
  //loop from 0 to NEARESTNEIGHBORS
  for(int neighborIdx = 0; neighborIdx < NEARESTNEIGHBORS; neighborIdx++)
  {
    // set currentMin
    currentMin = DEFAULT_MIN; //1000000000.0
    // Check each distance value in distanceArr
    for(int idx = 0; idx < N; idx++)
    {
        // Check if value at distanceArr[idx] is new min
        if(distanceArray[idx] < currentMin)
        {
          //check if not on first iteration of distArr loop
          if(neighborIdx != 0)
          {
            bool neighborAlreadyAdded = false;
            // loop through already saved values in KNNSet
            for(int counter = 0; counter < neighborIdx; counter++)
            {
              //check if idx already used as a min value
              if(idx*DIM == KNNSet[counter])
              {
                // check if idx has already been used
                neighborAlreadyAdded = true;
              }
            }
            // if idx not used already
            if(!neighborAlreadyAdded)
            {
              // set val of idx in KNNSet to idx of min val in distArr
              KNNSet[neighborIdx] = idx*DIM;
            }
          }
          // on first loop of KNN
          else
          {
            // if first loop assign
            KNNSet[neighborIdx] = idx*DIM;
          }
          //update current min to new min value
          currentMin = distanceArray[idx];
        }
    }
  }
}

void calcDistanceArrayCPU(float * dataset, float * distanceArray, float * newWordArr, const unsigned int N, const unsigned int DIM, const int workingLetterIndex)
{
    //write code here
    int row = 0;
    int wordIdx = 0;
    float totalDistSquared = 0.0;
    float totalDist;

    // loop through columns
    for(row = 0; row < N; row++)
    {
      totalDistSquared = 0.0;
      //for(wordIdx = 0; wordIdx < DIM; wordIdx++)
      for(wordIdx = 0; wordIdx < workingLetterIndex + 1; wordIdx++)
      {
        totalDistSquared += (newWordArr[wordIdx]- dataset[(row * DIM )+ wordIdx]) * (newWordArr[wordIdx]-dataset[(row * DIM) + wordIdx]);
      }
      // take the sqrtf
      totalDist = sqrt(totalDistSquared);
      //then add to distanceArr
      distanceArray[row] = totalDist;
    }
}

bool selectRandomNeighbor(float * dataset, float * newWordArray, int * KNNSet, const int newLocation, const int NUMNEIGHBORS, const int DIM)
{
  bool wordEnd = false;
  // get random neighbor between 0 and 9
  int randomNeighbor = rand() % NUMNEIGHBORS;

  //int letterLocation = freqLocation + 1;
  // get index of next word from KNNSet
  /*for (int i =0; i < NUMNEIGHBORS; i++)
  {
    printf("KNN value at index %d: %d\n", i, KNNSet[i]);
  }*/

  int datasetIndex = KNNSet[randomNeighbor];

  //printf("\nAttempt to access index %d in dataset (%d + %d).\n", datasetIndex + newLocation, datasetIndex, newLocation);
  
  float newValue = dataset[datasetIndex + newLocation];
  //float newLetter = dataset[randomNeighbor*DIM + letterLocation];


  newWordArray[newLocation] = newValue;
  //newWordArray[letterLocation] = newLetter;
  if (newLocation % 2 == 0 && (int)newValue == 12) {
    wordEnd = true;
    // If word has received '_' then pad out the rest of the word with '_'
    for (int i = newLocation; i < DIM; i++)
    {
      newWordArray[i] = 12.0; // 12.0 represents '_' when decoded
    }
  }

  return wordEnd;

}

void decodeWord(float * newWordArray, int DIM)
{
  printf("\n");
  int letterIndex = -1;
  char decodeArray[27] = {'e', 't', 'a', 'o', 'i', 'n', 's',
                          'h', 'r', 'd', 'l', 'c', '_', 'u',
                          'm', 'w', 'f', 'g', 'y', 'p', 'b',
                          'v', 'k', 'j', 'x', 'q', 'z'};
  for (int index = 0; index < DIM; index += 2)
  {
    letterIndex = (int)newWordArray[index];
    if (letterIndex < 27)
    {
      printf("%c", decodeArray[letterIndex]);
    }
    else
    {
      printf("_");
    }
  }
  printf("\n");
}

//For testing/debugging
/* void computeSumOfDistances(float * distanceArray, unsigned int N)
{
  double computeSumOfDistances=0;
  for (unsigned int i=0; i<N; i++)
  {
    for (unsigned int j=0; j<N; j++)
    {
      computeSumOfDistances+=(double)distanceArray[i*N+j];
    }
  }  

  printf("\nSum of distances: %f", computeSumOfDistances);
} */

//This is used to do post-processing in Python of bee statistics
//I left it in the starter file in case anyone else wants to tinker with the 
//distance matrix and the bees, but it is unnecessary for the assignment
void outputDistanceMatrixToFile(float * distanceMatrix, unsigned int N)
{

// Open file for writing
FILE * fp = fopen( "distance_matrix_output_shared.txt", "w" ); 

 for (int i=0; i<N; i++){
    for (int j=0; j<N; j++){
      if(j!=(N-1)){
        fprintf(fp, "%.3f,", distanceMatrix[i*N+j]);
      }
      else{
        fprintf(fp, "%.3f\n", distanceMatrix[i*N+j]);
      }
    }
    
  }   

  fclose(fp);
}

int translateLetter(char letterToTranslate)
{
    int numToReturn = -1;
    switch (letterToTranslate)
    {
        case 'e':
            numToReturn = 0;
            break;
        case 't':
            numToReturn = 1;
            break;
        case 'a':
            numToReturn = 2;
            break;
        case 'o':
            numToReturn = 3;
            break;
        case 'i':
            numToReturn = 4;
            break;
        case 'n':
            numToReturn = 5;
            break;
        case 's':
            numToReturn = 6;
            break;
        case 'h':
            numToReturn = 7;
            break;
        case 'r':
            numToReturn = 8;
            break;
        case 'd':
            numToReturn = 9;
            break;
        case 'l':
            numToReturn = 10;
            break;
        case 'c':
            numToReturn = 11;
            break;
        case '_':
            numToReturn = 12;
            break;
        case 'u':
            numToReturn = 13;
            break;
        case 'm':
            numToReturn = 14;
            break;
        case 'w':
            numToReturn = 15;
            break;
        case 'f':
            numToReturn = 16;
            break;
        case 'g':
            numToReturn = 17;
            break;
        case 'y':
            numToReturn = 18;
            break;
        case 'p':
            numToReturn = 19;
            break;
        case 'b':
            numToReturn = 20;
            break;
        case 'v':
            numToReturn = 21;
            break;
        case 'k':
            numToReturn = 22;
            break;
        case 'j':
            numToReturn = 23;
            break;
        case 'x':
            numToReturn = 24;
            break;
        case 'q':
            numToReturn = 25;
            break;
        case 'z':
            numToReturn = 26;
            break;
    }
    return numToReturn;
}

int translateLetterSequel(char letterToTranslate)
{
    int numToReturn = -1;
    switch (letterToTranslate)
    {
        case 'e':
            numToReturn = 0;
            break;
        case 't':
            numToReturn = 1;
            break;
        case 'a':
            numToReturn = 2;
            break;
        case 'o':
            numToReturn = 3;
            break;
        case 'i':
            numToReturn = 4;
            break;
        case 'n':
            numToReturn = 5;
            break;
        case 's':
            numToReturn = 6;
            break;
        case 'h':
            numToReturn = 7;
            break;
        case 'r':
            numToReturn = 8;
            break;
        case 'd':
            numToReturn = 9;
            break;
        case 'l':
            numToReturn = 10;
            break;
        case 'c':
            numToReturn = 11;
            break;
        case 'u':
            numToReturn = 12;
            break;
        case 'm':
            numToReturn = 13;
            break;
        case 'w':
            numToReturn = 14;
            break;
        case 'f':
            numToReturn = 15;
            break;
        case 'g':
            numToReturn = 16;
            break;
        case 'y':
            numToReturn = 17;
            break;
        case 'p':
            numToReturn = 18;
            break;
        case 'b':
            numToReturn = 19;
            break;
        case 'v':
            numToReturn = 20;
            break;
        case 'k':
            numToReturn = 21;
            break;
        case 'j':
            numToReturn = 22;
            break;
        case 'x':
            numToReturn = 23;
            break;
        case 'q':
            numToReturn = 24;
            break;
        case 'z':
            numToReturn = 25;
            break;
        case '_':
            numToReturn = 26;
            break;
    }
    return numToReturn;
}

//Query distance matrix with one thread per feature vector
/*Once the distance matrix has been constructed, it can be queried one or
more times. In this assignment, we will perform the same query on each point in the dataset. You will
determine the number of neighbors each point has within the search distance epsilon. In other words, after
the distance matrix has been computed (Figure 3(a)) you will scan the distance matrix and compute
whether dist(p0, p1) <= epsilon, and if so, count the number of instances for each point and store it in an
array that will be returned to the host. Figure 4 shows an example of the array that will be returned
to the host. You will add the elements in this array on the host to determine the total number of
instances that points were within epsilon of each other. */

//BEGIN GPU CODE
__device__ float generateRandomNumber(curandState *state, int endIdx)
{
  return curand(state) % (endIdx + 1);
}

__global__ void generateWordsBaseline(float * dataset, float * distanceArray, const unsigned int N, const unsigned int DIM)
{
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if(tid < N)
  {
    // variable declaration
    float *distanceArray = new float[N]; // local var for distance array
    float * newWordArr = new float[DIM]; // array for storing new word
    int *KNNSetArr = new float[NUMNEIGHBORS]; // array to store location of neighbors

    //Compute distance array for the current point

    double tstart = omp_get_wtime();
    int newWordIdx = 1;
    float encodedLetter;

    // assign all values on KNNSetArr to 0
    if(tid < NUMNEIGHBORS)
    {
      KNNSetArr[tid] = 0;
    }

    __syncthreads();
    // generate random encoded letter from dataset
    float randomN = generateRandomNumer(state, N);
    encodedLetter = dataset[randomN * DIM + 0];

    __syncthreads();

    if(tid < DIM)
    {
      //assign random letter to zero index
      if(tid == 0)
      {
        newWordArr[0] = encodedLetter;
      }
      else if( tid % 2 == 0)
      {
        newWordArr[tid] = generateRandomNumber(state, 20); // change to var
      }
      else
      {
        newWordArr[tid] = generateRandomNumber(state, 10) + tid;
      }

      __syncthreads();

      bool wordEnd = false;
      if(!wordEnd && tid > 0)
      {
        if(newWordIdx == tid)
        {
          // need calcDistanceArrayGPU written and called

          // need calcKNNGPU written and called

          // need to set end of word and call selectRandomNeighborGPU
        }

      }
    }
  }


if(tid == 0)
{
  // implement and call decodeWordGPU pass newWordArr and DIM
}

delete[] distanceArray;
delete[] KNNSetArr;
delete[] newWordArr;
}

/*
__global__ void queryDistanceMatrixBaseline(float * distanceMatrix, const unsigned int N, const unsigned int DIM, const float epsilon, unsigned int * resultSet)
{
  //write code here
  unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);

  if (tid < N)
  {
    resultSet[tid] = 0;
    for (unsigned int i = 0; i < N; i++)
    {
      // future modes could load in chunks of this array into shared memory
      if (distanceMatrix[tid*N+i] <= epsilon)
      {
        // future modes could use a local counter
        resultSet[tid] += 1;
      }
    }
  }
}

__global__ void queryDistanceMatrixRegisters(float * distanceMatrix, const unsigned int N, const unsigned int DIM, const float epsilon, unsigned int * resultSet)
{
  //write code here
  unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  unsigned int neighborCount = 0;

  if (tid < N)
  {
    for (unsigned int i = 0; i < N; i++)
    {
      // future modes could load in chunks of this array into shared memory
      if (distanceMatrix[i*N+tid] <= epsilon)
      {
        // future modes could use a local counter
        neighborCount++;
      }
    }
    resultSet[tid] = neighborCount;
  }
}

__global__ void queryDistanceMatrixReduction(float * distanceMatrix, const unsigned int N, const unsigned int DIM, const float epsilon, unsigned int * resultSet)
{
    //write code here
  unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  __shared__ int sharedMatrix[BLOCKSIZE];

  //Use local registers
  //unsigned int neighborCount = 0;
    // run through columns
    for (unsigned int i = 0; i < N; i++)
    {
      // check if the index in distance matrix is less than or equal to epsilon
        // tid * N gets us rows
          // + i runs us through the individual values in the row
      if (tid < N && distanceMatrix[tid*N+i] <= epsilon)
      {
        // increment count
        sharedMatrix[threadIdx.x] += 1;
      }
    }
  __syncthreads(); // finish threads writing to shared mem
  for(unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    if(threadIdx.x < stride){
      sharedMatrix[threadIdx.x] += sharedMatrix[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if(threadIdx.x==0)
  {
    atomicAdd(&resultSet[tid], sharedMatrix[0]);
  }
}

//One thread per feature vector -- baseline kernel
__global__ void distanceMatrixBaseline(float * dataset, float * distanceMatrix, const unsigned int N, const unsigned int DIM)
{
  //write code here

  // each thread assigned 1 point in the dataset
  // DIM=135000
  // dataset[rowCnt*DIM+colCnt]
  
  unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tid < N)
  {
    unsigned int tidRow = tid*DIM;
    float sumOfSquaredDiffs;
    for (unsigned int otherRow = 0; otherRow < N; otherRow++)
    {
      sumOfSquaredDiffs = 0;
      for (unsigned int column = 0; column < DIM; column++)
      {
        // sum square this point's column and other point's column
        sumOfSquaredDiffs += (dataset[tidRow+column] - dataset[otherRow*DIM+column]) * (dataset[tidRow+column] - dataset[otherRow*DIM+column]);
      }
      // square root the sum up to this point + write it to distanceMatrix
      distanceMatrix[tid*N+otherRow] = sqrtf(sumOfSquaredDiffs);
    }
  }
}

// Calculate half of the distance matrix and save it for two positions of the matrix
__global__ void distanceMatrixMirrorCalculation(float * dataset, float * distanceMatrix, const unsigned int N, const unsigned int DIM)
{
  unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tid < N)
  {
    unsigned int tidRow = tid*DIM;
    float sumOfSquaredDiffs;
    for (unsigned int otherRow = tid + 1; otherRow < N; otherRow++)
    {
      sumOfSquaredDiffs = 0;
      for (unsigned int column = 0; column < DIM; column++)
      {
        // sum square this point's column and other point's column
        sumOfSquaredDiffs += (dataset[tidRow+column] - dataset[otherRow*DIM+column]) * (dataset[tidRow+column] - dataset[otherRow*DIM+column]);
      }
      // square root the sum up to this point + write it to distanceMatrix
      float totalDistance = sqrtf(sumOfSquaredDiffs);
      // mirror of this element is given by otherRow*N + tid
      distanceMatrix[tid*N+otherRow] = totalDistance;
      distanceMatrix[otherRow*N + tid] = totalDistance;
    }
  }
}

__global__ void distanceMatrixMirrorCalculationInverted(float * dataset, float * distanceMatrix, const unsigned int N, const unsigned int DIM)
{
  unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tid < N)
  {
    unsigned int tidRow = tid*DIM;
    float sumOfSquaredDiffs;
    for (unsigned int otherRow = N-1; otherRow > tid; otherRow--)
    {
      sumOfSquaredDiffs = 0;
      for (unsigned int column = 0; column < DIM; column++)
      {
        // sum square this point's column and other point's column
        sumOfSquaredDiffs += (dataset[tidRow+column] - dataset[otherRow*DIM+column]) * (dataset[tidRow+column] - dataset[otherRow*DIM+column]);
      }
      // square root the sum up to this point + write it to distanceMatrix
      float totalDistance = sqrtf(sumOfSquaredDiffs);
      // mirror of this element is given by otherRow*N + tid
      distanceMatrix[tid*N+otherRow] = totalDistance;
      distanceMatrix[otherRow*N + tid] = totalDistance;
    }
  }
}

__global__ void distanceMatrixCalculationSharedMemory(float * dataset, float * distanceMatrix, const unsigned int N, const unsigned int DIM)
{
  unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  __shared__ int sharedMemoryBlock[WIDTH*HEIGHT];
  // declare a register array for HEIGHT other rows to accumulate to
  float sumOfSquaredDiffs[HEIGHT];
  // declare a register array for WIDTH number of ints from global dataset
  int dimensionSet[WIDTH];
  // declare shared memory array of ints for WIDTH*N elements to page into
  // enter loop to horizontally page chunks of dataset into shared memory
  for (unsigned int vStride = 0; vStride < N; vStride += HEIGHT)
  {
    // loop to reset all the sums to 0 since we're on a new set of rows
    for (unsigned int resetIndex=0; resetIndex < HEIGHT; resetIndex++)
    {
        sumOfSquaredDiffs[resetIndex] = 0.000;
    }
    // enter loop to vertically page chunks of dataset into shared memory
    for (unsigned int hStride = 0; hStride < DIM; hStride += WIDTH)
    {
      for (unsigned int registerIndex = 0; registerIndex < WIDTH; registerIndex++)
      {
        if (tid < N)
        {
          dimensionSet[registerIndex] = dataset[tid*DIM+hStride+registerIndex];
        }
      }

      // Load in a block of shared memory
      if (threadIdx.x < WIDTH)
      {
        for (unsigned int pagingIndex = 0; pagingIndex < HEIGHT; pagingIndex++)
        {
          sharedMemoryBlock[pagingIndex*WIDTH + threadIdx.x] = dataset[(vStride+pagingIndex)*DIM+hStride+threadIdx.x];
        }
      }
      __syncthreads();
      for (unsigned int otherRow = 0; otherRow < HEIGHT; otherRow++)
      {
        for (unsigned int column = 0; column < WIDTH; column++)
        {
          // sum square this point's column and other point's column
          if (tid < N)
          {
          sumOfSquaredDiffs[otherRow] += (dimensionSet[column] - sharedMemoryBlock[otherRow*WIDTH+column]) * (dimensionSet[column] - sharedMemoryBlock[otherRow*WIDTH+column]);
          }
        }
        // mirror of this element is given by otherRow*N + tid
        //distanceMatrix[tid*N+vStride+otherRow] += sumOfSquaredDiffs;
        //end of for loop for otherRow
        if (tid < N && hStride+WIDTH == DIM)
        {
          distanceMatrix[tid*N+otherRow+vStride] = sqrtf(sumOfSquaredDiffs[otherRow]);
        }

      }
      //end of for loop for horizontal stride
    } 
    //end of for loop for vertical stride
  }
}

__global__ void distanceMatrixMirrorCalculationSharedMemory(float * dataset, float * distanceMatrix, const unsigned int N, const unsigned int DIM)
{
  unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  __shared__ int sharedMemoryBlock[WIDTH*HEIGHT];
  // declare a register array for HEIGHT other rows to accumulate to
  float sumOfSquaredDiffs[HEIGHT];
  // declare a register array for WIDTH number of ints from global dataset
  int dimensionSet[WIDTH];
  // declare shared memory array of ints for WIDTH*N elements to page into
  // enter loop to horizontally page chunks of dataset into shared memory
  for (int vStride = N-HEIGHT; vStride >= (int)(tid/HEIGHT)*HEIGHT; vStride -= HEIGHT)
  {
    // loop to reset all the sums to 0 since we're on a new set of rows
    for (unsigned int resetIndex=0; resetIndex < HEIGHT; resetIndex++)
    {
        sumOfSquaredDiffs[resetIndex] = 0.000;
    }
    // enter loop to vertically page chunks of dataset into shared memory
    for (unsigned int hStride = 0; hStride < DIM; hStride += WIDTH)
    {
      for (unsigned int registerIndex = 0; registerIndex < WIDTH; registerIndex++)
      {
        if (tid < N)
        {
          dimensionSet[registerIndex] = dataset[tid*DIM+hStride+registerIndex];
        }
      }

      // Load in a block of shared memory
      if (threadIdx.x < WIDTH)
      {
        for (unsigned int pagingIndex = 0; pagingIndex < HEIGHT; pagingIndex++)
        {
          if (tid < N)
          {
            sharedMemoryBlock[pagingIndex*WIDTH + threadIdx.x] = dataset[(vStride+pagingIndex)*DIM+hStride+threadIdx.x];
          }
        }
      }
      __syncthreads();
      for (unsigned int otherRow = 0; otherRow < HEIGHT; otherRow++)
      {
        for (unsigned int column = 0; column < WIDTH; column++)
        {
          // sum square this point's column and other point's column
          if (tid < N)
          {
          sumOfSquaredDiffs[otherRow] += (dimensionSet[column] - sharedMemoryBlock[otherRow*WIDTH+column]) * (dimensionSet[column] - sharedMemoryBlock[otherRow*WIDTH+column]);
          }
        }
        // mirror of this element is given by otherRow*N + tid
        //distanceMatrix[tid*N+vStride+otherRow] += sumOfSquaredDiffs;
        //end of for loop for otherRow
        if (tid < N && hStride+WIDTH == DIM && tid != vStride+otherRow)
        {
            distanceMatrix[(tid*N)+otherRow+vStride] = sqrtf(sumOfSquaredDiffs[otherRow]);
            distanceMatrix[(vStride+otherRow)*N+tid] = sqrtf(sumOfSquaredDiffs[otherRow]);
        }

      }
      //end of for loop for horizontal stride
    } 
    //end of for loop for vertical stride
  }
}
*/