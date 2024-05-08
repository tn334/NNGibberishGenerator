//example of running the program: ./A5_similarity_search_starter 7490 5 10 bee_dataset_1D_feature_vectors.txt

#include <curand_kernel.h>
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

void decodeWord(float * wordToDecode, int offsetIndex, int DIM);

void outputDistanceMatrixToFile(float * distanceMatrix, unsigned int N);

void calcDistanceArrayCPU(float * dataset, float * distanceArray, float * newWordArray, const unsigned int N, const unsigned int DIM, const int wordIndex);

int translateLetter(char letterToTranslate);

int translateLetterSequel(char letterToTranslate);

//Part 1: Computing the distance matrix 

//Baseline kernel --- one thread per point/feature vector
__device__ float generateRandomNumbers(curandState *state, int endIdx);

__global__ void setupResultSetBaseline(float * resultSet,  float * dataset, const unsigned int WTG, const unsigned int DIM, const unsigned int N);

__global__ void generateDistanceMatrixBaseline(float * dataset, float * resultSet, float * distanceMatrix, const unsigned int colIdx,
                          const unsigned int WTG, const unsigned int N, const unsigned int DIM);

__global__ void generateKNNBaseline(float * dataset, float * resultSet, float * distanceMatrix, const unsigned int WTG,
                    const unsigned int NN, const unsigned int colIdx, const unsigned int N, const unsigned int DIM);

//Part 2: 
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
  //const unsigned int NUMNEIGHBORS = K;
  unsigned int WORDSTOGENERATE = NUMWORDSTOCREATE;
  //float epsilon=0;

  if (argc != 4) {
    fprintf(stderr,"Please provide the following on the command line: N (number of lines in the file), dimensionality (number of coordinates per point), number of neighbors, dataset filename.\n");
    exit(0);
  }

  sscanf(argv[1],"%d",&N);
  sscanf(argv[2],"%d",&DIM);
  printf("N: %d\n", N);
  printf("DIM: %d\n", DIM);
  printf("Number of neighbors: %d\n", NUMNEIGHBORS);
  strcpy(inputFname,argv[3]);

  checkParams(N, DIM);

  printf("\nAllocating the following amount of memory for the dataset: %f GiB\n", (sizeof(float)*N*DIM)/(1024*1024*1024.0));
  //printf("\nAllocating the following amount of memory for the distance matrix: %f GiB", (sizeof(float)*N*N)/(1024*1024*1024.0));
  

  float * dataset=(float*)malloc(sizeof(float*)*DIM*N);
  importDataset(inputFname, N, DIM, dataset);

  double tstart = omp_get_wtime();

  //CPU-only mode
  //It only computes the distance matrix but does not query the distance matrix
  if(MODE==0){
    // generate 100 words
    
    for (int index = 0; index < WORDSTOGENERATE; index++)
    {
      generateWordsCPU( dataset, N, DIM, NUMNEIGHBORS);
    }
    double tend = omp_get_wtime();
    printf("\n[MODE: %d, N: %d]\nTotal time: %f", MODE, N, tend-tstart);
    printf("\nReturning after computing on the CPU");
    return(0);
  }

  //Allocate memory for the dataset
  float * dev_dataset;
  gpuErrchk(cudaMalloc((float**)&dev_dataset, sizeof(float)*DIM*N));
  gpuErrchk(cudaMemcpy(dev_dataset, dataset, sizeof(float)*DIM*N, cudaMemcpyHostToDevice));

  //For part 1 that stores our results from GPU

    // Optimization set 1: baseline of both kernels


  //For part 2 for querying the distance matrix
  float * resultSet = (float *)calloc(WORDSTOGENERATE*DIM, sizeof(float));
  float * dev_resultSet;
  gpuErrchk(cudaMalloc((float**)&dev_resultSet, sizeof(float)*WORDSTOGENERATE*DIM));
  gpuErrchk(cudaMemcpy(dev_resultSet, resultSet, sizeof(float)*WORDSTOGENERATE*DIM, cudaMemcpyHostToDevice));

  float * dev_distanceMatrix;
  //unsigned int * dev_KNNSets;

  //Baseline kernels
  if(MODE==1){
  //float * distanceArray=(float*)malloc(sizeof(float*)*N);

  gpuErrchk(cudaMalloc((float**)&dev_distanceMatrix, sizeof(float)*N*WORDSTOGENERATE));
  //gpuErrchk(cudaMemCpy(dev_distanceArray, distanceArray, sizeof(float)*N, cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMalloc((unsigned int**)&dev_KNNSets, sizeof(unsigned int)*WORDSTOGENERATE*NUMNEIGHBORS));

    // generate our random starting words in resultSet baseline
  unsigned int BLOCKDIM = BLOCKSIZE;
  unsigned int NBLOCKS = ceil((WORDSTOGENERATE*DIM*1.0)/BLOCKSIZE*1.0);
  setupResultSetBaseline<<<NBLOCKS, BLOCKDIM>>>(dev_resultSet, dev_dataset, WORDSTOGENERATE, DIM, N);
  // Optimization set 1: baseline of both kernels

  // loop for DIM times
  for (int wordIdx = 1; wordIdx < DIM; wordIdx++) {
      // Generate distanceMatrix -> change distanceMatrix to be N*WORDSTOGENERATE
      // query distance matrix to find and select a nearest neighbor
      BLOCKDIM = BLOCKSIZE; 
      NBLOCKS = ceil((N*WORDSTOGENERATE*1.0)/(BLOCKSIZE*1.0));
      //Part 1: Compute distance matrix
      generateDistanceMatrixBaseline<<<NBLOCKS, BLOCKDIM>>>(dev_dataset, dev_resultSet, dev_distanceMatrix, wordIdx, WORDSTOGENERATE, N, DIM);

      cudaDeviceSynchronize();
      BLOCKDIM = BLOCKSIZE; 
      NBLOCKS = ceil((WORDSTOGENERATE*1.0)/(BLOCKSIZE*1.0));
      // Part 2: Generate the set of K nearest neighbors and select one to add to the built words
      // One thread each responsible for finding K nearest neighbors and selecting one at random
      generateKNNBaseline<<<NBLOCKS, BLOCKDIM>>>(dev_dataset, dev_resultSet, dev_distanceMatrix, WORDSTOGENERATE, NUMNEIGHBORS, wordIdx, N, DIM);
  }
  }


  //Note to reader: you can move querying the distance matrix outside of the mode
  //Part 2: Query distance matrix
  //queryDistanceMatrixBaseline<<<NBLOCKS,BLOCKDIM>>>(dev_distanceMatrix, N, DIM, epsilon, dev_resultSet);
  
  //Copy result set from the GPU
  gpuErrchk(cudaMemcpy(resultSet, dev_resultSet, sizeof(unsigned int)*WORDSTOGENERATE*DIM, cudaMemcpyDeviceToHost));

  // Decode each word in resultSet
  for (unsigned int i = 0; i < WORDSTOGENERATE; i++)
  {
    decodeWord(resultSet, i*DIM, DIM);
  }

  double tend=omp_get_wtime();

  printf("\n[MODE: %d, N: %d]\nTotal time: %f", MODE, N, tend-tstart);

  
  //For outputing the distance matrix for post processing (not needed for assignment --- feel free to remove)
  //float * distanceMatrix = (float*)calloc(N*N, sizeof(float));
  //gpuErrchk(cudaMemcpy(distanceMatrix, dev_distanceMatrix, sizeof(float)*N*N, cudaMemcpyDeviceToHost));
  //outputDistanceMatrixToFile(distanceMatrix, N);
 

  //Free memory here
  gpuErrchk(cudaFree(dev_dataset));
  gpuErrchk(cudaFree(dev_distanceMatrix));
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
void generateWordsCPU(float * dataset, unsigned int N, unsigned int DIM, const int NN)
{
  // distance array holds location values for comparison between our word and all stored words AKA N
  float * distanceArray = (float*)malloc(sizeof(float)*N);
  float * newWordArray = (float*)malloc(sizeof(float)*DIM); // random letter selected from a-z
  int newWordIdx = 1;
  float encodedLetter;
  //float nextEncodedLetter;
  double tstart = omp_get_wtime();
  int KNNSet[NN];

 // loop through all values of KNN
  for (int i = 0; i < NN; i++)
  {
    KNNSet[i] = -1;
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

  decodeWord(newWordArray, 0, DIM);

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
      for(wordIdx = 0; wordIdx < workingLetterIndex; wordIdx++)
      {
        totalDistSquared += (newWordArr[wordIdx]- dataset[(row * DIM )+ wordIdx]) * (newWordArr[wordIdx]-dataset[(row * DIM) + wordIdx]);
      }
      // take the sqrtf
      totalDist = sqrt(totalDistSquared);
      //then add to distanceArr
      distanceArray[row] = totalDist;
    }
}

bool selectRandomNeighbor(float * dataset, float * newWordArray, int * KNNSet, const int newLocation, const int NN, const int DIM)
{
  bool wordEnd = false;
  // get random neighbor between 0 and 9
  int randomNeighbor = rand() % NN;

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

void decodeWord(float * wordToDecode, int offsetIndex, int DIM)
{
  printf("\n");
  bool wordEnd = false;
  int letterIndex = -1;
  char decodeArray[27] = {'e', 't', 'a', 'o', 'i', 'n', 's',
                          'h', 'r', 'd', 'l', 'c', '_', 'u',
                          'm', 'w', 'f', 'g', 'y', 'p', 'b',
                          'v', 'k', 'j', 'x', 'q', 'z'};
  for (int index = 0; index < DIM; index += 2)
  {
    letterIndex = (int)wordToDecode[offsetIndex + index];
    if (letterIndex < 27 && !wordEnd)
    {
      printf("%c", decodeArray[letterIndex]);
      wordEnd = letterIndex == 12; 
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
  return (float)(curand(state) % (endIdx));
}

// setting up resultSet
__global__ void setupResultSetBaseline(float * resultSet, float * dataset, const unsigned int WTG, const unsigned int DIM, const unsigned int N)
{
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curandState state;
  curand_init(42, tid, 0, &state);

  if(tid < DIM * WTG)
  {
    //assign random letter to zero index
      if(tid % DIM == 0)
      {
        float randomN = generateRandomNumber(&state, N);
        float encodedLetter = dataset[(int)randomN * DIM + 0];
        // Assign the first letter to a value of a letter with the frequency of first letters in the dataset
        resultSet[tid] = encodedLetter;
      }
      else
      {
        // Assign letter vectors a value between 0 and 26, assign frequency vectors to a value between 0 and 13
        resultSet[tid] = generateRandomNumber(&state, 20) - ((tid % DIM) % 2) * tid / DIM;
      }
  }
}

__global__ void generateDistanceMatrixBaseline(float * dataset, float * resultSet, float * distanceMatrix, const unsigned int colIdx,
                          const unsigned int WTG, const unsigned int N, const unsigned int DIM)
{
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int resultRow = tid / N;
  unsigned int datasetRow = tid % N;
  if(tid < N*WTG)
  {
    int dimIdx;
    float totalDistSquared = 0.0;
    for(dimIdx = 0; dimIdx < colIdx + 1; dimIdx++)
    {
      totalDistSquared += (resultSet[resultRow*DIM + dimIdx]- dataset[(datasetRow * DIM )+ dimIdx]) * (resultSet[(resultRow*DIM) + dimIdx] - dataset[(datasetRow * DIM) + dimIdx]);
    }
    distanceMatrix[tid] = sqrtf(totalDistSquared);
  }
}

  // Part 2: Generate the set of K nearest neighbors and select one to add to the built words
  __global__ void generateKNNBaseline(float * dataset, float * resultSet, float * distanceMatrix, const unsigned int WTG,
                  const unsigned int NN, const unsigned int colIdx, const unsigned int N, const unsigned int DIM)
{
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  int KNNSet[NUMNEIGHBORS];
  for (int i = 0 ; i < NN; i++)
  {
    KNNSet[i] = -i;
  }
  float foundMin;
  bool skipIdx = false;
  curandState state;
  curand_init(42, tid, 0, &state);

  //resultSet instead of newWordArr
  if(tid < WTG)
  {
    //loop through one row in the distanceMatrix
    for(int minNeighborIdx = 0; minNeighborIdx < NN; minNeighborIdx++)
    { 
      foundMin = DEFAULT_MIN;
      for (int distanceIdx = 0 ; distanceIdx < N; distanceIdx++)
      {
        skipIdx = false;
        for (int foundMinIdx = 0; foundMinIdx < minNeighborIdx && !skipIdx; foundMinIdx++)
        {
          skipIdx = distanceIdx == KNNSet[foundMinIdx];
        }
        if (!skipIdx && distanceMatrix[tid*N + distanceIdx] < foundMin)
        {
          foundMin = distanceMatrix[tid*N + distanceIdx];
          KNNSet[minNeighborIdx] = distanceIdx;
        }
      } 
    }
    int randomNeighbor = (int)generateRandomNumber(&state, NN);
    resultSet[tid * DIM + colIdx] = dataset[KNNSet[randomNeighbor]*DIM + colIdx];
    
  }

}
