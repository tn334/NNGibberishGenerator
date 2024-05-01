//example of running the program: ./A5_similarity_search_starter 7490 5 10 ../../bee_dataset_1D_feature_vectors.txt

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>
#include <math.h>

//Define any constants here
//Feel free to change BLOCKSIZE
//#define BLOCKSIZE 128

using namespace std;

//function prototypes
//Some of these are for debugging so I did not remove them from the starter file

void printDataset(unsigned int N, unsigned int DIM, float * dataset);

void importDataset(char * fname, unsigned int N, unsigned int DIM, float * dataset);

int translateLetter(char letterToTranslate);

int main(int argc, char *argv[])
{
  char inputFname[500];
  unsigned int N=0;
  unsigned int DIM=0;
  unsigned int NUMNEIGHBORS = 0;
  float epsilon=0;

  if (argc != 5) {
    fprintf(stderr,"Please provide the following on the command line: N (number of lines in the file), dimensionality (number of coordinates per point), number of neighbors, dataset filename.\n");
    exit(0);
  }

  sscanf(argv[1],"%d",&N);
  sscanf(argv[2],"%d",&DIM);
  sscanf(argv[3],"%d",&NUMNEIGHBORS);
  strcpy(inputFname,argv[4]);

  printf("\nAllocating the following amount of memory for the dataset: %f GiB\n", (sizeof(float)*N*DIM)/(1024*1024*1024.0));
  //printf("\nAllocating the following amount of memory for the distance matrix: %f GiB", (sizeof(float)*N*N)/(1024*1024*1024.0));
  

  float * dataset=(float*)malloc(sizeof(float*)*N*DIM);
  printf("Made it past the malloc of the dataset\n");
  importDataset(inputFname, N, DIM, dataset);
  printf("Made it past the import of the dataset\n");
  printDataset(N, DIM, dataset);

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

        // after strtok, field should be a pointer to a string of the form a1833p77p15l3e1
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
            dataset[rowCnt*DIM+colCnt]=(float)numTemp;
          }   

        }
        rowCnt++;
    }

    fclose(fp);

}