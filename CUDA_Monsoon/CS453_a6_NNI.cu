#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>
#include <math.h>
#include <cuda.h>
#include <cudnn.h>

const int arrSize = 4096460; // number of characters in flattenedTree.txt
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

#define cudnnErrCheck(ans) { cudnnErrCheck_((ans), __FILE__, __LINE__); }
void cudnnErrCheck_(cudnnStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUDNN_STATUS_SUCCESS) {
      fprintf(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void initGPUData_ker(float *data, int numElements, float value) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid < numElements) {
      data[tid] = value;
   }
}

void initGPUData(float *data, int numElements, float value) {
   dim3 gridDim;
   dim3 blockDim;
   
   blockDim.x = 1024;
   gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
   
   initGPUData_ker <<< gridDim, blockDim >>> (data, numElements, value);
}

// load information from .txt file CPU
void loadDataFromFileCPU(const char* filename, int* myDataArr)
{
    FILE *myLanguageData;
    myLanguageData = fopen(filename, "r");
    if (!myLanguageData) {
        perror("Error opening file");
        return;
    }
    
    char line[1024]; // Buffer to hold each line from the file
    int i = 0;
    while(fgets(line, sizeof(line), myLanguageData)) {
        char val1;
        int val2, val3;
        char val4;
        if(sscanf(line, "%c, %d, %d, %c", &val1, &val2, &val3, &val4) == 4) {
            // Convert characters to ASCII values and store in array
            myDataArr[i++] = (int)val1;
            myDataArr[i++] = val2;
            myDataArr[i++] = val3;
            myDataArr[i++] = (int)val4;
        }
    }

    fclose(myLanguageData);
}

using namespace std;


//function prototypes


//Part 1: Kernel Prototypes

// kernel 1:


int main(int argc, char* argv[]) {

   int seqLength;


   int numLayers;
   int hiddenSize;
   int inputSize;
   int miniBatch;
   float dropout = 0.0;
   bool bidirectional = 0;
   int mode = 2;
   int persistent = 0;
   float paddingFill = 0.0;

   //int* myDataArr= new int [arrSize];

   //loadDataFromFileCPU("../Output/flattenedTree.txt", myDataArr);


   // output data
   FILE *fp;
   fp=fopen("result.txt","w");

   // seqLength just needs to be the longest path in the training data
   // numLayers should be 8 I think
   // 
   if (argc == 5) {
      seqLength = atoi(argv[1]);
      numLayers = atoi(argv[2]);
      hiddenSize = atoi(argv[3]);
      inputSize = hiddenSize;
      miniBatch = atoi(argv[4]);

   }
   else {
      printf("Usage:\n");
      printf("./RNN <seqLength> <numLayers> <hiddenSize> <miniBatch>\n");
      return 1;
   }
   int seqLengthArray[miniBatch];
   for (int i = 0; i < miniBatch; i++)
   {
      seqLengthArray[i] = seqLength;
   }

   // -------------------------   
   // Create cudnn context
   // -------------------------  
   cudnnHandle_t cudnnHandle;   
   cudnnErrCheck(cudnnCreate(&cudnnHandle));

   
   // -------------------------   
   // Set up inputs and outputs
   // -------------------------
   
   void *x;
   void *hx = NULL;
   void *cx = NULL;
   
   void *dx;
   void *dhx = NULL;
   void *dcx = NULL;
  
   void *y;
   void *hy = NULL;
   void *cy = NULL;
   
   void *dy;
   void *dhy = NULL;
   void *dcy = NULL;
   
   // Memory allocation. hx, cx, dhx, dcx, hy, cy, dhy and dcy can be NULL.
   gpuErrchk(cudaMalloc((void**)&x, seqLength * inputSize * miniBatch * sizeof(float)));
   gpuErrchk(cudaMalloc((void**)&hx, numLayers * hiddenSize * miniBatch * sizeof(float)));
   gpuErrchk(cudaMalloc((void**)&cx, numLayers * hiddenSize * miniBatch * sizeof(float)));
   
   gpuErrchk(cudaMalloc((void**)&dx, seqLength * inputSize * miniBatch * sizeof(float)));
   gpuErrchk(cudaMalloc((void**)&dhx, numLayers * hiddenSize * miniBatch * sizeof(float)));
   gpuErrchk(cudaMalloc((void**)&dcx, numLayers * hiddenSize * miniBatch * sizeof(float)));
   
   gpuErrchk(cudaMalloc((void**)&y, seqLength * hiddenSize * miniBatch * sizeof(float)));
   gpuErrchk(cudaMalloc((void**)&hy, numLayers * hiddenSize * miniBatch * sizeof(float)));
   gpuErrchk(cudaMalloc((void**)&cy, numLayers * hiddenSize * miniBatch * sizeof(float)));
   
   gpuErrchk(cudaMalloc((void**)&dy, seqLength * hiddenSize * miniBatch * sizeof(float)));
   gpuErrchk(cudaMalloc((void**)&dhy, numLayers * hiddenSize * miniBatch * sizeof(float)));
   gpuErrchk(cudaMalloc((void**)&dcy, numLayers * hiddenSize * miniBatch * sizeof(float)));

   // Set up tensor descriptors. x/y/dx/dy are arrays, one per time step.
   cudnnRNNDataDescriptor_t *xDesc, *yDesc;
   cudnnTensorDescriptor_t *dxDesc, *dyDesc;
   cudnnTensorDescriptor_t hxDesc, cxDesc;
   cudnnTensorDescriptor_t hyDesc, cyDesc;
   cudnnTensorDescriptor_t dhxDesc, dcxDesc;
   cudnnTensorDescriptor_t dhyDesc, dcyDesc;
   
   xDesc = (cudnnRNNDataDescriptor_t*)malloc(seqLength * sizeof(cudnnRNNDataDescriptor_t));
   yDesc = (cudnnRNNDataDescriptor_t*)malloc(seqLength * sizeof(cudnnRNNDataDescriptor_t));
   dxDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
   dyDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
   
   int dimA[3];
   int strideA[3];

   // In this example dimA[1] is constant across the whole sequence
   // This isn't required, all that is required is that it does not increase.
   for (int i = 0; i < seqLength; i++) {
      cudnnErrCheck(cudnnCreateRNNDataDescriptor(&xDesc[i]));
      cudnnErrCheck(cudnnCreateRNNDataDescriptor(&yDesc[i]));
      //cudnnErrCheck(cudnnCreateTensorDescriptor(&xDesc[i]));
      //cudnnErrCheck(cudnnCreateTensorDescriptor(&yDesc[i]));
      cudnnErrCheck(cudnnCreateTensorDescriptor(&dxDesc[i]));
      cudnnErrCheck(cudnnCreateTensorDescriptor(&dyDesc[i]));

      //maxSeqLength and batchSize

      dimA[0] = seqLength;
      dimA[1] = miniBatch;
      dimA[2] = 1;
     
      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;

      cudnnErrCheck(cudnnSetRNNDataDescriptor(xDesc[i], CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, seqLength, miniBatch, seqLength, seqLengthArray, (void*)&paddingFill));
      cudnnErrCheck(cudnnSetTensorNdDescriptor(dxDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
      
      
      dimA[0] = miniBatch;
      dimA[1] = hiddenSize;
      dimA[2] = 1;

      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;
      
      cudnnErrCheck(cudnnSetRNNDataDescriptor(yDesc[i], CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, seqLength, miniBatch, seqLength, seqLengthArray, (void*)&paddingFill));
      cudnnErrCheck(cudnnSetTensorNdDescriptor(dyDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
   }
   
   
   dimA[0] = numLayers;
   dimA[1] = miniBatch;
   dimA[2] = hiddenSize;
   
   strideA[0] = dimA[2] * dimA[1];
   strideA[1] = dimA[2];
   strideA[2] = 1;
   
   cudnnErrCheck(cudnnCreateTensorDescriptor(&hxDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&cxDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&hyDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&cyDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&dhxDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&dcxDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&dhyDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&dcyDesc));
   
   cudnnErrCheck(cudnnSetTensorNdDescriptor(hxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(cxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(hyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(cyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(dhxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(dcxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(dhyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(dcyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
  
  
   // -------------------------
   // Set up the dropout descriptor (needed for the RNN descriptor)
   // -------------------------
   unsigned long long seed = 1337ull; // Pick a seed.
   
   cudnnDropoutDescriptor_t dropoutDesc;
   cudnnErrCheck(cudnnCreateDropoutDescriptor(&dropoutDesc));
   
   // How much memory does dropout need for states?
   // These states are used to generate random numbers internally
   // and should not be freed until the RNN descriptor is no longer used
   size_t stateSize;
   void *states;
   cudnnErrCheck(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));
   
   gpuErrchk(cudaMalloc(&states, stateSize));
   
   cudnnErrCheck(cudnnSetDropoutDescriptor(dropoutDesc, cudnnHandle, dropout, states, stateSize, seed));
                             
   // -------------------------   
   // Set up the RNN descriptor
   // -------------------------
   cudnnRNNDescriptor_t rnnDesc;
   cudnnRNNMode_t RNNMode;
   cudnnRNNAlgo_t RNNAlgo;
   
   cudnnErrCheck(cudnnCreateRNNDescriptor(&rnnDesc));
   
   RNNMode = CUDNN_LSTM;
   
   // Persistent RNNs are only supported on Pascal+ GPUs.
   if      (persistent == 0) RNNAlgo = CUDNN_RNN_ALGO_STANDARD;
   else if (persistent == 1) RNNAlgo = CUDNN_RNN_ALGO_PERSIST_STATIC;
   else if (persistent == 2) RNNAlgo = CUDNN_RNN_ALGO_PERSIST_DYNAMIC;
      
   //cudnnErrCheck(cudnnSetRNNDescriptor_v6(cudnnHandle, rnnDesc, hiddenSize, numLayers, dropoutDesc,
   //                                       CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, 
   //                                       RNNMode, RNNAlgo, CUDNN_DATA_INT32));

  cudnnErrCheck(cudnnSetRNNDescriptor_v8( rnnDesc, RNNAlgo, RNNMode, CUDNN_RNN_SINGLE_INP_BIAS, 
                            CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                            CUDNN_DEFAULT_MATH, inputSize, hiddenSize, hiddenSize, numLayers, dropoutDesc, 0));
   
   
   // -------------------------
   // Set up parameters
   // -------------------------
   // This needs to be done after the rnn descriptor is set as otherwise
   // we don't know how many parameters we have to allocate
   void *w;   
   void *dw;   

   cudnnFilterDescriptor_t wDesc, dwDesc;
   
   cudnnErrCheck(cudnnCreateFilterDescriptor(&wDesc));
   cudnnErrCheck(cudnnCreateFilterDescriptor(&dwDesc));
   
   size_t weightSpaceSize;
   cudnnErrCheck(cudnnGetRNNWeightSpaceSize(cudnnHandle, rnnDesc, &weightSpaceSize));
   
   int dimW[3];   
   dimW[0] =  weightSpaceSize / sizeof(int32_t);
   dimW[1] = 1;
   dimW[2] = 1;
      
   cudnnErrCheck(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));   
   cudnnErrCheck(cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));   
   
   gpuErrchk(cudaMalloc((void**)&w,  weightSpaceSize));
   gpuErrchk(cudaMalloc((void**)&dw, weightSpaceSize));
   
   
   // -------------------------
   // Set up work space and reserved memory
   // -------------------------   
   void *workspace;
   void *reserveSpace;   
   
   size_t workSize;
   size_t reserveSize;

   // Need for every pass
   int testSeqLengthArray[3];
   cudnnRNNDataLayout_t testLayout;
   cudnnDataType_t testDataType;
   int testMaxSeqLength, testBatchSize, testVectorSize, testArrayLengthRequested;
   float paddingValue;

   // Assuming xDesc is already created and configured
   cudnnErrCheck(cudnnGetRNNDataDescriptor(*xDesc, &testDataType, &testLayout, &testMaxSeqLength, &testBatchSize, &testVectorSize, miniBatch, testSeqLengthArray, (void*)&paddingValue));

   printf("Data type: %d\n", testDataType);
   printf("testMaxSeqLength: %d\n", testMaxSeqLength);
   printf("testBatchSize: %d\n", testBatchSize);
   printf("testVectorSize: %d\n", testVectorSize);
   printf("paddingValue: %f\n", paddingValue);
   for (int i = 0; i < miniBatch; i++)
   {
      printf("Sequence length: %d\n", testSeqLengthArray[i]);
   }

   cudnnErrCheck(cudnnGetRNNTempSpaceSizes(cudnnHandle, rnnDesc, CUDNN_FWD_MODE_TRAINING, *xDesc, &workSize, &reserveSize));
   // Only needed in training, shouldn't be touched between passes.
   //cudnnErrCheck(cudnnGetRNNWorkspaceSize(cudnnHandle, rnnDesc, CUDNN_FWD_MODE_INFERENCE, xDesc, &workSize, &reserveSize));
    
   gpuErrchk(cudaMalloc((void**)&workspace, workSize));
   gpuErrchk(cudaMalloc((void**)&reserveSpace, reserveSize));
   
   // *********************************************************************************************************
   // Initialise weights and inputs
   // *********************************************************************************************************
   // We initialise to something simple.
   // Matrices are initialised to 1 / matrixSize, biases to 1, data is 1.
   initGPUData((float*)x, seqLength * inputSize * miniBatch, 1.f);
   if (hx != NULL) initGPUData((float*)hx, numLayers * hiddenSize * miniBatch, 1.f);
   if (cx != NULL) initGPUData((float*)cx, numLayers * hiddenSize * miniBatch, 1.f);
   
   initGPUData((float*)dy, seqLength * hiddenSize * miniBatch, 1.f);
   if (dhy != NULL) initGPUData((float*)dhy, numLayers * hiddenSize * miniBatch, 1.f);
   if (dcy != NULL) initGPUData((float*)dcy, numLayers * hiddenSize * miniBatch, 1.f);
      
   
   // Weights
   // 8 Layers for LSTM.
   // 2 for traditional RNN, 6 for GRU
   int numLinearLayers = 8;
   
   for (int layer = 0; layer < numLayers; layer++) {
      for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
         
         // linear layer matrix descriptor + variable initializations
         cudnnTensorDescriptor_t linLayerMatDesc;
         cudnnErrCheck(cudnnCreateTensorDescriptor(&linLayerMatDesc));
         cudnnErrCheck(cudnnSetTensorNdDescriptor(linLayerMatDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
         int *linLayerMat;

         // linear layer bias descriptor + variable initializations
         cudnnTensorDescriptor_t linLayerBiasDesc;
         cudnnErrCheck(cudnnCreateTensorDescriptor(&linLayerBiasDesc));
         cudnnErrCheck(cudnnSetTensorNdDescriptor(linLayerBiasDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
         int *linLayerBias;

         //cudnnErrCheck(cudnnGetRNNLinLayerMatrixParams( cudnnHandle, rnnDesc, layer, xDesc[0], wDesc, w,
                                                        //linLayerID, linLayerMatDesc, (void**)&linLayerMat));
         
         cudnnDataType_t dataType;
         int nbDims;
         int biasDimA[3];
         int matrixDimA[3];
         int biasStrideA[3];
         int matrixStrideA[3];
         // these calls get info about the bias descriptor and matrix descriptor dimensions so their data can be initialized
         cudnnErrCheck(cudnnGetTensorNdDescriptor(linLayerMatDesc, 3, &dataType, &nbDims, matrixDimA, matrixStrideA));                                             
         cudnnErrCheck(cudnnGetTensorNdDescriptor(linLayerBiasDesc, 3, &dataType, &nbDims, biasDimA, biasStrideA));
                  
         

         
         // This function is used to obtain the start address and shape of every RNN weight matrix and bias vector in each pseudo-layer within the recurrent network.
         cudnnGetRNNWeightParams( cudnnHandle, rnnDesc, layer, weightSpaceSize, w, 
                                                linLayerID, linLayerMatDesc, (void **)&linLayerMat, linLayerBiasDesc, (void **)&linLayerBias);

         //cudnnErrCheck(cudnnGetRNNLinLayerBiasParams( cudnnHandle, rnnDesc, layer, xDesc[0], wDesc, w, 
                                                      //linLayerID, linLayerBiasDesc, (void**)&linLayerBias));
         
         

         initGPUData((float*)linLayerMat, matrixDimA[0] * matrixDimA[1] * matrixDimA[2], 1.f / (float)(matrixDimA[0] * matrixDimA[1] * matrixDimA[2]));                             
         initGPUData((float*)linLayerBias, biasDimA[0] * biasDimA[1] * biasDimA[2], 1.f);

         cudnnErrCheck(cudnnDestroyTensorDescriptor(linLayerMatDesc));
         cudnnErrCheck(cudnnDestroyTensorDescriptor(linLayerBiasDesc));
      }
   }
   
   // *********************************************************************************************************
   // Dynamic persistent RNN plan (if using this algo)
   // *********************************************************************************************************
   cudnnPersistentRNNPlan_t rnnPlan;
   if (RNNAlgo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
      // Note: This step is expensive. Once completed the plan can be reused so long as the descriptor
      //       minibatch or datatype don't change.
      cudnnErrCheck(cudnnBuildRNNDynamic(cudnnHandle, rnnDesc, miniBatch));
      //cudnnErrCheck(cudnnCreatePersistentRNNPlan(rnnDesc, miniBatch, CUDNN_DATA_INT32, &rnnPlan));
      // Tell calls using this descriptor which plan to use.
      //cudnnErrCheck(cudnnSetPersistentRNNPlan(rnnDesc, rnnPlan));
   }
   
   // *********************************************************************************************************
   // At this point all of the setup is done. We now need to pass through the RNN.
   // *********************************************************************************************************
   gpuErrchk(cudaDeviceSynchronize());
   
   cudaEvent_t start, stop;
   float timeForward, timeBackward1, timeBackward2;
   gpuErrchk(cudaEventCreate(&start));
   gpuErrchk(cudaEventCreate(&stop));
   
   gpuErrchk(cudaEventRecord(start));   

   // If we're not training we use this instead
   /* cudnnErrCheck(cudnnRNNForwardInference(cudnnHandle, rnnDesc, seqLength,                                          
                                             xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, 
                                             yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSize)); */


   /*cudnnErrCheck(cudnnRNNForwardTraining(cudnnHandle, rnnDesc, seqLength,                                       
                                         xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, 
                                         yDesc, y, hyDesc, hy, cyDesc, cy, 
                                         workspace, workSize, reserveSpace, reserveSize));*/
   cudnnErrCheck(cudnnRNNForward( cudnnHandle, rnnDesc, CUDNN_FWD_MODE_TRAINING, NULL, *xDesc, x, *yDesc, y, hxDesc, hx, hy,
                    cxDesc, cx, cy, weightSpaceSize, w, workSize, workspace, reserveSize, reserveSpace));
                
   gpuErrchk(cudaEventRecord(stop));   
   gpuErrchk(cudaEventSynchronize(stop));
   gpuErrchk(cudaEventElapsedTime(&timeForward, start, stop));
   
   gpuErrchk(cudaEventRecord(start));
   

   /*cudnnErrCheck(cudnnRNNBackwardData(cudnnHandle, rnnDesc, seqLength,                                
                               yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, 
                               wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, 
                               dhxDesc, dhx, dcxDesc, dcx,
                               workspace, workSize, reserveSpace, reserveSize ));*/

   cudnnErrCheck(cudnnRNNBackwardData_v8( cudnnHandle, rnnDesc, NULL, *yDesc, y, dy, *xDesc, dx, hxDesc, hx, dhy, dhx, cxDesc, cx, dcy, dcx,
                            weightSpaceSize, w, workSize, workspace, reserveSize, reserveSpace));
   
   gpuErrchk(cudaEventRecord(stop));   
   gpuErrchk(cudaEventSynchronize(stop));
   gpuErrchk(cudaEventElapsedTime(&timeBackward1, start, stop));
   
   gpuErrchk(cudaEventRecord(start));
   
   // cudnnRNNBackwardWeights adds to the data in dw.
   gpuErrchk(cudaMemset(dw, 0, weightSpaceSize));
   
   /*cudnnErrCheck(cudnnRNNBackwardWeights( cudnnHandle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y,
                                    workspace, workSize, dwDesc, dw, reserveSpace, reserveSize ));*/

   cudnnErrCheck(cudnnRNNBackwardWeights_v8( cudnnHandle, rnnDesc, CUDNN_WGRAD_MODE_ADD, NULL, *xDesc, x, hxDesc, hx,
                               *yDesc, y, weightSpaceSize, dw, workSize, workspace, reserveSize, reserveSpace));

   gpuErrchk(cudaEventRecord(stop));   

   gpuErrchk(cudaEventSynchronize(stop));
   gpuErrchk(cudaEventElapsedTime(&timeBackward2, start, stop));

   
   int numMats = 8;
   
   //if (RNNMode == CUDNN_RNN_RELU || RNNMode == CUDNN_RNN_TANH) {
      //numMats = 2;
   //}
   //else if (RNNMode == CUDNN_LSTM) {
      //numMats = 8;
   //}
   //else if (RNNMode == CUDNN_GRU) {
      //numMats = 6;
   //}
   
   // Calculate FLOPS
   printf("Forward: %3.0f GFLOPS\n", numMats * 2ull * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeForward));
   printf("Backward: %3.0f GFLOPS, ", numMats * 4ull * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * (timeBackward1 + timeBackward2)));
   printf("(%3.0f GFLOPS), ", numMats * 2ull * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward1));
   printf("(%3.0f GFLOPS)\n", numMats * 2ull * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward2));

   // Calculate FLOPS
   fprintf(fp,"Forward: %3.0f GFLOPS\n", numMats * 2ull * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeForward));
   fprintf(fp,"Backward: %3.0f GFLOPS, ", numMats * 4ull * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * (timeBackward1 + timeBackward2)));
   fprintf(fp,"(%3.0f GFLOPS), ", numMats * 2ull * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward1));
   fprintf(fp,"(%3.0f GFLOPS)\n", numMats * 2ull * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward2));

   // Make double-sure everything is finished before we copy for result checking.
   cudaDeviceSynchronize();
   
   // *********************************************************************************************************
   // Print checksums.
   // *********************************************************************************************************
   /*
   if (true) {
      float* testOutputi;
      float* testOutputh;
      float* testOutputc;
      
      int biDirScale = 1;
      
      testOutputi = (float*)malloc(hiddenSize * seqLength * miniBatch * biDirScale * sizeof(float));
      testOutputh = (float*)malloc(hiddenSize * miniBatch * numLayers * biDirScale * sizeof(float));
      testOutputc = (float*)malloc(hiddenSize * miniBatch * numLayers * biDirScale * sizeof(float));
 
      gpuErrchk(cudaMemcpy(testOutputi, y, hiddenSize * seqLength * miniBatch * biDirScale * sizeof(float), cudaMemcpyDeviceToHost));
      if (hy != NULL) gpuErrchk(cudaMemcpy(testOutputh, hy, numLayers * hiddenSize * miniBatch * biDirScale * sizeof(float), cudaMemcpyDeviceToHost));
      if (cy != NULL && RNNMode == CUDNN_LSTM) gpuErrchk(cudaMemcpy(testOutputc, cy, numLayers * hiddenSize * miniBatch * biDirScale * sizeof(float), cudaMemcpyDeviceToHost));
      
      double checksumi = 0.f;
      double checksumh = 0.f;
      double checksumc = 0.f;
      
      for (int m = 0; m < miniBatch; m++) {
         double localSumi = 0;
         double localSumh = 0;
         double localSumc = 0;
         
         for (int j = 0; j < seqLength; j++) {
            for (int i = 0; i < hiddenSize * biDirScale; i++) {   
               localSumi += testOutputi[j * miniBatch * hiddenSize * biDirScale + m * hiddenSize * biDirScale + i];
            }
         }
         for (int j = 0; j < numLayers * biDirScale; j++) {
            for (int i = 0; i < hiddenSize; i++) {         
               if (hy != NULL) localSumh += testOutputh[j * hiddenSize * miniBatch + m * hiddenSize + i];
               if (cy != NULL) if (RNNMode == CUDNN_LSTM) localSumc += testOutputc[j * hiddenSize * miniBatch + m * hiddenSize + i];
            }
         }
                  
         checksumi += localSumi;
         checksumh += localSumh;
         checksumc += localSumc;
      }
      
      printf("i checksum %E     ", checksumi);
      fprintf(fp,"i checksum %E     ", checksumi);
      if (RNNMode == CUDNN_LSTM) { printf("c checksum %E     ", checksumc); fprintf(fp,"c checksum %E     ", checksumc); }
      printf("h checksum %E\n", checksumh);
      fprintf(fp,"h checksum %E\n", checksumh);
      
      free(testOutputi);
      free(testOutputc);
      free(testOutputh);
   }   
   
   if (true) {
      float* testOutputdi;
      float* testOutputdh;
      float* testOutputdc;

      int biDirScale = (bidirectional ? 2 : 1);
      
      testOutputdi = (float*)malloc(inputSize * seqLength * miniBatch * sizeof(float));
      testOutputdh = (float*)malloc(hiddenSize * miniBatch * numLayers * biDirScale * sizeof(float));
      testOutputdc = (float*)malloc(hiddenSize * miniBatch * numLayers * biDirScale * sizeof(float));
      gpuErrchk(cudaMemcpy(testOutputdi, dx, seqLength * miniBatch * inputSize * sizeof(float), cudaMemcpyDeviceToHost));
      if (dhx != NULL) gpuErrchk(cudaMemcpy(testOutputdh, dhx, numLayers * hiddenSize * miniBatch * biDirScale * sizeof(float), cudaMemcpyDeviceToHost));
      if (dcx != NULL) if (RNNMode == CUDNN_LSTM) gpuErrchk(cudaMemcpy(testOutputdc, dcx, numLayers * hiddenSize * miniBatch * biDirScale * sizeof(float), cudaMemcpyDeviceToHost));
      
      float checksumdi = 0.f;
      float checksumdh = 0.f;
      float checksumdc = 0.f;
      
      for (int m = 0; m < miniBatch; m++) {
         double localSumdi = 0;
         double localSumdh = 0;
         double localSumdc = 0;

         for (int j = 0; j < seqLength; j++) {
            for (int i = 0; i < inputSize; i++) {
               localSumdi += testOutputdi[j * miniBatch * inputSize + m * inputSize + i];
            }
         }

         for (int j = 0; j < numLayers * biDirScale; j++) {
            for (int i = 0; i < hiddenSize; i++) {         
               localSumdh += testOutputdh[j * hiddenSize * miniBatch + m * hiddenSize + i];
               if (RNNMode == CUDNN_LSTM) localSumdc += testOutputdc[j * hiddenSize * miniBatch + m * hiddenSize + i];
            }
         }         

         checksumdi += localSumdi;
         checksumdh += localSumdh;
         checksumdc += localSumdc;
         
      }
      
      printf("di checksum %E    ", checksumdi);
      fprintf(fp,"di checksum %E    ", checksumdi);
      if (RNNMode == CUDNN_LSTM) { printf("dc checksum %E    ", checksumdc); fprintf(fp,"dc checksum %E    ", checksumdc); }
      printf("dh checksum %E\n", checksumdh);
      fprintf(fp,"dh checksum %E\n", checksumdh);
      
      free(testOutputdi);
      free(testOutputdh);
      free(testOutputdc);
   }

   if (true) {
      float* testOutputdw;
      testOutputdw = (float*)malloc(weightSpaceSize);
 
      gpuErrchk(cudaMemcpy(testOutputdw, dw, weightSpaceSize, cudaMemcpyDeviceToHost));
      
      double checksumdw = 0.;
            
      for (int i = 0; i < weightSpaceSize / sizeof(float); i++) {
         checksumdw += testOutputdw[i];
      }
      
      printf("dw checksum %E\n", checksumdw);
      fprintf(fp,"dw checksum %E\n", checksumdw);
      
      free(testOutputdw);
   }

  */
   if (RNNAlgo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
      //cudnnDestroyPersistentRNNPlan(rnnPlan);
   }  
  
   cudaFree(x);
   cudaFree(hx);
   cudaFree(cx);
   cudaFree(y);
   cudaFree(hy);
   cudaFree(cy);
   cudaFree(dx);
   cudaFree(dhx);
   cudaFree(dcx);
   cudaFree(dy);
   cudaFree(dhy);
   cudaFree(dcy);
   cudaFree(workspace);
   cudaFree(reserveSpace);
   cudaFree(w);
   cudaFree(dw);
   
   cudnnDestroy(cudnnHandle);
   //delete[] myDataArr;
   fclose(fp);
   return 0;
}

// Kernel Implementation