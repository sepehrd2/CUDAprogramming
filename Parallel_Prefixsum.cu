// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 256 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void total(float *input, float *output, int len) {
  
  __shared__ float partialSum[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;
  if (start + t < len)
    partialSum[t] = input[start + t];
  else
    partialSum[t] = 0.0;
  
  if (start + blockDim.x + t < len){
  partialSum[blockDim.x + t] = input[start + blockDim.x + t];
  }
  else{
    partialSum[blockDim.x + t] = 0.0;
  }
  for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2)
  {
  __syncthreads();
  if (t < stride)
  partialSum[t] += partialSum[t + stride];
  } 
  output[blockIdx.x] = partialSum[0];
}

__global__ void scan(float *input, float *output, int len) {

  __shared__ float T[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;
  if (start + 2 * t < len)
    T[2 * t] = input[start + 2 * t];
  else
    T[2 * t] = 0.0;
  
  if (start + 2 * t + 1 < len){
  T[2 * t + 1] = input[start + 2 * t + 1];
  }
  else{
    T[2 * t + 1] = 0.0;
  }

  int stride = 1;
  while(stride < 2 * BLOCK_SIZE){
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if(index < 2 * BLOCK_SIZE && (index - stride) >= 0)
      T[index] += T[index - stride];
    stride *= 2;
  }

  stride = BLOCK_SIZE / 2;
  while(stride >= 1){
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if((index + stride) < 2 * BLOCK_SIZE)
      T[index + stride] += T[index];
    stride = int(stride * 0.5);
  }

  output[start + 2 * t]     = T[2 * t];
  output[start + 2 * t + 1] = T[2 * t + 1];
}

__global__ void scan_m(float *input, float *output, float *PSUM, int len) {
  
  __shared__ float T[2 * BLOCK_SIZE];  
  unsigned int t = threadIdx.x;
  int b = blockIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;
  if (start + 2 * t < len)
    T[2 * t] = input[start + 2 * t];
  else
    T[2 * t] = 0.0;
  
  if (start + 2 * t + 1 < len){
  T[2 * t + 1] = input[start + 2 * t + 1];
  }
  else{
    T[2 * t + 1] = 0.0;
  }

  int stride = 1;
  while(stride < 2 * BLOCK_SIZE){
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if(index < 2 * BLOCK_SIZE && (index - stride) >= 0)
      T[index] += T[index - stride];
    stride *= 2;
  }

  stride = BLOCK_SIZE / 2;
  while(stride >= 1){
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if((index + stride) < 2 * BLOCK_SIZE)
      T[index + stride] += T[index];
    stride = int(stride * 0.5);
  }
  __syncthreads();
 
  if(blockIdx.x == 0)
  {
    output[start + 2 * t]     = T[2 * t];
    output[start + 2 * t + 1] = T[2 * t + 1];

  }
  else
  {
    output[start + 2 * t]     = T[2 * t]     + PSUM[b - 1];
    output[start + 2 * t + 1] = T[2 * t + 1] + PSUM[b - 1];
  }
}

int main(int argc, char **argv) {
  
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceSum;
  float *devicePSum;

  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);
  int numSum = ceil(numElements/(2.0 * BLOCK_SIZE));
  int size   = numElements * sizeof(float);
  int size_m = numSum * sizeof(float);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, size));
  wbCheck(cudaMalloc((void **)&deviceOutput, size));
  wbCheck(cudaMalloc((void **)&deviceSum, size_m));
  wbCheck(cudaMalloc((void **)&devicePSum, size_m));
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, size, cudaMemcpyHostToDevice));
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid (ceil(numElements/(2.0 * BLOCK_SIZE)), 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 DimGrid_m (1, 1, 1);

  total<<<DimGrid, DimBlock>>>(deviceInput, deviceSum, numElements);
  scan<<<DimGrid_m, DimBlock>>>(deviceSum, devicePSum, numSum);
  scan_m<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, devicePSum, numElements);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost));
  
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(devicePSum);
  cudaFree(deviceSum);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

