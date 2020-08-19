#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define TILE_WIDTH 4
#define BLOCK_WIDTH (MASK_WIDTH + TILE_WIDTH - 1)
//@@ Define constant memory for device kernel here
__constant__ float MASK[MASK_WIDTH * MASK_WIDTH * MASK_WIDTH]; 

__global__ void conv3d(float *input, float *output, float * deviceKernel, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x; 
  int ty = threadIdx.y; 
  int tz = threadIdx.z; 
  
  int x = blockIdx.x * TILE_WIDTH + tx; 
  int y = blockIdx.y * TILE_WIDTH + ty; 
  int z = blockIdx.z * TILE_WIDTH + tz; 
  
  __shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];
  
  if((x >= 0 ) && (x < x_size) && (y >= 0) && (y < y_size) && (z >= 0) && (z < z_size))
  { 
    float SUM = 0.0f;
    
   for (int z_val = 0; z_val < MASK_WIDTH; z_val++)
   {
     for(int y_val = 0; y_val < MASK_WIDTH; y_val++)
     {
       for(int x_val = 0; x_val < MASK_WIDTH; x_val++)
       {
         int x_i, y_i, z_i; 
         x_i = x - MASK_RADIUS + x_val; 
         y_i = y - MASK_RADIUS + y_val; 
         z_i = z - MASK_RADIUS + z_val; 
          if((x_i >= 0 ) && (x_i < x_size) && (y_i >= 0) && (y_i < y_size) && (z_i >= 0) && (z_i < z_size))
          {
            N_ds[tz][ty][tx] = input[(z_i * x_size * y_size) + (y_i * x_size) + x_i];
            SUM += deviceKernel[(z_val * MASK_WIDTH * MASK_WIDTH) + (y_val * MASK_WIDTH) + x_val] * N_ds[tz][ty][tx]; 
          }
           else 
           {
             N_ds[tz][ty][tx] = 0.0f;
           }
           __syncthreads(); 
       
       }
     }
   }
    output[(z * x_size * y_size) + (y * x_size) + x] = SUM;  
  }
   
  
}
int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostKernel_1;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;
  float *deviceKernel;

  args = wbArg_read(argc, argv);

  
  hostInput  = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);


  // Import data
  int size_i   = inputLength * sizeof(float);
  int size_o   = (inputLength - 3) * sizeof(float);
  hostOutput   = (float*)malloc(size_i);
  hostKernel_1 = (float*)malloc(kernelLength * sizeof(float));


  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  wbLog(TRACE, "the kernel size ", kernelLength);
  wbLog(TRACE, "the input size " , inputLength);
  //for (int i = 0; i < 27; ++i)   wbLog(TRACE, "kernel " , hostKernel[i]);
  
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
    cudaMalloc(&deviceInput , size_o);
    cudaMalloc(&deviceOutput, size_o);
    cudaMalloc(&deviceKernel, kernelLength * sizeof(float));


  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceKernel, hostKernel, kernelLength * sizeof(float) , cudaMemcpyHostToDevice);


  cudaMemcpyToSymbol(MASK, hostKernel, kernelLength * sizeof(float) , cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");
  
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid (ceil(x_size/(1.0 * TILE_WIDTH)), ceil(y_size/(1.0 * TILE_WIDTH)), ceil(z_size/(1.0 * TILE_WIDTH)));
  dim3 DimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);
  
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, deviceKernel, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;

  wbSolution(args, hostOutput, inputLength);
  //cudaMemcpyFromSymbol(hostKernel_1, MASK, kernelLength * sizeof(float) , cudaMemcpyDeviceToHost);
  //for (int i = 0; i < 27; ++i)   wbLog(TRACE, "kernel " , hostKernel[i]);


  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

