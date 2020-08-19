#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32

__global__ void FToUI(float *input, unsigned char *output, int w, int h) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    int i = blockIdx.z * (w * h) + y * (w) + x;
    output[i] = (unsigned char) (255 * input[i]);
  }
}
__global__ void rgbToGray(unsigned char *input, unsigned char *output, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    int i = y * (w) + x;
    unsigned char r = input[3 * i];
    unsigned char g = input[3 * i + 1];
    unsigned char b = input[3 * i + 2];
    output[i] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
  }
}
__global__ void grayScaleToHist(unsigned char *input, unsigned int *output, int w, int h) {

  __shared__ unsigned int hist[HISTOGRAM_LENGTH];

  int tIdx = threadIdx.x + threadIdx.y * blockDim.x;
  if (tIdx < HISTOGRAM_LENGTH) {
    hist[tIdx] = 0;
  }

  __syncthreads();
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    int idx = y * (w) + x;
    unsigned char buff = input[idx];
    atomicAdd(&(hist[buff]), 1);
  }
  __syncthreads();
  if (tIdx < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[tIdx]), hist[tIdx]);
  }
}
__global__ void histToCDF(unsigned int *input, float *output, int w, int h) {
  __shared__ unsigned int cdf[HISTOGRAM_LENGTH];
  int x = threadIdx.x;
  cdf[x] = input[x];

  for (unsigned int stride = 1; stride <= HISTOGRAM_LENGTH / 2; stride *= 2) {
    __syncthreads();
    int idx = (x + 1) * 2 * stride - 1;
    if (idx < HISTOGRAM_LENGTH) {
      cdf[idx] += cdf[idx - stride];
    }
  }
  for (int stride = HISTOGRAM_LENGTH / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int idx = (x + 1) * 2 * stride - 1;
    if (idx + stride < HISTOGRAM_LENGTH) {
      cdf[idx + stride] += cdf[idx];
    }
  }

  __syncthreads();
  output[x] = cdf[x] / ((float) (w * h));
}
__global__ void equalize(unsigned char *inout, float *cdf, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    int idx = blockIdx.z * (w * h) + y * (w) + x;
    unsigned char buff = inout[idx];

    float equalized = 255 * (cdf[buff] - cdf[0]) / (1.0 - cdf[0]);
    float fin   = min(max(equalized, 0.0), 255.0);

    inout[idx] = (unsigned char) (fin);
  }
}
__global__ void UIToF(unsigned char *input, float *output, int w, int h) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    int i = blockIdx.z * (w * h) + y * (w) + x;
    output[i] = (float) (input[i] / 255.0);
  }
}

int main(int argc, char **argv) {

  wbArg_t args;
  const char *inputImageFile;

  int imageWidth;
  int imageHeight;
  int imageChannels;

  wbImage_t inputImage;
  wbImage_t outputImage;

  float *hostInputImageData;
  float *hostOutputImageData;

  float   *deviceImageFloat;
  unsigned char *deviceImageUChar;
  unsigned char *deviceImageUCharGrayScale;
  unsigned int  *deviceImageHistogram;
  float   *deviceImageCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage          = wbImport(inputImageFile);
  imageWidth          = wbImage_getWidth(inputImage);
  imageHeight         = wbImage_getHeight(inputImage);
  imageChannels       = wbImage_getChannels(inputImage);
  hostInputImageData  = wbImage_getData(inputImage);
  outputImage         = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  printf("%d, %d, %d\n", imageWidth, imageHeight, imageChannels);

  cudaMalloc((void**) &deviceImageFloat,          imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceImageUChar,          imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void**) &deviceImageUCharGrayScale, imageWidth * imageHeight *                 sizeof(unsigned char));
  cudaMalloc((void**) &deviceImageHistogram,      HISTOGRAM_LENGTH *                         sizeof(unsigned int));
  cudaMemset((void *) deviceImageHistogram, 0,    HISTOGRAM_LENGTH *                         sizeof(unsigned int));
  cudaMalloc((void**) &deviceImageCDF,            HISTOGRAM_LENGTH *                         sizeof(float));


  cudaMemcpy(deviceImageFloat, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid;
  dim3 dimBlock;

  dimGrid  = dim3(ceil(imageWidth/(1.0 * BLOCK_SIZE)), ceil(imageHeight/(1.0 * BLOCK_SIZE)), imageChannels);
  dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

  FToUI<<<dimGrid, dimBlock>>>(deviceImageFloat, deviceImageUChar, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid  = dim3(ceil(imageWidth/(1.0 * BLOCK_SIZE)), ceil(imageHeight/(1.0 * BLOCK_SIZE)), 1);
  dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

  rgbToGray<<<dimGrid, dimBlock>>>(deviceImageUChar, deviceImageUCharGrayScale, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid  = dim3(ceil(imageWidth/(1.0 * BLOCK_SIZE)), ceil(imageHeight/(1.0 * BLOCK_SIZE)), 1);
  dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

  grayScaleToHist<<<dimGrid, dimBlock>>>(deviceImageUCharGrayScale, deviceImageHistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid  = dim3(1, 1, 1);
  dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);

  histToCDF<<<dimGrid, dimBlock>>>(deviceImageHistogram, deviceImageCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid  = dim3(ceil(imageWidth/(1.0 * BLOCK_SIZE)), ceil(imageHeight/(1.0 * BLOCK_SIZE)), imageChannels);
  dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

  equalize<<<dimGrid, dimBlock>>>(deviceImageUChar, deviceImageCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid  = dim3(ceil(imageWidth/(1.0 * BLOCK_SIZE)), ceil(imageHeight/(1.0 * BLOCK_SIZE)), imageChannels);
  dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

  UIToF<<<dimGrid, dimBlock>>>(deviceImageUChar, deviceImageFloat, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, deviceImageFloat, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  wbSolution(args, outputImage);
  cudaFree(deviceImageFloat);
  return 0;
}

