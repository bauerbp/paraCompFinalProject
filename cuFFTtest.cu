#define NX 8
#define BATCH_SIZE 1

#include "cufft.h"

#include <math.h>
#include <stdio.h>

//#include "soundfile-2.2/libsoundfile.h"
typedef float2 Complex;

void testcuFFT(){

  cufftReal *h_signal = (cufftReal *)malloc(sizeof(cufftReal) * BATCH_SIZE);
  cufftComplex *h_data = (cufftComplex *)malloc(sizeof(cufftComplex) * (NX/2+1)*BATCH_SIZE);

  float ryanSignal [NX] = {0.0,
1.15443278102,
1.50377819535,
0.957393116649,
0.19925316202,
0.0408603874003,
0.663651234058,
1.44858683588};
  // Initalize the memory for the signal
  for (unsigned int i = 0; i < NX; ++i)
  {
    //h_signal[i].x = rand() / (float)RAND_MAX+1;
    h_signal[i] = ryanSignal[i];
    //h_signal[i].y = 0;
    printf("h_signal[%u]: %f\n", i, h_signal[i]);
  }

  cufftHandle plan;
  cufftComplex *d_data;
  cufftReal *d_signal;
  cudaMalloc((void**)&d_data, sizeof(cufftComplex)*(NX/2+1)*BATCH_SIZE);
  cudaMalloc((void**)&d_signal, sizeof(cufftReal)*NX);
  //cudaMalloc((void**)&d_signal, sizeof(cufftReal)*SIGNAL_SIZE);
  cudaMemcpy(d_signal, h_signal, sizeof(cufftReal)*NX, cudaMemcpyHostToDevice);

  free(h_signal);

  if(cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return; 
  }
  if(cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH_SIZE) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return; 
  } 

  // Use the CUFFT plan to transform the signal in place. 
  if(cufftExecR2C(plan, (cufftReal*)d_signal, d_data) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
    return; 
  }
  if(cudaThreadSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    return; 
  }
  
  cudaMemcpy(h_data,d_data,sizeof(cufftComplex)*BATCH_SIZE * (NX/2+1),cudaMemcpyDeviceToHost);
  for(unsigned int k=0; k<10; k++){
    //printf("h_data[%i]: %f\n",k,h_data[k].x);
    printf("h_data[%u]: %f\n", k, h_data[k].x);
  }

  cufftDestroy(plan);
  cudaFree(d_data);
}

int main(){
  testcuFFT(); 

  return 0;
}
