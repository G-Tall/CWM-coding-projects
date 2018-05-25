/**********************************************************************************
	This code performs a calculation of pi using the monte carlo method 
       using cuda GPU parallelisation. 
       
       Created by: George Tall
       Email: george.tall@seh.ox.ac.uk

/*********************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

__global__ void point_test(int *N, float *d_x, float *d_y, float *d_R, float *d_A){

	long int index = blockDim.x*blockIdx.x + threadIdx.x;
	//printf("%d\n", index);
        // Now each value of d_R is computed
        d_R[index] = d_x[index]*d_x[index] + d_y[index]*d_y[index];
        
        //printf("Thread %d d_R is %f \n", index, d_R[index]);
        
        // sync threads at this point to prevent deadlock
        __syncthreads();

	if(d_R[index] < 1.0f) atomicAdd(&d_A[blockIdx.x], 1);

        //printf("\nPoints in block %d = %d", blockIdx.x, d_A[blockIdx.x]);	
}

__global__ void area_reduction(float *d_A){
        
    // allocate shared memory 
    extern __shared__ float shared_array[];
    
    // copy passed array into shared array
    int tid = threadIdx.x;
    //long int index = blockIdx.x*blockDim.x + threadIdx.x;
    shared_array[tid] = d_A[tid];
      __syncthreads();
    
    for(long int d = blockDim.x/2; d > 0; d /= 2){
        if(tid < d){
            atomicAdd(&shared_array[tid], shared_array[tid+d]);
      }           
      __syncthreads();
    }
    __syncthreads();
    
    // if you're the first thread get the value from shared array
    if(tid == 0){
        d_A[0] = shared_array[0];
    }
}

int main() {
    
    // N is the number of random points.
    // area stores the number of random points that fall into
    // the area of the quadrant of a circle of radius 1
    //size_t N = 2^10;
    int N = 6536;
    float area=0;
    
    //initalize the GPU
    int nBlocks = N/256;
    int nThreads = 256;
    
    int deviceid = 0; // using GPU with id 0
    int devCount;
    // gets number of GPU available
    cudaGetDeviceCount(&devCount);
    // check if we have enough GPUs
    if(deviceid<devCount) {
       // tell CUDA that we want to use GPU 0
       cudaSetDevice(deviceid);
    }
    else return(1);

    //random variable gen
    curandGenerator_t gen;
    
    //pointers to host memory and device memory we have a pointer for a radius in the device to calculate that before conditional is operated
    //we are also going to have an area count per block to prevent confusion in the kernal if statement later
    float *h_x, *h_y, *h_A;
    float *d_x, *d_y, *d_R, *d_A;
    
    //allocate host memory
    h_x = (float*)malloc(N*sizeof(float));
    h_y = (float*)malloc(N*sizeof(float));
    h_A = (float*)malloc(nBlocks*sizeof(float));

    //allocate device memory
    cudaMalloc((void**)&d_x, N*sizeof(float));
    cudaMalloc((void**)&d_y, N*sizeof(float));
    cudaMalloc((void**)&d_R, N*sizeof(float));
    cudaMalloc((void**)&d_A, nBlocks*sizeof(float));

    // Create a pseudo-random number generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    
    // Set a seed
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    
    // Generate N pseudo random numbers on device for x
    curandGenerateUniform(gen, d_x, N);
    curandGenerateUniform(gen, d_y, N);

    // Kernal for testing if points lie in area or not
    point_test<<<nBlocks, nThreads>>>(&N, d_x, d_y, d_R, d_A);

    // Syncronise the device here in order for all the blocks to finish calculating their area data points
    cudaDeviceSynchronize();
    
    // Kernal for reducing the sum of the areas of the blocks
    area_reduction<<<nBlocks/nThreads, nThreads, nBlocks*sizeof(float)>>>(d_A);
 
    //Copy the generated numbers back to host
    // I've bought the other data back aside from h_A because it appears my reduction doesn't work
    cudaMemcpy(h_x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A, d_A, nBlocks*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < nBlocks; i++){
        printf("%f \n",h_A[i]);
    }
   
    area = h_A[0];

    printf("\nPi from reduction:\t%f\n", (4.0*area)/(float)N);

    // reset area to zero so that we can do a monte carlo method on the host, this now limits the number of points we can use
    area = 0;
    for(int i = 0; i < N; i++){
        if(h_x[i]*h_x[i] + h_y[i]*h_y[i] < 1.0f) area++;
    }

    printf("\nPi from host:\t%f\n", (4.0*area)/(float)N);
/*
    area = 0;	   
    for(int i=0; i<N; i++) {
        double x = ((double)rand())/RAND_MAX;
        double y = ((double)rand())/RAND_MAX;
	if(x*x + y*y <= 1.0) area++;
    }

    printf("\nPi:\t%f\n", (4.0*area)/(double)N);
*/  
    // Free memory on host and device
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_R); cudaFree(d_A);
    free(h_x); free(h_y); free(h_A);
   
    return(0);
}
