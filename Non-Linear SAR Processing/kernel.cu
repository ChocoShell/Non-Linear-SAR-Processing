/*
Programming the algorithms in Non-Linear SAR Processing paper
1. Choose a transmitted radar signal
4. Simulate measured SAR Data signal in fast and slow time (signal frequency, plane)
5. Convolution or FFT and IFFT for equation 6
6. Code equation 12
7. Section 3.5 mx, my = xi, yi = location of reflector

//Further Stuff
Read Section 4 and 5 as separate paper

//New Plan
0. Figure out how to plot 3d graphs in C++/CUDA
1. Randomly Generate Map (X v Y with amplitudes -> 250 x 100, x = 1000:2:1500, y = 100:200)
2. Plot Map on Figure 1
3. Generate Plane Function length 100 -> x in range [0,50] y in range [50,250]
4. Plot Plane on Figure 1
5. Generate Radar Signal (Simple Chirp)
6. Plot Radar Signal on Figure 2
7. Simulate SAR Data -> Get delay and attenuation(magnitude) from map(x,y), add gaussian noise to return signal
                     -> Do this 100 times 
8. Get sm(x,y,u) = sm(t,u) = s(x, y, u) * p*(-t) -> Same length use as function indices not matching delays
9. Integrate sm(x,y,u) over u to get f(x,y)
10. Plot f(x,y) compare with the first map

// Help from:
http://nehe.gamedev.net/tutorial/your_first_polygon/13002/
http://www.wikihow.com/Make-a-Cube-in-OpenGL

New New Plan
Write own convolution function
plot sM in matlab

*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <list>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

// CUDA Libraries
#include <cufft.h>
#include <cufftw.h>
#include <curand_kernel.h>

#include <vector_types.h>

using namespace std;

#define BLOCK_SIZE 64

#define PI 3.1415926535

// Old code
typedef struct { 
  int width; 
  int height; 
  cufftComplex* signal; 
} cufftComplexMatrix;

const long int spd_of_light = 299792458;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

void split_line(string& line, string delim, list<string>& values)
{
    size_t pos = 0;
    while ((pos = line.find(delim, (pos + 1))) != string::npos) {
        string p = line.substr(0, pos);
        values.push_back(p);
        line = line.substr(pos + 1);
    }

    if (!line.empty()) {
        values.push_back(line);
    }
}

__global__ void MV_complex_kernel(cufftComplex *matrix_signal, cufftComplex *vector_signal, const int width, const int batch)
{
	cufftComplex ducks;
	unsigned int r = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int c = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (width > r && batch > c)
	{
		unsigned int ind = (c * batch) + r;

		ducks.x = matrix_signal[ind].x * vector_signal[r].x - matrix_signal[ind].y * vector_signal[r].y;
		ducks.y = matrix_signal[ind].x * vector_signal[r].y + vector_signal[r].x * matrix_signal[ind].y;

		matrix_signal[ind] = ducks;
	}
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void shmem_reduce_kernel(float *d_out, const float *d_in)
{
	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ float sdata[];

	unsigned int s;
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	// load global data into shared memory
	sdata[tid] = d_in[id];

	// make sure entire block is loaded
	__syncthreads();  
	
	// Reduction in shared memory
	for (s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// only thread 0 writes result back to global mem for current block
	if(tid == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}
}

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void map_maker(curandState *states, float *out)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float result;

	curandState localState = states[id];

	result = sqrt((float) id*(BLOCK_SIZE * BLOCK_SIZE - id)) * (10 - curand_uniform(&localState)) /10;
	
	states[id] = localState;
	
	out[id] = result;
}

__global__ void generate_uniform_kernel(curandState *state, float *result, const float shift, const float normalizer)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random uniforms */
    x = shift + curand_uniform(&localState) * normalizer;
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] = x;
}

__global__ void flattenKernel(cufftComplex *matrix_signal, cufftComplex *out_signal, const int width, const int batch)
{
	cufftComplex ducks;
	unsigned int r = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (width > r)
	{
		for (int i = 0; i < batch; i ++)
		{
			ducks.x += matrix_signal[i * batch + r].x;
			ducks.y += matrix_signal[i * batch + r].y;
		}
		out_signal[r] = ducks;
	}
}

void map_maker_helper(curandState *states, float *hostOut)
{
	float *devOut;

	cudaMalloc((void **)&devOut, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));

	cudaMemset(devOut, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));

	cudaMalloc((void **)&states, BLOCK_SIZE * BLOCK_SIZE * sizeof(curandState));

	//setup_kernel<<<BLOCK_SIZE, BLOCK_SIZE>>>(states);

	map_maker<<<BLOCK_SIZE, BLOCK_SIZE>>>(states, devOut);

	cudaMemcpy(hostOut, devOut, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(devOut);
}

void generator_uniform_wrapper(curandState *devStates, float *hostOut, const float shift, const float normalizer)
{
	float *devOut;
	
	cudaMalloc((void **)&devOut, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));

	cudaMemset(devOut, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));

	cudaMalloc((void **)&devStates, BLOCK_SIZE * BLOCK_SIZE * sizeof(curandState));

	setup_kernel<<<BLOCK_SIZE, BLOCK_SIZE>>>(devStates);

	generate_uniform_kernel<<<BLOCK_SIZE, BLOCK_SIZE>>>(devStates, devOut, shift, normalizer );

	cudaMemcpy(hostOut, devOut, BLOCK_SIZE * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(devOut);
}

__global__ void conv_mat_vec_kernel(cufftComplex *matrix, cufftComplex *vector, cufftComplex *out, const unsigned int width, const unsigned int batch)
{
	unsigned int fid = threadIdx.x;
	unsigned int uid = blockIdx.x;
	unsigned int maxLen = 2*width - 1;
	float matX, matY, vecX, vecY;
	unsigned int start, end;
	
	cufftComplex sum;
	sum.x = 0;
	sum.y = 0;
	
	if (fid < maxLen) 
	{
		if (fid < width)
		{
			start = width - fid - 1; // greater or equal to this
			end   = width; // less than this
		}
		else
		{
			start = 0;
			end = maxLen - fid; //less than this
		}
		matX = matrix[uid*width + 437 - start].x;
		matY = matrix[uid*width + 437 - start].y;
		vecX = vector[start].x;
		vecY = vector[start].y;

		for(start; start < end; start++)
		{
			sum.x += matX*vecX - matY*vecY;
			sum.y += matX*vecY + matY*vecX;
		}
		out[uid * maxLen + fid] = sum;
	}
}

void convolveWithCuda(cufftComplex *unknown_signal_block, cufftComplex *template_signal, cufftComplex *hOut, const int width, const int batch)
{	
	cufftComplex *data, *temp, *out;

	// FFT of return signal matrix
	cudaMalloc((void**)&data, sizeof(cufftComplex)*width*batch);
	if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate data\n");
		return;
	}

	cudaMalloc((void**)&out, sizeof(cufftComplex)*(2*width - 1)*batch);
	if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate data\n");
		return;
	}

	cudaMalloc((void**)&temp, sizeof(cufftComplex)*width);
	if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate temp\n");
		return;
	}

	cudaMemcpy(data, unknown_signal_block, sizeof(cufftComplex)*batch*width, cudaMemcpyHostToDevice);
	cudaMemcpy(temp, template_signal, sizeof(cufftComplex)*width, cudaMemcpyHostToDevice);

	conv_mat_vec_kernel<<<160, 896>>>(data, devOut, out, width, batch);

	cudaMemcpy(hOut, out, sizeof(cufftComplex)*(2*width-1)*batch, cudaMemcpyDeviceToHost);

	cudaFree(data);
	cudaFree(temp);
	cudaFree(out);
	return;
}

int main()
{
	//new code
	cudaDeviceReset();
	cudaSetDevice(0);

	// Get all data
	ifstream fastTimeFilter ("fastTimeFilter.csv");
	ifstream imagsRaw ("imagsRaw.csv");
	ifstream realsRaw ("realsRaw.csv");
	// End of data read

	float d, i;
	int count = 437;
	cufftComplex *sRaw, *signal, *hOut;
	string value;
	hOut   = (cufftComplex *)malloc(sizeof(cufftComplex)*(438*2 - 1)*160);
	signal = (cufftComplex *)malloc(sizeof(cufftComplex)*438);
	sRaw   = (cufftComplex *)malloc(sizeof(cufftComplex)*438*160);
    list<string> values;

    while ( fastTimeFilter.good() )
    {
        getline ( fastTimeFilter, value, ',' ); // read a string until next comma: http://www.cplusplus.com/reference/string/getline/
        if (value.find('\n') != string::npos) {
            split_line(value, "\n", values);
        } else {
            values.push_back(value);
        }
    }

    list<string>::const_iterator it = values.begin();
    for (it = values.begin(); it != values.end(); it++) {
        string tmp = *it;
        d = stof(tmp.c_str(), NULL);
		it++;
		tmp = *it;
		i = stof(tmp.c_str(), NULL);
		signal[count].x = d;
		signal[count].y = -1 * i;
		count--;
        //cout << "Double val: " << right << showpoint << d << endl;
    }
	
	string value1;
    list<string> values1;
    while ( imagsRaw.good() )
    {
        getline ( imagsRaw, value1, ',' ); // read a string until next comma: http://www.cplusplus.com/reference/string/getline/
        if (value1.find('\n') != string::npos) {
            split_line(value1, "\n", values1);
        } else {
            values1.push_back(value1);
        }
    }

    it = values1.begin();
	count = 0;
    for (it = values1.begin(); it != values1.end(); it++) {
        string tmp = *it;
        d = stof(tmp.c_str(), NULL);
	    sRaw[count].y = d;
		count++;
        //cout << "Double val: " << right << showpoint << d << endl;
    }

	string value2;
	list<string> values2;
    while ( realsRaw.good() )
    {
        getline ( realsRaw, value2, ',' ); // read a string until next comma: http://www.cplusplus.com/reference/string/getline/
        if (value2.find('\n') != string::npos) {
            split_line(value2, "\n", values2);
        } else {
            values2.push_back(value2);
        }
    }

    it = values2.begin();
	count = 0;
    for (it = values2.begin(); it != values2.end(); it++) {
        string tmp = *it;
        d = stof(tmp.c_str(), NULL);
		sRaw[count].x = d;
		count++;
        //cout << "Double val: " << right << showpoint << d << endl;
    }

	convolveWithCuda(sRaw, signal, hOut, 438, 160);
	//hOut is sM after convolution to of raw data with p signal
	free(signal);
	free(sRaw);
	cudaDeviceReset();
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

//example function for s
float2 s()
{
	float2 total;

	//curandStatus_t curandGenerateUniform(generator, float *outputPtr, size_t num);
	float rand = 0;
	// Magnitude rand bound to some distance e.g., 400 to 450 km
	total.x = rand;
	// Phase bound by 0 to 2pi
	total.y = rand;
	return total;
}

//example function for sm
float2 sm( const float2 reflector, const float2 plane, const float u) 
{
	float delay;
	float2 total;
	
	delay = 2 * sqrt( ((reflector.x - plane.x)*(reflector.x - plane.x)) + ((reflector.y - plane.y)*(reflector.y - plane.y)) ) / spd_of_light;
	total.x = delay;
	total.y = u;
	return total;
}
