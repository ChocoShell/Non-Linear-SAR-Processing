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

// CUDA Libraries
#include <cufft.h>
#include <cufftw.h>
#include <curand_kernel.h>
#include <math_functions.h>

#include <vector_types.h>

#include <array>

using namespace std;

#define BLOCK_SIZE 32

#define PI 3.1415926535

const long int spd_of_light = 299792458;

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

__global__ void fftshift_kernel(cuDoubleComplex *d_signal,
                                 const unsigned int width, 
                                 const unsigned int batch)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= batch || col >= width)
        return;

    int newrow = (row + batch/2) % batch;
    int newcol = (col + width/2) % width;

    cuDoubleComplex oldVal = d_signal[col + width*row];

    __syncthreads();

    d_signal[newcol + width*newrow] = oldVal;
}

void fftshift(cuDoubleComplex *h_signal,
              const unsigned int width, const unsigned int batch)
{
    cuDoubleComplex *d_signal, curr;

    cudaMalloc((void**)&d_signal, sizeof(cuDoubleComplex)*width*batch);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for signal\n");
		return;
	}

    //Copying matrix onto device
    cudaMemcpy(d_signal, h_signal, sizeof(cuDoubleComplex)*width*batch,
               cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(width/threadsPerBlock.x + 1, batch/threadsPerBlock.y + 1);

    fftshift_kernel<<<numOfBlocks, threadsPerBlock>>>(d_signal, width, batch);

    cudaMemcpy(h_signal, d_signal, sizeof(cuDoubleComplex)*width*batch, cudaMemcpyDeviceToHost);

    cudaFree(d_signal);

    for(int x = 0; x < batch; x++)
    {
        for(int y = 0; y < width; y++)
        {
            curr = h_signal[x* width + y];
            printf("%g + (%gi), ", cuCreal(curr), cuCimag(curr));
        }
        cout << endl;
    }

    return;
}

__global__ void map_kernel(cuDoubleComplex *s_M, cuDoubleComplex *out,
                           const unsigned int width, const unsigned int batch,
                           const unsigned int max_x, const unsigned int max_y)
{   /*s_M is the fast-time matched filtered SAR signal
     *Because this signal is in discrete time and uses indices 
     *instead of values, we have to modify the time delay
     *function to fit between the range 1 and the width of s_M
    */

    // We will run a total of max_x * max_y threads,
    // looping through all possible slow time locations
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(max_y - 1 < row || max_x - 1 < col) return;

    int u;
    int delay;
    cuDoubleComplex val;
    val.x= 0;
    val.y =0;
    int uNormal = max_y/batch;
    for(u = 0; u < batch; u++)
    {   // The number 1.8821 is width/max(s_m), this will be changed later
        // It is used to evenly distribute the magnitudes of the map
        delay = lround(hypot( (double) row, (double) (col - u*uNormal))*0.01281);
        val.x += s_M[u*width + delay].x;
        val.y += s_M[u*width + delay].y;        
    }
    out[row*max_x + col] = val;
}

void mapMaker(cuDoubleComplex *s_M, cuDoubleComplex *mapOut, 
              const unsigned int width, const unsigned int batch,
              const unsigned int mapLength, const unsigned int mapWidth)
{
    cuDoubleComplex *dS_M, *dMapOut;

    //Allocating memory on GPU
    cudaMalloc((void**)&dS_M, sizeof(cuDoubleComplex)*width*batch);
	if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate data\n");
		return;
	}

    cudaMalloc((void**)&dMapOut, sizeof(cuDoubleComplex)*mapLength*mapWidth);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate data\n");
		return;
	}
    //Finished Allocation

    //Copying matrix onto device
    cudaMemcpy(dS_M, s_M, sizeof(cuDoubleComplex)*width*batch, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(mapLength/threadsPerBlock.x + 1, mapWidth/threadsPerBlock.y + 1);

    map_kernel<<<numOfBlocks, threadsPerBlock>>>(dS_M, dMapOut, width, batch, mapLength, mapWidth);

    cudaMemcpy(mapOut, dMapOut, sizeof(cuDoubleComplex)*mapLength*mapWidth, cudaMemcpyDeviceToHost);

    //Printing map values to console.
    int x,y;
    cuDoubleComplex curr;
    for(y = 0; y < mapWidth; y++)
    {
        for(x = 0; x < mapLength; x++)
        {
            curr = mapOut[y*mapLength + x];
            printf("%g + (%gi), ", cuCreal(curr), cuCimag(curr));
        }
        cout << endl;
    }
    
    cudaFree(dS_M);
    cudaFree(dMapOut);
    return;
}

__global__ void conv_mat_vec_kernel(cuDoubleComplex *matrix, 
                                    cuDoubleComplex *vector, 
                                    cuDoubleComplex *out, 
                                    const unsigned int width, 
                                    const unsigned int batch)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int maxLen = 2*width - 1;
    float matX, matY, vecX, vecY;
	unsigned int start, end;
	cuDoubleComplex sum;
	
    if (batch -1 < row || maxLen - 1 < col) return;

	sum.x = 0;
	sum.y = 0;
	
	if (col < width)
	{
		start = 0; // greater or equal to this
		end   = col +1; // less than this
	}
	else
	{
		start = col - width +1;
		end   = width; //less than this
	}

    //start and end act as Tau in the convolution equation
	for(start; start < end; start++)
	{
        matX = matrix[row*width+ start].x;
    	matY = matrix[row*width + start].y;
	    vecX = vector[col-start].x;
	    vecY = vector[col-start].y;
        sum.x += matX*vecX - matY*vecY;
		sum.y += matX*vecY + matY*vecX;
	}
	out[row*maxLen + col] = sum;	
}

void convolveWithCuda(cuDoubleComplex *unknown_signal_block, 
                      cuDoubleComplex *template_signal, 
                      cuDoubleComplex *sM, const unsigned int width, 
                      const unsigned int batch)
{	
	cuDoubleComplex *data, *temp, *out, curr;
    int x, y;
    int tots = 2*width - 1;

	//Allocating memory on CUDA device and checking for errrors
	cudaMalloc((void**)&data, sizeof(cuDoubleComplex)*width*batch);
	if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate data\n");
		return;
	}

	cudaMalloc((void**)&out, sizeof(cuDoubleComplex)*(2*width - 1)*batch);
	if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate data\n");
		return;
	}

	cudaMalloc((void**)&temp, sizeof(cuDoubleComplex)*width);
	if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate temp\n");
		return;
	}
    //Finished Allocating

    //Copying host variables onto the device
	cudaMemcpy(data, unknown_signal_block, sizeof(cuDoubleComplex)*batch*width, cudaMemcpyHostToDevice);
	cudaMemcpy(temp, template_signal,      sizeof(cuDoubleComplex)*width,       cudaMemcpyHostToDevice);
    
    //Setting up the number of threads to run
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks((2*width-1)/threadsPerBlock.x + 1, batch/threadsPerBlock.y + 1);

    conv_mat_vec_kernel<<< numOfBlocks, threadsPerBlock >>>(data, temp, out, width, batch);
    
    //Waiting until all threads are finished
    cudaDeviceSynchronize();
    
    int err = cudaGetLastError();

    //Retrieving data from device
	cudaMemcpy(sM, out, sizeof(cuDoubleComplex)*(2*width-1)*batch, cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to Synchronize\n");
        cout << err << endl;
		return;
	}

    /* Printing values to console
    for(x = 0; x < batch; x++)
    {
        for(y = 0; y < tots; y++)
        {
            curr = sM[x* tots + y];
            printf("%g + (%gi), ", cuCreal(curr), cuCimag(curr));
        }
        cout << endl;
    }*/

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
    int x,y;

    //Dimensions of sRaw data
    int width = 438;
    int batch = 160;

    //Dimensions of final map
    int mapLength = 382;
    int mapWidth  = 266;

    //Length of matched filter
    int tots = 2*width - 1;

    int count = width - 1;

	cuDoubleComplex curr, *sRaw, *signal, *sM, *mapOut;
	string value;

	signal = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex)*width);
	sRaw   = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex)*width*batch);
    
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
    //From fast time filter we get p*(-t)
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
	//Done Reading in values from files.

    // Output for convolution
    //sM = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex)*tots*batch);

	//convolveWithCuda(sRaw, signal, sM, width, batch);

    fftshift(sRaw, width, batch);
        
    //free(sM);
    free(sRaw);
    free(signal);
	return 0;
}
