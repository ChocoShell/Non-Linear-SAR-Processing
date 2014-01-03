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

//kernel functions
__global__ void square_kernel(cuComplex *d_vector, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= length || col >= width) {return;}

    d_out[width*row + col] = cuCmulf(d_vector[width*row + col], d_vector[width*row + col]);
}
__global__ void sqrt_abs_kernel(cuComplex *d_in, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= length || col >= width) {return;}

    d_out[width*row + col].x = rsqrtf(cuCabsf(d_in[width*row + col]));
    d_out[width*row + col].y = 0;
}
__global__ void real_to_imag_kernel(cuComplex *d_in, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= length || col >= width) {return;}
    
    d_out[width*row + col].x = 0;
    d_out[width*row + col].y = d_in[width*row + col].x;
}
__global__ void vec_vec_mult_kernel(cuComplex *d_vec1, cuComplex *d_vec2, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int newRow, ind;

    if (row*BLOCK_SIZE >= length || col >= width) {return;}

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        newRow = (BLOCK_SIZE*row) + i;
        if(newRow < length)
        {
            ind = col + width*newRow;
            d_out[ind] = cuCmulf(d_vec1[ind], d_vec2[ind]);
        }
    }
}
__global__ void vec_vec_mat_kernel(cuComplex *d_vec1, cuComplex *d_vec2, cuComplex *d_out, const unsigned int len_1, const unsigned int len_2)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= len_1 || col >= len_2) {return;}

    d_out[len_2*row + col] = cuCmulf(d_vec1[row], d_vec2[col]);
}
__global__ void sca_vec_add_kernel(const double K, cuComplex *d_vector, const unsigned length, const unsigned int width, const double M)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= length || col >= width) {return;}

    d_vector[width*row + col].x = K + M*d_vector[width*row + col].x;
    d_vector[width*row + col].y = M*d_vector[width*row + col].y;
}
__global__ void sca_vec_mult_kernel(const double K, cuComplex *d_vector, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int newRow;

    if (row*BLOCK_SIZE >= length || col >= width) {return;}

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        newRow = (BLOCK_SIZE*row) + i;
        if(newRow < length)
        {
            d_vector[col + width*newRow].x *= K;
            d_vector[col + width*newRow].y *= K;
        }
    }
}
__global__ void transpose_kernel(cuComplex *d_matrix, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int newRow;

    if (row*BLOCK_SIZE >= length || col >= width) {return;}

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        newRow = (BLOCK_SIZE*row) + i;
        if(newRow < length)
        {
            d_out[length*col + newRow] = d_matrix[col + width*newRow];
        }
    }
    return;
}
__global__ void fftshift_kernel(cuComplex *d_signal, cuComplex *d_out, const unsigned int width, const unsigned int batch, const int dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ((row >= batch) || (col >= width))
        return;

    int newcol = col;
    int newrow = row;

    if (dim != 1) 
        newcol = (col + width/2 +3) % width;

    if (dim != 2) 
        newrow = (row + batch/2 +3) % batch;
    
    d_out[newcol + newrow*width] = d_signal[col + row*width];
}
__global__ void map_kernel(cuComplex *s_M, cuComplex *out, const unsigned int width, const unsigned int batch, const unsigned int max_x, const unsigned int max_y)
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
    cuComplex val;
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
    return;
}
__global__ void mat_vec_mult_kernel(cuComplex *matrix, cuComplex *vector, cuComplex *out, const unsigned int width, const unsigned int batch)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch || col >= width) {return;}

    out[row*width + col] = cuCmulf(matrix[row*width + col], vector[col]);
}

// kernel helpers
void square(cuComplex *h_vector, cuComplex *h_out, const unsigned int length, const unsigned int width)
{
    cuComplex *d_vector, *d_out;

    cudaMalloc((void**)&d_vector, sizeof(cuComplex)*width*length);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_out, sizeof(cuComplex)*width*length);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMemcpy(d_vector, h_vector, sizeof(cuComplex)*width*length,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(width/threadsPerBlock.x + 1, length/threadsPerBlock.y + 1);

    square_kernel<<<numOfBlocks, threadsPerBlock>>>(d_vector, d_vector, length, width);

    cudaMemcpy(h_vector, d_out, sizeof(cuComplex)*width*length,
               cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to host failed\n");
		return;
	}

    cudaFree(d_out);
    cudaFree(d_vector);
}
void sqrt_abs(cuComplex *h_in, cuComplex *h_out, const unsigned int length, const unsigned int width)
{
    cuComplex *d_in, *d_out;

    cudaMalloc((void**)&d_in, sizeof(cuComplex)*width*length);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_out, sizeof(cuComplex)*width*length);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMemcpy(d_in, h_in, sizeof(cuComplex)*width*length,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(width/threadsPerBlock.x + 1, length/threadsPerBlock.y + 1);

    sqrt_abs_kernel<<<numOfBlocks, threadsPerBlock>>>(d_in, d_out, length, width);

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*width*length,
               cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to host failed\n");
		return;
	}

    cudaFree(d_in);
    cudaFree(d_out);
}
void real_to_imag(cuComplex *h_in, cuComplex *h_out, const unsigned int length, const unsigned int width)
{
    cuComplex *d_in, *d_out;

    cudaMalloc((void**)&d_in, sizeof(cuComplex)*width*length);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_out, sizeof(cuComplex)*width*length);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMemcpy(d_in, h_in, sizeof(cuComplex)*width*length,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(width/threadsPerBlock.x + 1, length/threadsPerBlock.y + 1);

    real_to_imag_kernel<<<numOfBlocks, threadsPerBlock>>>(d_in, d_out, length, width);

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*width*length,
               cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to host failed\n");
		return;
	}

    cudaFree(d_in);
    cudaFree(d_out);
}
void vec_vec_mult(cuComplex *h_vec1, cuComplex *h_vec2, const unsigned int length, const unsigned int width)
{
    //Element wise multiplication of 2 vectors, output is placed in h_vec1
    cuComplex *d_vec1, *d_vec2, *d_out;

    cudaMalloc((void**)&d_vec1, sizeof(cuComplex)*width*length);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_vec2, sizeof(cuComplex)*width*length);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_out, sizeof(cuComplex)*width*length);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    //Copying vectors onto device
    cudaMemcpy(d_vec1, h_vec1, sizeof(cuComplex)*width*length,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    cudaMemcpy(d_vec2, h_vec2, sizeof(cuComplex)*width*length,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(width/(threadsPerBlock.x + BLOCK_SIZE) + 1, length/threadsPerBlock.y + 1);

    vec_vec_mult_kernel<<<numOfBlocks, threadsPerBlock>>>(d_vec1, d_vec2, d_out, length, width);

    cudaMemcpy(h_vec1, d_out, sizeof(cuComplex)*width*length,
               cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to host failed\n");
		return;
	}

    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_out);
}
void vec_vec_mat(cuComplex *h_vec1, cuComplex *h_vec2, cuComplex *h_out, const unsigned int len_1, const unsigned int len_2)
{
    cuComplex *d_vec1, *d_vec2, *d_out;

    cudaMalloc((void**)&d_vec1, sizeof(cuComplex)*len_1);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_vec2, sizeof(cuComplex)*len_2);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_out, sizeof(cuComplex)*len_1*len_2);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    //Copying vectors onto device
    cudaMemcpy(d_vec1, h_vec1, sizeof(cuComplex)*len_1,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    cudaMemcpy(d_vec2, h_vec2, sizeof(cuComplex)*len_2,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(len_2/threadsPerBlock.x + 1, len_1/threadsPerBlock.y + 1);

    vec_vec_mat_kernel<<<numOfBlocks, threadsPerBlock>>>(d_vec1, d_vec2, d_out, len_1, len_2);

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*len_2*len_1,
               cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}
    
    cudaFree(d_out);
    cudaFree(d_vec1);
    cudaFree(d_vec2);
}
void sca_vec_add(const double K, cuComplex *h_vector, const unsigned length, const unsigned int width, const double M)
{
    cuComplex *d_vector;

    cudaMalloc((void**)&d_vector, sizeof(cuComplex)*width*length);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMemcpy(d_vector, h_vector, sizeof(cuComplex)*width*length,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(width/threadsPerBlock.x + 1, length/threadsPerBlock.y + 1);

    sca_vec_add_kernel<<<numOfBlocks, threadsPerBlock>>>(K, d_vector, length, width, M);

    cudaMemcpy(h_vector, d_vector, sizeof(cuComplex)*width*length,
               cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    cudaFree(d_vector);
}
void sca_vec_mult(const double K, cuComplex *h_vector, const unsigned int length, const unsigned int width)
{
    cuComplex *d_vector;

    cudaMalloc((void**)&d_vector, sizeof(cuComplex)*width*length);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    //Copying vector onto device
    cudaMemcpy(d_vector, h_vector, sizeof(cuComplex)*width*length,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(width/(threadsPerBlock.x + BLOCK_SIZE) + 1, length/threadsPerBlock.y + 1);

    sca_vec_mult_kernel<<<numOfBlocks, threadsPerBlock>>>(K, d_vector, length, width);

    cudaMemcpy(h_vector, d_vector, sizeof(cuComplex)*width*length,
               cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to host failed\n");
		return;
	}

    cudaFree(d_vector);
}
void transpose(cuComplex *h_matrix, const unsigned int width, const unsigned int batch)
{
    cuComplex *d_matrix, *d_out, curr;

    cudaMalloc((void**)&d_matrix, sizeof(cuComplex)*width*batch);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_out, sizeof(cuComplex)*width*batch);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for output\n");
		return;
	}

    //Copying matrix onto device
    cudaMemcpy(d_matrix, h_matrix, sizeof(cuComplex)*width*batch,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(width/(threadsPerBlock.x + BLOCK_SIZE) + 1, batch/threadsPerBlock.y + 1);

    transpose_kernel<<<numOfBlocks, threadsPerBlock>>>(d_matrix, d_out, width, batch);

    cudaMemcpy(h_matrix, d_out, sizeof(cuComplex)*width*batch, cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to host failed\n");
		return;
	}

    cudaFree(d_matrix);
    cudaFree(d_out);

    return;
}
void fftshift(cuComplex *h_signal, const unsigned int width, const unsigned int batch, const int dim)
{
    cuComplex *d_signal, *d_out, curr;

    cudaMalloc((void**)&d_signal, sizeof(cuComplex)*width*batch);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for signal\n");
		return;
	}

    cudaMalloc((void**)&d_out, sizeof(cuComplex)*width*batch);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for output\n");
		return;
	}

    //Copying matrix onto device
    cudaMemcpy(d_signal, h_signal, sizeof(cuComplex)*width*batch,
               cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(width/threadsPerBlock.x + 1, batch/threadsPerBlock.y + 1);

    fftshift_kernel<<<numOfBlocks, threadsPerBlock>>>(d_signal, d_out, width, batch, dim);

    cudaMemcpy(h_signal, d_out, sizeof(cuComplex)*width*batch,
               cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_signal);
    
    return;
}
void mapMaker(cuComplex *s_M, cuComplex *mapOut, const unsigned int width, const unsigned int batch, const unsigned int mapLength, const unsigned int mapWidth)
{// Multiplies vector of certain width by each row in matrix
    cuComplex *dS_M, *dMapOut;

    //Allocating memory on GPU
    cudaMalloc((void**)&dS_M, sizeof(cuComplex)*width*batch);
	if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate data\n");
		return;
	}

    cudaMalloc((void**)&dMapOut, sizeof(cuComplex)*mapLength*mapWidth);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate data\n");
		return;
	}
    //Finished Allocation

    //Copying matrix onto device
    cudaMemcpy(dS_M, s_M, sizeof(cuComplex)*width*batch, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(mapLength/threadsPerBlock.x + 1, mapWidth/threadsPerBlock.y + 1);

    map_kernel<<<numOfBlocks, threadsPerBlock>>>(dS_M, dMapOut, width, batch, mapLength, mapWidth);

    cudaMemcpy(mapOut, dMapOut, sizeof(cuComplex)*mapLength*mapWidth, cudaMemcpyDeviceToHost);

    //Printing map values to console.
    cudaFree(dS_M);
    cudaFree(dMapOut);
    return;
}
void mat_vec_mult(cuComplex *h_matrix, cuComplex *h_vector, cuComplex *h_out, const unsigned int width, const unsigned int batch)
{
    cuComplex *d_matrix, *d_vector, *d_out;

    cudaMalloc((void**)&d_matrix, sizeof(cuComplex)*width*batch);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_vector, sizeof(cuComplex)*width);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_out, sizeof(cuComplex)*width*batch);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMemcpy(d_matrix, h_matrix, sizeof(cuComplex)*width*batch,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    cudaMemcpy(d_vector, h_vector, sizeof(cuComplex)*width,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
    }

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(width/threadsPerBlock.x + 1, batch/threadsPerBlock.y + 1);

    mat_vec_mult_kernel<<<numOfBlocks, threadsPerBlock>>>(d_matrix, d_vector, d_out, width, batch);

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*width*batch, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_matrix);
    cudaFree(d_vector);
}

// Produces Compression Constants
void comp_decomp(const float Xc, cuComplex *uc, const int length,  cuComplex *u, const int u_len, cuComplex *k, const int width)
{
    cuComplex *compression, *decompression;

    compression = (cuComplex *)malloc(sizeof(cuComplex)*length*width);
    decompression = (cuComplex *)malloc(sizeof(cuComplex)*u_len*width);

    // fftshift uc
    fftshift(uc, length, 1, 0);
    
    // Square each element
    square(uc, uc, length, 1);
    square(u, u, u_len, 1);

    // add constant to vector
    sca_vec_add(Xc*Xc, uc, length, 1, 1);
    sca_vec_add(Xc*Xc, u, u_len, 1, 1);

    // sqrt(abs complex vector)
    sqrt_abs(uc, uc, length, 1);
    sqrt_abs(u, u, u_len, 1);

    // subtract contant from vector
    sca_vec_add(-1.0*Xc, uc, length, 1, 1.0);
    // Xc - u
    sca_vec_add(Xc, u, u_len, 1, -1.0);

    // change real vector imaginary vector
    real_to_imag(k, k, width, 1);
    sca_vec_mult(2.0, k, width, 1);

    // mult vec vec to matrix
    vec_vec_mat(uc, k, compression, length, width);
    vec_vec_mat(k, u, decompression, width, u_len);

    // exp mat

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

	cuComplex curr, *sRaw, *d_sRaw, *signal, *d_signal, *sM, *mapOut, *out_signal;
    cufftHandle plan;
	
	signal = (cuComplex *)malloc(sizeof(cuComplex)*width);
	sRaw   = (cuComplex *)malloc(sizeof(cuComplex)*width*batch);
    out_signal = (cuComplex *)malloc(sizeof(cuComplex)*width*batch);
    
    //Copying Data from CSV files into memory
    string value;
    list<string> values;
    int count = width - 1;

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

    //---------------------------------------------------------------------------------------------------------------------

    // Output for convolution
    //sM = (cuComplex *)malloc(sizeof(cuComplex)*tots*batch);

	//convolveWithCuda(sRaw, signal, sM, width, batch);

    transpose(sRaw, width, batch);

    cudaMalloc((void**)&d_sRaw, sizeof(cuComplex)*width*batch);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate data\n");
		return;
	}

    cudaMemcpy(d_sRaw, sRaw, sizeof(cuComplex)*width*batch,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    cudaMalloc((void**)&d_signal, sizeof(cuComplex)*width);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate data\n");
		return;
	}

    cudaMemcpy(d_signal, signal, sizeof(cuComplex)*width,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device for signal\n");
		return;
	}

    cufftPlan1d(&plan, width, CUFFT_C2C, batch);
    cufftExecC2C(plan, d_sRaw, d_sRaw, CUFFT_FORWARD);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: cufft failed\n");
		return;
	}

    cudaMemcpy(sRaw, d_sRaw, sizeof(cuComplex)*width*batch,
               cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy from device failed\n");
		return;
	}
    cufftDestroy(plan);
    
    fftshift(sRaw, width, batch, 2);

    fftshift(signal, width, 1, 0);

    mat_vec_mult(sRaw, signal, sRaw, width, batch);

    for(int x = 0; x < batch; x++)
    {
        for(int y = 0; y < width; y++)
        {
            curr = sRaw[x* width + y];
            printf("%g + (%gi), ", cuCrealf(curr), cuCimagf(curr));
        }
        cout << endl;
    }

    cudaFree(d_sRaw);
    cudaFree(d_signal);
    free(sRaw);
    free(signal);
    free(out_signal);
	return 0;
}
