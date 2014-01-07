/*
Programming the algorithms in Non-Linear SAR Processing paper
1. Choose a transmitted radar signal
4. Simulate measured SAR Data signal in fast and slow time (signal frequency, plane)
5. Convolution or FFT and IFFT for equation 6
6. Code equation 12
7. Section 3.5 mx, my = xi, yi = location of reflector

New Not Dumb Plan


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
#include "utils.h"

#include <array>

using namespace std;

#define BLOCK_SIZE 32

#define PI 3.14159265358979323846

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

void csv_real_reader(string filename, cuComplex *signal, boolean isReal, boolean zero)
{
    float d;
    int count = 0;
    ifstream csv (filename);
    
    //Copying Data from CSV files into memory
    string value;
    list<string> values;
    while ( csv.good() )
    {
        // read a string until next comma: http://www.cplusplus.com/reference/string/getline/
        getline ( csv, value, ',' ); 
        if (value.find('\n') != string::npos)
            split_line(value, "\n", values);
        else
            values.push_back(value);
    }

    list<string>::const_iterator it = values.begin();
    for (it = values.begin(); it != values.end(); it++) {
        string tmp = *it;
        d = stof(tmp.c_str(), NULL);
        if (isReal) {
		    signal[count].x = d;
            if (zero) {
                signal[count].y = 0;
            }
        } else {
            signal[count].y = d;
            if (zero) {
                signal[count].x = 0;
            }
        }
		count++;
    }
}

//kernel functions
__global__ void square_kernel(cuComplex *d_vector, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= length || col >= width) {return;}

    d_out[width*row + col].x = cuCabsf(d_vector[width*row + col]) * cuCabsf(d_vector[width*row + col]);
    d_out[width*row + col].y = 0.0;
}
__global__ void sqrt_abs_kernel(cuComplex *d_in, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= length || col >= width) {return;}

    d_out[width*row + col].x = sqrtf(cuCabsf(d_in[width*row + col]));
    d_out[width*row + col].y = 0;
}
__global__ void exp_mat_kernel(cuComplex *d_in, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= length || col >= width) {return;}

    float s, c;
    float e = expf(d_in[width*row + col].x);
    sincosf(d_in[width*row + col].y, &s, &c);

    d_out[width*row + col].x = c * e;
    d_out[width*row + col].y = s * e;
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

    if (row >= width || col >= length) {return;}

    d_out[width*col + row] = d_matrix[length*row + col];
        
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
__global__ void pad_kernel(cuComplex *d_in, cuComplex *d_out, const unsigned int length, const unsigned int width, const unsigned int padInd, const unsigned int padLength)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= width || col >= (padLength+length)) {return;}
    int newind = row*(length+padLength) + col;

    if (col < padInd)
        d_out[newind] = d_in[row*length + col];
    else if (col < padLength + padInd)
        d_out[newind] = make_cuComplex(0,0);
    else 
        d_out[newind] = d_in[row*length + col - padLength];

    return;
}
__global__ void vec_copy2mat_kernel(cuComplex *d_in, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= width || col >= length) {return;}

    d_out[length*row + col] = d_in[col];

    return;
}
__global__ void vec_vec_add_kernel(cuComplex *d_mat1, cuComplex *d_mat2, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= width || col >= length) {return;}

    int ind = length*row + col;

    d_out[ind] = cuCaddf(d_mat1[ind], d_mat2[ind]);

    return;
}
__global__ void sca_max_kernel(float K, cuComplex *d_in, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= width || col >= length) {return;}

    int ind = length*row + col;

    if (d_in[ind].x > K) 
    { 
        d_out[ind] = d_in[ind];
    }
    else 
    {
        d_out[ind].x = K;
        d_out[ind].y = 0.0;
    }

    return;
}
__global__ void is_pos_kernel(cuComplex *d_in, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= width || col >= length) {return;}

    int ind = length*row + col;

    if(d_in[ind].x > 0.0) 
        d_out[ind] = make_cuComplex(1.0, 0.0);
    else
        d_out[ind] = make_cuComplex(0.0, 0.0);
}
__global__ void round_vec_kernel(cuComplex *d_in, cuComplex *d_out, const unsigned int length, const unsigned int width)
{
    //d_in is real
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (row >= width || col >= length) {return;}

    int ind = length*row + col;

    d_out[ind] = make_cuComplex(roundf(d_in[ind].x), 0.0);

    return;
}
__global__ void spatial_inter_kernel(cuComplex *filteredSignal, float *kx, float *GridValues, int *rowidx, const unsigned int length, const unsigned int mapLength, const unsigned int mapWidth, const int nInterpSideLobes, const float dkx, const float kxs, cuComplex *out)
{
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int stack = blockDim.z * blockIdx.z + threadIdx.z;
    
    if ( col >= length || row >= mapWidth || stack >= (2*nInterpSideLobes -1)) {return;}
    
    float snc;

    int ind = length*row + col;
    int idxout = rowidx[ind] + stack;
    float sliceRange = GridValues[idxout] - kx[ind];
    
    if (sliceRange/dkx == 0.0f)
        snc = (0.54 + 0.46*cos( sliceRange * PI/kxs ));
    else
        snc = (0.54 + 0.46*cos( sliceRange * PI/kxs ))*sin(PI*sliceRange/dkx)/(PI*sliceRange/dkx);

    int image_ind = idxout + mapLength*row;
    out[image_ind].x = out[image_ind].x + filteredSignal[ind].x*snc;
    out[image_ind].y = out[image_ind].y + filteredSignal[ind].y*snc;

    return;
}
__global__ void cuComplex2Int_kernel(cuComplex *h_in, int *h_out, const unsigned int length, const unsigned int width)
{
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

     if (row >= width || col >= length) {return;}

     unsigned int ind = length*row + col;

     h_out[ind] = lroundf(cuCabsf(h_in[ind]));

    return;
}
__global__ void cuComplex2float_kernel(cuComplex *h_in, float *h_out, const unsigned int length, const unsigned int width)
{
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

     if (row >= width || col >= length) {return;}

     unsigned int ind = length*row + col;

     h_out[ind] = cuCabsf(h_in[ind]);

    return;
}
__global__ void float2cuComplex_kernel(float *h_in, cuComplex *h_out, const unsigned int length, const unsigned int width)
{
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

     if (row >= width || col >= length) {return;}

     unsigned int ind = length*row + col;

     h_out[ind].x = h_in[ind];
     h_out[ind].y = 0.0;

    return;
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

    square_kernel<<<numOfBlocks, threadsPerBlock>>>(d_vector, d_out, length, width);

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*width*length,
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
void exp_mat(cuComplex *h_in, cuComplex *h_out, const unsigned int length, const unsigned int width)
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

    exp_mat_kernel<<<numOfBlocks, threadsPerBlock>>>(d_in, d_out, length, width);

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
void vec_vec_mult(cuComplex *h_vec1, cuComplex *h_vec2, cuComplex *h_out, const unsigned int length, const unsigned int width)
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
    dim3 numOfBlocks(width/threadsPerBlock.x + 1, length/(threadsPerBlock.y + BLOCK_SIZE) + 1);

    vec_vec_mult_kernel<<<numOfBlocks, threadsPerBlock>>>(d_vec1, d_vec2, d_out, length, width);

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*width*length,
               cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to host failed\n");
		return;
	}

    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_out);
    return;
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
    dim3 numOfBlocks(width/threadsPerBlock.x + 1, length/(threadsPerBlock.y + BLOCK_SIZE) + 1);

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
    cuComplex *d_matrix, *d_out;

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
    dim3 numOfBlocks(width/threadsPerBlock.x + 1, batch/threadsPerBlock.y + 1);

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
void fftshift(cuComplex *h_signal, cuComplex *h_out, const unsigned int width, const unsigned int batch, const int dim)
{
    cuComplex *d_signal, *d_out;

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

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*width*batch,
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
void pad(cuComplex *h_in, cuComplex *h_out, const unsigned int length, const unsigned int width, const unsigned int padInd, const unsigned int padLength)
{//(sRaw, padded_data, batch, width, batch/2, mapLength - batch);
    
    cuComplex *d_in, *d_out;
    int newLength = padLength+length;

    cudaMalloc((void**)&d_in, sizeof(cuComplex)*width*length);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_out, sizeof(cuComplex)*width*newLength);
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
    dim3 numOfBlocks(newLength/threadsPerBlock.x + 1, width/threadsPerBlock.y + 1);

    pad_kernel<<<numOfBlocks, threadsPerBlock>>>(d_in, d_out, length, width, padInd, padLength);

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*width*newLength, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);
    return;
}
void vec_copy2mat(cuComplex *h_in, cuComplex *h_out, const unsigned int length, const unsigned int width)
{
    //Copies input vector of certain length, over and over to fit in a length X width matrix
    cuComplex *d_in, *d_out;
    
    cudaMalloc((void**)&d_in, sizeof(cuComplex)*length);
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

    cudaMemcpy(d_in, h_in, sizeof(cuComplex)*length,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(length/threadsPerBlock.x + 1, width/threadsPerBlock.y + 1);

    vec_copy2mat_kernel<<<numOfBlocks, threadsPerBlock>>>(d_in, d_out, length, width);

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*width*length, cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy from device failed\n");
		return;
	}

    cudaFree(d_out);
    cudaFree(d_in);
    return;
}
void vec_vec_add(cuComplex *h_mat1, cuComplex *h_mat2, cuComplex *h_out, const unsigned int length, const unsigned int width)
{
    cuComplex *d_mat1, *d_mat2, *d_out;

    cudaMalloc((void**)&d_mat1, sizeof(cuComplex)*length*width);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_mat2, sizeof(cuComplex)*length*width);
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
    
    cudaMemcpy(d_mat1, h_mat1, sizeof(cuComplex)*length*width, cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    cudaMemcpy(d_mat2, h_mat2, sizeof(cuComplex)*length*width, cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(length/threadsPerBlock.x + 1, width/threadsPerBlock.y + 1);
    
    vec_vec_add_kernel<<<numOfBlocks, threadsPerBlock>>>(d_mat1, d_mat2, d_out, length, width);

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*width*length, cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy from device failed\n");
		return;
	}

    return;
}
void sca_max(float K, cuComplex *h_in, cuComplex *h_out, const unsigned int length, const unsigned int width)
{
    cuComplex *d_in, *d_out;

    cudaMalloc((void**)&d_in, sizeof(cuComplex)*length*width);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMalloc((void**)&d_out, sizeof(cuComplex)*length*width);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate memory for matrix\n");
		return;
	}

    cudaMemcpy(d_in, h_in, sizeof(cuComplex)*length*width, cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(length/threadsPerBlock.x + 1, width/threadsPerBlock.y + 1);
    
    sca_max_kernel<<<numOfBlocks, threadsPerBlock>>>(K, d_in, d_out, length, width);

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*width*length, cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy from device failed\n");
		return;
	}

    cudaFree(d_in);
    cudaFree(d_out);
    return;
}
void is_pos(cuComplex *h_in, cuComplex *h_out, const unsigned int length, const unsigned int width)
{
    cuComplex *d_in, *d_out;

    cudaMalloc((void**)&d_in, sizeof(cuComplex)*length*width);
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

    cudaMemcpy(d_in, h_in, sizeof(cuComplex)*length*width,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(length/threadsPerBlock.x + 1, width/threadsPerBlock.y + 1);

    is_pos_kernel<<<numOfBlocks, threadsPerBlock>>>(d_in, d_out, length, width);

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*width*length, cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy from device failed\n");
		return;
	}

    cudaFree(d_out);
    cudaFree(d_in);
    return;
}
void round_vec(cuComplex *h_in, cuComplex *h_out, const unsigned int length, const unsigned int width)
{
    cuComplex *d_in, *d_out;

    cudaMalloc((void**)&d_in, sizeof(cuComplex)*length*width);
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

    cudaMemcpy(d_in, h_in, sizeof(cuComplex)*length*width,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(length/threadsPerBlock.x + 1, width/threadsPerBlock.y + 1);

    round_vec_kernel<<<numOfBlocks, threadsPerBlock>>>(d_in, d_out, length, width);

    cudaMemcpy(h_out, d_out, sizeof(cuComplex)*width*length, cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy from device failed\n");
		return;
	}

    cudaFree(d_out);
    cudaFree(d_in);
    return;
}
void cuComplex2Int(cuComplex *h_in, int *h_out, const unsigned int length, const unsigned int width)
{
    cuComplex *d_in;
    int *d_out;

    checkCudaErrors(cudaMalloc((void**)&d_in, sizeof(cuComplex)*length*width));
    
    checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(int)*length*width));
    
    checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(cuComplex)*length*width, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(length/threadsPerBlock.x + 1, width/threadsPerBlock.y + 1);

    cuComplex2Int_kernel<<<numOfBlocks, threadsPerBlock>>>(d_in, d_out, length, width);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(int)*length*width, cudaMemcpyDeviceToHost));
    
    cudaFree(d_out);
    cudaFree(d_in);
    return;
}
void cuComplex2float(cuComplex *h_in, float *h_out, const unsigned int length, const unsigned int width)
{
    cuComplex *d_in;
    float *d_out;

    checkCudaErrors(cudaMalloc((void**)&d_in, sizeof(cuComplex)*length*width));
    
    checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(float)*length*width));
    
    checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(cuComplex)*length*width, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(length/threadsPerBlock.x + 1, width/threadsPerBlock.y + 1);

    cuComplex2float_kernel<<<numOfBlocks, threadsPerBlock>>>(d_in, d_out, length, width);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(float)*length*width, cudaMemcpyDeviceToHost));
    
    cudaFree(d_out);
    cudaFree(d_in);
    return;
}
void float2cuComplex(float *h_in, cuComplex *h_out, const unsigned int length, const unsigned int width)
{
    float *d_in;
    cuComplex *d_out;

    checkCudaErrors(cudaMalloc((void**)&d_in, sizeof(float)*length*width));

    checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(cuComplex)*length*width));
        
    checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(float)*length*width, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(length/threadsPerBlock.x + 1, width/threadsPerBlock.y + 1);

    float2cuComplex_kernel<<<numOfBlocks, threadsPerBlock>>>(d_in, d_out, length, width);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(cuComplex)*length*width, cudaMemcpyDeviceToHost));
    
    cudaFree(d_out);
    cudaFree(d_in);
    return;
}

// Produces Compression Constants
void comp_decomp(const float Xc, cuComplex *uc, const int length,  cuComplex *u, const int u_len, cuComplex *k, const int width, cuComplex *compression, cuComplex *decompression)
{
    cuComplex *f_uc, *f_u, *f_k;

    f_u  = (cuComplex *)malloc(sizeof(cuComplex)*u_len);
    f_uc = (cuComplex *)malloc(sizeof(cuComplex)*length);
    f_k  = (cuComplex *)malloc(sizeof(cuComplex)*width);

    // fftshift
    fftshift(uc, f_uc, length, 1, 0);
    fftshift(u, f_u, u_len, 1, 0);

    // Square each element
    square(f_uc, f_uc, length, 1);
    square(f_u, f_u, u_len, 1);

    // add constant to vector
    sca_vec_add(Xc*Xc, f_uc, length, 1, 1);
    sca_vec_add(Xc*Xc, f_u, u_len, 1, 1);

    // sqrt(abs complex vector)
    sqrt_abs(f_uc, f_uc, length, 1);
    sqrt_abs(f_u, f_u, u_len, 1);

    // subtract contant from vector
    sca_vec_add(-1.0*Xc, f_uc, length, 1, 1.0);
    // Xc - u
    sca_vec_add(Xc, f_u, u_len, 1, -1.0);

    // change real vector imaginary vector
    real_to_imag(k, f_k, width, 1);
    sca_vec_mult(2.0, f_k, width, 1);

    // mult vec vec to matrix
    vec_vec_mat(f_uc, f_k, compression, length, width);
    vec_vec_mat(f_k, f_u, decompression, width, u_len);

    // exp mat
    exp_mat(compression, compression, width, length);
    exp_mat(decompression, decompression, u_len, width);

    free(f_u);
    free(f_uc);
    free(f_k);
    return;
}

void fft(cuComplex *h_matrix, cuComplex *h_out, const unsigned int length, const unsigned int width, int direction)
{   // One dimensional fft along length
    cuComplex *d_matrix;
    cufftHandle plan;

    cudaMalloc((void**)&d_matrix, sizeof(cuComplex)*length*width);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Failed to allocate data\n");
		return;
	}

    cudaMemcpy(d_matrix, h_matrix, sizeof(cuComplex)*length*width,
               cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy to device failed\n");
		return;
	}

    cufftPlan1d(&plan, length, CUFFT_C2C, width);
    cufftExecC2C(plan, d_matrix, d_matrix, direction);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: cufft failed\n");
		return;
	}

    cudaMemcpy(h_out, d_matrix, sizeof(cuComplex)*length*width,
               cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: Memcpy from device failed\n");
		return;
	}

    cufftDestroy(plan);
    cudaFree(d_matrix);
    return;
}

void fast_time_block(cuComplex *sRaw, cuComplex *fast_time_filter, const unsigned int length, const unsigned int width, 
                     const unsigned int mapWidth, cuComplex *fsSpotLit)
{
    cuComplex *fast_time_filter_copy, *sRaw_copy, *compression, *decompression, *uc, *u, *k;
    float Xc = 1000.0;

    sRaw_copy = (cuComplex *)malloc(sizeof(cuComplex)*length*width);
    fast_time_filter_copy = (cuComplex *)malloc(sizeof(cuComplex)*length);
    

    fft(sRaw, sRaw_copy, length, width, CUFFT_FORWARD);
    fftshift(sRaw_copy, sRaw_copy, length, width, 2);

    fftshift(fast_time_filter, fast_time_filter_copy, length, 1, 0);
    mat_vec_mult(sRaw_copy, fast_time_filter_copy, sRaw_copy, length, width);

    free(fast_time_filter_copy);
    
    uc = (cuComplex *)malloc(sizeof(cuComplex)*width);
    k  = (cuComplex *)malloc(sizeof(cuComplex)*length);
    u  = (cuComplex *)malloc(sizeof(cuComplex)*mapWidth);
    compression = (cuComplex *)malloc(sizeof(cuComplex)*length*width);
    decompression = (cuComplex *)malloc(sizeof(cuComplex)*length*mapWidth);
    
    csv_real_reader("u.csv",   u, true, true);
    csv_real_reader("uc.csv", uc, true, true);
    csv_real_reader("k.csv",   k, true, true);
    comp_decomp(Xc, uc, width, u, mapWidth, k, length, compression, decompression);

    free(u);
    free(uc);

    vec_vec_mult(sRaw_copy, compression, sRaw_copy, length, width);

    free(compression);

    transpose(sRaw_copy, length, width);

    fft(sRaw_copy, sRaw_copy, width, length, CUFFT_FORWARD);

    pad(sRaw_copy, fsSpotLit, width, length, width/2, mapWidth - width);

    sca_vec_mult(382.0/160.0, fsSpotLit, mapWidth, length);

    fft(fsSpotLit, fsSpotLit, mapWidth, length, CUFFT_INVERSE);
    sca_vec_mult(1.0/mapWidth, fsSpotLit, mapWidth, length);

    vec_vec_mult(fsSpotLit, decompression, fsSpotLit, mapWidth, length);

    free(decompression);
    
    fft(fsSpotLit, fsSpotLit, mapWidth, length, CUFFT_FORWARD);

    fftshift(fsSpotLit, fsSpotLit, mapWidth, length, 2);

    transpose(fsSpotLit, mapWidth, length);

    free(sRaw_copy);
}

void SpatialInterpolate(cuComplex *filteredSignal, float *wn, cuComplex *GridValues,
                        float dkx, float kxs, const int length, const int mapWidth, const int mapLength, 
                        cuComplex *outSignal, cuComplex *idxout)
{
    const int nInterpSidelobes = 8;
    const float pi = 3.1415926;
    
    cuComplex *row, *sliceRange;
    cuComplex *d_filteredSignal, *d_outSignal;
    int *rowidx, *d_rowidx;

    float *d_wn, *d_gridvalues;
    row = (cuComplex *)malloc(sizeof(cuComplex)*length*mapWidth);
    sliceRange = (cuComplex *)malloc(sizeof(cuComplex)*length*mapWidth);
    rowidx = (int *)malloc(sizeof(int)*length*mapWidth);
    
    float2cuComplex(wn, row, length, mapWidth);

    float negGridValue = cuCabsf(cuCmulf(make_cuComplex(-1.0,0.0), GridValues[0]));

    sca_vec_add(negGridValue, row, length, mapWidth, 1.0);

    sca_vec_mult(1.0/dkx, row, length, mapWidth);

    round_vec(row, row, length, mapWidth);

    sca_vec_add(-1.0*nInterpSidelobes, row, length, mapWidth, 1.0);

    cuComplex2Int(row, rowidx, length, mapWidth);
    //rowidx is complete at this line

    //Setting up Cuda memory
    checkCudaErrors(cudaMalloc((void**)&d_wn, sizeof(float)*length*mapWidth));
    checkCudaErrors(cudaMalloc((void**)&d_rowidx, sizeof(int)*length*mapWidth));
    checkCudaErrors(cudaMalloc((void**)&d_gridvalues, sizeof(float)*mapLength));
    checkCudaErrors(cudaMalloc((void**)&d_outSignal, sizeof(cuComplex)*mapLength*mapWidth));
    checkCudaErrors(cudaMalloc((void**)&d_filteredSignal, sizeof(cuComplex)*length*mapWidth));

    checkCudaErrors(cudaMemcpy(d_wn, wn, sizeof(float)*mapWidth*length, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rowidx, rowidx, sizeof(int)*mapWidth*length, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_gridvalues, GridValues, sizeof(float)*mapLength, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filteredSignal, filteredSignal, sizeof(cuComplex)*mapWidth*length, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 numOfBlocks(length/threadsPerBlock.x + 1, mapWidth/threadsPerBlock.y + 1, (2*nInterpSidelobes - 1)/threadsPerBlock.z + 1);
    //->
    spatial_inter_kernel<<<numOfBlocks, threadsPerBlock>>>(d_filteredSignal, 
        d_wn,
        d_gridvalues, 
        d_rowidx, length, 
        mapLength, mapWidth, nInterpSidelobes, dkx, kxs, 
        d_outSignal);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(outSignal, d_outSignal, sizeof(cuComplex)*mapWidth*mapLength, cudaMemcpyDeviceToHost));

    free(rowidx);
    checkCudaErrors(cudaFree(d_wn));
    checkCudaErrors(cudaFree(d_rowidx));
    checkCudaErrors(cudaFree(d_gridvalues));
    checkCudaErrors(cudaFree(d_outSignal));
    checkCudaErrors(cudaFree(d_filteredSignal));
    return;
}

int main()
{
	//new code
	cudaDeviceReset();
	cudaSetDevice(0);

	// Get all data
	ifstream fastTimeFilter ("fastTimeFilter.csv");
	// End of data read

	float d, i;
    float m = 382;
    float mc = 160;
    float Xc = 1000.0;
    float dkx = 0.0785;
    float kxs = 0.6283;
    float kxMin = 5.8462;

    int nInterpSidelobes = 8;

    //Dimensions of sRaw data
    int width = 438;
    int batch = 160;

    //Dimensions of final map
    int mapLength = 382;
    int mapWidth  = 266;

	cuComplex curr, *sRaw, *signal, *mapOut, *out_signal, *u, *uc, *k, *ku0, *padded_data, *fsSpotLit;
    cuComplex *kmat, *ku0mat, *kx, *kx_gt_zero, *kx_work, *filteredSignal, *no_interpolation_image;
    cuComplex *finalImage, *idxout;
    float *kx_float;

	u  = (cuComplex *)malloc(sizeof(cuComplex)*mapLength);
    uc = (cuComplex *)malloc(sizeof(cuComplex)*batch);
    k  = (cuComplex *)malloc(sizeof(cuComplex)*width);
    ku0 = (cuComplex *)malloc(sizeof(cuComplex)*mapLength);
    signal = (cuComplex *)malloc(sizeof(cuComplex)*width);
	sRaw   = (cuComplex *)malloc(sizeof(cuComplex)*width*batch);
    out_signal = (cuComplex *)malloc(sizeof(cuComplex)*width*batch);
    padded_data = (cuComplex *)malloc(sizeof(cuComplex)*width*mapLength);
    fsSpotLit   = (cuComplex *)malloc(sizeof(cuComplex)*width*mapLength);

    csv_real_reader("u.csv",   u, true, true);
    csv_real_reader("uc.csv", uc, true, true);
    csv_real_reader("k.csv",   k, true, true);
    csv_real_reader("ku0.csv", ku0, true, true);
    csv_real_reader("imagsRaw.csv", sRaw, false, false);
    csv_real_reader("realsRaw.csv", sRaw, true, false);

    //Copying Data from CSV files into memory
    string value;
    list<string> values;
    while ( fastTimeFilter.good() )
    {
        // read a string until next comma: http://www.cplusplus.com/reference/string/getline/
        getline ( fastTimeFilter, value, ',' ); 
        if (value.find('\n') != string::npos) {
            split_line(value, "\n", values);
        } else {
            values.push_back(value);
        }
    }
    //From fast time filter we get p*(-t)
    list<string>::const_iterator it = values.begin();
    int count = 0;
    for (it = values.begin(); it != values.end(); it++) {
        string tmp = *it;
        d = stof(tmp.c_str(), NULL);
		it++;
		tmp = *it;
		i = stof(tmp.c_str(), NULL);
		signal[count].x = d;
		signal[count].y = i;
		count++;
        //cout << "Double val: " << right << showpoint << d << endl;
    }
	
	//Done Reading in values from files.

    //---------------------------------------------------------------------------------------------------------------------

    // Two-D Matched Fitler
    
    square(k, k, width, 1);
    square(ku0, ku0, mapLength, 1);

    sca_vec_mult(4.0, k, width, 1);
    sca_vec_mult(-1.0, ku0, mapLength, 1);

    fast_time_block(sRaw, signal, width, batch, mapLength, fsSpotLit);

    kmat = (cuComplex *)malloc(sizeof(cuComplex)*mapLength*width);
    ku0mat = (cuComplex *)malloc(sizeof(cuComplex)*mapLength*width);
    kx     = (cuComplex *)malloc(sizeof(cuComplex)*mapLength*width);
    kx_work= (cuComplex *)malloc(sizeof(cuComplex)*mapLength*width);
    kx_gt_zero = (cuComplex *)malloc(sizeof(cuComplex)*mapLength*width);
    filteredSignal = (cuComplex *)malloc(sizeof(cuComplex)*mapLength*width);
    no_interpolation_image = (cuComplex *)malloc(sizeof(cuComplex)*mapLength*width);
    
    vec_copy2mat(k, kmat, width, mapLength);
    vec_copy2mat(ku0, ku0mat, mapLength, width);

    transpose(ku0mat, mapLength, width);

    vec_vec_add(kmat, ku0mat, kx, width, mapLength);
    sca_max(0, kx, kx, width, mapLength);

    sqrt_abs(kx, kx, width, mapLength);
    // kx is kx at this point

    is_pos(kx, kx_gt_zero, width, mapLength);

    sqrt_abs(kmat, kmat, width, mapLength);

    sca_vec_mult(-1.0, kmat, width, mapLength);

    vec_vec_add(kx, kmat, kx_work, width, mapLength);
    
    sca_vec_mult(Xc, kx_work, width, mapLength);

    csv_real_reader("ku0.csv", ku0, true, true);

    fftshift(ku0, ku0, mapLength, 1, 0);

    vec_copy2mat(ku0, ku0mat, mapLength, width);

    transpose(ku0mat, mapLength, width);
    
    vec_vec_add(kx_work, ku0mat, kx_work, width, mapLength);

    sca_vec_add(0.25*PI, kx_work, width, mapLength, 1.0);

    real_to_imag(kx_work, kx_work, width, mapLength);

    exp_mat(kx_work, kx_work, width, mapLength);

    vec_vec_mult(kx_work, kx_gt_zero, kx_work, width, mapLength);

    vec_vec_mult(kx_work, fsSpotLit, filteredSignal, width, mapLength);

    fft(filteredSignal, no_interpolation_image, width, mapLength, CUFFT_INVERSE);
    transpose(no_interpolation_image, width, mapLength);
    fft(no_interpolation_image, no_interpolation_image, mapLength, width, CUFFT_INVERSE);
    transpose(no_interpolation_image, mapLength, width);

    sca_vec_mult(1.0/width, no_interpolation_image, width, mapLength);
    sca_vec_mult(1.0/mapLength, no_interpolation_image, width, mapLength);

    kx_float = (float *)malloc(sizeof(float)*mapLength*width);
    cuComplex2float(kx, kx_float, width, mapLength);
    finalImage = (cuComplex *)malloc(sizeof(cuComplex)*mapLength*mapWidth);
    // Spatial Interpolate
    //SpatialInterpolate(filteredSignal, kx_float, (kxMin + ( (-nInterpSidelobes-2):(nx-nInterpSidelobes-3) ) * dkx), dkx, kxs, width, mapWidth, mapLength, finalImage, idxout);
    //-10 to 255 start end size
    // sca vec mult dkx
    // sca vec add kxMin
    // transpose?
    // = GridValues

    for(int x = 0; x < mapLength; x++)
    {
        for(int y = 0; y < width; y++)
        {
            curr = no_interpolation_image[x*width + y];
            printf("%g + (%gi), ", cuCrealf(curr), cuCimagf(curr));
        }
        cout << endl;
    }

    free(u);
    free(k);
    free(uc);
    free(ku0);
    free(kmat);
    free(sRaw);
    free(ku0mat);
    free(signal);
    free(kx_work);
    free(kx_float);
    free(fsSpotLit);
    free(out_signal);
    free(finalImage);
    free(padded_data);    
    free(no_interpolation_image);
	return 0;
}
