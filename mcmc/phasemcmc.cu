/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>

using namespace std;

// CUDA runtime
#include <cuda_runtime.h>

#include <cuda_profiler_api.h>
#define __int32 int
// Helper functions and utilities to work with CUDA

__global__ void countGenotypes1(const char * __restrict__ a, const char* __restrict__ b, int* result, int ypitch)
{
	a += 2504 * (threadIdx.x + blockDim.x * blockIdx.x);
	b += 2504 * (threadIdx.y + blockDim.y * blockIdx.y);
	int flag = 0;
	result += ypitch * (threadIdx.y + threadIdx.y + blockDim.y * blockIdx.y) + threadIdx.x + blockDim.x * blockIdx.x;
	*result = 0;
	for (int i = 0; i < 2504; i++)
	{
		int geno = (*a);
		geno *= 3;
		geno += *b;
		flag |= 1 << geno;

		if (flag == 511)
		{
			*result = 1;
			break;
		}
		a++;
		b++;
	}
}

__global__ void countGenotypes2(const char * __restrict__ a, const char* __restrict__ b, int* result, int ypitch)
{
	a += 2504 * (threadIdx.x + blockDim.x * blockIdx.x);
	b += 2504 * (threadIdx.y + blockDim.y * blockIdx.y);
	int flag = 0;
	result += ypitch * (threadIdx.y + threadIdx.y + blockDim.y * blockIdx.y) + threadIdx.x + blockDim.x * blockIdx.x;
	*result = 0;
	const char* enda = a + 626;
	for (; a != enda; a++,b++)
	{
#pragma unroll 4
		for (int k = 0; k < 4; k++)
		{
		int geno = (((*a) >> (k * 2)) & 3);
		geno *= 3;
		geno += ((*b >> (k * 2)) & 3);
		flag |= 1 << geno;
		}

		if (flag == 511)
		{
			*result = 1;
		//	break;
		}
	}
}

const int BLOCK_SIZE = 12;
const int GENOSBY4 = 640;

__device__ unsigned int bfe_ptx(unsigned int x, unsigned int bit, unsigned int numBits) {
unsigned int result;
asm("bfe.u32 %0, %1, %2, %3;" :
"=r"(result) : "r"(x), "r"(bit), "r"(numBits));
return result;
}


__global__ void countGenotypes3(const unsigned char * __restrict__ a, const unsigned char* __restrict__ b, int* result, int ypitch)
{
	__shared__ unsigned char as[BLOCK_SIZE][GENOSBY4];
	__shared__ unsigned char bs[BLOCK_SIZE][GENOSBY4];
	a += 2512 * (threadIdx.y + blockDim.y * blockIdx.y);
	b += 2512 * (threadIdx.y + blockDim.y * blockIdx.y);
	for (int k = threadIdx.x * (GENOSBY4 / (BLOCK_SIZE - 1) / 16) * 16; k < GENOSBY4; k+=16)
	{
		*((uint4*) &as[threadIdx.y][k]) = *((uint4*) (&a[k]));
		*((uint4*) &bs[threadIdx.y][k]) = *((uint4*) (&b[k]));
	}
	__syncthreads();

	int flag = 0;
	result += ypitch * (threadIdx.y + threadIdx.y + blockDim.y * blockIdx.y) + threadIdx.x + blockDim.x * blockIdx.x;
	*result = 0;
	const unsigned char* nowa = &as[threadIdx.x][0];
	const unsigned char* nowb = &bs[threadIdx.y][0];
	const unsigned char* enda = &as[threadIdx.x][0] + GENOSBY4;
	for (; nowa != enda; nowa++,nowb++)
	{
#pragma unroll 4
		for (int k = 0; k < 4; k++)
		{
		int geno = bfe_ptx(*nowa, k * 2, 2);
			//(((*nowa) >> (k * 2)) & 3);
		geno *= 3;
		//geno += ((*nowb >> (k * 2)) & 3);
		geno += bfe_ptx(*nowb, k * 2, 2);
		flag |= 1 << geno;
		}

		if (flag == 511)
		{
			*result = 1;
			break;
		}
	}

}

__global__ void countGenotypes4(const unsigned char * __restrict__ a, const unsigned char* __restrict__ b, int* result, int ypitch)
{
	__shared__ unsigned char as[BLOCK_SIZE][GENOSBY4];
	__shared__ unsigned char bs[BLOCK_SIZE][GENOSBY4];
	a += 2512 * (threadIdx.y + blockDim.y * blockIdx.y);
	b += 2512 * (threadIdx.y + blockDim.y * blockIdx.y);
	for (int k = threadIdx.x * (GENOSBY4 / (BLOCK_SIZE - 1) / 16) * 16; k < GENOSBY4; k+=16)
	{
		*((uint4*) &as[threadIdx.y][k]) = *((uint4*) (&a[k]));
		*((uint4*) &bs[threadIdx.y][k]) = *((uint4*) (&b[k]));
	}
	__syncthreads();

	int flag = 0;
	result += ypitch * (threadIdx.y + threadIdx.y + blockDim.y * blockIdx.y) + threadIdx.x + blockDim.x * blockIdx.x;
	*result = 0;
	const unsigned char* nowa = &as[threadIdx.x][0];
	const unsigned char* nowb = &bs[threadIdx.y][0];
	const unsigned char* enda = &as[threadIdx.x][0] + GENOSBY4;
	for (; nowa != enda; nowa+=4,nowb+=4)
	{
		unsigned int aval = *(unsigned int*)nowa;
		unsigned int bval = *(unsigned int*)nowb;
#pragma unroll 16
		for (int k = 0; k < 16; k++)
		{
			
		//int geno = bfe_ptx(aval, k * 2, 2);
		int geno = (((aval) >> (k * 2)) & 3);
		geno *= 3;
		geno += ((bval >> (k * 2)) & 3);
		//geno += bfe_ptx(bval, k * 2, 2);
		flag |= 1 << geno;
		}

		if (flag == 511)
		{
			*result = 1;
			break;
		}
	}

}

__global__ void countGenotypes99(const unsigned char * __restrict__ a, const unsigned char* __restrict__ b, int* result, int ypitch)
{
	__shared__ uint4 as[BLOCK_SIZE][GENOSBY4 / 4];
	a += 2512 * (threadIdx.y + blockDim.y * blockIdx.y);
	b += 2512 * (threadIdx.y + blockDim.y * blockIdx.y);
#pragma unroll
	for (int k = threadIdx.x * 16; k < GENOSBY4; k += 16)
	{
		as[threadIdx.y][k / 4] = *((uint4*)(&a[k]));
		//bs[threadIdx.y][k / 4] = *((uint4*)(&b[k]));
	}
	for (int k = 0; k < 10; k++)
		*result = as[threadIdx.x][k].x;
}

__global__ void countGenotypes5(const unsigned char * __restrict__ a, const unsigned char* __restrict__ b, __int32* result)
{
	__shared__ unsigned char as[BLOCK_SIZE][GENOSBY4];
	__shared__ unsigned char bs[BLOCK_SIZE][GENOSBY4];
	a += 640 * (threadIdx.y + blockDim.x * blockIdx.x);
	b += 640 * (threadIdx.y + blockDim.y * blockIdx.y);
	for (int k = threadIdx.x * 8; k < GENOSBY4; k += BLOCK_SIZE * 8)
//	for (int k = 0; k < GENOSBY4; k += 8)
	{
		*((uint2*)&as[threadIdx.y][k]) = *((uint2*)(&a[k]));
		*((uint2*)&bs[threadIdx.y][k]) = *((uint2*)(&b[k]));
	}
	__syncthreads();

	int flag = 0;
	int oldflag = 0;
	int count[16] = {0};
	const unsigned char* __restrict__ nowa = &as[threadIdx.x][0];
	const unsigned char* __restrict__ nowb = &bs[threadIdx.y][0];
	const unsigned char* __restrict__ enda = &as[threadIdx.x][0] + GENOSBY4;
#pragma unroll 2
	for (; nowa != enda; nowa += 16, nowb += 16)
	{
		uint4 aval4 = *(uint4*)nowa;
		uint4 bval4 = *(uint4*)nowb;

		int geno[16];

#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			unsigned int aval;
			unsigned int bval;

			switch (i)
			{
			case 0:
				aval = aval4.x;
				bval = bval4.x;
				break;
			case 1:
				aval = aval4.y;
				bval = bval4.y;
				break;
			case 2:
				aval = aval4.z;
				bval = bval4.z;
				break;
			case 3:
				aval = aval4.w;
				bval = bval4.w;
				break;
			}

#pragma unroll 16
			for (int k = 0; k < 16; k++)
			{

				//int geno = bfe_ptx(aval, k * 2, 2);
				geno[k] = (((aval) >> (k * 2)) & 3);
			}

#pragma unroll 16
			for (int k = 0; k < 16; k++)
			{
				geno[k] *= 3;
			}

#pragma unroll 16
			for (int k = 0; k < 16; k++)
			{
				geno[k] += ((bval >> (k * 2)) & 3);
			}

#pragma unroll 16
			for (int k = 0; k < 16; k++)
			{
				//geno += bfe_ptx(bval, k * 2, 2);
				flag |= 1 << geno[k];
				count[k] += geno[k] == 4;
			}
		}

		if (flag >= 511)
		{
			//*result = 1;
			break;
		}
	}

	//if (/*(flag & 511) == 511*/ flag != 4096)
	if ((flag & 511) != 511)
	{
	int count2 = 0;
#pragma unroll 16
	for (int i = 0; i < 16; i++)
{
count2 += count[i];
}

	if (count2 < 400) return;
		int index = atomicAdd(result, 4);
		result[index] = threadIdx.x + blockDim.x * blockIdx.x;
		result[index + 1] = threadIdx.y + blockDim.y * blockIdx.y;
		result[index + 2] = flag & 511;
		result[index + 3] = count2;
	}
}

__global__ void countGenotypes5b(const unsigned char * __restrict__ a, const unsigned char* __restrict__ b, __int32* result, int ypitch)
{
	__shared__ unsigned char as[BLOCK_SIZE][GENOSBY4];
	__shared__ unsigned char bs[BLOCK_SIZE][GENOSBY4];
	a += 2512 * (threadIdx.y + blockDim.y * blockIdx.y);
	b += 2512 * (threadIdx.y + blockDim.y * blockIdx.y);
	for (int k = threadIdx.x * 16; k < GENOSBY4; k+=16)
	{
		*((uint4*) &as[threadIdx.y][k]) = *((uint4*) (&a[k]));
		*((uint4*) &bs[threadIdx.y][k]) = *((uint4*) (&b[k]));
	}
	__syncthreads();

	int flag = 0;
	result += ypitch * (threadIdx.y + threadIdx.y + blockDim.y * blockIdx.y) + threadIdx.x + blockDim.x * blockIdx.x;
	*result = 0;
	const unsigned char* __restrict__ nowa = &as[threadIdx.x][0];
	const unsigned char* __restrict__ nowb = &bs[threadIdx.y][0];
	const unsigned char* __restrict__ enda = &as[threadIdx.x][0] + GENOSBY4;
	for (; nowa != enda; nowa+=16,nowb+=16)
	{
		uint4 aval4 = *(uint4*)nowa;
		uint4 bval4 = *(uint4*)nowb;
		
		int geno[16];

#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			unsigned int aval;
			unsigned int bval;

			switch (i)
			{
			case 0:
				aval = aval4.x;
				bval = bval4.x;
				break;
			case 1:
				aval = aval4.y;
				bval = bval4.y;
				break;
			case 2:
				aval = aval4.z;
				bval = bval4.z;
				break;
			case 3:
				aval = aval4.w;
				bval = bval4.w;
				break;
			}
		
		#pragma unroll 16
		for (int k = 0; k < 16; k++)
		{
			
		//int geno = bfe_ptx(aval, k * 2, 2);
		geno[k] = (((aval) >> (k * 2)) & 3);
		}

		#pragma unroll 16
		for (int k = 0; k < 16; k++)
		{
			geno[k] *= 3;
		}

		#pragma unroll 16
		for (int k = 0; k < 16; k++)
		{
			geno[k] += ((bval >> (k * 2)) & 3);
		}
		int oldflag = flag;

		#pragma unroll 16
		for (int k = 0; k < 16; k++)
		{
			int add = 1 << geno[k];
			if (!(oldflag & add))
			{
		//geno += bfe_ptx(bval, k * 2, 2);
			flag |= add;
			}
		}

		if (oldflag == 511)
		{
			//*result = 1;
			break;
		}
	}
	}
	
	if (flag == 511)
	{
			//*result = flag;
		as[0][0] = 0;
	}
}


__global__ void countGenotypes6(const unsigned char * __restrict__ a, const unsigned char* __restrict__ b, int* result)
{
	__shared__ unsigned char as[BLOCK_SIZE][GENOSBY4];
	__shared__ unsigned char bs[BLOCK_SIZE][GENOSBY4];
	a += 2512 * (threadIdx.y + blockDim.y * blockIdx.y);
	b += 2512 * (threadIdx.y + blockDim.y * blockIdx.y);
	for (int k = threadIdx.x * 16; k < GENOSBY4; k+=16)
	{
		*((uint4*) &as[threadIdx.y][k]) = *((uint4*) (&a[k]));
		*((uint4*) &bs[threadIdx.y][k]) = *((uint4*) (&b[k]));
	}
	__syncthreads();

	int flags[16] = {0};
	int geno2[16] = {0};
	*result = 0;
	const unsigned char* __restrict__ nowa = &as[threadIdx.x][0];
	const unsigned char* __restrict__ nowb = &bs[threadIdx.y][0];
	const unsigned char* __restrict__ enda = &as[threadIdx.x][0] + GENOSBY4;
	for (; nowa != enda; nowa+=16,nowb+=16)
	{
		uint4 aval4 = *(uint4*)nowa;
		uint4 bval4 = *(uint4*)nowb;
		
		unsigned int geno[16] = {0};

#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			unsigned int aval;
			unsigned int bval;

			switch (i)
			{
			case 0:
				aval = aval4.x;
				bval = bval4.x;
				break;
			case 1:
				aval = aval4.y;
				bval = bval4.y;
				break;
			case 2:
				aval = aval4.z;
				bval = bval4.z;
				break;
			case 3:
				aval = aval4.w;
				bval = bval4.w;
				break;
			}

		aval = aval ^ ((aval & 0xAAAAAAAA) >> 1);
		
		#pragma unroll 16
		for (int k = 0; k < 16; k++)
		{
			flags[k] |= geno2[k];
		}

		#pragma unroll 16
		for (int k = 0; k < 16; k++)
		{
			
		//int geno = bfe_ptx(aval, k * 2, 2);
			//flag |= geno[k];	
		geno[k] = (((aval) >> (k * 2)) & 3);
		}

		#pragma unroll 16
		for (int k = 0; k < 16; k++)
		{
			geno[k] += geno[k];
			flags[k] |= flags[(k - 7) & 15];
		}

		#pragma unroll 16
		for (int k = 0; k < 16; k++)
		{
			geno[k] += ((bval >> (k * 2)) & 3);
		}

		#pragma unroll 16
		for (int k = 0; k < 16; k++)
		{
			geno2[k] = 1 << geno[k];
		}

		if (flags[15] == 511)
		{
			//*result = 1;
			//break;
		}

		/*#pragma unroll 16
		for (unsigned int k = 0; k < 16; k++)
		{
		//geno += bfe_ptx(bval, k * 2, 2);
		
		}*/
		}

		
	}
}

__global__ void doit()
{

}

struct data
{
unsigned int* buffer;
vector<int> poses;

data(char* filename)
{
FILE* in = fopen(filename, "r");
int size;
fread(&size, sizeof(int), 1, in);

poses.resize(size);
fread(&poses[0], sizeof(int) * size, 1, in);

buffer = new unsigned int[poses.size() * 160];
fread(buffer, sizeof(int), poses.size() * 160, in);
}
};


int SIDE = 256;
int SHORT_SIDE = 12;

#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>


struct likelihoodgetter
{
double __device__ likelihood(int count, int mask, cufftComplex val)
{	
	if (!mask) return 0;
	double val2 = val.x * val.x + val.y * val.y;
	double term = 0;
/*	if (val2 < 1e-9)
	{
		term = -val.x * val.x;val2 = 1e-9;
	}
	else*/
	{
		term += log(val2) * count - val2;
	}

//	term -= val.y * val.y;
	return term;
}

double __device__ operator ()(thrust::tuple<int, int, cufftComplex> t)
{
return likelihood(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t));
}
} likelihooder;

void readtxt(char* name, int* data)
{
FILE* in = fopen(name, "r");
printf("FILE %d\n", in);
fflush(stdout);
for (int x = 0; x < SIDE * SIDE; x++)
{
	fscanf(in, "%d", &data[x]);
}
}

boost::mt19937 rander;
	boost::uniform_real<> dist(0.f, 1.f);
boost::variate_generator<boost::mt19937, boost::uniform_real<> > generator(rander, dist);
float randval(float min, float max)
{
	return generator() * (max - min) + min;
}

int main(int argc, char** argv)
{
	cufftComplex* data = (cufftComplex*) calloc(SIDE * SIDE, sizeof(cufftComplex));
	printf("Data %d\n", data);
	int* mask = (int*) malloc(sizeof(int) * SIDE * SIDE);
	int* truePattern = (int*) malloc(sizeof(int) * SIDE * SIDE);
	
	readtxt(argv[1], truePattern);
	int sum = 0;
	for (int x = 0; x < SIDE * SIDE; x++)
	{
	sum += truePattern[x];
		mask[x] = 1;
	}

/*	int index = (SHORT_SIDE / 2) * (SIDE + 1);
	data[index].x = sqrt(sum) / SIDE / SIDE;
	data[index].y = 0;*/

	int seed;
	sscanf(argv[2], "%d", &seed);
	generator.engine().seed(seed);

	

	int* d_truePattern;
	int* d_mask;
	cufftComplex* d_data;
	cufftComplex* d_pattern;
	cufftHandle plan;
	cufftPlan2d(&plan, SIDE, SIDE, CUFFT_C2C);

	cudaMalloc((void**) &d_data, sizeof(cufftComplex) * SIDE * SIDE);
	cudaMalloc((void**) &d_pattern, sizeof(cufftComplex) * SIDE * SIDE);
	cudaMalloc((void**) &d_mask, sizeof(int) * SIDE * SIDE);
	cudaMalloc((void**) &d_truePattern, sizeof(int) * SIDE * SIDE);

	cudaMemcpy(d_data, data, sizeof(cufftComplex) * SIDE * SIDE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_truePattern, truePattern, sizeof(int) * SIDE * SIDE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask, sizeof(int) * SIDE * SIDE, cudaMemcpyHostToDevice);

	thrust::device_ptr<int> t_mask = thrust::device_pointer_cast(d_mask);
	thrust::device_ptr<int> t_truePattern = thrust::device_pointer_cast(d_truePattern);
	thrust::device_ptr<cufftComplex> t_pattern = thrust::device_pointer_cast(d_pattern);

	double oldLikelihood = -INFINITY;

	int iter = 0;
	int forced = 0;
	int chosen = 0;
	int skips = 0;
	for (int y = 0; y < SHORT_SIDE; y++)
	{
		for (int x = 0; x < SHORT_SIDE; x++)
		{
			int index = y * SIDE + x;
			float angle = randval(0, M_PI * 2);
			float val = randval(1e-10, 15);
			//cufftComplex old = data[index];

			//data[index].x = val * cos(angle);
			//data[index].y = val * sin(angle);
		}
	}

	double temperatureFactor = 10000000;
	while (true)
{
	if (iter % 100 == 99)
	{
		if (forced < chosen * 1.05 || chosen + forced > skips * 5) temperatureFactor /= pow(10, 1.0/5);
		else
			temperatureFactor *= pow(10, 1.0/5);
		printf("Outer iteration %d %lf %d %d\t%lf\n", iter, oldLikelihood, forced, chosen, log(temperatureFactor)/log(10));
		fflush(stdout);
		forced = 0;
		chosen = 0;
		skips = 0;
	}
	iter++;
			if (temperatureFactor < 1) temperatureFactor = 1;
	for (int y = 0; y < SHORT_SIDE; y++)
	{
		for (int x = 0; x < SHORT_SIDE; x++)
		{
			int index = y * SIDE + x;
			float angle = randval(0, M_PI * 2);
			float val = randval(1e-10, 15);
			cufftComplex old = data[index];

			if (randval(0, 1) < 0.5)
			{
				data[index].x = val * cos(angle);
				data[index].y = val * sin(angle);
			}
			else
			{
			float& what = randval(0, 1) < 0.5 ? data[index].x : data[index].y;
			if (what == 0) continue;
			float step = randval(0.0f, 0.01f);
			if (randval(0, 1) < 0.5)
{
			if (randval(0, 1) < 0.5)   
			{
				what /= 1 + step;
			}
			else
			{
				what *= 1 + step;
			}
}
else
{
			step *= 0.1;
			if (randval(0, 1) < 0.5)   
			{
				what -= step;
			}
			else
			{
				what += step;
			}
}
}

			val = sqrt(data[index].x * data[index].x + data[index].y * data[index].y);
			angle = atan2(data[index].x, data[index].y);


			double nowsum = 0;
			for (int y = 0; y < SHORT_SIDE; y++)
			{
				for (int x = 0; x < SHORT_SIDE; x++)
				{
					int index2 = y * SIDE + x;
					nowsum += data[index2].x * data[index2].x + data[index2].y * data[index2].y;
				}
			}
			int x2, y2, otherindex;
			double factor;
			double basesum;

			//do
			{
				x2 = (int) randval(0, SHORT_SIDE);
				y2 = (int) randval(0, SHORT_SIDE);
				otherindex = y2 * SIDE + x2;
				basesum = data[otherindex].x * data[otherindex].x + data[otherindex].y * data[otherindex].y;
				if (basesum == 0) basesum = 1;
				double diff = sum * 1.0 / SIDE / SIDE - nowsum;
				factor = (basesum + diff) / basesum;
				
			} //while (factor < 0);
			if (factor < 0) factor = 1e-9;
			factor = sqrt(factor);

//			printf("%lf %lf %lf %lf\n", basesum, nowsum, sum * 1.0 / SIDE / SIDE, factor);

			cufftComplex otherold = data[otherindex];
/*			data[otherindex].x *= factor;
			data[otherindex].y *= factor;*/


//			cudaMemcpy(&d_data[index], &data[index], sizeof(cufftComplex), cudaMemcpyHostToDevice);
			cudaMemcpy(d_data, data, sizeof(cufftComplex) * SIDE * SHORT_SIDE, cudaMemcpyHostToDevice);

			cufftExecC2C(plan, d_data, d_pattern, CUFFT_FORWARD);
			double newLikelihood = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(t_truePattern, t_mask, t_pattern)),
                                         thrust::make_zip_iterator(thrust::make_tuple(&t_truePattern[SIDE * SIDE], &t_mask[SIDE * SIDE], &t_pattern[SIDE * SIDE])), likelihooder,
					 (double) 0, thrust::plus<double>()) + 34154. * SIDE * SIDE;
			if (newLikelihood > oldLikelihood)
			{
//				printf("FORCED ACCEPT: x: %d, y: %d - %f at %lf : %f %lf %lf %d\n", x, y, val, angle, newLikelihood, nowsum, nowsum - basesum + basesum * factor * factor, iter);
				forced++;
				oldLikelihood = newLikelihood;				
				continue;
			}

			double alpha = exp((newLikelihood - oldLikelihood) / temperatureFactor);
			if (randval(0, 1) < alpha)
			{
//				printf("CHOSEN ACCEPT: x: %d, y: %d - %f at %f : %lf %lf\n", x, y, val, angle, newLikelihood, alpha);
				chosen++;
				oldLikelihood = newLikelihood;
				continue;
			}
			else
			skips++;
/*			nowsum = 1 / nowsum;
			for (int y = 0; y < SHORT_SIDE; y++)
			{
				for (int x = 0; x < SHORT_SIDE; x++)
				{
					int index2 = y * SIDE + x;
					data[index2].x *= nowsum;
					data[index2].y *= nowsum;
				}
			}*/
//			printf("SKIP: x: %d, y: %d - %f at %f : %lf %lf\n", x, y, val, angle, newLikelihood, log(alpha));
			data[otherindex] = otherold;
			data[index] = old;
			cudaMemcpy(&d_data[index], &data[index], sizeof(cufftComplex), cudaMemcpyHostToDevice);
                }                 
	}
	if (iter % 10000 == 0)
{
	char tlf[255];
	sprintf(tlf, "iterZ_%05d_%05d", seed,iter);
	FILE* ut = fopen(tlf, "w");
	for (int y = 0; y < SIDE; y++)
{
	for (int x = 0; x < SIDE; x++)
{
	fprintf(ut, "%f%c", data[y * SIDE + x].x, (x == SIDE - 1) ? '\n' : ' ');
}
}

	for (int y = 0; y < SIDE; y++)
{
	for (int x = 0; x < SIDE; x++)
{
	fprintf(ut, "%f%c", data[y * SIDE + x].y, (x == SIDE - 1) ? '\n' : ' ');
}
}
	fclose(ut);
//	return 0;

}
}
}

/*
First real run 40 max
FORCED ACCEPT: x: 3, y: 18 - 0.022504 at 1.092070 : 2242061312.000000
                                                    941699328.000000
                                                    2004372992

Outer iteration 1000

Second real run 3052 max
Outer iteration 5635
FORCED ACCEPT: x: 1, y: 8 - 1.981631 at 2.325389 : 2212167168.000000
*/