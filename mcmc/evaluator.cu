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

// Total side of pattern
int SIDE = 256;
// Side of non-zero rect in real/autocorrelation space
// (definition dependent on the problem we are solving)
int SHORT_SIDE = 25;
int elem_count = 2 * (SHORT_SIDE * SHORT_SIDE / 2 + SHORT_SIDE % 2);

#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

#include <dlib/optimization.h>

using namespace dlib;

#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>


struct likelihoodgetter
{
double __device__ likelihood(int count, int mask, cufftComplex val)
{	
	if (!mask) return 0;
	double val2 = /*val.x * val.x + val.y * val.y*/ val.x;
	double term = 0;
	if (val2 <= 1e-10)
	{
		term = /*-val.x * val.x;val2 = 1e-9*/ -1e9 + val2 * 1e9;
	}
	else
	{
		term += log(val2) * count - val2;
	}
	term -= fabs(val.y) * 1;

	

//	term -= val.y * val.y;
	return term;
}

double __device__ likelihoodold(int count, int mask, cufftComplex val)
{	
	if (!mask) return 0;
	double val2 = sqrt(val.x * val.x + val.y * val.y);
	double term = 0;
	if (!isfinite(val2)) val2 = 1e-9;
	if (val2 < 1e-9) val2 = 1e-9;
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
return -likelihood(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t));
}
} likelihooder;


typedef matrix<double,0,1> column_vector;

// Keeps the information necessary to do the evaluation
struct evaluator
{
	cufftComplex* data;
	cufftComplex* d_data;
	cufftComplex* d_pattern;
	int* d_truePattern;
	int* d_mask;
	cufftHandle plan;

	evaluator()
	{
		cudaMalloc((void**) &d_data, sizeof(cufftComplex) * SIDE * SIDE);
		cudaMalloc((void**) &d_pattern, sizeof(cufftComplex) * SIDE * SIDE);
		cudaMalloc((void**) &d_mask, sizeof(int) * SIDE * SIDE);
		cudaMalloc((void**) &d_truePattern, sizeof(int) * SIDE * SIDE);
		data = (cufftComplex*) calloc(SIDE * SIDE, sizeof(cufftComplex));

		printf("Memset: %d\n", cudaMemset(d_data, 0, sizeof(cufftComplex) * SIDE * SIDE));
		cufftPlan2d(&plan, SIDE, SIDE, CUFFT_C2C);
	}

	double calc(const column_vector& shortData)
	{
		 column_vector::const_iterator i = shortData.begin();
		 column_vector::const_iterator iorig = i;
		 double sum = 0;
		 for (int y = 0; y < SHORT_SIDE / 2 + SHORT_SIDE % 2; y++)
		 {
			for (int x = 0; x < SHORT_SIDE; x++)
			{
				sum += fabs(*i);
				data[y * SIDE + x].x = *(i);
				data[(SIDE * (SHORT_SIDE - y - 1)) + SHORT_SIDE - x - 1].x = *(i);
				i++;

				data[y * SIDE + x].y = *(i);
				data[(SIDE * (SHORT_SIDE - y - 1)) + SHORT_SIDE - x - 1].y = -*(i);
				sum += fabs(*i);
				i++;
				if (i - iorig >= elem_count) goto end;
			}
		}
end:;

//		cudaMemcpy(d_data, data, sizeof(cufftComplex) * SIDE * SHORT_SIDE, cudaMemcpyHostToDevice);
		cudaMemcpy2D(d_data, sizeof(cufftComplex) * SIDE, &data[SHORT_SIDE / 2 * SIDE + SHORT_SIDE / 2], sizeof(cufftComplex) * SIDE, (SHORT_SIDE / 2 + SHORT_SIDE % 2) * sizeof(cufftComplex), SHORT_SIDE / 2 + SHORT_SIDE % 2, cudaMemcpyHostToDevice);
		cudaMemcpy2D(&d_data[SIDE - SHORT_SIDE / 2], sizeof(cufftComplex) * SIDE, &data[SHORT_SIDE / 2 * SIDE], sizeof(cufftComplex) * SIDE, SHORT_SIDE / 2 * sizeof(cufftComplex), SHORT_SIDE / 2 + SHORT_SIDE % 2, cudaMemcpyHostToDevice);
		cudaMemcpy2D(&d_data[(SIDE - SHORT_SIDE / 2) * SIDE], sizeof(cufftComplex) * SIDE, &data[SHORT_SIDE / 2], sizeof(cufftComplex) * SIDE, (SHORT_SIDE / 2 + SHORT_SIDE % 2) * sizeof(cufftComplex), SHORT_SIDE / 2, cudaMemcpyHostToDevice);
		cudaMemcpy2D(&d_data[(SIDE - SHORT_SIDE / 2) * (SIDE + 1)], sizeof(cufftComplex) * SIDE, &data[0], sizeof(cufftComplex) * SIDE, SHORT_SIDE / 2 * sizeof(cufftComplex), SHORT_SIDE / 2, cudaMemcpyHostToDevice);

		cufftExecC2C(plan, d_data, d_pattern, CUFFT_FORWARD);
		thrust::device_ptr<int> t_mask = thrust::device_pointer_cast(d_mask);
		thrust::device_ptr<int> t_truePattern = thrust::device_pointer_cast(d_truePattern);
		thrust::device_ptr<cufftComplex> t_pattern = thrust::device_pointer_cast(d_pattern);
		double val = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(t_truePattern, t_mask, t_pattern)),
                                         thrust::make_zip_iterator(thrust::make_tuple(&t_truePattern[SIDE * SIDE], &t_mask[SIDE * SIDE], &t_pattern[SIDE * SIDE])), likelihooder,
					 (double) 0, thrust::plus<double>()) /*+ 34154. * SIDE * SIDE*/;
		printf("VAL %le %le\n", sum, val);
		static int count = 0;
		count++;
		if (!(count & 255)) fflush(stdout);
		return val;
	}

	double operator() (const column_vector& shortData) const {
	       return ((evaluator*)this)->calc(shortData);
	}



	void setTarget(int* pattern, int* mask)
	{
		cudaMemcpy(d_truePattern, pattern, sizeof(int) * SIDE * SIDE, cudaMemcpyHostToDevice);
		cudaMemcpy(d_mask, mask, sizeof(int) * SIDE * SIDE, cudaMemcpyHostToDevice);
	}

	~evaluator()
	{
		cudaFree(d_data);
		cudaFree(d_pattern);
		cudaFree(d_truePattern);
		cudaFree(d_mask);
	}
};


void initjackdaw()
{    
}

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
     int* mask = (int*) malloc(sizeof(int) * SIDE * SIDE);
     int* truePattern = (int*) malloc(sizeof(int) * SIDE * SIDE);

	readtxt(argv[1], truePattern);
	int sum = 0;
	for (int x = 0; x < SIDE * SIDE; x++)
	{
	sum += truePattern[x];
		mask[x] = 1;
	}
evaluator eval;
     eval.setTarget(truePattern, mask);



/*	int index = (SHORT_SIDE / 2) * (SIDE + 1);
	data[index].x = sqrt(sum) / SIDE / SIDE;
	data[index].y = 0;*/

	int seed;
	sscanf(argv[2], "%d", &seed);
	generator.engine().seed(seed);

	

	initjackdaw();

	double oldLikelihood = -INFINITY;

	int iter = 0;
	int forced = 0;
	int chosen = 0;
	int skips = 0;
	column_vector guess;
	guess.set_size(elem_count, 1);
	for (int y = 0; y < SHORT_SIDE / 2 + SHORT_SIDE % 2; y++)
	{
		for (int x = 0; x < SHORT_SIDE; x++)
		{
			int index = y * SIDE + x;
			if ((y * SHORT_SIDE + x) * 2 >= elem_count) goto end;
			float angle = randval(0, M_PI * 2);
			float val = randval(1e-10, 15);
			//cufftComplex old = data[index];

//			guess((y * SHORT_SIDE + x) * 2) = val * cos(angle);
//			guess((y * SHORT_SIDE + x) * 2 + 1) = val * sin(angle);
			guess((y * SHORT_SIDE + x) * 2) = 0.1;
			guess((y * SHORT_SIDE + x) * 2 + 1) = 0.0;
		}
	}
end:;

        printf("Maximum: %lf\n", find_min_bobyqa(eval, guess, 2 * SHORT_SIDE * SHORT_SIDE + 10, uniform_matrix<double>(elem_count,1,-500), uniform_matrix<double>(elem_count,1, 500),
			 	 	10, 1e-2, 1e6));

	
        printf("Maximum: %lf\n", find_min_using_approximate_derivatives(cg_search_strategy(),
                                               objective_delta_stop_strategy(1e-7).be_verbose(),
                                               eval, guess, 1e30));


        printf("Maximum: %lf\n", find_min_using_approximate_derivatives(bfgs_search_strategy(),
                                               objective_delta_stop_strategy(1e-7).be_verbose(),
                                               eval, guess, 1e30));


/*        printf("Maximum: %lf\n", find_min_using_approximate_derivatives(newton_search_strategy(),
                                               objective_delta_stop_strategy(1e-7).be_verbose(),
                                               eval, guess, 1e30));*/

        printf("Maximum: %lf\n", find_min_using_approximate_derivatives(cg_search_strategy(),
                                               objective_delta_stop_strategy(1e-9).be_verbose(),
                                               eval, guess, 1e30));

        printf("Maximum: %lf\n", find_min_using_approximate_derivatives(bfgs_search_strategy(),
                                               objective_delta_stop_strategy(1e-9).be_verbose(),
                                               eval, guess, 1e30));

/*        printf("Maximum: %lf\n", find_min_using_approximate_derivatives(newton_search_strategy(),
                                               objective_delta_stop_strategy(1e-9).be_verbose(),
                                               eval, guess, 1e30));*/


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