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
const int SIDE = 256;
// Side of non-zero rect in real/autocorrelation space
// (definition dependent on the problem we are solving)
const int SHORT_SIDE = 25;
const int elem_count = SIDE * SIDE;

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


#define cufftComplex cufftDoubleComplex

double factor = 1;

struct likelihoodgetter
{
double factor;

double __device__ likelihood(int count, int mask, cufftComplex val)
{	
	if (!mask) return 0;
	double val2 = /*val.x * val.x + val.y * val.y*/ val.x;
	double term = 0;
	const double lim = 1e-10;
	if (val2 <= lim)
	{
		term += (val2 - lim) * factor;
		val2 = lim;
	}
	term += log(val2) * count - val2;
	//term -= (val2 - count) * (val2 - count);	
	term -= fabs(val.y) * 1;

	

//	term -= val.y * val.y;
//	term -= 34154;
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

struct supportpenalty
{
double factor;

  double __device__ operator ()(thrust::tuple<int, cufftComplex> val)
{
  int x = thrust::get<0>(val) % SIDE;
  int y = thrust::get<0>(val) / SIDE;
  if (!((x < SHORT_SIDE / 2 || x >= SIDE - SHORT_SIDE / 2 - (SHORT_SIDE % 2)) &&
      (y < SHORT_SIDE / 2 || y >= SIDE - SHORT_SIDE / 2 - (SHORT_SIDE % 2)))) {
    cufftComplex input = thrust::get<1>(val);
    return factor * (input.x * input.x + input.y * input.y);
  }
  return 0;
}
} supportpenalizer;

struct gradientgetter {
  double __device__ gradient(int count, int mask, cufftComplex val, cufftComplex gTempl)
  {
    if (!mask) return 0;
    double val2 = /*val.x * val.x + val.y * val.y*/ val.x;
    double term = 0;

    term -= 2 * (val2 - count) * gTempl.x;
    if (val.y > 0) term -= gTempl.y;
    if (val.y < 0) term += gTempl.y;

    return term;
  }

double __device__ operator ()(thrust::tuple<int, int, cufftComplex, cufftComplex> t)
{
return -gradient(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t), thrust::get<3>(t));

}
} gradienter;


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

  thrust::device_ptr<int> t_mask;
  thrust::device_ptr<int> t_truePattern;
  thrust::device_ptr<cufftComplex> t_pattern;
  thrust::device_ptr<cufftComplex> t_data;

	evaluator()
	{
		cudaMalloc((void**) &d_data, sizeof(cufftComplex) * SIDE * SIDE);
		cudaMalloc((void**) &d_pattern, sizeof(cufftComplex) * SIDE * SIDE);
		cudaMalloc((void**) &d_mask, sizeof(int) * SIDE * SIDE);
		cudaMalloc((void**) &d_truePattern, sizeof(int) * SIDE * SIDE);
		data = (cufftComplex*) calloc(SIDE * SIDE, sizeof(cufftComplex));

		t_mask = thrust::device_pointer_cast(d_mask);
		t_truePattern = thrust::device_pointer_cast(d_truePattern);
		t_pattern = thrust::device_pointer_cast(d_pattern);
		t_data = thrust::device_pointer_cast(d_data);

		printf("Memset: %d\n", cudaMemset(d_data, 0, sizeof(cufftComplex) * SIDE * SIDE));
		cufftPlan2d(&plan, SIDE, SIDE, CUFFT_Z2Z);
	}

  void dofft(const column_vector& shortData)
  {
		 column_vector::const_iterator i = shortData.begin();
		 column_vector::const_iterator iorig = i;
		 double sum = 0;

		 for (; i - iorig != elem_count; i++) {
		     sum += (*i) * (*i);
		 }
		 /*sum /= sqrt(sum);
		 sum *= fabs(iorig[elem_count]) * 1e-5;*/
		 sum = 1;
		 i = shortData.begin();	
		 for (int y = 0; y < SIDE; y++)
		 {
			for (int x = 0; x < SIDE; x++)
			{
				data[y * SIDE + x].x = *(i);
				i++;
				if (i - iorig >= elem_count) goto end;
			}
		}
end:;

		cudaMemcpy(d_data, data, sizeof(cufftComplex) * SIDE * SIDE, cudaMemcpyHostToDevice);
/*		cudaMemcpy2D(d_data, sizeof(cufftComplex) * SIDE, &data[SHORT_SIDE / 2 * SIDE + SHORT_SIDE / 2], sizeof(cufftComplex) * SIDE, (SHORT_SIDE / 2 + SHORT_SIDE % 2) * sizeof(cufftComplex), SHORT_SIDE / 2 + SHORT_SIDE % 2, cudaMemcpyHostToDevice);
		cudaMemcpy2D(&d_data[SIDE - SHORT_SIDE / 2], sizeof(cufftComplex) * SIDE, &data[SHORT_SIDE / 2 * SIDE], sizeof(cufftComplex) * SIDE, SHORT_SIDE / 2 * sizeof(cufftComplex), SHORT_SIDE / 2 + SHORT_SIDE % 2, cudaMemcpyHostToDevice);
		cudaMemcpy2D(&d_data[(SIDE - SHORT_SIDE / 2) * SIDE], sizeof(cufftComplex) * SIDE, &data[SHORT_SIDE / 2], sizeof(cufftComplex) * SIDE, (SHORT_SIDE / 2 + SHORT_SIDE % 2) * sizeof(cufftComplex), SHORT_SIDE / 2, cudaMemcpyHostToDevice);
		cudaMemcpy2D(&d_data[(SIDE - SHORT_SIDE / 2) * (SIDE + 1)], sizeof(cufftComplex) * SIDE, &data[0], sizeof(cufftComplex) * SIDE, SHORT_SIDE / 2 * sizeof(cufftComplex), SHORT_SIDE / 2, cudaMemcpyHostToDevice);*/
		
		cufftExecZ2Z(plan, d_data, d_pattern, CUFFT_FORWARD);
  }

	double calc(const column_vector& shortData)
	{
	  dofft(shortData);

	  double val = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(t_truePattern, t_mask, t_data)),
						thrust::make_zip_iterator(thrust::make_tuple(&t_truePattern[SIDE * SIDE], &t_mask[SIDE * SIDE], &t_data[SIDE * SIDE])), likelihooder,
						(double) 0, thrust::plus<double>()) /*+ 34154. * SIDE * SIDE*/;

	  val += thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0), t_pattern)),
						thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(SIDE * SIDE), &t_pattern[SIDE * SIDE])), supportpenalizer,
						(double) 0, thrust::plus<double>()) /*+ 34154. * SIDE * SIDE*/;
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

struct derivator
{
  evaluator& eval1;
  evaluator eval2;


  derivator(evaluator& eval1) : eval1(eval1) {}
  const column_vector operator() (const column_vector& shortData) const
  {
    eval1.dofft(shortData);
    column_vector shortData2;
    column_vector toReturn;
    shortData2.set_size(shortData.size());
    toReturn.set_size(shortData.size());
    for (int k = 0; k < elem_count; k++)
      {
	shortData2(k) = 0;
      }
    for (int k = 0; k < elem_count; k++)
      {
	shortData2(k) = 1;
	((derivator*) this)->eval2.dofft(shortData2);
	toReturn(k) =  thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(eval1.t_truePattern, eval1.t_mask, eval1.t_pattern, eval2.t_pattern)),
					      thrust::make_zip_iterator(thrust::make_tuple(&eval1.t_truePattern[SIDE * SIDE], &eval1.t_mask[SIDE * SIDE], &eval1.t_pattern[SIDE * SIDE], &eval2.t_pattern[SIDE * SIDE])), gradienter,
					 (double) 0, thrust::plus<double>()) /*+ 34154. * SIDE * SIDE*/;
	shortData2(k) = 0;
      }

      return toReturn;
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
	double val;
	fscanf(in, "%lf", &val);
	data[x] = val;
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
	  derivator deriv(eval);
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

	column_vector lower;
	column_vector upper;
	lower.set_size(elem_count, 1);
	upper.set_size(elem_count, 1);
	//	guess(elem_count) = 1;
	for (int y = 0; y < SIDE; y++)
	{
		for (int x = 0; x < SIDE; x++)
		{
			float val = /*randval(1e-10, 15)*/ truePattern[y * SIDE + x];
			//cufftComplex old = data[index];
			if (val <= 1e-9) val = 1e-2;
			guess(y * SIDE + x) = val;
			lower(y * SIDE + x) = 1e-9;
			upper(y * SIDE + x) = 1e6;
		}
	}

	//lower(SIDE * (SIDE / 2) + SIDE / 2) = 1e5;
end:;

/*        printf("Maximum: %lf\n", find_min_bobyqa(eval, guess, 2 * SHORT_SIDE * SHORT_SIDE + 10, uniform_matrix<double>(elem_count + 1,1,-25000), uniform_matrix<double>(elem_count + 1,1, 25000),
			 	 	10000, 1e-2, 1e6));*/

	
/*        printf("Maximum: %lf\n", find_min_using_approximate_derivatives(newton_search_strategy(),
                                               gradient_norm_stop_strategy(1e-7).be_verbose(),
                                               eval, guess, -1e30));*/

factor = 1.0 / 256;
//	for (; factor <= 1; factor *= 2)
{
	likelihooder.factor = factor;	
	supportpenalizer.factor = factor;
        printf("Maximum3: %lf %lf\n", find_min_box_constrained(lbfgs_search_strategy(5),
                                               objective_delta_stop_strategy(1e-7, 2000).be_verbose(),
							       eval, derivative(eval), guess, lower, upper), factor);
/*        printf("Maximum2: %lf %lf\n", find_min(cg_search_strategy(),
                                               objective_delta_stop_strategy(1e-7, 2000).be_verbose(),
                                               eval, deriv, guess, -1e30), factor);*/
}



for (int y = 0; y < SIDE; y++)
  {
    for (int x = 0; x < SIDE; x++)
      {
	printf("%lf\t", (double) eval.data[y * SIDE + x].x);
      }
    printf("\n");
  }

for (int y = 0; y < SIDE; y++)
  {
    for (int x = 0; x < SIDE; x++)
      {
	printf("%lf\t", (double) eval.data[y * SIDE + x].y);
      }
    printf("\n");
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
