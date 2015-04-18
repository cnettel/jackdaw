
#include <stdio.h>
#include <boost/multi_array.hpp>
#include <H5Cpp.h>

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "math_constants.h"

using boost::multi_array;
using boost::extents;

const int NX = 414;
const int NY = 414;

struct idealsphere : public thrust::unary_function<int, float>
{
  float rfactor;
  float roffset;
  float baseoffsetx;
  float baseoffsety;
  float lfactor;
  float offsetx, offsety;
  float r;
  thrust::device_ptr<short> dPhotons;
  thrust::device_ptr<float> dLambdas;

  idealsphere(thrust::device_vector<short>& dPhotons, thrust::device_vector<float>& dLambdas) :
  dPhotons(dPhotons.data()), dLambdas(dLambdas.data()) {}
  
  __host__ __device__ float operator () (const int& data) {
    if (!dLambdas[data]) return 0;

    float x = (data % NX) - offsetx;
    float y = (data / NX) - offsety;
    
    float q = sqrt(x * x + y * y);
    float val = 3 * (sinpif(2 * q * r)  - 2 * ((float) CUDART_PI_F) * q * r * cospif(2 * q * r));
    float den = (2 * ((float) CUDART_PI_F) * q * r);
    den = den * den * den;
    
    val /= den;
    val = val * val;
    
    return val;
  }
};

struct likelihood : public thrust::unary_function<int, float>
{
  idealsphere& spherer;
  float factor;
__host__ __device__ likelihood(idealsphere& spherer, float factor) : spherer(spherer), factor(factor) {}

  __host__ __device__ float operator () (const int& data) {
    if (!spherer.dLambdas[data]) return 0;

    float intensity = spherer(data) * factor + spherer.dLambdas[data] * spherer.lfactor;
    float val = 0;
    if (spherer.dPhotons[data])
      {
	val += spherer.dPhotons[data] * log(intensity);
      }
    val -= intensity;

    return val;
  }
};


__global__ void computeintensity(float* target, idealsphere myspherer, float psum, float lsum)
{
  myspherer.r = exp(myspherer.roffset + (threadIdx.z + blockIdx.z * blockDim.z) * myspherer.rfactor);
  myspherer.offsetx = (threadIdx.x + blockIdx.x * blockDim.x) + myspherer.baseoffsetx;
  myspherer.offsety = (blockIdx.y + myspherer.baseoffsety);
  myspherer.lfactor = 1.0 / sqrt(lsum) * threadIdx.y + 1.0;

  int idx =  (threadIdx.x + blockIdx.x * blockDim.x)  + (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x + (threadIdx.z + blockIdx.z * blockDim.z)  * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
  float intensity = thrust::reduce(thrust::seq,
				   thrust::make_transform_iterator(thrust::make_counting_iterator(0), myspherer),
				   thrust::make_transform_iterator(thrust::make_counting_iterator(NY * NX), myspherer));

  float intensityfactor = (psum - lsum * myspherer.lfactor) / intensity;
  if (intensityfactor < 1e-9) intensityfactor = 1e-9;
  likelihood likelihooder(myspherer, intensityfactor);
  float likelihood = thrust::reduce(thrust::seq,
				   thrust::make_transform_iterator(thrust::make_counting_iterator(0), likelihooder),
				   thrust::make_transform_iterator(thrust::make_counting_iterator(NY * NX), likelihooder));
  target[idx] = likelihood;
}

int main()
{
  //  H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/Background_RNA/HITS_RNApol/ToFhits.h5", H5F_ACC_RDONLY);
  H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/Background_RNA/HITS_RNApol/Hits.h5", H5F_ACC_RDONLY);
//  H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/20sizes_RNA/forCARL_PDB/HITS_PDB.h5", H5F_ACC_RDONLY);
  H5::Group group = file.openGroup("with_geometry");
  H5::DataSet lambdas = group.openDataSet("lambdas");
  H5::DataSet photons = group.openDataSet("ph_count");
  H5::DataSpace lambdaSpace = lambdas.getSpace();
  H5::DataSpace photonSpace = photons.getSpace();
  
  hsize_t count[3] = {1, NY, NX};
  hsize_t offset[3] = {0, 0, 0};
  lambdaSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
  photonSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
  H5::DataSpace memSpace(3, count);

  boost::multi_array<float, 2> lambdaVals(extents[NY][NX]);
  boost::multi_array<short, 2> photonVals(extents[NY][NX]);
  thrust::device_vector<short> dPhotons(NY * NX);
  thrust::device_vector<float> dLambdas(NY * NX);
  thrust::device_vector<float> dIntensity(50 * 300 * 1200);
  thrust::host_vector<float> hIntensity(50 * 300 * 1200);

  idealsphere spherer(dPhotons, dLambdas);

  for (int img = 0; img < 2779; img++)
    {
      offset[0] = img;
      lambdaSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
      photonSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
      lambdas.read(lambdaVals.data(), H5::PredType::NATIVE_FLOAT, memSpace, lambdaSpace);
      photons.read(photonVals.data(), H5::PredType::NATIVE_SHORT, memSpace, photonSpace);

      int psum = 0;
      double lsum = 0;
      for (int y = 0; y < NY; y++)
	{
	  for (int x = 0; x < NX; x++)
	    {
	      if ((y > 195 && y < 231) || y < 36 || ((x < 250 /* || (x < 300 && (y > 124 && y < 331))*/ ) && y > 55))
	      {
		photonVals[y][x] = 0;
		lambdaVals[y][x] = 0;
	      }
	      psum += photonVals[y][x];
	      lsum += lambdaVals[y][x];
	    }
	}
      
      dim3 grid(25, 40, 24);
      dim3 block(2, 7, 50);

      spherer.rfactor = 0.005;
      spherer.roffset = -10;
      spherer.baseoffsetx = NX / 2 - 30 - 0.5;
      spherer.baseoffsety = NY / 2 - 10 - 0.5;
      dPhotons.assign(photonVals.data(), photonVals.data() + NY * NX);
      dLambdas.assign(lambdaVals.data(), lambdaVals.data() + NY * NX);
      
      computeintensity<<<grid, block>>>(dIntensity.data().get(), spherer, psum, lsum);
      hIntensity.assign(dIntensity.begin(), dIntensity.end());

      float minval = 1e30;
float maxval = -1e30;
      int maxidx = 0;
      for (int k = 0; k < 280 * 50 * 1200; k++)
{
	if (hIntensity[k] > maxval)
	{
		maxval = hIntensity[k];
		maxidx = k;
	}
	if (hIntensity[k] < minval) minval = hIntensity[k];
}
	int maxR = maxidx / 280 / 50;
	int maxX = maxidx % 50;
	int maxY = (maxidx / 50) % 280;

      printf("%d %d %lf %g %g %g %d %d %d %d\n", img, psum, lsum, minval, maxval, hIntensity[0], maxR, maxX, maxY, cudaGetLastError());
      fflush(stdout);
    }
}
