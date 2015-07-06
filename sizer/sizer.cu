#include <stdio.h>
#include <boost/multi_array.hpp>
#include <H5Cpp.h>

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <string>
#include "math_constants.h"

using boost::multi_array;
using boost::extents;
using namespace std;

const int NX = 414;
const int NY = 414;

struct idealsphere : public thrust::unary_function<int, float>
{
  float rfactor;
  float roffset;
  float baseoffsetx;
  float baseoffsety;
  float lfactor;
  float ebase;
  float offsetx, offsety;
//  float arraybase;
  float r;
  int tid;
  int extrax;
  int extray;
  thrust::device_ptr<short> dPhotons;
  thrust::device_ptr<float> dLambdas;

  idealsphere(thrust::device_vector<short>& dPhotons, thrust::device_vector<float>& dLambdas) :
  dPhotons(dPhotons.data()), dLambdas(dLambdas.data()) {}
  
  __host__ __device__ float operator () (const int& data) {
    if (!dLambdas[data]) return 0;
    int rawx = data % NX;
    int rawy = data / NX;

    if (rawx < 211) rawx += 4;
    if (rawy < 219) rawy += 4;
    /*if (rawx < 211) rawx += 37;
    if (rawy < 219) rawy += 11;*/
//    if (rawx < 233 - extrax * 3) rawx += /*29*/ -18 + extrax * 3;
//    if (rawy < 226 - extray * 2) rawy += /*3*/ -12 + extray * 2;*/

    float x = rawx - offsetx;
    float y = rawy - offsety;
    
    float q = sqrt(x * x + y * y);
float val;
    float den = (2 * ((float) CUDART_PI_F) * q * r);
    if (r > 0.5e-4)
{
    if ( r < 1e-4)
    val = 3 * den;
else
    val = 3 * (sinpif(2 * q * r)  - 2 * ((float) CUDART_PI_F) * q * r * cospif(2 * q * r));

    den = den * den * den;
    
    val /= den;
    val = val * val;
}
else
val = 1.0f;
    
    return val;
  }
};

struct likelihood : public thrust::unary_function<int, float>
{
  idealsphere& spherer;
  float factor;
  
__host__ __device__ likelihood(idealsphere& spherer, float factor) : spherer(spherer), factor(factor) {}

  __host__ __device__ float getIntensity(const int& data) {
  	float lbase = spherer.dLambdas[data] * spherer.lfactor;
	lbase += spherer.ebase;
//	if (lbase < 2e-3) lbase = 2e-3;
  	float intensity = spherer(data) * factor + lbase;

	return intensity;
  }

  __host__ __device__ float operator () (const int& data) {
    if (!spherer.dLambdas[data]) return 0;

    float intensity = getIntensity(data);
    float val = 0;
    if (spherer.dPhotons[data])
{   
      val += spherer.dPhotons[data] * log(intensity);
}
    val -= intensity;
//    float weight = spherer.dPhotons[data] > 0;
//    float prob = 1.0f - exp(-intensity);
//    if (spherer.dPhotons[data])
      {
//      val += (weight/* - prob*/) * log(prob);
      }
//      else
//    val += ((1.0f - weight)/* - (1.0f - prob)*/) * (-intensity);

    return val;
  }
};

struct intensitygetter : public thrust::unary_function<int, float>
{
  likelihood& likelihooder;

  __host__ __device__ intensitygetter(likelihood& likelihooder) : likelihooder(likelihooder) {}

  __host__ __device__ float operator () (const int& data)
  {
    if (!likelihooder.spherer.dLambdas[data]) return 0;

    return likelihooder.getIntensity(data);
  }
};

__device__ likelihood getObjects(idealsphere& myspherer, unsigned int tx, unsigned int ty, unsigned int zval, unsigned int bx, unsigned int by, float lsum, float psum)
{
  myspherer.r = exp(myspherer.roffset + (zval) * myspherer.rfactor);
  myspherer.offsetx = (tx) * 2 + myspherer.baseoffsetx;
  myspherer.offsety = (ty * 2 + myspherer.baseoffsety);
  myspherer.extrax = bx;
  myspherer.extray = by;
  //myspherer.lfactor = 0.25 / sqrt(lsum) * (((int) bx 0)) + 1.0;
  //myspherer.lfactor =  1.00 * pow(1.04, bx - 2 * 0.5);
//  myspherer.ebase = (0 + bx) * 1e-4;
  myspherer.ebase = -lsum * (myspherer.lfactor - 1) / 46215;
  myspherer.ebase = 0;
//    myspherer.lfactor = 1.0;

  float intensity = thrust::reduce(thrust::seq,
				   thrust::make_transform_iterator(thrust::make_counting_iterator(0), myspherer),
				   thrust::make_transform_iterator(thrust::make_counting_iterator(NY * NX), myspherer));

  float intensityfactor = (psum - lsum * myspherer.lfactor) / intensity;
  if (intensityfactor < 1e-13)
{
/*target[idx] = -1e10;
intens[idx] = 0;*/
intensityfactor = 1e-13;
}
  //intensityfactor *= pow(1.01, myspherer.tid - 128 * 0.5);
  likelihood likelihooder(myspherer, intensityfactor);

  return likelihooder;
}

__global__ void getExpLambdas(float* target, float* target2, int maxidx, idealsphere myspherer, float psum, float lsum, dim3 block, dim3 grid)
{
  int tx = maxidx % block.x;
  int bx = (maxidx / block.x) % grid.x;
  int ty = (maxidx / block.x / grid.x) % block.y;
  int by = (maxidx / block.x / grid.x / block.y) % grid.y;
  int zval = (maxidx / block.x / grid.x / block.y / grid.y);
  likelihood likelihooder = getObjects(myspherer, tx, ty, zval, bx, by, lsum, psum);
  intensitygetter intensities(likelihooder);

  thrust::tabulate(thrust::device, target, target + NX * NY, intensities);
  thrust::tabulate(thrust::device, target2, target2 + NX * NY, likelihooder);
}

__global__ void computeintensity(float* target, float* intens, idealsphere myspherer, float psum, float lsum)
{
  likelihood likelihooder = getObjects(myspherer, threadIdx.x, threadIdx.y, threadIdx.z + blockIdx.z * blockDim.z, blockIdx.x, blockIdx.y, lsum, psum);
  float likelihood1 = thrust::reduce(thrust::seq,
				   thrust::make_transform_iterator(thrust::make_counting_iterator(0), likelihooder),
				   thrust::make_transform_iterator(thrust::make_counting_iterator(NY * NX), likelihooder));

  int idx =  (threadIdx.x + blockIdx.x * blockDim.x)  + (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x + (threadIdx.z + blockIdx.z * blockDim.z)  * gridDim.x * blockDim.x * gridDim.y * blockDim.y;	
  target[idx] = likelihood1;
  intens[idx] = likelihooder.factor;
}

int main()
{
//    H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/Background_RNA/HITS_RNApol/ToFhits.h5", H5F_ACC_RDONLY);
//  H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_MS2/ALL_runsRNApol_differentGainmapAndMask/HITS354/HITS.h5", H5F_ACC_RDONLY);
//H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_MS2/ALL_runsMS2/HITS_MS2/HITS.h5", H5F_ACC_RDONLY);
//H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_MS2/ALL_runsTBSV/HITS_TBSV/HITS.h5", H5F_ACC_RDONLY);
// RNA pol H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/BackgroundProva/HITS_RNApol/Hits.h5", H5F_ACC_RDONLY);

//H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_OmRV/ALLrunsOmRV/HITSOmRV/HITS.h5", H5F_ACC_RDONLY);
//H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_OmRV/ALLrunsOmRV/HITSOmRV/HITS1.h5", H5F_ACC_RDONLY);
//H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/ADUsim_70nm/HITS.h5", H5F_ACC_RDONLY);
//H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/ADUsim_RNAPII_600mm/HITS.h5", H5F_ACC_RDONLY);
//         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/ADUsim_12nm/HITS.h5", H5F_ACC_RDONLY);
//         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/ADUsim_RNAPII_600mm_1022/HITS.h5", H5F_ACC_RDONLY);
//         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/ADUsim_RDV/HITS.h5", H5F_ACC_RDONLY);
//         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/OmRVdata1/HITS.h5", H5F_ACC_RDONLY);

         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/OmRV/HITS3sigma.h5", H5F_ACC_RDONLY);
//         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/RNAPII/HITSbackgrand.h5", H5F_ACC_RDONLY);

//           H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/RNAPII/HITS.h5", H5F_ACC_RDONLY);
//         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/ADUsim_70nmICO/HITS.h5", H5F_ACC_RDONLY);
//  H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/12nm_SIM/HITS/HITS.h5", H5F_ACC_RDONLY);
//  H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/20sizes_RNA/forCARL_PDB/HITS_PDB.h5", H5F_ACC_RDONLY);
  H5::Group group = file.openGroup("with_geometry");
  H5::DataSet lambdas = group.openDataSet("lambdas");
  H5::DataSet photons = group.openDataSet("photon_count");
  H5::DataSpace lambdaSpace = lambdas.getSpace();
  H5::DataSpace photonSpace = photons.getSpace();
  
  hsize_t count[3] = {1, NY, NX};
  hsize_t offset[3] = {0, 0, 0};
  hsize_t fullsize[3];
  lambdaSpace.getSimpleExtentDims(fullsize);

//  H5::H5File expectedLambdaFile(string(getenv("SLURM_JOB_ID")) + ".h5", H5F_ACC_TRUNC);
//  H5::DataSet expectedLambdas = expectedLambdaFile.createDataSet("explambdas", H5::PredType::NATIVE_FLOAT, lambdaSpace);
//  H5::DataSpace expectedLambdaSpace = expectedLambdas.getSpace();
//  H5::DataSet logLs = expectedLambdaFile.createDataSet("logl", H5::PredType::NATIVE_FLOAT, lambdaSpace);
//  H5::DataSpace logLsSpace = expectedLambdas.getSpace();

  lambdaSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
//  logLsSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
  photonSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
  H5::DataSpace memSpace(3, count);

  boost::multi_array<float, 2> lambdaVals(extents[NY][NX]);
  thrust::host_vector<float> expLambdaVals(NY * NX);
  thrust::host_vector<float> logLsVals(NY * NX);
  boost::multi_array<short, 2> photonVals(extents[NY][NX]);
  thrust::device_vector<short> dPhotons(NY * NX);
  thrust::device_vector<float> dLambdas(NY * NX);
  thrust::device_vector<float> dExpLambdas(NY * NX);
  thrust::device_vector<float> dLogLs(NY * NX);
  thrust::device_vector<float> dIntensity(175000000);
  thrust::host_vector<float> hIntensity(175000000);
  thrust::device_vector<float> dIntensity2(175000000);
  thrust::host_vector<float> hIntensity2(175000000);

  idealsphere spherer(dPhotons, dLambdas);
  char* taskid = getenv("SLURM_ARRAY_TASK_ID");
  int tid;
  sscanf(taskid, "%d", &tid);
  spherer.lfactor = 1.0 * pow(1.01, (tid) - 64);
  spherer.tid = tid;

  for (int img = 0; img < fullsize[0]; img++)
    {
      offset[0] = img;
      lambdaSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
//      expectedLambdaSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
      photonSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
      lambdas.read(lambdaVals.data(), H5::PredType::NATIVE_FLOAT, memSpace, lambdaSpace);
      photons.read(photonVals.data(), H5::PredType::NATIVE_SHORT, memSpace, photonSpace);
      int psum = 0;
      double lsum = 0;
      for (int y = 0; y < NY; y++)
	{
	  for (int x = 0; x < NX; x++)
	    {
/*	      if (((y > 195 && y < 231) || y < 36 || ((x < 255  || (x < 300 && (y > 124 && y < 325))) && (y > 92 || x > 170 || x < 90)) && !(x<55 && y > 374))
	      || y < 105 || y > 343)
	      {
		photonVals[y][x] = 0;
		lambdaVals[y][x] = 0;
	      }*/
	      if (x > 393 || y > 411 || (/*(x + y > 700) ||*/ (y < 235 || (x < 300 && y < 314) || x < 255)  && !(y < 183 && y > 39 && x > 290 && x < 393) &&
	      !(x < 170 && x > 60 && y < 98 && y > 40) &&
	      !(x < 290 && x > 254 && y < 120 && y > 39) &&
	      !(x < 88 && x > 19 && y < 130 && y > 97)))
//	      if ((y < 233 || (x < 350 && y < 370) || x < 255))
	      {
		photonVals[y][x] = 0;
		lambdaVals[y][x] = 0;
	      }
	      psum += photonVals[y][x];
	      lsum += lambdaVals[y][x];
	    }
	}
      
//      if (psum - lsum < 2500) continue;
      dim3 grid(1, 1, 1350);
      dim3 block(32, 32, 1);

      spherer.rfactor = 0.005;
      spherer.roffset = -10;
      spherer.baseoffsetx = NX / 2 - 10 - 29 - 0.5; // good val -10
      spherer.baseoffsety = NY / 2 + 10 - 35 - 0.5; // good val +10
      dPhotons.assign(photonVals.data(), photonVals.data() + NY * NX);
      dLambdas.assign(lambdaVals.data(), lambdaVals.data() + NY * NX);
      
      computeintensity<<<grid, block>>>(dIntensity.data().get(), dIntensity2.data().get(), spherer, psum, lsum);
      hIntensity.assign(dIntensity.begin(), dIntensity.end());
      hIntensity2.assign(dIntensity2.begin(), dIntensity2.end());

      float minval = 1e30;
float maxval = -1e30;
      int maxidx = 0;
      float maxint = 0;
      for (int k = 0; k < grid.y * block.y * grid.x * block.x * grid.z * block.z; k++)
{
	if (hIntensity[k] > maxval)
	{
		maxval = hIntensity[k];
		maxint = hIntensity2[k];
		maxidx = k;
	}
	if (hIntensity[k] < minval) minval = hIntensity[k];
}

/*      dim3 unigrid(1,1,1);
      dim3 uniblock(1,1,1);
      getExpLambdas<<<unigrid, uniblock>>>(dExpLambdas.data().get(), dLogLs.data().get(), maxidx, spherer, psum, lsum, block, grid);
      expLambdaVals.assign(dExpLambdas.begin(), dExpLambdas.end());*/
//      logLsVals.assign(dLogLs.begin(), dLogLs.end());
//      expectedLambdas.write(expLambdaVals.data(), H5::PredType::NATIVE_FLOAT, memSpace, expectedLambdaSpace);
//      logLs.write(logLsVals.data(), H5::PredType::NATIVE_FLOAT, memSpace, expectedLambdaSpace);
      //expectedLambdaFile.flush(H5F_SCOPE_LOCAL);
      //hIntensity2.assign(dIntensity2.begin(), dIntensity2.end());
	int maxR = maxidx / grid.y / block.y / grid.x / block.x;
	int maxX = maxidx % block.x;
	int maxY = (maxidx / (grid.x * block.x)) % block.y;
	int maxI = (maxidx / (block.x * grid.x * block.y)) % grid.y;
	int maxI2 = (maxidx / block.x) % grid.x;

      printf("%d %d %lf %g %g %g %d %d %d %d %g %d %d\n", img, psum, lsum, minval, maxval, hIntensity[0], maxR, maxX, maxY, cudaGetLastError(), maxint, maxI, maxI2);
      fflush(stdout);
    }
}
