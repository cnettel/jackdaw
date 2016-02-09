#include <stdio.h>
#include <boost/multi_array.hpp>
#include <H5Cpp.h>

#include <cufft.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <string>
#include <set>
#include "math_constants.h"

using boost::multi_array;
using boost::extents;
using namespace std;

const int NX = 414;
const int NY = 414;

const int FX = 2048;
const int FY = 2048;
const int BS = 2;

__device__ __host__ float sinc(float val)
{
	if (val == 0) return 1;
	return sinpif(val) / (val * CUDART_PI_F);
}

struct realsphere : public thrust::unary_function<int, cufftComplex>
{
  float r;
  float sigma;
  float sigmag;
  float x;
  float y;

  __host__ __device__ cufftComplex operator () (const int& data) {
    int rawx = data & (FX - 1);
    int rawy = data / FX;
    float qx = rawx - x - FX * 0.5;
    float qy = rawy - y - FY * 0.5;

    cufftComplex val;
    val.y = 0.f;


    float r2 = r * r - qx * qx - qy * qy;
    if (r2 < 0.f)
    {
	val.x = 0.f;
	return val;
    }
    else
	r2 = sqrt(r2);

    
    qx += x;
    qy += y;
    //qy /= 1.8;
    //qx /= 1.5;
    //qx /= 1.8;
    float gr = qx * qx + qy * qy;


    float factor = (/*0.01f + 0.99f **/ expf(- gr / (2 * sigmag * sigmag)));
//    factor *= sinc(qx / sigma * 0.5) * sinc(qy / sigma * 0.5);

    //float factor = 1.0f;

    //if (factor < 0.3f) factor = 0.3f;
    val.x = r2 * factor;

    return val;
  }
};


struct absfunc : public thrust::unary_function<cufftComplex, float>
{
  __host__ __device__ float operator () (const cufftComplex& val) {
      return /* sqrt*/ (val.x * val.x + val.y * val.y);
  }
} sqabser;

struct idealsphere : public thrust::unary_function<int, float>
{
  int offsetx, offsety;
  float baseoffsetx, baseoffsety;
//  float arraybase;
  float r;
  float lfactor;
  float ebase;
  int tid;
  int extrax;
  int extray;
  thrust::device_ptr<short> dPhotons;
  thrust::device_ptr<float> dPattern;
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

    int x = rawx - offsetx;
    int y = rawy - offsety;
    if (x < 0) x += FX;
    if (y < 0) y += FY;

    return dPattern[y * FX + x];
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
	if (lbase < 1e-9) lbase = 1e-9;
  	float intensity = spherer(data) * factor + lbase;

	return intensity;
  }

  __host__ __device__ float operator () (const int& data) {
    if (!spherer.dLambdas[data]) return 0;

    float intensity = getIntensity(data);
    // BAGLIVO STYLE!!!
    if (!spherer.dPhotons[data]) return 0;
    return spherer.dPhotons[data] * log(intensity / spherer.dPhotons[data]);


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

__device__ likelihood getObjects(idealsphere& myspherer, unsigned int tx, unsigned int ty, unsigned int zval, unsigned int bx, unsigned int by, float* lsum, int* psum, long long* phc)
{
//  myspherer.r = exp(myspherer.roffset + (zval) * myspherer.rfactor);
  myspherer.offsetx = (tx * 2.f) /** 1.19f*/ + myspherer.baseoffsetx;
  myspherer.offsety = (ty * 2.f /** 1.19f*/ + myspherer.baseoffsety);
  myspherer.extrax = bx;
  myspherer.extray = by;
  myspherer.dPhotons = &myspherer.dPhotons[zval * NY * NX];
  myspherer.dLambdas = &myspherer.dLambdas[zval * NY * NX];
  //myspherer.lfactor = 0.25 / sqrt(lsum) * (((int) bx 0)) + 1.0;
//  myspherer.lfactor =  1.00 * pow(1.1f, -1.f + bx * 2.f);
  float fittedPhc = phc[zval * 3 + 2];
  float minPhc = max(1e-5, -6 * sqrt(0.6 * fittedPhc) + 0.6 * fittedPhc);
  float maxPhc = 6 * sqrt(1.4 * fittedPhc) + 1.4 * fittedPhc;
  myspherer.lfactor = (1.0f / phc[zval * 3]) * (minPhc + (maxPhc - minPhc) / 24 * bx); 
//myspherer.lfactor = (1.0f / phc[zval * 3]) * (phc[zval * 3 + 2]); 
//  myspherer.ebase = (0 + bx) * 1e-4;
  myspherer.ebase = -lsum[zval] * (myspherer.lfactor - 1) / 46215;
  myspherer.ebase = 0;
//    myspherer.lfactor = 1.0;

  float intensity = thrust::reduce(thrust::seq,
				   thrust::make_transform_iterator(thrust::make_counting_iterator(0), myspherer),
				   thrust::make_transform_iterator(thrust::make_counting_iterator(NY * NX), myspherer));
  float diff = (psum[zval] - lsum[zval] * myspherer.lfactor);
  float intensityfactor = diff / intensity;
  if (intensity == 0) intensityfactor = 1e6f;
  if (diff <= 0 || intensityfactor < 1e-17)
{
/*target[idx] = -1e10;
intens[idx] = 0;*/
intensityfactor = 0;
}
//  intensityfactor *= pow(1.1f, -1.f + by * 2.f);
  likelihood likelihooder(myspherer, intensityfactor);

  return likelihooder;
}

/*__global__ void getExpLambdas(float* target, float* target2, int maxidx, idealsphere myspherer, int psum, float lsum, dim3 block, dim3 grid)
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
}*/

__global__ void computeintensity(float* target, float* intens, idealsphere myspherer, int* psum, float* lsum, long long* phc)
{
  int zval = threadIdx.z + blockIdx.z * blockDim.z;
  likelihood likelihooder = getObjects(myspherer, threadIdx.x, threadIdx.y, zval, blockIdx.x, blockIdx.y, lsum, psum, phc);
  float likelihood1 = -1e30f;
  if (likelihooder.factor > 1e-17)
  {
  likelihood1 = thrust::reduce(thrust::seq,
				   thrust::make_transform_iterator(thrust::make_counting_iterator(0), likelihooder),
				   thrust::make_transform_iterator(thrust::make_counting_iterator(NY * NX), likelihooder));
  }
 


  int idx =  (threadIdx.x + blockIdx.x * blockDim.x)  + (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x + (threadIdx.z + blockIdx.z * blockDim.z)  * gridDim.x * blockDim.x * gridDim.y * blockDim.y;	
  target[idx] = likelihood1;
  intens[idx] = likelihooder.factor /* * myspherer.tid * myspherer.tid * 4 */;
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

//         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/RNAPII_1/HITS3sigma.h5", H5F_ACC_RDONLY);
/*         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/OmRV1/HITS4sigma.h5", H5F_ACC_RDONLY);
	   H5::H5File phcfile("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/OmRV1/photonCount.hdf5", H5F_ACC_RDONLY);*/


//         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_TBSV/TBSV/HITS4sigma.h5", H5F_ACC_RDONLY);
///scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_RNAPII/RNAPII
//RIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_RNAPII/RNAPII/HITS4sigma.h5"
//         H5::H5File phcfile("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_TBSV/TBSV/photon_count.h5", H5F_ACC_RDONLY);
///scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_RNAPII/RNAPII/
///scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_RNAPII/RNAPII/
// "/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_MS2/MS2_1/


//         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_RNAPII/RNAPII/HITS4sigma.h5", H5F_ACC_RDONLY);
///scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_RNAPII/RNAPII
//RIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_RNAPII/RNAPII/HITS4sigma.h5"
//         H5::H5File phcfile("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_RNAPII/RNAPII/photon_count.h5", H5F_ACC_RDONLY);


         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_OmRV/OmRV/HITS4sigma.h5", H5F_ACC_RDONLY);
///scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_RNAPII/RNAPII
//RIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_RNAPII/RNAPII/HITS4sigma.h5"
         H5::H5File phcfile("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/Run_OmRV/OmRV/photon_count.h5", H5F_ACC_RDONLY);


//         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/OmRV/HITS3sigma.h5", H5F_ACC_RDONLY);
//         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/RNAPII/HITSbackgrand.h5", H5F_ACC_RDONLY);

//           H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/CodeCleanUp/stathitf/RNAPII/HITS.h5", H5F_ACC_RDONLY);
//         H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/ADUsim_70nmICO/HITS.h5", H5F_ACC_RDONLY);
//  H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/May2013_RNApol/12nm_SIM/HITS/HITS.h5", H5F_ACC_RDONLY);
//  H5::H5File file("/scratch/fhgfs/alberto/MPI/TODO/EXPERIMENTAL/MASK_90000px/20sizes_RNA/forCARL_PDB/HITS_PDB.h5", H5F_ACC_RDONLY);
  H5::Group group = file.openGroup("with_geometry");
  H5::DataSet lambdas = group.openDataSet("lambdas");
  H5::DataSet photons = group.openDataSet("photon_count");
  H5::DataSet phcframe = phcfile.openDataSet("phc");
  H5::DataSpace lambdaSpace = lambdas.getSpace();
  H5::DataSpace photonSpace = photons.getSpace();
  H5::DataSpace phcSpace = phcframe.getSpace();
  
  hsize_t count[3] = {BS, NY, NX};
  hsize_t offset[3] = {0, 0, 0};
  hsize_t phcoffset[2] = {0, 0};
  hsize_t phccount[2] = {BS, 3};
  hsize_t zerooffset[3] = {0, 0, 0};
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
  H5::DataSpace phcMemSpace(2, phccount);

  boost::multi_array<float, 3> lambdaVals(extents[BS][NY][NX]);
  boost::multi_array<float, 3> lambdaValsZero(extents[BS][NY][NX]);
  thrust::host_vector<float> expLambdaVals(NY * NX);
  thrust::host_vector<float> logLsVals(NY * NX);
  boost::multi_array<short, 3> photonVals(extents[BS][NY][NX]);
  thrust::device_vector<short> dPhotons(BS * NY * NX);
  thrust::device_vector<float> dLambdas(BS * NY * NX);
  thrust::device_vector<float> dExpLambdas(NY * NX);
  thrust::device_vector<float> dLogLs(NY * NX);
  thrust::device_vector<float> dIntensity(BS * 1 * 48 * 32 * 32);
  thrust::host_vector<float> hIntensity(BS * 1 * 48 * 32 * 32);
  thrust::device_vector<float> dIntensity2(BS * 1 * 48 * 32 * 32);
  thrust::host_vector<float> hIntensity2(BS * 1 * 48 * 32 * 32);
  thrust::device_vector<int> dpsum(BS);
  thrust::device_vector<float> dlsum(BS);
  thrust::host_vector<long long> hPhc(BS * 3);
  thrust::device_vector<long long> dPhc(BS * 3);

  thrust::device_vector<cufftComplex> d_complexSpace(FY * FX);
  thrust::device_vector<float> dPattern(FY * FX);
  cufftHandle plan;
  cufftPlan2d(&plan, FX, FY, CUFFT_C2C);
  cufftHandle sizePlan[4097];
  for (int k = 1; k <= 4096; k++)
    {
      cufftPlan2d(&sizePlan[k], FX, FY, CUFFT_C2C);
    }

  idealsphere spherer(dPhotons, dLambdas);
  char* taskid = getenv("SLURM_ARRAY_TASK_ID");
  int tid;
  sscanf(taskid, "%d", &tid);
  spherer.lfactor = 1.0 /** pow(1.01, (tid) - 64)*/;
  spherer.dPattern = dPattern.data();
  spherer.tid = tid;
  realsphere reals;
  //  reals.sigma = /*12 + tid * 4*/ tid * 6;
  reals.sigma = 80;
  reals.sigmag = 30;
  int rc = 0;
  char tlf[255];
  FILE* already = fopen("349126", "r");
  set<int> alreadyset;
  while (fgets(tlf, 255, already))
    {
      int v;
      if (sscanf(tlf, "%d", &v) == 1)
	{
	  alreadyset.insert(v);
	}
    }

  for (int img = 0; img < fullsize[0]; img+=BS, rc++)
    {
      if (alreadyset.find(img) != alreadyset.end()) continue;
      if (rc % 96 != tid) continue;
      int end = min((int) fullsize[0], img + BS);
	int imgcount = end - img;
	count[0] = imgcount;
      offset[0] = img;
      phccount[0] = imgcount;
      phcoffset[0] = img;
      lambdaSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
      memSpace.selectHyperslab(H5S_SELECT_SET, count, zerooffset);
//      expectedLambdaSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
      photonSpace.selectHyperslab(H5S_SELECT_SET, count, offset);
      phcMemSpace.selectHyperslab(H5S_SELECT_SET, phccount, zerooffset);
      phcSpace.selectHyperslab(H5S_SELECT_SET, phccount, phcoffset);
      lambdas.read(lambdaVals.data(), H5::PredType::NATIVE_FLOAT, memSpace, lambdaSpace);
      
      if (img == 0)
      {
	lambdaValsZero = lambdaVals;
      }     
      photons.read(photonVals.data(), H5::PredType::NATIVE_SHORT, memSpace, photonSpace);
      phcframe.read(hPhc.data(), H5::PredType::NATIVE_INT64, phcMemSpace, phcSpace);
      fprintf(stderr, "%lld %lld %lld\n", hPhc[0], hPhc[1], hPhc[2]);
      int psum[BS] = {0};
      float lsum[BS] = {0};
      for (int j = 0; j < imgcount; j++)
	{
	int dpsum = 0;
	double dlsum = 0;
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
	      if (x > 393 || y > 411 || (/*(x + y > 700) ||*/ (y < 235 || (x < 300 && y < 314) || x < 255)  && !(y < 183 && y > 15 && x > 2790 && x < 393) &&
	      !(x < 170 && x > 2 && y < 180 && y > 35) &&
	      !(x < 290 && x > 254 && y < 120 && y > 35) &&
	      !(x < 88 && x > 19 && y < 130 && y > 97)) &&
//	      !(x < 45 && x > 1 && y < 414 && y > 362) &&
	      !(x > 252 && x < 393 && y < 411 && y > 234) &&
	      !(x > 252 && x < 393 && y < 186 && y > 15))
//	      if ((y < 233 || (x < 350 && y < 370) || x < 255))
	      /*if (!((x < 45 && x > 1 && y < 414 && y > 362)) &&
	          !((x < 170 && x > 1 && y < 24 && y > 15)) &&
		  !((x > 255 && x < 390 && y < 24 && y > 15)) &&
		  !(y > 15 && x > 1 && x + y < 81))*/
	      {
		photonVals[j][y][x] = 0;
		lambdaVals[j][y][x] = 0;
	      }
	      if ((y - 230) * (y-230) + (x-200) * (x-200) < 10000)
	      {
		photonVals[j][y][x] = 0;
		lambdaVals[j][y][x] = 0;
	      }

	      dpsum += photonVals[j][y][x];
	      dlsum += lambdaVals[j][y][x];
	    }
	}
	psum[j] = dpsum;
	lsum[j] = dlsum;
	}
      
//      if (psum - lsum < 2500) continue;
      dim3 grid(48, 1, imgcount);
      dim3 block(22, 22, 1);

      int base = (grid.y * block.y * grid.x * block.x * block.z);

      float rfactor = 0.0025;
      float roffset = -10;
      spherer.baseoffsetx = NX / 2 - 10 - 21 - 0.5; // good val -10
      spherer.baseoffsety = NY / 2 + 10 - 13 - 0.5; // good val +10
      dPhotons.assign(photonVals.data(), photonVals.data() + imgcount * NY * NX);
      dLambdas.assign(lambdaVals.data(), lambdaVals.data() + imgcount * NY * NX);
      dPhc.assign(hPhc.data(), hPhc.data() + imgcount * 3);

      dlsum.assign(&lsum[0], &lsum[BS]);
      dpsum.assign(&psum[0], &psum[BS]);


      float minval[BS];
      float maxval[BS];
      int maxidx[BS];
      float maxint[BS];
      int maxr[BS];
      int maxdx[BS];
      int maxdy[BS];
      fill(&maxdx[0], &maxdx[BS], 0);
      fill(&maxdy[0], &maxdy[BS], 0);
      fill(&maxr[0], &maxr[BS], 0);
      fill(&maxint[0], &maxint[BS], 0);
      fill(&maxidx[0], &maxidx[BS], 0);
      fill(&minval[0], &minval[BS], 1e30);
      fill(&maxval[0], &maxval[BS], -1e30);
      
      for (int r = 0; r < 1200; r++)
	{
	  if (r > 350) r += 4;
	  if (r > 600) r += 5;
	  if (r > 1200) r += 10;
	  if (r > 1500) r += 20;
	  //float r2 = r * 1.1e-4 * 0.1;
	  float r2 = r * 0.1;
	  reals.r = r2;
	  spherer.r = r2;
	  for (int dx = 0; dx <= 0.6 * reals.sigma; dx+=4)
	    {
	      for (int dy = -0.6 * reals.sigma; dy <= 0.6 * reals.sigma; dy+= 4)
		{
		  if (dx == 0 && dy > 0) continue;
/*		  int dx = 0;
		  int dy = 0;*/
		  reals.x = dx;
		  reals.y = dy;
		  thrust::tabulate(thrust::device, d_complexSpace.data(), d_complexSpace.data() + FX * FY, reals);
		  cufftExecC2C(plan, d_complexSpace.data().get(), d_complexSpace.data().get(), CUFFT_FORWARD);
		  thrust::transform(thrust::device, d_complexSpace.begin(), d_complexSpace.end(), dPattern.begin(), sqabser);
		  computeintensity<<<grid, block>>>(dIntensity.data().get(), dIntensity2.data().get(), spherer, dpsum.data().get(), dlsum.data().get(), dPhc.data().get());
		  hIntensity.assign(dIntensity.begin(), dIntensity.end());
		  hIntensity2.assign(dIntensity2.begin(), dIntensity2.end());
		  for (int k = 0; k < grid.y * block.y * grid.x * block.x * grid.z * block.z; k++)
		    {
			int img = k / base;
		      if (hIntensity[k] > maxval[img])
			{
			  maxval[img] = hIntensity[k];
			  maxint[img] = hIntensity2[k];
			  maxidx[img] = k % base;
			  maxr[img] = r;
			  maxdx[img] = dx;
			  maxdy[img] = dy;
			}
		      if (hIntensity[k] < minval[img]) minval[img] = hIntensity[k];
		    }
		}
	    }
	  fprintf(stderr, "%d r: %d\n", img, r);
	}
	  for (int subimg = 0; subimg < imgcount; subimg++)
	    {
        int maxR = /*maxidx / grid.y / block.y / grid.x / block.x*/ maxr[subimg];
	int maxX = maxidx[subimg] % block.x;
	int maxY = (maxidx[subimg] / (grid.x * block.x)) % block.y;
	int maxI = (maxidx[subimg] / (block.x * grid.x * block.y)) % grid.y;
	int maxI2 = (maxidx[subimg] / block.x) % grid.x;
	printf("%d %d %lf %g %g %g %d %d %d %d %g %d %d %d %d\n", img + subimg, psum[subimg], lsum[subimg], minval[subimg], maxval[subimg], hIntensity[subimg * base], maxR, maxX, maxY, cudaGetLastError(), maxint[subimg], maxI, maxI2, maxdx[subimg], maxdy[subimg]); 
	    }
      fflush(stdout);





/*      dim3 unigrid(1,1,1);
      dim3 uniblock(1,1,1);
      getExpLambdas<<<unigrid, uniblock>>>(dExpLambdas.data().get(), dLogLs.data().get(), maxidx, spherer, psum, lsum, block, grid);
      expLambdaVals.assign(dExpLambdas.begin(), dExpLambdas.end());*/
//      logLsVals.assign(dLogLs.begin(), dLogLs.end());
//      expectedLambdas.write(expLambdaVals.data(), H5::PredType::NATIVE_FLOAT, memSpace, expectedLambdaSpace);
//      logLs.write(logLsVals.data(), H5::PredType::NATIVE_FLOAT, memSpace, expectedLambdaSpace);
      //expectedLambdaFile.flush(H5F_SCOPE_LOCAL);
      //hIntensity2.assign(dIntensity2.begin(), dIntensity2.end());

    }
}
