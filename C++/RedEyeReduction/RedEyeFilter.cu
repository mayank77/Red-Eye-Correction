//-----------------------------------------------------------------------------
// File: RedEyeReduction.cu
//
// 1. First we have to find out how likely a pixel belongs to a red eye. To achieve that
//    we need to create a score for each pixel comparing image to a red-eye template.
// 2. Next, we need to sort pixels in ascending order which will allow us to identify pixels
//	  to alter. Each score is associated with a position; when we sort scores we must move
//    the positions accordingly.
//
// 
//
//-----------------------------------------------------------------------------

#include "stdafx.h"
#include "Utilities.h"

using namespace Bisque;

using std::vector;

using thrust::device_vector;
using thrust::host_vector;
using thrust::tuple;
using thrust::unary_function;
using thrust::make_tuple;
using thrust::make_zip_iterator;
using thrust::raw_pointer_cast;

// assert is only supported for compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef assert
#define assert(arg)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

/*
// Print kernel
__global__ void printfKernel(float f)
{
	float data = f * threadIdx.x;
	printf("Thread %d, f = %f.\n", threadIdx.x, data);
}

// Assert Kernel
__global__ void assertKernel()
{
	int is_one = 1;
	int should_be_one = 0;

	// This will have no effect
	assert(is_one);
	//printf("Thread %d.\n", threadIdx.x);

	// This will halt kernel execution
	assert(should_be_one);
}
*/

// struct RedEyeData: contains references to data on the GPU
struct RedEyeData
{
	device_vector<uchar4>			image;				// original picture copied to the device
	device_vector<unsigned char>	red;				// red channel
	device_vector<unsigned char>	green;				// green channel
	device_vector<unsigned char>	blue;				// blue channel
	int								rows;				// number of rows
	int								cols;				// number of cols
	int								pixels;				// number of pixels
	int								halfWidth;			// center point in x dimension
	int								halfHeight;			// center point in y dimension
};

// struct Mean: stores mean values for each channel
struct RedEyeMean
{
	float red;
	float green;
	float blue;
};

// Forward declarations: executed on the host
void RedEye_ComputeCrossCorrelation	(device_vector<float> & combined, const RedEyeData& d_image, const RedEyeData& d_template, const RedEyeMean& d_mean);
void RedEye_ComputeTemplateMean		(RedEyeMean& d_mean, const RedEyeData& d_template);
void RedEye_CpuRadixSort			(host_vector<unsigned int>& sortedCoord, const device_vector<float>& correlatedPixels, const device_vector<unsigned int>& originalCoords, int numPixels);
void RedEye_RemoveRedness			(host_vector<uchar4>&	modifiedImage, RedEyeData& d_image, const device_vector<unsigned int>& coords);
void RedEye_SeparateChannels		(RedEyeData& d_image, RedEyeData& d_template, const vector<uchar4>&	image, const vector<uchar4>& redEyeTemplate);
void RedEye_SortCoordinates			(device_vector<unsigned int>& sortedCoords, device_vector<float>& correlatedPixels, int numPixels);

// Kernels
__global__ void redEyeKernel_normalized_cross_correlation(
	float*					result,					// return value: normalized color channel values 
	const unsigned char* 	image,					// input channel from the original image
	const unsigned char*	redEyeTemplate,			// input channel from the red eye template
	int						imageRows,				// image rows
	int						imageCols,				// image cols
	int						templateCols,			// red eye template cols
	int						templateHalfHeight,		// red eye template center in x dimaension
	int						templateHalfWidth,		// red eye template center in y dimension
	float					templatePixelsInverse,	// red eye template inverse of the number of pixels
	float					templateMean			// red eye template mean value (for a channel: rgb)
	);

__device__ void redEyeKernel_partition_by_bit(
	unsigned int*			pixels, 
	unsigned int			bit
	);

template<typename T> 
__device__ T redEyeKernel_plus_scan(T* x);

__global__ void redEyeKernel_remove_redness(
	unsigned char*			red,					// return value: red channel after redness is removed
	const unsigned char*	green,					// green channel
	const unsigned char*	blue,					// blue channel
	const unsigned int*		coordinates,			// red eye pixel coordinates
	int						numCoords,				// number of coordinates to scan for replacing red effect, usually a small box
	int						numPixels,				// number of pixels in the image
	int						rows,					// image rows
	int						cols,					// image cols
	int						templateHalfHeight,		// red eye template center in x dimaension
	int						templateHalfWidth		// red eye template center in y dimension
	);

__global__ void redEyeKernel_sort_correlated_coordinates(unsigned int* keys);

//
// Helper structures to split/combine images
//

// struct SplitChannels
struct SplitChannels : unary_function< uchar4, tuple<unsigned char, unsigned char, unsigned char> >
{
	__host__ __device__
	tuple<unsigned char, unsigned char, unsigned char> operator() (uchar4 pixel)
	{
		return make_tuple(pixel.x, pixel.y, pixel.z);
	}
};

// struct CombineChannels
struct CombineChannels : unary_function< tuple<unsigned char, unsigned char, unsigned char>, uchar4 >
{
	__host__ __device__
	uchar4 operator() (tuple<unsigned char, unsigned char, unsigned char> t)
	{
		return make_uchar4(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t), 255);
	}
};

// struct CombineResponses
struct CombineResponses : unary_function< float, tuple<float, float, float> >
{
	__host__ __device__
	float operator() (tuple<float, float, float> t)
	{
		return thrust::get<0>(t) * thrust::get<1>(t) * thrust::get<2>(t);
	}
};

//
// Kernels
//

// Cross-correlates original image with red-eye templates and normalizes results.
// Does it for each individual channel: red, green, and blue.
// In signal processing, cross-correlation is a measure of similarity of two waveforms as a function of a time-lag applied to one of them. 
// This is also known as a sliding dot product or sliding inner-product. It is commonly used for searching a long-signal for a shorter, known feature. 
// It also has applications in pattern recognition, single particle analysis, electron tomographic averaging, cryptanalysis, and neurophysiology.
// This is a 'naive', simplified algorithm.
__global__
void redEyeKernel_normalized_cross_correlation(
	float*					result,					// return value: normalized color channel values 
	const unsigned char* 	image,					// input channel from the original image
	const unsigned char*	redEyeTemplate,			// input channel from the red eye template
	int						imageRows,				// image rows
	int						imageCols,				// image cols
	int						templateCols,			// red eye template cols
	int						templateHalfHeight,		// red eye template center in x dimaension
	int						templateHalfWidth,		// red eye template center in y dimension
	float					templatePixelsInverse,	// red eye template inverse of the number of pixels
	float					templateMean			// red eye template mean value (for a channel: rgb)
	)
{
	int col	=  blockIdx.x * blockDim.x + threadIdx.x;		// current column
	int row	=  blockIdx.y * blockDim.y + threadIdx.y;		// current row

	if ((row >= imageRows) || (col >= imageCols))
	{
		return;
	}

	int idx	= col + imageCols * row;									// current pixel index

	//
	// Compute image mean the size of red-eye template box, 
	// do it for each pixel similar to gaussian blur algorithm.
	//
	float colorSum = 0;

	// To align memory, we step row by row and iterate over all columns
	for (int y = -templateHalfHeight; y <= templateHalfHeight; ++y)
	{
		for (int x = -templateHalfWidth; x <= templateHalfWidth; ++x)
		{
			int2 offsetIdx = make_int2(col + x, row + y);
			int2 clampIdx  = make_int2( min( imageCols - 1, max( 0, offsetIdx.x ) ), min( imageRows - 1, max( 0, offsetIdx.y ) ) );				// clamp index
			int  pixel     = clampIdx.x + ( imageCols * clampIdx.y );																			// 1d offset

			float value = static_cast<float>( image[pixel] );
			colorSum += value;
		}
	}

	// This mean is an average of sum of colors (per RGB channel) around the image pixel divided by total number of pixels in a red-eye template
	float imageMean = colorSum * templatePixelsInverse;

	//
	// Compute sums in order to match signals
	//
	float crossSum			= 0;
	float imageSumDiff		= 0;
	float templateSumDiff	= 0;

	for (int y = -templateHalfHeight; y <= templateHalfHeight; ++y)
	{
		for (int x = -templateHalfWidth; x <= templateHalfWidth; ++x)
		{
			// image color offset from mean
			int2 offsetIdx			= make_int2( col + x, row + y );
			int2 clampIdx			= make_int2( min( imageCols - 1, max( 0, offsetIdx.x ) ), min( imageRows - 1, max( 0, offsetIdx.y ) ) );	// clamp index
			int  pixel				= clampIdx.x + ( imageCols * clampIdx.y );																	// 1d offset

			float imageColor		= static_cast<float>( image[pixel] );
			float imageDiff			= imageColor - imageMean;

			// template color offset from mean
			offsetIdx				= make_int2( templateHalfWidth + x, templateHalfHeight + y );
			pixel					= offsetIdx.x + ( templateCols * offsetIdx.y );

			float templateColor		= static_cast<float>(redEyeTemplate[pixel]);
			float templateDiff		= templateColor - templateMean;

			// Match signals
			float imageSquare		= imageDiff		* imageDiff;
			float templateSquare	= templateDiff	* templateDiff;
			float crossProduct		= imageColor	* templateDiff;

			crossSum			   += crossProduct;
			imageSumDiff		   += imageSquare;
			templateSumDiff		   += templateSquare;
		}
	}

	//
	// Compute final result
	//
	float finalResult = 0;

	if (imageSumDiff != 0 && templateSumDiff != 0)
	{
		finalResult = crossSum / sqrt(imageSumDiff * templateSumDiff);
	}

	result[idx] = finalResult;
}

// Plus scan
//
// Replaces input array by the prefix sums of the elements in it. The prefix sum is the sum
// of all elements up to and including that element. The sum operation can be replaced by 
// any binary associative operator, for example multiplication.
//
// The function returns a new value that will replace element in the array at that position.
// 
// Example:
//                 A =  3  1  7  0  4  1  6  3 
//   
// Successive iterations yield
//     offset = 1  A =  3  4  8  7  4  5  7  9
//     offset = 2  A =  3  4 11 11 12 12 11 14
//     offset = 4  A =  3  4 11 11 15 16 22 25
//   
// When it is finished it will have taken log N steps and used N log N adds.
// This means that it is not work-efficient, since the sequential algorithm uses N adds.
//
template<typename T>
__device__
T redEyeKernel_plus_scan(T* x)
{
	unsigned int i = threadIdx.x;		// id of thread executing this instance
	unsigned int n = blockDim.x;		// total number of threads in this block

	for (unsigned int offset = 1; offset < n; offset *= 2)			// same as <<= 1
	{
		T t;

		if (i >= offset)
		{
			t = x[i - offset];
		}

		__syncthreads();

		// 
		if (i >= offset)
		{
			x[i] = t + x[i];			// i.e. x[i] = x[i] + x[i - 1]
		}

		__syncthreads();
	}

	return x[i];
}

// Helper function for radix sort
//
__device__ 
void redEyeKernel_partition_by_bit(unsigned int* values, unsigned int bit)
{
	unsigned int i = threadIdx.x;
	unsigned int n = blockDim.x;
	unsigned int x = values[i];					// decimal number, we store it so that we could find a new place for it and put it back into the array at a new position
	unsigned int b = (x >> bit) & 1;			// binary number
	
	// Replace each x with a 0 or 1 bit
	// Remember that we are looping from 0 to 31 which is the max number of digits in an integer.
	// 0 or 1 represent binary value for the whole number at the position in the loop.
	//
	// Note that we do that in order to sort values in the array. We will replace the binary value
	// back with the decimal that we cache temporarily before we exit the function.

	values[i] = b;
	__syncthreads();
	
	// Entire array now consists of 0's and 1's.
	// values[i] = 0 if the bit at position bit in the value was 0 and 1 othrewise.

	// Compute number of true bits (1-bits) up to and including values[i],
	// transforming values[] so that values[idx] contains the sum of the 1-bits
	// from values[0] .. values[idx]

	// plus_scan(values) calculates a histogram and returns the total number of '1' bits
	// for all values where value is in the array with the index less then i. 
	// The return value is assigned to 'sum'.

	unsigned int sum = redEyeKernel_plus_scan(values);

	// The plus_scan function does not return here until all threads have reached the
	// __syncthreads call in the last iteration of its loop. Therefore, when it does return,
	// we know that the entire array has had the prefix sums computed, and that values[size-1]
	// is the sum of all elements in the array, which happens to be the number of 1-bits
	// in the current bit position.

	// oneBits after the scan is the total number of 1-bits in the entire array.
	// zeroBits is the sum of all 0 bits

	unsigned int oneBits  = values[n - 1];
	unsigned int zeroBits = n - oneBits;

	__syncthreads();

	// No we can put the cached decimal value back into the array in the right location
	// satisfying the condition that all values with 0 in the current bit position precede
	// all those with 1. The resulting array must be stable which means that previous sort order
	// must not be changed. For example, if I have a UI with 2 columns: last name and first name - 
	// and I sort by last name, then sort by first, I want to see a list sorted by last name and
	// then first names sortd in ascending order without breaking last name order.

	if (b == 1)
	{
		values[sum - 1 + zeroBits] = x;
	}
	else
	{
		values[i - sum] = x;
	}
}

// Takes in correlated and sorted coordinates of the red color and replaces them with the
// averaged blue and green channel color. We can replace red with any other color instead...
// We only iterate on a small subset of colors in the image, less then 40 in this case.
// That is, we only use 40 threads to replace 40 values!
//
__global__ 
void redEyeKernel_remove_redness(
	unsigned char*			red,					// return value: red channel after redness is removed
	const unsigned char*	green,					// green channel
	const unsigned char*	blue,					// blue channel
	const unsigned int*		coordinates,			// red eye pixel coordinates
	int						numCoords,				// number of coordinates to scan for replacing red effect, usually a small box
	int						numPixels,				// number of pixels in the image
	int						rows,					// image rows
	int						cols,					// image cols
	int						templateHalfHeight,		// red eye template center in x dimaension
	int						templateHalfWidth		// red eye template center in y dimension
    )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= numCoords)
	{
		return;
	}

	// Each thread grabs a coordinate to replace
	unsigned int idx = coordinates[numPixels - tid - 1];
	ushort2		 i2d = make_ushort2(idx % cols, idx / cols);
  
	for (int y = i2d.y - templateHalfHeight; y <= i2d.y + templateHalfHeight; ++y)
	{
		for (int x = i2d.x - templateHalfWidth; x <= i2d.x + templateHalfWidth; ++x)
		{
			int2 offset		= make_int2(x, y);
			int2 clamped2d	= make_int2( min( cols - 1, max( 0, offset.x ) ), min( rows - 1, max( 0, offset.y ) ) );
			int	 clampedIdx	= cols * clamped2d.y + clamped2d.x;

			unsigned char g = green[clampedIdx];
			unsigned char b = blue[clampedIdx];

			unsigned int avg = (g + b) / 2;

			red[clampedIdx] = (unsigned char)avg;
		}
	}
}

// Radix sort on the correlatedPixels.
// Input is 1-dimensional array. We are interested in coordinates of the pixels
//
// Radix sort has a fixed number of iterations and a consistent execution flow. 
// It works by sorting based on the least significant bit and then working up to the most significant bit. 
// With a 32-bit integer, using a single radix bit, you will have 32 iterations of the sort, no matter how large the dataset. 
// Let’s consider an example with the following dataset:
//		{ 122, 10, 2, 1, 2, 22, 12, 9 }
// The binary representation of each of these would be
//		122 = 01111010
//		 10 = 00001010
//		 2 = 00000010
//		 22 = 00010010
//		 12 = 00001100
//		 9 = 00001001
// In the first pass of the list, all elements with a 0 in the least significant bit (the right side) would form the first list. 
// Those with a 1 as the least significant bit would form the second list. Thus, the two lists are
//		0 = { 122, 10, 2, 22, 12 }
//		1 = { 9 }
// The two lists are appended in this order, becoming
//		{ 122, 10, 2, 22, 12, 9 }
// The process is then repeated for bit one, generating the next two lists based on the ordering of the previous cycle:
//		0 = { 12, 9 }
//		1 = { 122, 10, 2, 22 }
//
// After each partitioning loop we have to aquire a barrier to allow all threads operating on the same bit to complete.
//
__global__ 
void redEyeKernel_sort_correlated_coordinates(unsigned int*	keys)
{
	for (int bit = 0; bit < 32; ++bit)
	{
		redEyeKernel_partition_by_bit(keys, bit);
		__syncthreads();
	}
}

//
// Main
//

// Entry function into the red-eye kernels
// This is Main for this file
void RedEyeFilter_Main(
	host_vector<uchar4>&	modifiedImage,
	const vector<uchar4>&	image, 
	int						imageRows,
	int						imageCols,
	const vector<uchar4>&	redEyeTemplate,
	int						redEyeRows,
	int						redEyeCols
	)
{
	RedEyeData d_image;				// original image
	RedEyeData d_template;			// red eye template
	RedEyeMean d_mean;				// template mean values for each channel (sum of color values / number of pixels)

	// d_image and d_template are global variables used in this cu file
	d_image		.rows		= imageRows;
	d_image		.cols		= imageCols;
	d_image		.pixels		= imageRows * imageCols;

	d_template	.rows		= redEyeRows;
	d_template	.cols		= redEyeCols;
	d_template	.pixels		= redEyeRows * redEyeCols;
	d_template	.halfWidth	= (redEyeCols - 1) / 2;
	d_template	.halfHeight	= (redEyeRows - 1) / 2;

	// Separate channels
	RedEye_SeparateChannels(d_image, d_template, image, redEyeTemplate);

	// Compute mean for each channel
	RedEye_ComputeTemplateMean(d_mean, d_template);

	// Cross-correlate original image with red-eye template and normalize result
	device_vector<float> correlatedPixels(d_image.pixels);

	RedEye_ComputeCrossCorrelation(correlatedPixels, d_image, d_template, d_mean);

	// Create 1D coordinates that will be attached to the keys
	device_vector<unsigned int> sortedCoords(d_image.pixels);
	thrust::sequence( sortedCoords.begin(), sortedCoords.end() );				// [0, ..., numPixels - 1]

	// Sort pixels in assending order. That will tell us which pixels must be replaced
	RedEye_SortCoordinates(sortedCoords, correlatedPixels, d_image.pixels);

	// Remove redness
	RedEye_RemoveRedness(modifiedImage, d_image, sortedCoords);

	// Clean-up GPU resources
	// We'd need to do that if the vectors were global.
	//d_image.blue .clear();
	//d_image.green.clear();
	//d_image.red	 .clear();

	//d_image.blue .shrink_to_fit();
	//d_image.green.shrink_to_fit();
	//d_image.red	 .shrink_to_fit();
}

//
// Utility functions for red-eye reduction
//

// Removes red-eye pixels - calls a kernel to do that
void RedEye_RemoveRedness(
	host_vector<uchar4>&				modifiedImage, 
	RedEyeData&							d_image, 
	const device_vector<unsigned int>&	coords)
{
	const char* fn = "RedEye_RemoveRedness";
	OutputDebugString(L"   Remove redness\n");

	cudaError ce = cudaSuccess;

	// red channel after applying the filter, copy data from the original image
	// kernel will replace red pixels only
	device_vector<unsigned char> red = d_image.red;

	// We only need to replace color in the eyes, so only a small number of threads is required
#define RED_EYE_FILTER_MAX_THREADS				40
#define RED_EYE_FILTER_TEMPLATE_HALF_HEIGHT		9
#define RED_EYE_FILTER_TEMPLATE_HALF_WIDTH		9

	const dim3 block(256);
	const dim3 grid( static_cast<int>( ceilf( static_cast<float>(RED_EYE_FILTER_MAX_THREADS) / static_cast<float>(block.x) )));
	
	redEyeKernel_remove_redness<<<grid, block>>>(
		raw_pointer_cast(red.data()),
		raw_pointer_cast(d_image.green.data()),
		raw_pointer_cast(d_image.blue.data()),
		raw_pointer_cast(coords.data()),
		RED_EYE_FILTER_MAX_THREADS,
		d_image.pixels,
		d_image.rows,
		d_image.cols,
		RED_EYE_FILTER_TEMPLATE_HALF_HEIGHT, 
		RED_EYE_FILTER_TEMPLATE_HALF_WIDTH
		);
	
	ce = cudaDeviceSynchronize();																	CHECK_CUDA_ERROR(ce, fn, "Remove red-eye kernel call failed.");
  
	// Combine new red channel with original blue and green
	device_vector<uchar4> d_combined(d_image.pixels);

	thrust::transform(
		make_zip_iterator( make_tuple(red.begin(), d_image.green.begin(), d_image.blue.begin()) ),
		make_zip_iterator( make_tuple(red.end(), d_image.green.end(), d_image.blue.end()) ),
		d_combined.begin(),
		CombineChannels()
		);

	// Copy back to the CPU
	modifiedImage = d_combined;

#if 0
	host_vector<unsigned int> r(red.begin(), red.end());
	host_vector<unsigned int> orig(d_image.red.begin(), d_image.red.end());

	wchar_t text[MAX_PATH];

	for (int i = 0; i < 1100; ++i)
	{
		swprintf_s(text, L"   i=%d %d, %d\n", i, orig[i], r[i]);
		OutputDebugString(text);
	}
#endif
}

// Splits original picture into channels, copies images and channels to the device.
void RedEye_SeparateChannels(
	RedEyeData&				d_image,
	RedEyeData&				d_template,
	const vector<uchar4>&	image, 
	const vector<uchar4>&	redEyeTemplate
	)
{
	OutputDebugString(L"   Separate channels\n");

	// Copy image and red-eye template to the device
	d_image.image.resize(d_image.pixels);
	thrust::copy(image.begin(), image.end(), d_image.image.begin());

	d_template.image.resize(d_template.pixels);
	thrust::copy(redEyeTemplate.begin(), redEyeTemplate.end(), d_template.image.begin());

	// Separate image into channels
	d_image.red		.resize(d_image.pixels);
	d_image.green	.resize(d_image.pixels);
	d_image.blue	.resize(d_image.pixels);

	thrust::transform(
		d_image.image.begin(),
		d_image.image.end(),
		make_zip_iterator( make_tuple( d_image.red.begin(), d_image.green.begin(), d_image.blue.begin() ) ),
		SplitChannels()
		);

	// transform template
	d_template.red	.resize(d_template.pixels);
	d_template.green.resize(d_template.pixels);
	d_template.blue	.resize(d_template.pixels);

	thrust::transform(
		d_template.image.begin(),
		d_template.image.end(),
		make_zip_iterator(make_tuple(d_template.red.begin(), d_template.green.begin(), d_template.blue.begin())),
		SplitChannels()
		);

#if 0
	using thrust::host_vector;

	host_vector<uchar4> tmp(d_image.image.begin(), d_image.image.begin() + d_image.pixels);
	host_vector<unsigned int> r(d_image.red.begin(), d_image.red.begin() + d_image.pixels);
	host_vector<unsigned int> g(d_image.green.begin(), d_image.green.begin() + d_image.pixels);
	host_vector<unsigned int> b(d_image.blue.begin(), d_image.blue.begin() + d_image.pixels);
	//host_vector<uchar4> tmp(d_template.image.begin(), d_template.image.begin() + d_template.pixels);

	for (int i = 0; i < 100; ++i)
	{
		wchar_t text[MAX_PATH];
		swprintf_s(text, L"   %d: x=%d, y=%d, z=%d w=%d --- r=%d, g=%d, b=%d\n", i, tmp[i].x, tmp[i].y, tmp[i].z, tmp[i].w, r[i], g[i], b[i]);
		OutputDebugString(text);
	}
#endif

#if 0
	int r = 0;
	int g = 0;
	int b = 0;

	using thrust::host_vector;
	host_vector<unsigned int> red  (d_template.red.begin(),   d_template.red.begin()   + d_template.pixels);
	host_vector<unsigned int> green(d_template.green.begin(), d_template.green.begin() + d_template.pixels);
	host_vector<unsigned int> blue (d_template.blue.begin(),  d_template.blue.begin()  + d_template.pixels);

	for (int i = 0; i < d_template.pixels; ++i)
	{
		//r += redEyeTemplate[i].x;
		//g += redEyeTemplate[i].y;
		//b += redEyeTemplate[i].z;
		r += red[i];
		g += green[i];
		b += blue[i];
	}

	// Sum:  r=195523, g=121098, b=125712
	wchar_t text[MAX_PATH];
	swprintf_s(text, L"   Sum:  r=%d, g=%d, b=%d\n", r, g, b);
	OutputDebugString(text);
#endif
}

// Computes mean for each channel of the red eye template image
void RedEye_ComputeTemplateMean(RedEyeMean& d_mean, const RedEyeData& d_template)
{
	OutputDebugString(L"   Compute template mean\n");

	// Compute sum of all pixel values for each channel
	int red   = thrust::reduce(d_template.red.begin(),   d_template.red.end(),   (int)0, thrust::plus<int>());
	int green = thrust::reduce(d_template.green.begin(), d_template.green.end(), (int)0, thrust::plus<int>());
	int blue  = thrust::reduce(d_template.blue.begin(),  d_template.blue.end(),  (int)0, thrust::plus<int>());

	// Mean
	double d = static_cast<double>(d_template.pixels);

	d_mean.red   = static_cast<float>(static_cast<double>(red)   / d);
	d_mean.green = static_cast<float>(static_cast<double>(green) / d);
	d_mean.blue  = static_cast<float>(static_cast<double>(blue)  / d);

#if 0
	// Sum:  r=195523, g=121098, b=125712
	// Mean: r=179.543625, g=111.201103, b=115.438019
	wchar_t text[MAX_PATH];

	swprintf_s(text, L"  Sum:  r=%d, g=%d, b=%d\n", red, green, blue);
	OutputDebugString(text);
	swprintf_s(text, L"  Mean: r=%f, g=%f, b=%f\n", d_mean.red, d_mean.green, d_mean.blue);
	OutputDebugString(text);
#endif
}

// Cross-correlates original image with red-eye template and normalizes results
// In signal processing, cross-correlation is a measure of similarity of two waveforms as a function of a time-lag applied to one of them. 
// This is also known as a sliding dot product or sliding inner-product. It is commonly used for searching a long-signal for a shorter, known feature. 
// It also has applications in pattern recognition, single particle analysis, electron tomographic averaging, cryptanalysis, and neurophysiology.
void RedEye_ComputeCrossCorrelation(
	device_vector<float> &	combined, 
	const RedEyeData&		d_image, 
	const RedEyeData&		d_template, 
	const RedEyeMean&		d_mean
	)
{
	const char* fn = "RedEye_ComputeCrossCorrelation";
	OutputDebugString(L"   Compute cross correlation\n");

	cudaError ce = cudaSuccess;

	device_vector<float> red  (d_image.pixels);
	device_vector<float> green(d_image.pixels);
	device_vector<float> blue (d_image.pixels);

	// 32 x 32 = 1024 threads per block, max on my Quadro 5000M
	// However, if I set max block size it breaks the kernel, must be due to register pressure...
	const dim3 block(16, 16, 1);
	const dim3 grid(
		static_cast<int>(ceilf( static_cast<float>(d_image.cols) / static_cast<float>(block.x) )),
		static_cast<int>(ceilf( static_cast<float>(d_image.rows) / static_cast<float>(block.y) ))
		);

	static const float templatePixelsInverse = 1.f / static_cast<float>(d_template.pixels);

	// red
	redEyeKernel_normalized_cross_correlation<<<grid, block>>>(
		raw_pointer_cast(red.data()),							// return value: normalized color channel values 
		raw_pointer_cast(d_image.red.data()),					// input channel from the original image
		raw_pointer_cast(d_template.red.data()),				// input channel from the red eye template
		d_image.rows,											// image rows
		d_image.cols,											// image cols
		d_template.cols,										// template cols
		d_template.halfHeight,									// template center in x dimaension
		d_template.halfWidth,									// template center in y dimension
		templatePixelsInverse,									// inverse number of template pixels
		d_mean.red												// template mean value
		);

	ce = cudaDeviceSynchronize();																						CHECK_CUDA_ERROR(ce, fn, "Normalized cross correlation failed.");

	// green
	redEyeKernel_normalized_cross_correlation<<<grid, block>>>(
		raw_pointer_cast(green.data()),							// return value: normalized color channel values 
		raw_pointer_cast(d_image.green.data()),					// input channel from the original image
		raw_pointer_cast(d_template.green.data()),				// input channel from the red eye template
		d_image.rows,											// image rows
		d_image.cols,											// image cols
		d_template.cols,										// template cols
		d_template.halfHeight,									// template center in x dimaension
		d_template.halfWidth,									// template center in y dimension
		templatePixelsInverse,									// inverse number of template pixels
		d_mean.green											// template mean value
		);

	ce = cudaDeviceSynchronize();																						CHECK_CUDA_ERROR(ce, fn, "Normalized cross correlation failed.");

	// blue
	redEyeKernel_normalized_cross_correlation<<<grid, block>>>(
		raw_pointer_cast(blue.data()),							// return value: normalized color channel values 
		raw_pointer_cast(d_image.blue.data()),					// input channel from the original image
		raw_pointer_cast(d_template.blue.data()),				// input channel from the red eye template
		d_image.rows,											// image rows
		d_image.cols,											// image cols
		d_template.cols,										// template cols
		d_template.halfHeight,									// template center in x dimaension
		d_template.halfWidth,									// template center in y dimension
		templatePixelsInverse,									// inverse number of template pixels
		d_mean.blue												// template mean value
		);

	ce = cudaDeviceSynchronize();																						CHECK_CUDA_ERROR(ce, fn, "Normalized cross correlation failed.");

	//
	// Combine channels by multiplying them together.
	//
	thrust::transform( make_zip_iterator( make_tuple( red.begin(), green.begin(), blue.begin() ) ),
		 			   make_zip_iterator( make_tuple( red.end(),   green.end(),   blue.end()   ) ),
		 			   combined.begin(),
		 			   CombineResponses()
					 );

	// Find min/max of the response
	typedef device_vector<float>::iterator it;
	thrust::pair<it, it> min_max = thrust::minmax_element(combined.begin(), combined.end());

	float bias = *min_max.first;

	// Make all numbers positive to allow sorting without bit twiddling
	thrust::transform(	combined.begin(), 
						combined.end(), 
						thrust::make_constant_iterator(-bias), 
						combined.begin(), 
						thrust::plus<float>() 
					);

#if 0
	using thrust::host_vector;
	host_vector<float> r(red.begin(),   red.begin()   + d_image.pixels);
	host_vector<float> g(green.begin(), green.begin() + d_image.pixels);
	host_vector<float> b(blue.begin(),  blue.begin()  + d_image.pixels);
	host_vector<float> c(combined.begin(),  combined.begin()  + d_image.pixels);

	wchar_t text[MAX_PATH];

	for (int i = 0; i < 1089; ++i)
	{
		swprintf_s(text, L"  i=%d; r=%f, g=%f, b=%f --- c=%f\n", i, r[i], g[i], b[i], c[i]);
		OutputDebugString(text);

	}

	swprintf_s(text, L"  .. bias = %f\n", bias);
	OutputDebugString(text);
#endif
}

// Sort pixels in ascending order to identify which need to be replaced.
//
// Function implements Parallel Radix Sort
//
// Basic idea is to construct a histogram on each pass of how many of each "digit" there are.
// Then we scan this histogram so that we know where to put the output of each digit.
// For example, the first 1 must come after all the 0s, so we have to know how many 0s there are 
// to be able to start moving 1s into the correct position.
//
// 1) Histogram of the number of occurrences of each digit
// 2) Exclusive Prefix Sum of Histogram
// 3) Determine relative offset of each digit
//    For example [0 0 1 1 0 0 1]
//			   -> [0 1 0 1 2 3 2]
//
// 4) Combine results of steps 2 & 3 to determine the final output location for each element
//    and move it there.
//
// LSB Radix sort is an out-of-place sort and you will need to ping-pong values between the input
// and output buffers. Final sorted results are placed into the output buffer.
//
void RedEye_SortCoordinates(device_vector<unsigned int>& sortedCoords, device_vector<float>& correlatedPixels, int numPixels)
{
	const char* fn = "RedEye_SortCoordinates";
	OutputDebugString(L"   Sort cross-correlated pixels\n");

	cudaError ce = cudaSuccess;

	using thrust::device_ptr;
	using thrust::raw_pointer_cast;
	using thrust::stable_sort_by_key;

#if 0
	// Test run on the CPU so that we could compare results of the GPU sort

	host_vector<unsigned int> h_sortedCoord(numPixels);
	RedEye_CpuRadixSort(h_sortedCoord, correlatedPixels, sortedCoords, numPixels);

	wchar_t text[MAX_PATH];

	//for (int i = 0; i < 1089; ++i)
	//{
	//	swprintf_s(text, L"  i=%d; %d\n", i, h_sortedCoord[i]);
	//	OutputDebugString(text);

	//}
#endif

	//
	// Sort pixels on the GPU
	//

	// First cast floats to unsigned int
	// will produce strange numbers like 1034332492
	unsigned int* tmp;
	ce = cudaMalloc(&tmp, sizeof(unsigned int) * numPixels);																				CHECK_CUDA_ERROR(ce, fn, "Failed to allocate memory on the device.");
	ce = cudaMemcpy(tmp, raw_pointer_cast(correlatedPixels.data()), sizeof(unsigned int) * numPixels, cudaMemcpyDeviceToDevice);			CHECK_CUDA_ERROR(ce, fn, "Failed to copy correlated pixels.");

	device_vector<unsigned int> d_keys(device_ptr<unsigned int>(tmp), device_ptr<unsigned int>(tmp) + numPixels);

	// Sort pixels using thrust
	stable_sort_by_key(d_keys.begin(), d_keys.end(), sortedCoords.begin());

#if 0

	host_vector<unsigned int> h_keys(d_keys.begin(), d_keys.end());
	host_vector<unsigned int> h_values(sortedCoords.begin(), sortedCoords.end());

	wchar_t text[MAX_PATH];

	for (int i = 0; i < 1089; ++i)
	{
		swprintf_s(text, L"  i=%d %d %d\n", i, h_keys[i], h_values[i]);
		OutputDebugString(text);
	}
#endif

	// * * * * * * *
	//
	// NOTE 
	// 
	// Code below is commented and not used, it is to try out my own algorithm and to experiment with radix sort
	//
	// An implementation of the radix sort is available in the Thrust library shipped with v4.0 onwards of the CUDA SDK so you don’t have to implement your own radix sort.
	// We'll do it anyway..
	// http://back40computing.googlecode.com/svn/wiki/documents/PplGpuSortingPreprint.pdf
	// http://mgarland.org/files/papers/nvr-2008-001.pdf
	// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
	//

#if 0
	// This is a copy of the original pixel values before kernel call for testing
	host_vector<unsigned int> p1(device_ptr<unsigned int>(d_pixel), device_ptr<unsigned int>(d_pixel) + numPixels);
	unsigned int dec = p1[0];

    // 1040146228 -> 00111101111111110101111100110100
	for (int bit = 0; bit < 32; ++bit)
	{
		unsigned int binary = (dec >> bit) & 1;			// binary number
		swprintf_s(text, L"  i=%d; %d %d\n", bit, dec, binary);
		OutputDebugString(text);
	}
#endif

	// Following radix sort algorithm only works for one block.
	// Needs more time to exapand it to work with many...
	/*
	const char* fn = "RedEye_SortCoordinates";

	std::list<unsigned int> h_k;
	h_k.push_back(10);
	h_k.push_back(9);
	h_k.push_back(8);
	h_k.push_back(7);
	h_k.push_back(6);
	h_k.push_back(5);
	h_k.push_back(4);
	h_k.push_back(3);
	h_k.push_back(2);
	h_k.push_back(1);
	h_k.push_back(0);

	device_vector<unsigned int> d_k(h_k.begin(), h_k.end());

	dim3 block(h_k.size());
	const dim3 grid(
		static_cast<int>(ceilf( static_cast<float>(h_k.size()) / static_cast<float>(block.x) ))
		);

	redEyeKernel_sort_correlated_coordinates<<<grid, block, block.x * sizeof(unsigned int)>>>(
		raw_pointer_cast(d_k.data()),
		numPixels
		);
	
	ce = cudaDeviceSynchronize();																						CHECK_CUDA_ERROR(ce, fn, "RedEye_SortCoordinates failed.");
	*/
#if 0
	host_vector<unsigned int>	  pk(d_k.begin(),  d_k.end());

	for (int i = 0; i < h_k.size(); ++i)
	{
		swprintf_s(text, L"  i=%d %d\n", i, pk[i]);
		OutputDebugString(text);
	}
#endif	
}

// Reference sort executed on the CPU for correctness check
// 
// This is a simple non-optimized calcuation that only deals with power-of-2 radices.
//
void RedEye_CpuRadixSort(
	host_vector<unsigned int>&			sortedCoord,			// output: returns sorted coordinates of pixels
	const device_vector<float>&			correlatedPixels,		// Normalized and correlated between red-eye template and a picture pixels
	const device_vector<unsigned int>&	originalCoords,					// Coordinates of the pixels
	int									numPixels				// Number of pixels
	)
{
	const char* fn = "RedEye_CpuRadixSort";
	OutputDebugString(L"   Radix sort on the CPU\n");

	cudaError_t ce = cudaSuccess;

	using thrust::device_ptr;
	using thrust::raw_pointer_cast;

	//
	// Copy data to the host
	//
	// We do not care here about the pixel values, we only care that their relative values do not change and 
	// we do care about pixel positions. For that reason, we copy bits from floats into integers 
	// resulting in wrong pixel values but preserving their order.
	//
	host_vector<float>			c1	(correlatedPixels.begin(), correlatedPixels.begin() + numPixels);
	host_vector<unsigned int>	c2	(correlatedPixels.begin(), correlatedPixels.begin() + numPixels);		// will produce all 0's
	host_vector<unsigned int>	pos (originalCoords.begin(), originalCoords.begin() + numPixels);

	// First cast floats to unsigned int
	// will produce strange numbers like 1034332492
	unsigned int* d_color;
	ce = cudaMalloc(&d_color, sizeof(unsigned int) * numPixels);																			CHECK_CUDA_ERROR(ce, fn, "Failed to allocate memory on the device.");
	ce = cudaMemcpy(d_color, raw_pointer_cast(correlatedPixels.data()), sizeof(unsigned int) * numPixels, cudaMemcpyDeviceToDevice);		CHECK_CUDA_ERROR(ce, fn, "Failed to copy pixels.");

	host_vector<unsigned int> color(device_ptr<unsigned int>(d_color), device_ptr<unsigned int>(d_color) + numPixels);

#if 0
	wchar_t text[MAX_PATH];

	for (int i = 0; i < 1089; ++i)
	{
		swprintf_s(text, L"  i=%d; c=%f : %d : %d --- p=%d\n", i, c1[i], c2[i], color[i], pos[i]);
		OutputDebugString(text);

	}
#endif

	const int numBits = 1;
	const int numBins = 1 << numBits;		// Same as 1 * 2

	//using std::vector;
	vector<unsigned int> binHistogram(numBins);
	vector<unsigned int> binScan(numBins);

	unsigned int* srcValue = color.data();
	unsigned int* srcCoord = pos.data();

	// This buffer will store sorted pixel colors.
	// We do not care of them, we only care about corresponding coordinates, so we can discard this buffer.
	host_vector<unsigned int> sortedValue(numPixels);
	
	unsigned int* dstValue = sortedValue.data();
	unsigned int* dstCoord = sortedCoord.data();

	// A simple radix sort, only guaranteed to work for numBits that are multiples of 2
	for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits)
	{
		// Keep multiplying by 2
		unsigned int mask = (numBins - 1) << i;

		// Zero out both bins
		binHistogram[0] = 144U;
		memset(binHistogram.data(), 0, sizeof(unsigned int) * numBins);
		memset(binScan.data(), 0, sizeof(unsigned int) * numBins);

		// Perform histogram of data & mask into bins
		for (int j = 0; j < numPixels; ++j)
		{
			unsigned int bin = (srcValue[j] & mask) >> i;
			binHistogram[bin]++;
		}

		// Perform exclusive prefix sum (scan) on binHistogram to get 
		// starting location for each bin
		for (unsigned int j = 1; j < numBins; ++j)
		{
			binScan[j] = binScan[j - 1] + binHistogram[j - 1];
		}

		// Gather everything into correct locations
		for (int j = 0; j < numPixels; ++j)
		{
			unsigned int bin = (srcValue[j] & mask) >> i;
			unsigned int idx = binScan[bin];

			dstValue[idx] = srcValue[j];
			dstCoord[idx] = srcCoord[j];

			binScan[bin]++;
		}

		// Swap the buffers
		using std::swap;

		swap(dstValue, srcValue);
		swap(dstCoord, srcCoord);
	}

	// Copy into output buffer
	std::copy(color.begin(), color.end(), sortedValue.begin());
	std::copy(pos.begin(),   pos.end(),   sortedCoord.begin());

#if 0
	wchar_t text[MAX_PATH];

	for (int i = 0; i < 1089; ++i)
	{
		swprintf_s(text, L"  i=%d; %d %d\n", i, sortedValue[i], sortedCoord[i]);
		OutputDebugString(text);

	}
#endif
}
