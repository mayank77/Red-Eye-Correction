#include "stdafx.h"
#include "RedEyeFilter.h"

#define TRACE_RED_EYE 1

using namespace Bisque;

// Forward declarations
void RedEyeFilter_Main(
	thrust::host_vector<uchar4>&	modifiedImage,
	const vector<uchar4>&			image, 
	int								imageRows,
	int								imageCols,
	const vector<uchar4>&			redEyeTemplate,
	int								redEyeRows,
	int								redEyeCols
	);

// Ctor
RedEyeFilter::RedEyeFilter(const dim3& block)
	: m_block(block)
{
}

RedEyeFilter::~RedEyeFilter(void)
{
}

// Entry function responsible for orchestrating removal of red eye form an image
void RedEyeFilter::ApplyFilter(const string& imagePath, const string& outputPath, const string& redEyeTemplate)
{
    const char* fn = "RedEyeFilter::ApplyFilter";
	OutputDebugString(L" Here we go! Apply  Filter\n");

	cudaError ce = cudaFree(nullptr);																			CHECK_CUDA_ERROR(ce, fn, "Could not free CUDA memory.");

#if TRACE_RED_EYE
	using std::chrono::duration_cast;
	using std::chrono::high_resolution_clock;
	using std::chrono::microseconds;
	using std::chrono::milliseconds;
	using std::chrono::time_point;

	using std::cout;
	using std::endl;

	GpuTimer gpuTimer;
	time_point<high_resolution_clock> start;
	time_point<high_resolution_clock> stop;

	start = high_resolution_clock::now();
#endif

	//
	// Load image
	//
	LoadPicture(imagePath, m_picture);
	LoadPicture(redEyeTemplate, m_redEye);

#if TRACE_RED_EYE
	stop = high_resolution_clock::now();
	long long ms = duration_cast<milliseconds>(stop - start).count();
	long long us = duration_cast<microseconds>(stop - start).count();

	cout << "Loaded images in " << us << "us (" << ms << "ms)" << endl;

	start = high_resolution_clock::now();
#endif

	//
	// Initialize CUDA and extract image and red eye template channels
	//
	using thrust::host_vector;
	host_vector<uchar4> modifiedImage(m_picture.pixels);

	ProcessImage(modifiedImage);

#if TRACE_RED_EYE
	stop = high_resolution_clock::now();
	ms = duration_cast<milliseconds>(stop - start).count();
	us = duration_cast<microseconds>(stop - start).count();

	cout << "Removed red-eyes from the image in " << us << "us (" << ms << "ms)" << endl;

	start = high_resolution_clock::now();
#endif

	//
	// Save final image
	//
	SaveImage(modifiedImage, outputPath);

#if TRACE_RED_EYE
	stop = high_resolution_clock::now();
	ms = duration_cast<milliseconds>(stop - start).count();
	us = duration_cast<microseconds>(stop - start).count();

	cout << "Saved copy of the image in " << us << "us (" << ms << "ms)" << endl;

	start = high_resolution_clock::now();
#endif

	// cleanup
	Release();
}

// Loads phot with red eye and eye template, splits it into channels
void RedEyeFilter::LoadPicture(const string& imagePath, image_host_t& data)
{
    //const char* fn = "RedEyeFilter::LoadPicture";
	OutputDebugString(L"   Load picture\n");

	using cv::Mat;
	using cv::cvtColor;
	using cv::imread;
	using cv::imwrite;
	using concurrency::parallel_for;

	// Load image (image is stored in bgra format on the disk)
	Mat image = imread(imagePath.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty())
	{
		string msg = "Could not open image file: " + imagePath;
		throw runtime_error(msg);
	}

	if ((image.channels() != 3) || (!image.isContinuous()))
	{
		string msg = "Image " + imagePath + " has incorrect format.";
		throw runtime_error(msg);
	}

	data.cols	= image.cols;
	data.rows	= image.rows;
	data.pixels	= data.rows * data.cols;

	// Copy bgr mat image to rgba channels
	unsigned char* tmp = image.ptr<unsigned char>(0);				// temporary pointer
	auto&          rgb = data.image;								// alias to picture

	rgb.resize(data.pixels);

	//for (int i = 0; i < m_picture.pixels; ++i)
	parallel_for(int(0), data.pixels, [&rgb, &tmp](int i)
	{
		rgb[i].x = tmp[i * 3 + 2];
		rgb[i].y = tmp[i * 3 + 1];
		rgb[i].z = tmp[i * 3 + 0];
		rgb[i].w = 255;												// opaque
	});

#if 0
	for (int i = 0; i < 100; ++i)
	{
		wchar_t text[MAX_PATH];
		//swprintf_s(text, L"   %d: x=%d, y=%d, z=%d w=%d\n", i, m_picture.image[i].x, m_picture.image[i].y, m_picture.image[i].z, m_picture.image[i].w);
		swprintf_s(text, L"   %d: x=%d, y=%d, z=%d w=%d\n", i, rgb[i].x, rgb[i].y, rgb[i].z, rgb[i].w);
		OutputDebugString(text);
	}
#endif

}

// Preprocesses image
void RedEyeFilter::ProcessImage(thrust::host_vector<uchar4>& modifiedImage)
{
    //const char* fn = "RedEyeFilter::ProcessImage";
	OutputDebugString(L"   Process image\n");

	// Run CUDA kernels
	RedEyeFilter_Main(
		modifiedImage,										// return value: this is a copy of an image with red-eyes removed
		m_picture.image, 
		m_picture.rows,
		m_picture.cols, 
		m_redEye.image, 
		m_redEye.rows,
		m_redEye.cols
		);
}

// Releases resources
void RedEyeFilter::Release()
{
    //const char* fn = "RedEyeFilter::Release";
	OutputDebugString(L"   Free resources\n");
}

// Saves image to disk
void RedEyeFilter::SaveImage(thrust::host_vector<uchar4>& modifiedImage, const string& outputPath)
{
	OutputDebugString(L"   Save image copy\n");

	int size[] = { m_picture.rows, m_picture.cols };

	using cv::Mat;
	using cv::cvtColor;
	using cv::imwrite;

	Mat rgba(2, size, CV_8UC4, reinterpret_cast<void*>(modifiedImage.data()));
	Mat bgr;

	cvtColor(rgba, bgr, CV_RGBA2BGR);
	imwrite(outputPath.c_str(), bgr);
}
