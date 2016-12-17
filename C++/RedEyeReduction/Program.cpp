// program.cpp
//
// Main entry point into the application.
//

#include "stdafx.h"
#include "RedEyeFilter.h"

using std::cout;
using std::cerr;
using std::endl;
using std::exception;
using std::string;

using namespace Bisque;

// Forward declarations
void SetDevice(dim3& block);

// Main entry into the application
int main(int argc, char** argv)
{
#if _DEBUG
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

	// Test input parameters: required path to 2 images, original and modified.
	string imagePath;
	string outputPath;
	string redEyeTemplate;

	if (argc > 3)
	{
		imagePath		= string(argv[1]);
		outputPath		= string(argv[2]);
		redEyeTemplate	= string(argv[3]);
	}
	else
	{
		cerr << "Please provide input and output image file names (paths) and red eye template image name (path) as arguments to this application." << endl;
		cerr << "Usage: ./hw input_file output_file red_eye_template_file" << endl;
		exit(EXIT_FAILURE);
	}

	// Show device info and pick the best available device
	dim3 block;
	SetDevice(block);

	// Apply red eye filter
	using std::unique_ptr;
	unique_ptr<RedEyeFilter> filter(new RedEyeFilter(block));

	try
	{
		filter->ApplyFilter(imagePath, outputPath, redEyeTemplate);
	}
	catch(exception& e)
	{
		cerr << endl << "ERROR: " << e.what() << endl;
		exit(EXIT_FAILURE);
	}

	cout << "Done... Resetting device!" << endl << endl;

	// Reset device before exiting for profiling and tracing tools to show compete traces
	cudaError_t ce = cudaDeviceReset();
	if (ce != cudaSuccess)
	{
		cerr << "cudaDeviceReset failed!" << endl;
		return 1;
	}

	return 0;
}

// Show device information, set preferred device
/*
Using device: 0
	Name:                    Quadro 5000M
	Compute version:         2.0
	Global memory:           2047.69 mb
	Const memory:            64 kb
	L2 cache size:           512 kb
	Clock rate:              810 mhz
	Timeout enabled:         true
	Multiprocessors:         10
	Max grid size:           65535 : 65535 : 65535
	Max threads per SM:      1536
	Max threads per block:   1024
	Registers per block:     32768
	Shared memory per block: 48 kb
	Memory bus width:        256 bits
	Memory clock rate:       1200 mhz
	Compute mode:            Default
*/
void SetDevice(dim3& block)
{
	// Only show stats on the first run of this code
	static bool showStats = true;

	if (!showStats)
		return;
	
	showStats = false;

	// Set device and display properties
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		cerr << "ERROR: Your system does not have CUDA." << endl;
		exit(EXIT_FAILURE);
	}

	int device;
	for (device = 0; device < deviceCount; ++device)
	{
		cudaDeviceProp p;
		cudaError ce = cudaGetDeviceProperties(&p, device);

		if (ce != cudaSuccess)
		{
			cerr << "ERROR: Device query failed." << endl;
			exit(EXIT_FAILURE);
		}

		cout << "\n\nUsing device: " << device << endl;
		cout << "   Name:                    " << p.name << endl;
		cout << "   Compute capability:      " << p.major << "." << p.minor << endl;
		cout << "   Warp Size:               " << p.warpSize << endl;
		cout << "   Global memory:           " << p.totalGlobalMem / static_cast<float>(1024 * 1024) << " mb" << endl;
		cout << "   Const memory:            " << p.totalConstMem / static_cast<float>(1024) << " kb" << endl;
		cout << "   L2 cache size:           " << p.l2CacheSize / static_cast<float>(1024) << " kb" << endl;
		cout << "   Clock rate:              " << p.clockRate / 1000.f << " mhz"<< endl;
		cout << "   Timeout enabled:         " << (p.kernelExecTimeoutEnabled == 1 ? "true" : "false") << endl;
		cout << "   Multiprocessors:         " << p.multiProcessorCount << endl;
		cout << "   Max grid size:           " << p.maxGridSize[0] << " : " << p.maxGridSize[1] << " : " << p.maxGridSize[2] << endl;
		cout << "   Max threads per SM:      " << p.maxThreadsPerMultiProcessor << endl;
		cout << "   Max threads per block:   " << p.maxThreadsPerBlock << endl;
		cout << "   Registers per block:     " << p.regsPerBlock << endl;
		cout << "   Shared memory per block: " << p.sharedMemPerBlock / static_cast<float>(1024) << " kb" << endl;
		cout << "   Memory bus width:        " << p.memoryBusWidth << " bits" << endl;
		cout << "   Memory clock rate:       " << p.memoryClockRate / 1000.f << " mhz" << endl;
		cout << "   Compute mode:            " << (p.computeMode == 0 ? "Default" : "Exclusive or Prohibitive") << endl;
		cout << "   PCI Bus id:              " << p.pciBusID << endl;
		cout << "   PCI Device id:           " << p.pciDeviceID << endl;
		cout << "   PCI Domain id:           " << p.pciDomainID << endl;
		cout << "   Number of async engines: " << p.asyncEngineCount << endl;
		cout << "   Discrete GPU:            " << (p.integrated == 0 ? "Yes" : "No") << endl;
		cout << "   Can map host memory:     " << (p.canMapHostMemory == 1 ? "Yes" : "No") << endl;
		cout << "   Concurrent kernels:      " << (p.concurrentKernels == 1 ? "Yes" : "No") << endl;
		cout << "   ECC supported:           " << (p.ECCEnabled == 1 ? "Yes" : "No") << endl;
		cout << "   Unified addressing:      " << (p.unifiedAddressing == 1 ? "Yes" : "No") << endl;
		cout << endl << endl;

		// Set best device here
		if (device == 0)
		{			
			ce = cudaSetDevice(0);
			if (ce != cudaSuccess)
			{
				cerr << "ERROR: failed to set CUDA device!" << endl;
				exit(EXIT_FAILURE);
			}

			block.x = 32;
			block.y = p.maxThreadsPerBlock / block.x;
			block.z = 1;
		}
	}
}
