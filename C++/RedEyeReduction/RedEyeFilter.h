//-----------------------------------------------------------------------------
// File: RedEyeFilter.h
//
// Desc: Removes red-eye artifact from an image.
//
//-----------------------------------------------------------------------------

#pragma once

namespace Bisque
{
	using std::string;
	using std::vector;
	using cv::Mat;

	// class RedEyeFilter
	class RedEyeFilter
	{
	public:
		RedEyeFilter(const dim3& block);
		~RedEyeFilter();

		void ApplyFilter(const string& imagePath, const string& outputPath, const string& redEyeTemplate);

	private:

		// struct image_host_t: contains references to data on the CPU
		struct image_host_t
		{
			vector<uchar4>	image;						// original picture with red eyes
			int				cols;
			int				rows;
			int				pixels;						// number of pixels, has to be computed when the image is loaded
		};

	private:
		void LoadPicture	(const string& imagePath, image_host_t& image);
		void ProcessImage	(thrust::host_vector<uchar4>& modifiedImage);
		void Release		();
		void SaveImage		(thrust::host_vector<uchar4>& modifiedImage, const string& outputPath);

	private:
		// Member variables
		dim3 m_block;
		image_host_t	m_picture;							// data related to original picture that requires red eye removal
		image_host_t	m_redEye;							// contains red-eye template image
	};
}