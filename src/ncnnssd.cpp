#include "ncnnssd.h"
#include "face_detect.id.h"
#include "face_detect.mem.h"

ncnnssd::ncnnssd()
{
	m_facenet = nullptr;
	m_facenet = new ncnn::Net;
	m_facenet->load_param(face_detect_param_bin);
	m_facenet->load_model(face_detect_bin);

	mean_vals[0] = 127.5f;
	mean_vals[1] = 127.5f;
	mean_vals[2] = 127.5f;

	norm_vals[0] = 1.0;
	norm_vals[1] = 1.0;
	norm_vals[2] = 1.0;
}

ncnnssd::~ncnnssd(){
	if (m_facenet){
		delete m_facenet;
		m_facenet = nullptr;
	}
}

int ncnnssd::clipBorder(int i, int lower, int upper)
{
	if(i< lower){ i = lower; }
	if(i> upper){ i = upper; }
	return i;
}

bool ncnnssd::detect(unsigned char*pInBGRData, int nInRows, int nInCols, std::vector<ssdFaceRect> &rtfaces,
	                 int ntargetrows, int ntargetcols, float fminscore) {
	if (!m_facenet)
	{
		return false;
	}
	ncnn::Mat indata = ncnn::Mat::from_pixels_resize(pInBGRData, ncnn::Mat::PIXEL_BGR, nInCols, nInRows, ntargetcols, ntargetrows);
	indata.substract_mean_normalize(mean_vals, norm_vals);
	ncnn::Mat outfeats;
	ncnn::Extractor ex = m_facenet->create_extractor();
	#ifdef __LINUX__
	ex.set_num_threads(1);
	#endif
	//ex.input("data", indata);
	//ex.extract("detection_out", outfeats);
	ex.set_light_mode(true);
	ex.input(face_detect_param_id::BLOB_data, indata);
	ex.extract(face_detect_param_id::BLOB_detection_out, outfeats);
	//printf("outfeats: %d %d %d\n", outfeats.w, outfeats.h, outfeats.c);
	
	rtfaces.clear();
	for (int i = 0; i < outfeats.h; i++)
	{
		const float* values = outfeats.row(i);
		ssdFaceRect face;
		//face.nlabel = values[0];
		face.confidence = values[1];

		if(false)
		{
			float x1 = values[2] * nInCols;
			float y1 = values[3] * nInRows;
			float x2 = values[4] * nInCols;
			float y2 = values[5] * nInRows;

			face.x = clipBorder(x1, 0, nInCols);
			face.y = clipBorder(y1, 0, nInRows);
			face.w = clipBorder(x2, 0, nInCols) - face.x;
			face.h = clipBorder(y2, 0, nInRows) - face.y;
		}
		else
		{
			float x1 = values[2] * nInCols;
			float y1 = values[3] * nInRows;
			float x2 = values[4] * nInCols;
			float y2 = values[5] * nInRows;

			float cx = (x2 + x1)/2.0;
			float cy = (y2 + y1)/2.0;
			float mm = std::max(y2 - y1, x2 - x1) / 2.0;
			x1 = cx - mm * 0.9;
			y1 = cy - mm * 0.9;
			x2 = cx + mm * 0.9;
			y2 = cy + mm * 0.9;

			face.x = clipBorder(x1, 0, nInCols);
			face.y = clipBorder(y1, 0, nInRows);
			face.w = clipBorder(x2, 0, nInCols) - face.x;
			face.h = clipBorder(y2, 0, nInRows) - face.y;

		}
		
		//face.x = values[2] * nInCols;
		//face.y = values[3] * nInRows;
		//face.w = values[4] * nInCols - values[2] * nInCols;
		//face.h = values[5] * nInRows - values[3] * nInRows;

		rtfaces.push_back(face);
	}
	return true;
}
