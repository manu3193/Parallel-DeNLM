/*
 * NoAdaptiveUSM.hpp
 *
 *  Created on: Nov 1, 2015
 *      Author: davidp
 */

#ifndef NOADAPTIVEUSM_HPP_
#define NOADAPTIVEUSM_HPP_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "highgui/highgui.hpp"
#include "tools.hpp"
using namespace cv;

class NoAdaptiveUSM{

public:
	Mat NoAdaptiveUSM(const Mat& U, float lambda, const int kernelLen, const float sigma);

private:
	Mat generateLoGKernel(const int kernelLen, const float sigma);
};
#endif /* NOADAPTIVEUSM_HPP_ */
