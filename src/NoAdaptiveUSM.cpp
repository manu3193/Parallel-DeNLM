/*
 * NoAdaptiveUSM.cpp
 *
 *  Created on: Oct 28, 2015
 *      Author: davidp
 */

#include "NoAdaptiveUSM.hpp"

Mat NoAdaptiveUSM::NoAdaptiveUSM(const Mat& U, const float lambda, const int kernelLen, const float sigma){

	Point anchor( -1 , -1 );
	double delta = 0;
	int ddepth = -1;

	Mat F(U.size(),U.type());

	Mat kernel = this->generateLoGKernel(const int size, const float sigma);

	filter2D(U, F, ddepth , kernel, anchor, delta, BORDER_DEFAULT );

	//normalization
	double minZ,maxZ;
	minMaxLoc(abs(Z),&minZ,&maxZ);

	double minU,maxU;
	minMaxLoc(U,&minU,&maxU);

	Z = maxU * (Z / maxZ);

	//Unsharp masking
	return (U + lambda1 * Z);
}


Mat NoAdaptiveUSM::generateLoGKernel(const int size, const float sigma){

}