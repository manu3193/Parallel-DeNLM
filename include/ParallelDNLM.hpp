/*
 * PARALLELDNLM.hpp
 *
 *  Created on: Nov 5, 2015
 *      Author: davidp
 *  Modified on: Sep 24, 2018
 * 		Author: manzumbado
 */

#ifndef PARALLELDNLM_HPP_
#define PARALLELDNLM_HPP_

#include <string>
//include <bfilterDeceived.hpp>
#include <DNLMFilter.hpp>
#include <NoAdaptiveUSM.hpp>

using namespace std;

class ParallelDNLM{
public:
	void help();
	Mat processImage(const Mat& U);

private:
	NoAdaptiveUSM nal;
	DNLMFilter nlmfd;

	Mat filterDNLM(const Mat& U, int wSize, int wSize_n, int sigma_r, int lambda, int kernelLen, float kernelStd);
};
#endif /* PARALLELDNLM_HPP_ */
