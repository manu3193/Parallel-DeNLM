/*
 * main.cpp
 *
 *  Created on: Oct 27, 2015
 *      Author: davidp
 */
#include "parallelDeWAFF.hpp"
#include <ctime>
using namespace std;
  

int main(int argc, char* argv[]){
	ParallelDeWAFF deWAFF;

	//Check input arguments
    if (argc != 2){
        cout << "ERROR: Not enough parameters" << endl;
        deWAFF.help();
        return -1;
    }

    //Open input video file
    const string inputFile = argv[1];
	VideoCapture videoReader(inputFile);
	if (!videoReader.isOpened()){
		cerr << "ERROR: Could not open the input video for read: " << inputFile << endl;
		return -1;
	}

    // Acquire input information: frame rate, number of frames, codec and size
	int fps = static_cast<int>(videoReader.get(CV_CAP_PROP_FPS));
    int nFrames = static_cast<int>(videoReader.get(CV_CAP_PROP_FRAME_COUNT));
    int fourcc = static_cast<int>(videoReader.get(CV_CAP_PROP_FOURCC));
    Size inputSize = Size((int) videoReader.get(CV_CAP_PROP_FRAME_WIDTH),
                            (int) videoReader.get(CV_CAP_PROP_FRAME_HEIGHT));

    // Transform from int to char via Bitwise operators
    char EXT[] = {(char)(fourcc & 0XFF) , (char)((fourcc & 0XFF00) >> 8),(char)((fourcc & 0XFF0000) >> 16),(char)((fourcc & 0XFF000000) >> 24), 0};

	// Find extension point
    string::size_type pAt = inputFile.find_last_of('.');

    // Form the new name with container
    const string outputFile = inputFile.substr(0, pAt) + "_DeWAFF.avi";

	// Set the output size
	Rect outRegion(19,19,inputSize.width-40,inputSize.height-40);

	//Open output video
	VideoWriter videoWriter;
	videoWriter.open(outputFile, CV_FOURCC('M','J','P','G'), fps , outRegion.size(), true);
    if (!videoWriter.isOpened()){
        cout  << "ERROR: Could not open the output video for write: " << outputFile << endl;
        return -1;
    }

    cout << "### Input Video Information ###" << endl
    	 << "Frame resolution: Width=" << inputSize.width << "  Height=" << inputSize.height << endl
         << "Number of frames: " << nFrames << endl
         << "Codec type: " << EXT << endl << endl;

	//Create the Laplacian of Gaussian mask once
    NoAdaptiveLaplacian* nAL = deWAFF.getNAL();
    Mat h =  Tools::fspecialLoG(17, 0.005);
    nAL->setMask(-h);
clock_t begin = clock();
	Mat U,F1;
	for(int frame=1;frame<=nFrames;frame++){
		cout << "Processing frame " << frame << " of " << nFrames << "." << endl;

		//Read one frame from input video
		if(!videoReader.read(U)){
			cout << "Could not read frame from video file" << endl;
			break;
		}


	    	
	        


		//time start
		F1 = deWAFF.processImage(U);
		//time end


 		

		
		//Display output frame
		//Tools::showImg(F1);

		//Cut frame to match output video size
		F1 = F1(outRegion);

		//Write frame to output video
		videoWriter.write(F1);
	}
clock_t end = clock();
  		double elapsed_secs =  ((double) (end - begin)) / CLOCKS_PER_SEC;
cout << "Time to process a frame "  << elapsed_secs << endl;
	//Release video resources
	videoReader.release();
	videoWriter.release();
	return 0;
}

NoAdaptiveLaplacian* ParallelDeWAFF::getNAL(){
	return &(this->nal);
}

void ParallelDeWAFF::help(){
    cout
        << "------------------------------------------------------------------------------" << endl
        << "Usage:"                                                                         << endl
        << "./ParallelDeWAFF inputvideoName"                                                << endl
        << "------------------------------------------------------------------------------" << endl
        << endl;
}

//used parameters for the paper IWOBI 2014
Mat ParallelDeWAFF::processImage(const Mat& U){
    //Set parameters for processing
    int wRSize = 21;
    int wSize_n=3;
    double sigma_s = wRSize/1.5;
    int sigma_r = 13;
    int lambda = 1.7;

    Mat fBilD = filterDeceivedNLM(U, wRSize, wSize_n, sigma_s, sigma_r, lambda);
    //Mat fBilD = filterDeceivedBilateral(U, wRSize, sigma_s, sigma_r, lambda);
    //Mat fBilD = filterDGF(lambda, wRSize, 1, sigma_r, U);

    return fBilD;
}

//Input image must be from 0 to 255
Mat ParallelDeWAFF::filterDeceivedBilateral(const Mat& U, int wSize, double sigma_s, int sigma_r, int lambda){
	//double amps[] = {lambda*0.3, lambda, lambda*0.2};
	//int trap1[] = {5, 20, 35, 90};
	//int trap2[] = {70, 100, 150, 255};

	//double amps = {lambda*0, lambda, lambda};
	//int trap1 = {1, 1, 255, 255};
	//int trap2 = {254, 254, 255, 255};

    Mat Unorm;
    //The image has to to have values from 0 to 1
    U.convertTo(Unorm,CV_32FC3,1.0/255.0);
    //[L, alfaMat, Vnorm] = adaptiveLaplacian(Unorm, amps, trap1, trap2);

    Mat L = this->nal.noAdaptiveLaplacian(Unorm, lambda);
    Mat F = this->bfd.bfilterDeceived(Unorm, L, wSize, sigma_s, sigma_r);

    //putting back everything
    F.convertTo(F,CV_8UC3,255);
    return F;
}

//Input image must be from 0 to 255
Mat ParallelDeWAFF::filterDeceivedNLM(const Mat& U, int wSize, int wSize_n, double sigma_s, int sigma_r, int lambda){


    Mat Unorm;
    //convert to grayscale
    //cvtColor(U, grey, CV_BGR2GRAY);
    //The image has to to have values from 0 to 1
    U.convertTo(Unorm,CV_32FC3,1.0/255.0);
    //[L, alfaMat, Vnorm] = adaptiveLaplacian(Unorm, amps, trap1, trap2);

    Mat L = this->nal.noAdaptiveLaplacian(Unorm, lambda);
    Mat F = this->nlmfd.nlmfilterDeceived(Unorm, L, wSize, sigma_s, sigma_r);

    //putting back everything
    F.convertTo(F,CV_8UC1,255);
    return F;
}
