/*
 * main.cpp
 *
 *  Modified on: Nov 07, 2016
 *      Authors: manzumbado, davidp
 */
#include <ParallelDNLM.hpp>
#include <ctime>    
using namespace std;
  

int main(int argc, char* argv[]){
    ParallelDNLM deWAFF;

    //Check input arguments
    if (argc != 2){
        cout << "ERROR: Not enough parameters" << endl;
        deWAFF.help();
        return -1;
    }else{

    //Open input video file
    const string inputFile = argv[1];
    // Find extension point
    string::size_type pAt = inputFile.find_last_of('.');

    // Form the new name with container
    const string outputFile = inputFile.substr(0, pAt) + "_DeNLM.jpg";



    Mat U,F1;
    U = imread(inputFile, CV_LOAD_IMAGE_COLOR);
    //Read one frame from input video
        if(!U.data){
            cout << "Could not read image from file." << endl;
            return -1;
        }
    //time start
    //clock_t begin = clock();
    F1 = deWAFF.processImage(U);
    //clock_t end = clock();
    //time end

    //Write image to output file.
    imwrite(outputFile, F1);
    //double elapsed_secs =  ((double) (end - begin)) / CLOCKS_PER_SEC;
    //cout << "Time to process an image: "  << elapsed_secs << endl;
    return 0;
    }
}

void ParallelDeWAFF::help(){
    cout
        << "------------------------------------------------------------------------------" << endl
        << "Usage:"                                                                         << endl
        << "./dnlmfilter inputImageName"                                                << endl
        << "------------------------------------------------------------------------------" << endl
        << endl;
}

//used parameters for the paper CONCAPAN 2016
Mat ParallelDeWAFF::processImage(const Mat& U){
    //Set parameters for processing
    int wRSize = 21;
    int wSize_n=7;
    int kernelLen = 21;
    float kernel_std = 0.6;
    int sigma_r = 3; 
    int lambda = 1.7;


    Mat fDeceivedNLM = filterDeceivedNLM(U, wRSize, wSize_n, sigma_r, lambda, kernelLen, kernelStd);

    return fDeceivedNLM;
}

//Input image must be from 0 to 255
Mat ParallelDeWAFF::filterDeceivedNLM(const Mat& U, int wSize, int wSize_n, int sigma_r, int lambda, int kernelLen, float kernelStd){
    Mat Unorm;
    //The image has to have values from 0 to 1
    U.convertTo(Unorm,CV_32FC3,1.0/255.0);
    //[L, alfaMat, Vnorm] = adaptiveLaplacian(Unorm, amps, trap1, trap2);

    Mat L = this->nal.noAdaptiveUSM(Unorm, lambda);
    Mat F = this->nlmfd.nlmfilterDeceived(Unorm, L, wSize, wSize_n, sigma_r);

    //putting back everything
    F.convertTo(F,CV_8UC1,255);
    return F;
}
