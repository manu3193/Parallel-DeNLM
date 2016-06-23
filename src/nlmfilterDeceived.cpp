/*
 * nlmfilterDeceived.cpp
 *
 *  Created on: Jun 20, 2016
 *      Author: davidp, manzumbado
 */


#include "nlmfilterDeceived.hpp"

// Pre-process input and select appropriate filter.
Mat NLMFilterDeceived::nlmfilterDeceived(const Mat& A, const Mat& L, int w, int w_n, double sigma_s, int sigma_r){
    double minA, maxA;
    Tools::minMax(A,&minA,&maxA);
    int type = A.type();

    if (!(type == CV_32FC3) || minA < 0 || maxA > 1){
       cerr << "Input image A must be a double precision matrix of size NxMx1 on the closed interval [0,1]." << endl;
    }

    // Apply either grayscale nlm filtering.
    Mat B;
    if (type == CV_32FC3 )
       //B = bfltColorDeceived(A, L, w, sigma(1),sigma(2));//?????????????????
        B = this->nlmfltBWDeceived(A, L, w, w_n, sigma_s, sigma_r);
    else
        ;
    return B;
}

//Implements bilateral filter for color images.
//sigma range is multiplied by 100
Mat NLMFilterDeceived::nlmfltBWDeceived(const Mat& A, const Mat& L, int w, int w_n, double sigma_d, int sigma_r){
    Mat B,C,D;

    //Convert input BGR image to CIELab color space.
    //CIELab 'a' and 'b' values go from -127 to 127
    cout << "Using the CIELab color space." << endl;
    cvtColor(A,B,CV_BGR2Lab);

    //L =  filterUM_laplacianLAB(A, lambda);
    C = this->nlmfilBW_deceived(B, L, w, w_n, sigma_d,sigma_r);

    //Convert filtered image back to sRGB color space.
    cvtColor(C,D,CV_Lab2BGR);
    Tools::showImg(D);

    return D;
}

Mat NLMFilterDeceived::nlmfilBW_deceived(const Mat& A, const Mat& Laplacian, int w, int w_n, double sigma_d, int sigma_r){
    int iMin, iMax, jMin, jMax;
    Mat B, F, G, H, I, L, S, E;

	Mat1i X,Y;
	vector<Mat> channels(3);
	Vec3f pixel;
	double norm_F;

    //Pre-compute Gaussian domain weights.
    Tools::meshgrid(Range(-w,w),Range(-w,w),X,Y);
    pow(X,2,X);
    pow(Y,2,Y);
    S = X+Y;
    S.convertTo(S,CV_32F);
    S /= (-2*pow(sigma_d,2));

    exp(S,G);

    //Apply bilateral filter.
    omp_set_num_threads(128);
    B = Mat::zeros(A.size(),A.type());
    cout << "Applying the deceived bilateral filter..." << endl;
    
	//#pragma omp parallel for private(I,iMin,iMax,jMin,jMax,pixel,E,channels,H,F,norm_F,L) shared(A,B,G,Laplacian,w,sigma_d,sigma_r)
    for(int i = 1; i < A.rows-1; i++){
       for(int j = 1; j < A.cols-1; j++){


             //Extract local region.
             iMin = max(i - w - w_n,0);
             iMax = min(i + w + w_n,A.rows-1);
             jMin = max(j - w - w_n,0);
             jMax = min(j + w + w_n,A.cols-1);

             //Compute Gaussian range weights.
             //done in the three layers
             I = A(Range(iMin,iMax), Range(jMin,jMax));
             cout << "I = "<< endl << " "  << I << endl << endl;
             E= CalcEuclideanDistMat(I, w_n, i, j);

             exp(E / (-2 * pow(sigma_r,2)),H);

             cout << E.rows << " e " << E.cols<< endl;
             cout << H.rows << " h " << H.cols<< endl;

             cout << "E = "<< endl << " "  << E << endl << endl;

             cout << "H = "<< endl << " "  << H << endl << endl;



             //Calculate NLM filter response.
             F = H.mul(G(Range(iMin-i+w, iMax-i+w), Range(jMin-j+w, jMax-j+w)));
             norm_F = sum(F).val[0];

             //The laplacian deceive consists on weighting the gaussian function with
             //the original image, and using the image values of the laplacian image.
             L = Laplacian(Range(iMin,iMax), Range(jMin,jMax));
             split(L,channels);
             B.at<Vec3f>(i,j)[0] = (sum(sum(F.mul(channels[0])))/norm_F).val[0];
       }
    }
    split(Laplacian,channels);
    Tools::showImg(channels[0]);
    return B;

}

Mat NLMFilterDeceived::CalcEuclideanDistMat(const Mat& I, int w_n, int i, int j){
    vector<float> v_z, v_w; //Neighborhood vectors
    Mat Z, W, E;
    int size_x,size_y; //Window dimensions

    //NLM------------------------------------------------------------------------
            size_x=I.rows;
            size_y=I.cols;
             //Current Pixel neighborhood
             Z= I(Range(i-w_n,i+w_n),Range(j-w_n,j+w_n));
             //Vectorized neighborhood
             v_z.assign((float*)Z.datastart, (float*)Z.dataend);
             //Create output Mat
             E.create(size_x-2,size_y-2, DataType<float>::type);
             cout << "E2 = "<< endl << " "  << E << endl << endl;

             //Visit each pixel in the window I
             for(int n=1; n<size_x-2; n++){
                 for(int m=1; m<size_y-2; m++){

                    //Get pixel mini-window
                    W = I(Range(n-w_n,n+w_n), Range(m-w_n,m+w_n));
                    cout << "W = "<< endl << " "  << W << endl << endl;
                    //vectorized neighborhood
                    v_w.assign((float*)W.datastart, (float*)W.dataend);

                    if(v_z.size() != v_w.size()){
                        break;
                    }
                    E.at<float>(n-1,m-1) = (float) norm(v_z,v_w,NORM_L2);
                 }
             }
    return E;
}
