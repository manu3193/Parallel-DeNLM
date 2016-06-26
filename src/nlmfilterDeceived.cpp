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

    // Apply grayscale nlm filtering.
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
    //Vec3f pixel;
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
    omp_set_num_threads(4);
    B = Mat::zeros(A.size(),A.type());
    cout << "Applying the deceived nlm filter..." << endl;
    
    #pragma omp parallel for private(I,iMin,iMax,jMin,jMax/*,pixel*/,E,channels,H,F,norm_F,L) shared(A,B,G,Laplacian,w,sigma_d,sigma_r)
    for(int i = 0; i < A.rows-1; i++){
       for(int j = 0; j < A.cols-1; j++){

           float val; //result value
           //Extract local region.
           iMin = max(i - w,0);
           iMax = min(i + w,A.rows-1);
           jMin = max(j - w,0);
           jMax = min(j + w,A.cols-1);


           //Exclude borders
           if((i>=w+w_n) && (j>=w+w_n) && (i<=A.rows-1-w-w_n) && (j<=A.cols-1-w-w_n)){

               I = A(Range(iMin-w_n,iMax+w_n+1), Range(jMin-w_n,jMax+w_n+1)); /*********************************************************/
               //Compute Gaussian range weights.
               //done in the three layers
               split(I,channels);
               /*
               //Debug
               //cout << "I = "<< " "  << channels[0] << endl << endl;
               cout << "Ai = "<< " "  << i << " "<< "Aj =" << " " << j << endl;
               cout.precision(9);
               cout << "Pixel neighborhood in A = " << endl <<fixed<< A.at<Vec3f>(i-1,j-1)[0] << ", "  <<fixed<< A.at<Vec3f>(i-1,j)[0] << ", " <<fixed<<A.at<Vec3f>(i-1,j+1)[0] << endl;
               cout << "                          " << endl <<fixed<< A.at<Vec3f>(i,j-1)[0] << ", "  <<fixed<< A.at<Vec3f>(i,j)[0] << ", "<<fixed <<A.at<Vec3f>(i,j+1)[0] << endl;
               cout << "                          " << endl <<fixed<< A.at<Vec3f>(i+1,j-1)[0] << ", "  <<fixed<< A.at<Vec3f>(i+1,j)[0] << ", "<<fixed <<A.at<Vec3f>(i+1,j+1)[0] << endl;

               //cout << "I iMin = " << " "  << iMin << " "<< "I iMax =" << " " << iMax << endl;
               //cout << "I jMin = " << " "  << jMin << " " << "I jMax =" << " " << jMax << endl;
               */
               E= CalcEuclideanDistMat(channels[0], w_n, i, j, iMin, jMin);
               exp(E / (-2 * pow(sigma_r,2)),H);

               /*
               //Debug
               cout << E.rows << " e " << E.cols<< endl;
               cout << H.rows << " h " << H.cols<< endl;

               cout << "I = "<< endl << " "  << I.rows << " " << I.cols << endl;

               cout << "H = "<< endl << " "  << H.rows << " " << H.cols << endl;

               cout << "G = "<< endl << " "  << G(Range(iMin-i+w, iMax-i+w), Range(jMin-j+w, jMax-j+w)).rows << " "<< G(Range(iMin-i+w, iMax-i+w), Range(jMin-j+w, jMax-j+w)).cols << endl << endl;
               */

               //Calculate NLM filter response.
               F = H.mul(G(Range(iMin-i+w, iMax-i+w), Range(jMin-j+w, jMax-j+w)));
               norm_F = sum(F).val[0];

               //The laplacian deceive consists on weighting the gaussian function with
               //the original image, and using the image values of the laplacian image.
               L = Laplacian(Range(iMin,iMax), Range(jMin,jMax));
               split(L,channels);
               val =(sum(sum(F.mul(channels[0])))/norm_F).val[0];

             //Create black border
             }else{
                 val=0;
             }            
                 B.at<Vec3f>(i,j)[0] = val;
       }
    }
    return B;

}

Mat NLMFilterDeceived::CalcEuclideanDistMat(const Mat& I, int w_n, int i, int j, int iMin, int jMin){
    int nMin_w, nMax_w, mMin_w, mMax_w, nMin_z, nMax_z, mMin_z, mMax_z;
    Mat Z, W, O;
    int size_x,size_y; //Window dimensions

    //NLM------------------------------------------------------------------------
    size_x=I.rows;
    size_y=I.cols;

    //Extract pixel z neighborhood local region.
    nMin_z = max(i - iMin - w_n+1,0);
    nMax_z = min(i - iMin + w_n+1,I.cols-1);
    mMin_z = max(j - jMin - w_n+1,0);
    mMax_z = min(j - jMin + w_n+1,I.cols-1);

    //Debug
    /*
    int Ii, Ij;
    Ii=i-iMin+w_n;
    Ij=j-jMin+w_n;
    cout << "nMin_z = "<< " "  << nMin_z << endl << "nMax_z" << " " << nMax_z << endl;
    cout << "mMin_z = " << " "  << mMin_z << endl << "mMax_z" << " " << mMax_z << endl;
    cout << "Ii = "<< " "  << Ii << " " << "Ij=" << " " << Ij << endl;
    */

    //Current Pixel z neighborhood
    Z= I(Range(nMin_z,nMax_z+1),Range(mMin_z,mMax_z+1));


    /*
    //Debug
    float pix=I.at<float>(i-iMin+w_n,j-jMin+w_n);
    cout << "Pixel Value in I = " << " "  << pix << endl;
    cout << "Z = "<< endl << " "  << Z << endl << endl;
    */

    //Create output Mat
    O.create(size_x-1-2*w_n,size_y-1-2*w_n, DataType<float>::type);

     //Visit each pixel in the window I
     for(int n=1; n<size_x-2; n++){
         for(int m=1; m<size_y-2; m++){

            //Extract pixel w neighborhood local region.
            nMin_w = max(n - w_n,0);
            nMax_w = min(n + w_n,I.rows-1);
            mMin_w = max(m - w_n,0);
            mMax_w = min(m + w_n,I.cols-1);
            //Get pixel mini-window
            W = I(Range(nMin_w,nMax_w+1), Range(mMin_w,mMax_w+1));
            /*
            //Debug
            cout << "W = "<< endl << " "  << W << endl << endl;
            */
            float res=(float) norm(W,Z,NORM_L2);
            //cout << "dist = "<< endl << " "  << res  << endl;
            O.at<float>(n-w_n,m-w_n) = res;
         }
     }
     /*
      //Debug
     cout << "O = "<< endl << " "  << O << endl << endl;
     */
     return O;

}
