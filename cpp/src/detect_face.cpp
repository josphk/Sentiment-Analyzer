#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "RaspiCamCV.h"

#include <iostream>
#include <stdio.h>

void display(IplImage* img);

String faceCascadePath = "../../cascades/haarcascade.xml";
cv::CascadeClassifier faceCascade;
int pressedKey = 0;

int main() {
  RASPIVID_CONFIG* config = (RASPIVID_CONFIG*)malloc(sizeof(RASPIVID_CONFIG));
  config -> width = 320;
  config -> height = 240;
  config -> bitrate = 0;
  config -> framerate = 0;
  config -> monochrome = 1;

  RaspiCamCvCapture * camera = raspiCamCvCreateCameraCapture2(0, config);
  free(config);

  IplImage* image = raspiCamCvQueryFrame(camera);
  cvNamedWindow("Display", CV_WINDOW_AUTOSIZE);

  if (faceCascade.load(faceCascadePath)) {
    display(image);
  }

  return 0;
}

void display(IplImage& img) {
    do {
      std::vector<Rect> faces;
      IplImage* grey;

      cvCvtColor(img, grey, CV_BGR2GRAY);
      cvEqualizeHist(grey, grey);

      faceCascade.detectMultiScale(grey, faces, 1.1, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));

      for (size_t i = 0; i < faces.size(); i++) {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( img, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
      }

      cvShowImage("Display", img);
      pressedKey = cvWaitKey(1);
    } while (pressedKey != 27);
}
