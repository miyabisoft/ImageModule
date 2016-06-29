//
//  Imagesmodule.h
//  ImageModule
//
//  Created by setomasakazu on 2015/06/27.
//  Copyright (c) 2015年 setomasakazu. All rights reserved.
//

#ifndef __ImageModule__Imagesmodule__
#define __ImageModule__Imagesmodule__

#include <stdio.h>
#include <iostream>
#include <string>

#endif /* defined(__ImageModule__Imagesmodule__) */

extern "C" {
    void UpdateTexture(char* data, int width, int height);
    void DetectHsv(unsigned char* src, unsigned char* dest, int width, int height);
    void convertToCannyTextureA(char* src, int width, int height, float threshold1, float threshold2);
    void OpticalFlow1(unsigned char* src, unsigned char* pre, int width, int height);
    void OpticalFlow2(unsigned char* src, int width, int height);
    void OpticalFlow3(unsigned char* src, unsigned char* pre, int width, int height);
    void createCheckTexture(unsigned char* arr, int w, int h, int ch);
    int SetTextureOfCam1(unsigned char* src, int width,  int height);
}
