#ifndef IMAGESPLITTER_CXX
#define IMAGESPLITTER_CXX

#include "ImageSplitter.h"

void ImageSplitter::setDepth(size_t depth){
  if ( depth < _MAX_DEPTH ){
    _depth = depth;
  }
}

void ImageSplitter::setXPixels(size_t pixVal){
  if (pixVal < _MAX_PIXELS){
    _x_pixels = pixVal;
  }
}
void ImageSplitter::setYPixels(size_t pixVal){
  if (pixVal < _MAX_PIXELS){
    _y_pixels = pixVal;
  }
}


#endif
