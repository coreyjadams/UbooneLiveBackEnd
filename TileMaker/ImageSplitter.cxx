#ifndef IMAGESPLITTER_CXX
#define IMAGESPLITTER_CXX

#include "ImageSplitter.h"

namespace ulbe {


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

// void makeTiles(std::string fileName){
//   // To make the files, first we open the existing png:
//   png::image<png::rgb_pixel> image(fileName);
//   image.write("output.png");
// }

void ImageSplitter::makeTiles(PyObject * array){
  
  // Make sure the input is a numpy array:
  PyArrayObject * data = (PyArrayObject *) (array);

  // Take in the numpy array for the whole image and then split it up into sub arrays
  
  // Get the shape of the array and verify it works:
  npy_intp * shape;
  try{
    shape = PyArray_SHAPE(data);
  }
  catch (const std::exception& e){
    std::cout << e.what() << std::endl;
  }

  // Get the data type for this array:
  auto datatype = PyArray_DTYPE(data);
  std::cout << datatype -> type_num << std::endl;
  std::cout << "numpy double: " << NPY_DOUBLE << std::endl;

  std::cout << "sizeof(shape) is " << sizeof(shape) << std::endl;
  std::cout << "sizeof(double) is " << sizeof(double) << std::endl;

  std::cout << "shape is " <<  * shape << std::endl;

}

} // ulbe

#endif
