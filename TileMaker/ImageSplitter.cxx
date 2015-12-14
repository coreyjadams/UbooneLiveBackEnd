#ifndef IMAGESPLITTER_CXX
#define IMAGESPLITTER_CXX

#include "ImageSplitter.h"

#include <cmath>

namespace ulbe {

ImageSplitter::ImageSplitter() {
  _base_depth = 4;
  _x_pixels = 256;
  _y_pixels = 256;
  _x_scaling = 1;
  _y_scaling = 1;
  import_array();
}

ImageSplitter::~ImageSplitter() {
  for (auto & image : imageArray) {
    if (image)
      delete image;
  }
}

void ImageSplitter::setDepth(size_t depth) {
  if ( depth < _MAX_DEPTH ) {
    _base_depth = depth;
  }
}

void ImageSplitter::setXPixels(size_t pixVal) {
  if (pixVal < _MAX_PIXELS) {
    _x_pixels = pixVal;
  }
}
void ImageSplitter::setYPixels(size_t pixVal) {
  if (pixVal < _MAX_PIXELS) {
    _y_pixels = pixVal;
  }
}

void ImageSplitter::setXScaling(size_t scaleVal) {
  if (scaleVal < _MAX_SCALING) {
    _x_scaling = scaleVal;
  }
}
void ImageSplitter::setYScaling(size_t scaleVal) {
  if (scaleVal < _MAX_SCALING) {
    _y_scaling = scaleVal;
  }
}

PyObject * ImageSplitter::getImageAtPos(size_t level, size_t x_pos, size_t y_pos) {
  PyObject * returnNull = nullptr;
  if (x_pos < pow(2, _base_depth)) {
    if (y_pos < pow(2, _base_depth) ) {
      // Convert the wire data to numpy arrays:
      int n_dim = 2;
      int * dims = new int[n_dim];
      dims[0] = _y_pixels * _y_scaling;
      dims[1] = _x_pixels * _x_scaling;
      int data_type = PyArray_FLOAT;

      // Get the pointer to vector:
      std::vector<float> * image = imageMap[level][x_pos][y_pos];
      if (image == 0) {
        // std::cout << "image is 0" <<std::endl;
        // Need to make the image.
        genImage(level, x_pos, y_pos);
      }

      size_t _image_index = x_pos + y_pos * pow(2, _base_depth);
      return (PyObject *) PyArray_FromDimsAndData(n_dim, dims, data_type, (char*) &
             ((*imageArray.at(_image_index))[0]) );
    }
  }
  else {
    return returnNull;
  }
}


bool ImageSplitter::acceptInput(const std::vector<float> & inputVec,
                                size_t _x_len_in, size_t _y_len_in) {
  // For this image to work, it has to have at least _x_pixels*_y_pixels*nImages
  // data points
  // size_t min_points = _x_pixels * _y_pixels * pow(2, 2 * _base_depth);
  size_t n_images = pow(2, 2 * _base_depth);
  size_t n_images_per_axes = pow(2, _base_depth);
  size_t n_pixels_per_image = _x_pixels * _y_pixels;  //( x is time ticks, y is wires)

  // Make sure there is enough space to write out these images:
  imageArray.resize(n_images);
  for (auto & image : imageArray) {
    if (image) delete image;
    image = new std::vector<float>(n_pixels_per_image);
  }

  // Parse the image and separate it into the base level images

  for (size_t point = 0; point < inputVec.size(); point++) {
    // Determine where this point goes:
    // These positions are the x and y in the global image
    size_t y_pos = point / _x_len_in;  //purposefully losing the remainder here by dividing size_t/size_t
    size_t x_pos = point % _x_len_in;  //x pos is the remainder

    // Knowing global x and y, map that to which image and the relative x and y
    size_t _y_image_index = y_pos / _y_pixels;
    size_t _x_image_index = x_pos / _x_pixels;

    // Now we know which image it goes into, so figure out what it's location in that index is:
    size_t _image_y_pos = y_pos % _y_pixels;
    size_t _image_x_pos = x_pos % _x_pixels;

    size_t _image_index = _x_image_index + _y_image_index * pow(2, _base_depth);
    size_t _local_image_point = _image_x_pos + _image_y_pos * _x_pixels;

    // Ok, copy the point into it's new image:
    imageArray.at(_image_index) -> at(_local_image_point) = inputVec[point];

  }


  // testSplit();

  return true;

}

void ImageSplitter::testSplit() {
  // for (auto & image : imageArray){
  //   std::cout << "Image length is " << image->size() << std::endl;
  // }
  // size_t i = 0, offset = 0;
  // while (i < 25) {
  //   std::cout << i << ": " << imageArray.at(85)->at(i + offset) << "\n";
  //   i++;
  // }
  // std::cout << "Total number of images: " << imageArray.size() << std::endl;

  std::cout << "x pixels: " << _x_pixels << std::endl;
  std::cout << "y pixels: " << _y_pixels << std::endl;

  size_t offset = 1200;
  size_t start_wire = 0;
  // Print out some test info:
  for (size_t i = start_wire; i < start_wire + 3; i++) {
    std::cout << imageArray[0]->at(offset + 0 + i * _x_pixels) << ", "
              << imageArray[0]->at(offset + 1 + i * _x_pixels) << ", "
              << imageArray[0]->at(offset + 2 + i * _x_pixels) << ", "
              << imageArray[0]->at(offset + 3 + i * _x_pixels) << ", "
              << imageArray[0]->at(offset + 4 + i * _x_pixels) << ", "
              << imageArray[0]->at(offset + 5 + i * _x_pixels) << ", "
              << imageArray[0]->at(offset + 6 + i * _x_pixels) << ", "
              << imageArray[0]->at(offset + 7 + i * _x_pixels) << ", "
              << imageArray[0]->at(offset + 8 + i * _x_pixels) << ", "
              << imageArray[0]->at(offset + 9 + i * _x_pixels) << ", "
              << std::endl;
  }



  return;
}

// void makeTiles(std::string fileName){
//   // To make the files, first we open the existing png:
//   png::image<png::rgb_pixel> image(fileName);
//   image.write("output.png");
// }

// void ImageSplitter::makeTiles(PyObject * array){

//   // Make sure the input is a numpy array:
//   PyArrayObject * data = (PyArrayObject *) (array);

//   // Take in the numpy array for the whole image and then split it up into sub arrays

//   // Get the shape of the array and verify it works:
//   npy_intp * shape;
//   try{
//     shape = PyArray_SHAPE(data);
//   }
//   catch (const std::exception& e){
//     std::cout << e.what() << std::endl;
//   }

//   // Get the data type for this array:
//   auto datatype = PyArray_DTYPE(data);
//   std::cout << datatype -> type_num << std::endl;
//   std::cout << "numpy double: " << NPY_DOUBLE << std::endl;

//   std::cout << "sizeof(shape) is " << sizeof(shape) << std::endl;
//   std::cout << "sizeof(double) is " << sizeof(double) << std::endl;

//   std::cout << "shape is " <<  * shape << std::endl;

// }

void ImageSplitter::genImage(size_t level, size_t x_pos, size_t y_pos) {

  // This function is the core worker function of this class.  It's recursive.
  // When an image is requested at level L, do the following:
  // if L == _base_depth, generate the image with any necessary scaling
  // if L > _base_depth, use the image at L-1 to generate the image and scale to the correct size
  // if L < _base_depth, use the 4 images at L+1 that can combine to form the image, and scale.
  // For the second two options, check if the needed images exist.  If not, call genImage.

  // Don't need this since size_t >= 0 always
  // // Don't allow the level to go beyond 0:
  // if (level < 0)
  //   return;

  // Don't allow it beyond max depth:
  if (level > _MAX_DEPTH)
    return;

  // Return immediately if the needed image already exists:
  if (imageMap[level][x_pos][y_pos] != 0) {
    return;
  }

  // Check the base level:
  if (level == _base_depth) {
    if (imageMap[level][x_pos][y_pos] == 0) {
      // initialize the vector:
      imageMap[level][x_pos][y_pos] = new std::vector<float>(_x_pixels * _y_pixels * _x_scaling * _y_scaling);
      // Populate the vector:
      size_t N = pow(2, _base_depth);
      scaleInPlace(imageArray[x_pos + y_pos * N], imageMap[level][x_pos][y_pos]);

    }
    return;
  }
  // Handle higher up depths
  if (level < _base_depth) {
    // In this case, we're asking for something above the _base_depth
    // This means we have to merge 4 images together
    std::vector<std::vector<float> * > sourceImages;
    // If we're asking for x,y in level L, we need (2x,2y),(2x+1,2y),(2x,2y+1),(2x+1,2y+1)
    // Make sure these images exist:
    if (imageMap[level + 1][2 * x_pos][2 * y_pos] == 0) {
      genImage(level + 1, 2 * x_pos, 2 * y_pos);
    }
    sourceImages.push_back(imageMap[level + 1][2 * x_pos][2 * y_pos]);

    if (imageMap[level + 1][2 * x_pos + 1][2 * y_pos] == 0) {
      genImage(level + 1, 2 * x_pos + 1, 2 * y_pos);
    }
    sourceImages.push_back(imageMap[level + 1][2 * x_pos + 1][2 * y_pos]);

    if (imageMap[level + 1][2 * x_pos][2 * y_pos + 1] == 0) {
      genImage(level + 1, 2 * x_pos, 2 * y_pos + 1);
    }
    sourceImages.push_back(imageMap[level + 1][2 * x_pos][2 * y_pos + 1]);

    if (imageMap[level + 1][2 * x_pos + 1][2 * y_pos + 1] == 0) {
      genImage(level + 1, 2 * x_pos + 1, 2 * y_pos + 1);
    }
    sourceImages.push_back(imageMap[level + 1][2 * x_pos + 1][2 * y_pos + 1]);

    // Now that all of the inputs are here, initialize the output:
    imageMap[level][x_pos][y_pos] = new std::vector<float>(_x_pixels * _y_pixels * _x_scaling * _y_scaling);

    mergeAndScale(sourceImages, imageMap[level][x_pos][y_pos]);
  }
  // Handle lower down depths:
  if (level > _base_depth) {

    // Make sure the image above this one exists:
    genImage(level - 1, x_pos / 2, y_pos / 2);
    auto inputVec = imageMap[level - 1][x_pos / 2][y_pos / 2];

    // For this instance, we can create all 4 subimages in one pass.
    // So do it.
    std::vector<std::vector<float> *> outputVecs;
    imageMap[level][2 * (x_pos / 2)][2 * (y_pos / 2)]
      = new std::vector<float>(_x_pixels * _y_pixels * _x_scaling * _y_scaling);
    outputVecs.push_back(imageMap[level][2 * (x_pos / 2)][2 * (y_pos / 2)]);

    imageMap[level][2 * (x_pos / 2) + 1][2 * (y_pos / 2)]
      = new std::vector<float>(_x_pixels * _y_pixels * _x_scaling * _y_scaling);
    outputVecs.push_back(imageMap[level][2 * (x_pos / 2) + 1][2 * (y_pos / 2)]);

    imageMap[level][2 * (x_pos / 2)][2 * (y_pos / 2) + 1]
      = new std::vector<float>(_x_pixels * _y_pixels * _x_scaling * _y_scaling);
    outputVecs.push_back(imageMap[level][2 * (x_pos / 2)][2 * (y_pos / 2) + 1]);

    imageMap[level][2 * (x_pos / 2) + 1][2 * (y_pos / 2) + 1]
      = new std::vector<float>(_x_pixels * _y_pixels * _x_scaling * _y_scaling);
    outputVecs.push_back(imageMap[level][2 * (x_pos / 2) + 1][2 * (y_pos / 2) + 1]);

    spiltAndScale(inputVec, outputVecs);
  }


}

void ImageSplitter::scaleInPlace(std::vector<float> * inputVec, std::vector<float> * outputVec) {
  // Loop over every point of the input vector and decide where in the output vector it goes
  for (size_t point = 0; point < inputVec -> size(); point ++) {
    // Take this point and decide what it's x and y positions are in the original vector:
    size_t x, y;
    x = point % _x_pixels;
    y = point / _x_pixels;
    // Write this point to the output vector with the correct scaling.
    // Since x points are adjacent in memory loop over y on the outside
    size_t x_new, y_new;
    x_new = x * _x_scaling;
    y_new = y * _y_scaling;
    size_t len_new = _x_pixels * _x_scaling;

    // outputVec -> at(point) = inputVec->at(point);
    for (size_t y_scale = 0; y_scale < _y_scaling; y_scale ++) {
      for (size_t x_scale = 0; x_scale < _x_scaling; x_scale ++) {
        outputVec -> at(x_new + x_scale + (y_new + y_scale)*len_new) = inputVec->at(point);
        if (point == 0 || point == 1200) {
          std::cout << "Mapping " << point << " to "
                    << x_new << " + " << x_scale << " + " << "( "
                    << y_new << " + " << y_scale << ") * " << len_new << " = "
                    << x_new + x_scale + (y_new + y_scale)*len_new
                    << "\n\t(" << x << "," << y << ") to (" << x_new + x_scale << ", " << y_new + y_scale << ")"
                    << std::endl;
        }
      }
    }
    // if (point == 1200) {
    //   exit(-1);
    // }

  }
  return;
}

void ImageSplitter::spiltAndScale(std::vector<float> * inputVec,
                                  std::vector<std::vector<float>*> outputVecs) {
  // Take the image from input Vec and split it into 4 images, scaled by 2 in x and y
  // Again, loop over the original vector and figure out where we are,
  // then write that to the correct output vector

  for (size_t point = 0; point < inputVec -> size(); point ++) {
    // Take this point and decide what it's x and y positions are in the original vector:
    size_t x, y;
    x = point % _x_pixels;
    y = point / _x_pixels;

    size_t out_index;
    // Determine which output image this goes to:
    if (x < _x_pixels / 2) {
      if (y < _y_pixels / 2) {
        out_index = 0;
      }
      else {
        out_index = 2;
      }
    }
    else {
      if (y < _y_pixels / 2) {
        out_index = 1;
      }
      else {
        out_index = 3;
      }
    }

    x = point % (_x_pixels / 2);
    y = point / (_x_pixels / 2);
    // Write this point to the output vector with the correct scaling.
    // Since x points are adjacent in memory loop over y on the outside
    for (size_t y_scale = 0; y_scale <  2; y_scale ++) {
      for (size_t x_scale = 0; x_scale <  2; x_scale ++) {
        outputVecs[out_index] -> at(x + x_scale + (_y_pixels * _y_scaling) * (y + y_scale))
          = inputVec->at(point);
      }
    }
  }
  return;
}

void ImageSplitter::mergeAndScale(std::vector<std::vector<float> * > inputVecs,
                                  std::vector<float> * outputVec) {
  std::cout << "Entering merge and scale" << std::endl;
  // This one, at least, is easy.  We take the input vectors and simply average them.
  // Then scale the output vector as needed.
  for (size_t point = 0; point < inputVecs.front()->size(); point ++) {
    float result = 0.25 * (inputVecs[0]->at(point)
                           + inputVecs[1]->at(point)
                           + inputVecs[2]->at(point)
                           + inputVecs[3]->at(point)
                          );

    size_t x, y;
    x = point % _x_pixels;
    y = point / _x_pixels;

    // Write this point to the output vector
    outputVec -> at(x  + (_y_pixels) * (y)) = result;

  }
  return;
}

} // ulbe

#endif
