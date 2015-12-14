/**
 * \file ImageSplitter.h
 *
 * \ingroup TileMaker
 *
 * \brief Class def header for a class ImageSplitter
 *
 * @author cadams
 */

/** \addtogroup TileMaker

    @{*/
#ifndef IMAGESPLITTER_H
#define IMAGESPLITTER_H

#include <iostream>
#include <vector>
#include <map>

// struct _object;
// typedef _object PyObject;

// #include "png.hpp"


// #ifndef __CINT__
#include "Python.h"
#include "numpy/arrayobject.h"
// #endif

namespace ulbe {


class color {
public:
  size_t red;
  size_t green;
  size_t blue;
};


/**
   \class ImageSplitter
   User defined class ImageSplitter ... these comments are used to generate
   doxygen documentation!
 */
class ImageSplitter {

public:

  /// Default constructor
  ImageSplitter();

  /// Default destructor
  ~ImageSplitter();

  void setDepth(size_t);

  void setXPixels(size_t);
  void setYPixels(size_t);

  void setXScaling(size_t);
  void setYScaling(size_t);

  bool acceptInput(const std::vector<float> &, size_t x_len, size_t y_len);

  PyObject * getImageAtPos(size_t level, size_t x_pos, size_t y_pos);

private:

  // Need to define the output tile pixel sizes.  No need to constrain x and y to match:
  size_t _x_pixels;
  size_t _y_pixels;

  size_t _x_scaling;
  size_t _y_scaling;

  size_t _base_depth;

  size_t _depth;

  const static size_t _MAX_DEPTH = 8;
  const static size_t _MAX_PIXELS = 2000;
  const static size_t _MAX_SCALING = 6;

  const color & getColor(float);

  // Need to define the color scale here:
  float colorMin, colorMax;
  std::vector<float> colorAnchors;
  std::vector<color> colorPoints;

  void testSplit();

  void genImage(size_t level, size_t x_pos, size_t y_pos);

  // This is an array of the images at native resolution, split into
  // sections of _x_pixels by _y_pixels
  // If it isn't an even division, it's padded on the ends
  // the length of the vector is 2**(2*_depth) so it grows quite quickly
  // Access the image at (x,y) coordinates in depth level (L) with
  // the index imageArray[x + y*2**(L)]
  std::vector<std::vector<float> * > imageArray;

  // This map contains the actual images to be exported.
  // On access, it will generate it from other entries in the map recursively.
  std::map<int, std::map<int, std::map<int, std::vector<float>*> > > imageMap;

  // Worker functions to generate images from already produced images:
  void mergeAndScale(std::vector<std::vector<float> * > inputVecs,
                     std::vector<float> * outputVec);
  void spiltAndScale(std::vector<float> * inputVec,
                     std::vector<std::vector<float>*> outputVecs);
  void scaleInPlace(std::vector<float> * inputVec, std::vector<float> * outputVec);

};

} // ulbe


#endif
/** @} */ // end of doxygen group

