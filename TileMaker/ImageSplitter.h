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


struct _object;
typedef _object PyObject;

// #include "png.hpp"


// #ifndef __CINT__
#include "Python.h"
#include "numpy/arrayobject.h"
// #endif

namespace ulbe {


class color{
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
class ImageSplitter{

public:

  /// Default constructor
  ImageSplitter(){}

  /// Default destructor
  ~ImageSplitter(){}

  void makeTiles(PyObject *);

  // void makeTiles(std::string fileName);

  void setDepth(size_t);

  void setXPixels(size_t);
  void setYPixels(size_t);


private:

  // Need to define the output tile pixel sizes.  No need to constrain x and y to match:
  size_t _x_pixels;
  size_t _y_pixels;

  size_t _depth;

  const static size_t _MAX_DEPTH = 5;
  const static size_t _MAX_PIXELS = 500;

  const color & getColor(float);

  // Need to define the color scale here:
  float colorMin, colorMax;
  std::vector<float> colorAnchors;
  std::vector<color> colorPoints;
};

} // ulbe


#endif
/** @} */ // end of doxygen group 

