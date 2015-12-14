# Import all the dependancies:
#
import Image
import matplotlib

# Needed for the process to read data and filter noise:
import larlite
from ROOT import evd, ulbe

# Needed for array manipulation and color mapping:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

import sys
import time


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


folder = "/home/cadams/larlite/UserDev/UbooneLiveBackEnd/outputImages/"

# For informational purposes:
# aspectratio = 5.439709882141784
# targetSize = (int(aspectratio*3456),9600)


# This defines the official microboone color map, in a way usable by
# matplotlib colormaps
cdict = {'red':
         ((0.00000, 022./255, 022./255),
          (0.33333, 000./255, 000./255),
          (0.47000, 076./255, 076./255),
          (0.64500, 000./255, 000./255),
          (0.79100, 254./255, 254./255),
          (1.00000, 255./255, 255./255)),
         'green':
         ((0.00000, 030./255, 030./255),
          (0.33333, 181./255, 181./255),
          (0.47000, 140./255, 140./255),
          (0.64500, 206./255, 206./255),
          (0.79100, 209./255, 209./255),
          (1.00000, 000./255, 000./255)),
         'blue':
         ((0.00000, 151./255, 151./255),
          (0.33333, 226./255, 226./255),
          (0.47000, 043./255, 043./255),
          (0.64500, 024./255, 024./255),
          (0.79100, 065./255, 065./255),
          (1.00000, 000./255, 000./255))
         }


print bcolors.OKBLUE, "Initializing data ...", bcolors.ENDC

# This section creates an instance of the data reader and generates the
# numpy array:

reader = evd.DrawUbSwiz()

# Set up the noise filter and initialize
reader.SetCorrectData(True)
reader.SetSaveData(False)
reader.SetStepSizeByPlane(48, 0)
reader.SetStepSizeByPlane(48, 1)
reader.SetStepSizeByPlane(96, 2)
reader.initialize()

# Give the file to the reader, go to the first event
_file = "/data_linux/noiseStudy/larsoft.root"
reader.setInput(_file)
reader.goToEvent(0)

# # Fetch the data from the reader and transpose it to get the axes correct
_orig_array = reader.getArrayByPlane(2)
_orig_array = np.transpose(_orig_array)


depth = 5

_y_max = 9600 / pow(2, depth)
_x_max = 3456 / pow(2, depth)

# _y_max = 75
# _x_max = 5*27

n_steps_1d = pow(2, depth)


# This takes the data and maps it on to an RGBalpha array
# Create a color map using the official color scheme
ubooneEVCmap = LinearSegmentedColormap('ubooneEVDCmap', cdict)
# register the color map:
plt.register_cmap(cmap=ubooneEVCmap)

# this is a utility that maps arrays into rgb arrays
scalarM = cm.ScalarMappable()
scalarM.set_cmap('ubooneEVDCmap')
# Set the levels of the color mapping:
scalarM.set_clim(vmin=-10, vmax=200)

# This function maps the data onto the color scheme and makes and RGB array
array = scalarM.to_rgba(_orig_array)

array = np.uint8(array*255)

# Now we have the array of wires by time ticks set up.
# Create an image of the data:
base_image = Image.fromarray(array)

# resize it to the top level image:
temp_image = base_image.resize((5*_x_max, int(1*_y_max)), Image.ANTIALIAS)

# Write the base image to file:
temp_image.save(folder+"tile_0_0-0.png")

print bcolors.OKBLUE, "Baselevel image created.   \
  Creating images for level 1 ...", bcolors.ENDC


# Now make some sub images

# We'll need to know the x and y dimensions of the top level image:
_x_dims_original, _y_dims_original = base_image.size

# levelList = [5]
levelList = [2, 3, 4, 5, 6, 7]
for level in levelList:
    print bcolors.OKBLUE, "Making images for level ", level, bcolors.ENDC

    n_steps_1d = pow(2, level)

    # # Take the original image and resize it to encompass the entire picture in this view:
    # _this_base_image = base_image.resize((_x_max*n_steps_1d,_y_max*n_steps_1d),Image.ANTIALIAS)

    _x_crop_dim = _x_dims_original / n_steps_1d
    _y_crop_dim = _y_dims_original / n_steps_1d

    total_images = n_steps_1d**2
    n_equal_signs = 25
    i = 0
    start = time.time()
    for x in xrange(n_steps_1d):
        for y in xrange(n_steps_1d):
            # We're on subregion (x,y) of the image.

            # Crop the original to this area.
            _x_crop_start = x * _x_crop_dim
            _y_crop_start = y * _y_crop_dim
            # Define the area to crop (x, y, x+xlen,y+ylen)
            _crop_area = (_x_crop_start, _y_crop_start,
                          _x_crop_start+_x_crop_dim, _y_crop_start+_y_crop_dim)

            # Make a new, cropped image:
            temp_image = base_image.crop(_crop_area)
            # Resize the image to the right aspect ratio and number of pixels
            temp_image = temp_image.resize(
                (5*_x_max, int(1*_y_max)), Image.ANTIALIAS)

            # Write the base image to file:
            _temp_name = 'tile_{lev}_{row}-{col}.png'.format(
                lev=level, row=x, col=y)
            temp_image.save(folder+_temp_name)
            i += 1

            # Print out a status bar update for each update in x:
            if(i % (5) == 0):
                sys.stdout.write(bcolors.OKGREEN + "\r\t["
                                 + "=" * (n_equal_signs * i / total_images)
                                 + " " *
                                 (n_equal_signs *
                                  (total_images - i) / total_images)
                                 + "]" + str(100*i / total_images) + "%")
                sys.stdout.flush()

    end = time.time()
    sys.stdout.write("\r\t[" + "="*(n_equal_signs) + "] 100%, "
                     + '{0:.2f}'.format(end-start) + " (s) \n"+bcolors.ENDC)
    sys.stdout.flush()
    # print bcolors.ENDC
