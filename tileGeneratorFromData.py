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

# Not informational:
# Define the maximum x and y distances:
_x_max = 1175
# _x_max = 500
_y_max = 600
# _y_max = 256

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
# _orig_array = reader.getArrayByPlane(2)
# _orig_array = np.transpose(_orig_array)


print bcolors.OKBLUE, "Done reading data, pass it to splitter", bcolors.ENDC

depth = 3

_y_max = 9600 / pow(2, depth)
_x_max = 3456 / pow(2, depth)

print(_x_max, _y_max)

scanner = ulbe.ImageSplitter()
scanner.setDepth(depth)
scanner.setXPixels(_y_max)
scanner.setYPixels(_x_max)
scanner.setXScaling(1)
scanner.setYScaling(5)
data = reader.getArrayByPlane(2)
vector = reader.getDataByPlane(2)
# Print out a few pieces of the data to test the slicing:
# [wire][timetick]
# start_wire = 0
# wire = 1
# time = 500
# offset = time + 9600*wire
# print data.shape
# print vector.size()
# print data[wire][time+0]," vs ",vector[offset+0]
# print data[wire][time+1]," vs ",vector[offset+1]
# print data[wire][time+2]," vs ",vector[offset+2]
# print data[wire][time+3]," vs ",vector[offset+3]
# print data[wire][time+4]," vs ",vector[offset+4]

# offset = 0
# for w in xrange(3):
#     wtemp = w+start_wire
#     print(data[wtemp][offset],
#           data[wtemp][offset+1],
#           data[wtemp][offset+2],
#           data[wtemp][offset+3],
#           data[wtemp][offset+4],
#           data[wtemp][offset+5],
#           data[wtemp][offset+6],
#           data[wtemp][offset+7],
#           data[wtemp][offset+8],
#           data[wtemp][offset+9])


if not scanner.acceptInput(reader.getDataByPlane(2), 9600, 3456):
    print("Something went wrong reading in the data to the splitter.")
    exit()


print bcolors.OKBLUE, "Success splitting data at level ", depth, bcolors.ENDC


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

# # This function
# array = scalarM.to_rgba(_orig_array)

# array = np.uint8(array*255)

# # Now we have the array of wires by time ticks set up.
# # Create an image of the data:
# base_image = Image.fromarray(array)

# # resize it to the top level image:
# temp_image = base_image.resize((_x_max,_y_max),Image.ANTIALIAS)

# # Write the base image to file:
# temp_image.save(folder+"tile_0_0-0.png")

# print bcolors.OKBLUE, "Baselevel image created.  Creating images for
# level 1 ...", bcolors.ENDC


# Now make some sub images

# We'll need to know the x and y dimensions of the top level image:
# _x_dims_original, _y_dims_original = base_image.size

levelList = range(depth+1)
levelList.reverse()
print levelList
for level in levelList:
    print "Making images for level ", level
    # Divide the image up into 2^(level + 1) regions.
    # For each region, crop the image and save a copy.
    n_steps_1d = 2**(level)
    for x in xrange(n_steps_1d):
        for y in xrange(n_steps_1d):
            image = scanner.getImageAtPos(level, x, y)
            image = np.uint8(255*scalarM.to_rgba(np.transpose(image)))
            # image = np.uint8(255*scalarM.to_rgba(image))
            im = Image.fromarray(image)
            # pixels = im.load()
            # print pixels[5,5]
            im.save(
                folder+"tile_{lev}_{x}-{y}.png".format(lev=level, x=x, y=y))
        print "Finished x row ", x, " in level ", level

    # # _x_crop_size = _x_dims_original / n_steps_1d
    # # _y_crop_size = _y_dims_original / n_steps_1d
    # _x_crop_size = _x_max / n_steps_1d
    # _y_crop_size = _y_max / n_steps_1d

    # # Take the original image and resize it to encompass the entire picture in this view:
    # # _this_base_image = base_image.resize((_x_max*n_steps_1d,_y_max*n_steps_1d),Image.ANTIALIAS)

    # # print _this_base_image.size

    # # # Work with the original numpy array to crop and down-sample
    # # Nbig_x,Nbig_y = _orig_array.shape
    # # Nsmall_x = _x_max * (level + 1)
    # # Nsmall_y = _y_max * (level + 1)
    # # _small_array = _orig_array.reshape([Nsmall_x, Nbig_x/Nsmall_x, Nsmall_y, Nbig_y/Nsmall_y]).mean(3).mean(1)

    # # print _small_array.shape
    # # break

    # for x in xrange(n_steps_1d):
    #   for y in xrange(n_steps_1d):
    #     # We're on subregion (x,y) of the image.

    #     # Crop the original to this area.
    #     # _x_crop_start = x * _x_crop_size
    #     # _y_crop_start = y * _y_crop_size
    #     _x_crop_start = x * _x_max
    #     _y_crop_start = y * _y_max
    #     # Define the area to crop (x, y, x+xlen,y+ylen)
    #     # _crop_area = (_x_crop_start,_y_crop_start,_x_crop_start+_x_crop_size,_y_crop_start+_y_crop_size)
    #     _crop_area = (_x_crop_start,_y_crop_start,_x_crop_start+_x_max,_y_crop_start+_y_max)
    #     # Crop the image
    #     temp_image = _this_base_image.crop(_crop_area)
    #     # Resize the image to the right aspect ratio and number of pixels
    #     # temp_image = temp_image.resize((_x_max,_y_max),Image.ANTIALIAS)

    #     # Write the base image to file:
    #     _temp_name = 'tile_{lev}_{row}-{col}.png'.format(lev=level+1,row=x,col=y)
    #     print bcolors.OKBLUE, "Saving file", _temp_name, " ...", bcolors.ENDC

    #     temp_image.save(folder+_temp_name)
    # print bcolors.OKBLUE, "Done saving file", _temp_name, ".", bcolors.ENDC
