import ROOT
import larlite
from ROOT import evd, ulbe
import numpy as np

aspectratio = 5.439709882141784

targetSize = (int(aspectratio*3456),9600)

print targetSize

reader = evd.DrawUbSwiz()

# Set up the noise filter and initialize
reader.SetCorrectData(True)
reader.SetSaveData(False)
reader.SetStepSizeByPlane(48, 0)
reader.SetStepSizeByPlane(48, 1)
reader.SetStepSizeByPlane(96, 2)
reader.initialize()

_file = "/data_linux/noiseStudy/larsoft.root"
reader.setInput(_file)
reader.goToEvent(0)

print "Done reading data, pass it to splitter"

scanner = ulbe.ImageSplitter()
scanner.setDepth(5)
scanner.setXPixels(300)
scanner.setYPixels(108)
if not scanner.acceptInput(reader.getDataByPlane(2),9600,3456):
    print ("Something went wrong reading in the data to the splitter.")
    exit()

print "Success!"

# Try making some images:

# array = reader.getArrayByPlane(2)
# array = np.transpose(array)

# print array.max()
# print array[0][0]
# print array.min()

# # # Scale the array to between 0 and 1:
# # array = array - array.min()
# # print array.max()
# # print array[0][0]
# # print array.min()

# # array = array * (1.0 / array.max())
# # print array.max()
# # print array[0][0]
# # print array.min()

# Try with matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# cdict_basic = {'red':   ((0.0, 0.0, 0.0),
#                    (0.5, 0.0, 0.1),
#                    (1.0, 1.0, 1.0)),

#          'green': ((0.0, 0.0, 0.0),
#                    (1.0, 0.0, 0.0)),

#          'blue':  ((0.0, 0.0, 1.0),
#                    (0.5, 0.1, 0.0),
#                    (1.0, 0.0, 0.0))
#         }

cdict = { 'red':  
               ((0.00000, 022./255,022./255),
                (0.33333, 000./255,000./255),
                (0.47000, 076./255,076./255),
                (0.64500, 000./255,000./255),
                (0.79100, 254./255,254./255),
                (1.00000, 255./255,255./255)),
          'green':
               ((0.00000, 030./255,030./255),
                (0.33333, 181./255,181./255),
                (0.47000, 140./255,140./255),
                (0.64500, 206./255,206./255),
                (0.79100, 209./255,209./255),
                (1.00000, 000./255,000./255)),
          'blue':
               ((0.00000, 151./255,151./255),
                (0.33333, 226./255,226./255),
                (0.47000, 043./255,043./255),
                (0.64500, 024./255,024./255),
                (0.79100, 065./255,065./255),
                (1.00000, 000./255,000./255))
}
            # {'ticks': [(0, (22, 30, 151, 255)),
            #            (0.33333, (0, 181, 226, 255)),
            #            (0.47, (76, 140, 43, 255)),
            #            (0.645, (0, 206, 24, 255)),
            #            (0.791, (254, 209, 65, 255)),
            #            (1, (255, 0, 0, 255))],

blue_red1 = LinearSegmentedColormap('ubooneEVDCmap', cdict)
plt.register_cmap(cmap=blue_red1)

from matplotlib import cm

a = cm.ScalarMappable()
a.set_cmap('ubooneEVDCmap')
a.set_clim(vmin=-10,vmax=200)

# Now make images:

import Image

for x in xrange(16):
    for y in xrange(16):
        image = scanner.getImageAtPos(6, x,y)
        # image = scanner.getImageAtPos(5, x,y)
        image = np.uint8(255*a.to_rgba(np.transpose(image)))
        im = Image.fromarray(image)
        # pixels = im.load()
        # print pixels[5,5]
        im.save("test_6_{x}-{y}.png".format(x=x,y=y))
    print "Finished x row ", x

# output = a.to_rgba(array)


# output = np.uint8(output*255)



# import Image
# im = Image.fromarray(output)

# # Fix the aspect ratio:
# # print im.shape
# im = im.resize((18800/4,9600/4),Image.ANTIALIAS)

# im.save("test2.png")