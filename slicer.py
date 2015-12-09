import sys, os

import image_slicer

import time

layers = 5

filein = 'evt.png'

# keep track of total time emlpoyed slicing images
tbegin = time.time()

outpath = '/home/david/Desktop/croft/'
file_prefix = 'tile'

for l in xrange(layers):
    
    image_slicer.slice(filein, 4**(l+1), l+1, True, file_prefix, outpath)
    print 'done slicing level %i -> produced %i images'%(l,4**(l+1))
    os.system('cp %s %s/%s'%(filein,outpath,'tile_0_0_0.png'))

tend = time.time()

dt = tend-tbegin
print 'total time employed: %.02f'%dt
