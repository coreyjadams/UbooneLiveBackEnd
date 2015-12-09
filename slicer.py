import sys, os

import image_slicer

import time

layers = 6

filein = 'evt.png'

# keep track of total time emlpoyed slicing images
tbegin = time.time()

for l in xrange(layers):
    
    image_slicer.slice(filein, 4**(l+1), l, True)
    print 'done slicing level %i -> produced %i images'%(l,4**(l+1))

tend = time.time()

dt = tend-tbegin
print 'total time employed: %.02f'%dt
