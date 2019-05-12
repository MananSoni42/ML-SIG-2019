import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from  skimage import filters, morphology,transform

# lambda function to scale an array to [min,max]
scale = lambda arr,min,max: min + (max-min)*(arr-np.min(arr))/(np.max(arr)-np.min(arr))
# lambda function get a PIL image from an array
get = lambda arr: Image.fromarray(scale(arr,0,255))

# use commmand line arguements for image path
try:
    path = sys.argv[1]
except:
    print('No path specified!')
    print('Usage: python3 get_track.py path_to_image')
    exit(0)

# open image and convert to Black and white (0-255)
im = Image.open(path).convert('L')
# display image
# im.show()
# invert image colors for better results and convert to an array for easier manipulation
im = 255 - np.array(im)

# apply median filter to remove salt and pepper noise
im = filters.median(im)
# use gaussian filter to smooth out the image
im = filters.gaussian(im,sigma=2)
im = scale(im,0,1)
# Reduce thick lines to single pixel lines for easier analysis
final_im =np.array(morphology.skeletonize_3d(im)).astype(int)
# display image
get(final_im).show()
y,x = np.where(final_im==255)
#print(x[0],y[0])
plt.plot(x,y,'ko')
plt.show()
