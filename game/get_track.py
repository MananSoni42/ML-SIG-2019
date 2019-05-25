import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from  skimage import filters, morphology,transform
import csv

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# lambda function to scale an array to [min,max]
scale = lambda arr,min,max: min + (max-min)*(arr-np.min(arr))/(np.max(arr)-np.min(arr))
# lambda function get a PIL image from an array
get = lambda arr: Image.fromarray(scale(arr,0,255))

# use commmand line arguements for image path
try:
    path = sys.argv[1]
    out_file = sys.argv[2]
except:
    print('No path specified!')
    print('Usage: python3 get_track.py path_to_image out_file')
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
im = filters.gaussian(im,sigma=2.2)
im = scale(im,0,1)
# Reduce thick lines to single pixel lines for easier analysis
# Returns a thresholded image i.e All values are either 255 or 0
final_im =np.array(morphology.skeletonize_3d(im)).astype(int)

# Get the x and y co-ordinates (numpy and PIL have transpose conventions)
y,x = np.where(final_im==255)

# Divide the y values into upper and lower parts of the track
x_final = []
y_up = []
y_down = []
for pt in np.sort(x):
    try:
        y1,y2 = y[np.where(x==pt)]
        x_final.append(pt)
        y_up.append(max(y1,y2))
        y_down.append(min(y1,y2))
    except:
        pass

# Use a moving average to smooth out the array to remove any minor discontinuities
# Window size is 1% of track length to minimize the amount of data lost
win = int(0.01*len(x_final))
y_up = running_mean(y_up,win)
y_down = running_mean(y_down,win)
x_final = x_final[:-win+1]

# write the results to a csv file
with open(out_file,'w') as f:
    writer = csv.writer(f)
    for i in range(len(x_final)):
        writer.writerow([x[i],y_down[i],y_up[i]])

# Plot the track using matplotlib
plt.title('Visualization of track')
plt.scatter(x_final,y_down,color='r')
plt.scatter(x_final,y_up,color='b')
plt.show()
