#!/usr/bin/env python
import numpy as np
from PIL import Image
import sys
from skimage import filters, morphology, transform


def get_track(path="./test_1.jpg"):
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    # lambda function to scale an array to [min,max]
    scale = lambda arr, min, max: min + (max - min) * (arr - np.min(arr)) / (
        np.max(arr) - np.min(arr)
    )
    # lambda function get a PIL image from an array
    get = lambda arr: Image.fromarray(scale(arr, 0, 255))
    # open image and convert to Black and white (0-255)
    im = Image.open(path).convert("L")
    # display image
    # im.show()
    # invert image colors for better results and convert to an array for easier manipulation
    im = 255 - np.array(im)
    # apply median filter to remove salt and pepper noise
    im = filters.median(im)
    # use gaussian filter to smooth out the image
    im = filters.gaussian(im, sigma=2.2)
    im = scale(im, 0, 1)
    # Reduce thick lines to single pixel lines for easier analysis
    # Returns a thresholded image i.e All values are either 255 or 0
    final_im = np.array(morphology.skeletonize_3d(im)).astype(int)
    # Get the x and y co-ordinates (numpy and PIL have transpose conventions)
    y, x = np.where(final_im == 255)
    # Divide the y values into upper and lower parts of the track
    x_final = []
    y_up = []
    y_down = []
    for pt in np.sort(x):
        try:
            y1, y2 = y[np.where(x == pt)]
            x_final.append(pt)
            y_up.append(max(y1, y2))
            y_down.append(min(y1, y2))
        except:
            pass
    # Use a moving average to smooth out the array to remove any minor discontinuities
    # Window size is 1% of track length to minimize the amount of data lost
    win = int(0.01 * len(x_final))
    y_up = running_mean(y_up, win)
    y_down = running_mean(y_down, win)
    x_final = x_final[: -win + 1]
    return x_final, y_up, y_down


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class car:
    eyes = np.array([[0, 1], [0, -1]])

    def __init__(self, track):
        self.track = track
        self.track_at = lambda x: (
            np.interp(x, self.track[0], self.track[1]),
            np.interp(x, self.track[0], self.track[2]),
        )
        self.pos = np.array((track[0][0], (track[1][0] + track[2][0]) / 2))
        self.accl = np.zeros((2,))
        self.vel = np.zeros((2,))
        self.last_pos = np.zeros((2,))
        self.pos_history = [np.copy(self.pos)]
        self.vel_history = [np.copy(self.vel)]
        self.accl_history = [np.copy(self.accl)]
        self.end = self.track[0][-1]

    def is_legal(self):
        t = self.track_at(self.pos[0])
        return (t[0] > self.pos[1] > t[1]) and (self.pos[0] < self.end)

    def update(self):
        self.last_pos = np.copy(self.pos)
        self.pos += self.vel
        if self.is_legal():
            self.vel += self.accl
        else:
            self.pos = self.last_pos
            self.vel = np.zeros((2,))
            self.accl = np.zeros((2,))
        self.pos_history.append(np.copy(self.pos))
        self.vel_history.append(np.copy(self.vel))
        self.accl_history.append(np.copy(self.accl))

    def get_surrounding():
        # rotate eyes
        if np.linalg.norm(self.vel) != 0:
            v1 = normalize(self.vel)
            rot_matrix = np.array([v1, [-v1[0], v1[1]]]).T
        else:
            rot_matrix = np.eye(2)
            # 2 is the num of dimensions
        rotated_eyes = rot_matrix.dot(eyes)
        # TODO get return distances in the direction of eyes
        # TODO add max speed limit


"""
my_car1 =car(get_track())
my_car1.accl = np.array((0, -0.3))
my_car1.vel = np.array((3.0,0))
for i in range(100):
    my_car1.update()
plt.scatter(*zip(*my_car1.pos_history))
plt.show()
"""
