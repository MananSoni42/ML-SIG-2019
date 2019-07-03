#!/usr/bin/env python
import numpy as np
from PIL import Image
import sys
import csv
import matplotlib.pyplot as plt
from pprint import pprint

# utility to read csv track files into numpy arrays
def read_track(in_file='sample_path.csv', scale=[1000,600]):
    x = []
    y1 = []
    y2 = []
    with open(in_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(float(row[0]))
            y1.append(float(row[1]))
            y2.append(float(row[2]))
    return scale[0] * np.array(x), scale[1] * np.array(y1), scale[1]*np.array(y2)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class car:
    def __init__(self, track):
        self.eyes = np.array([[0, 1], [0, -1], [1, 1], [1, -1]])
        self.track = track
        self.track_at = lambda x: (
            np.interp(x, self.track[0], self.track[1]),
            np.interp(x, self.track[0], self.track[2]),
        )
        #self.pos = np.array((track[0][0], (track[1][0] + track[2][0]) / 2))
        self.pos = np.array((track[0][0], track[1][0]+10))
        self.max_dist = 1000
        self.accl = np.zeros((2,))
        self.vel = np.zeros((2,))
        self.max_vel = np.array([25, 25])
        self.last_pos = np.zeros((2,))
        self.pos_history = [np.copy(self.pos)]
        self.vel_history = [np.copy(self.vel)]
        self.accl_history = [np.copy(self.accl)]
        self.end = self.track[0][-1]

    def is_legal(self):
        t = self.track_at(self.pos[0])
        return (t[0] < self.pos[1] < t[1]) and (self.pos[0] < self.end)

    def get_rot_mat(self, th):
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def accl_function(self, dists):
        du, dd, _, _ = dists
        #TODO

        # Approach 1
        self.accl = np.array([0.01,(du-dd)/(du+dd)])

        #Approach 2
        """
        if dd > 100:
            self.accl = np.array([0.01,-0.1])
        elif 30 < dd <= 100:
            self.accl = np.array([0.01,5])
        else:
            self.accl = np.array([0.01,5])
        """    
        # Approach 3
        """
        if du > 100:
            self.accl = np.array([0.01,0.2])
        elif 30 < du <= 100:
            self.accl = np.array([0.01,-1])
        else:
            self.accl = np.array([0.01,-3])
        """
    def run(self):
        self.accl_function(self.get_surrounding())
        self.update()

    def update(self):
        self.last_pos = np.copy(self.pos)

        if self.vel[0] > self.max_vel[0]:
            self.vel[0] = self.max_vel[0]
        if self.vel[0] < 0:
            self.vel[0] = 0
        if self.vel[1] > self.max_vel[1]:
            self.vel[1] = self.max_vel[1]
        if self.vel[1] < -self.max_vel[1]:
            self.vel[1] = -self.max_vel[1]

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
        # plt.scatter(self.get_surrounding(),50,color='y')

    def get_surrounding(self):
        if np.linalg.norm(self.vel) != 0:
            v1 = normalize(self.vel)
            rot_matrix = np.array([v1, [-v1[1], v1[0]]]).T
        else:
            rot_matrix = np.eye(2)
        # 2 is the num of dimensions
        rotated_eyes = self.eyes.dot(rot_matrix.T)

        xvals = np.arange(self.track[0].shape[0])
        vision_tensor = rotated_eyes * xvals[:, np.newaxis, np.newaxis]
        vision_tensor += np.expand_dims(self.pos, 1).T

        dists = np.zeros(self.eyes.shape[0])
        for i in range(self.eyes.shape[0]):
            # take upper track if line of vision is above 0
            if self.eyes[i][1] > 0:
                # find point on vision_tensor closest to track
                idx = np.argwhere(np.diff(np.sign(vision_tensor[:,i,1] - self.track[2]))).flatten()
                #plt.scatter(*vision_tensor[idx,i].reshape(2,))

            # else take lower track
            else:
                # find point on vision_tensor closest to track
                idx = np.argwhere(np.diff(np.sign(vision_tensor[:,i,1] - self.track[1]))).flatten()
                #plt.scatter(*vision_tensor[idx,i].reshape(2,))

            try:
                dists[i] = np.sqrt(np.sum(np.square(vision_tensor[idx,i].reshape(2,)-self.pos.reshape(2,))))
            except ValueError:
                dists[i] = self.max_dist
        return dists

"""
track = read_track('test_2.csv')
my_car1 = car(track)
#my_car1.accl = np.array((0.0, 0.0))
#my_car1.vel = np.array((15.0, 0.0))

for i in range(5000):
    my_car1.run()

plt.plot(track[0], track[1], c="k")
plt.plot(track[0], track[2], c="k")
plt.plot(*zip(*my_car1.pos_history))

plt.gca().set_xlim(-10,1010)
plt.gca().set_ylim(-10,1010)
plt.show()
"""
