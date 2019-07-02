#!/usr/bin/env python
import numpy as np
from PIL import Image
import sys
import csv
import matplotlib.pyplot as plt
from pprint import pprint

# utility to read csv track files into numpy arrays
def read_track(in_file="./test2.csv", scale=1000):
    x = []
    y1 = []
    y2 = []
    with open(in_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(float(row[0]))
            y1.append(float(row[1]))
            y2.append(float(row[2]))
    return scale * np.array(x), scale * np.array(y1), scale * np.array(y2)


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
        self.pos = np.array((track[0][0], (track[1][0] + track[2][0]) / 2))
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
        # rotate eyes
        if np.linalg.norm(self.vel) != 0:
            v1 = normalize(self.vel)
            rot_matrix = np.array([v1, [-v1[1], v1[0]]]).T
        else:
            rot_matrix = np.eye(2)
            # 2 is the num of dimensions
        rotated_eyes = self.eyes.dot(rot_matrix.T)
        # rotated_eyes_on_car = rotated_eyes + np.expand_dims(self.pos, 1).T
        # pprint(rotated_eyes_on_car)
        eye_ray = np.arange(50) * 30 + 1
        vision_tensor = rotated_eyes * eye_ray[:, np.newaxis, np.newaxis]
        vision_tensor += np.expand_dims(self.pos, 1).T
        # pprint(vision_tensor)
        # pprint(vision_tensor.reshape(12, 2))
        # https://stackoverflow.com/questions/28766692/intersection-of-two-graphs-in-python-find-the-x-value
        plt.scatter(*zip(*vision_tensor.reshape((200, 2))))
        plt.scatter(*self.pos)
        # pprint(vision_tensor[:, :, :1])
        # I know that the names are getting a bit fancy :P
        # pprint(self.track_at(vision_tensor[:, :, :1]) - vision_tensor[:, :, 1:])

        pprint(
            np.sign(self.track_at(vision_tensor[:, :, :1]) - vision_tensor[:, :, 1:])
        )
        # pprint(
        #     np.argwhere(
        #         np.diff(
        #             np.sign(
        #                 self.track_at(vision_tensor[:, :, :1]) - vision_tensor[:, :, 1:]
        #             )
        #         )
        #     ).flatten()
        # )
        # rot_mat = self.get_rot_mat(90)

        # if np.linalg.norm(self.vel) == 0.0:
        #     print("--")
        #     dist = self.track_at(self.pos)
        # else:
        # upper track
        # p1 = self.pos
        # p2 = self.pos + (np.dot(rot_mat, self.vel.T)).T
        # # print(p1,p2)
        # p3 = np.vstack([track[0], track[2]]).T

        # dist = np.abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))
        # return track[0][np.argmin(dist)]


track = read_track("./test2.csv", scale=30000)
my_car1 = car(track)
my_car1.accl = np.array((0.0, 0.0))
my_car1.vel = np.array((15.0, 0.0))

# for i in range(100):
#     my_car1.update()

my_car1.get_surrounding()

"""
for i in range(np.array(my_car1.pos_history)[:,0].shape[0]):
        plt.arrow(np.array(my_car1.pos_history)[i,0],np.array(my_car1.pos_history)[i,1],
            np.array(my_car1.vel_history)[i,0],np.array(my_car1.vel_history)[i,1],color='r')
"""

plt.scatter(track[0], track[1], c="k")
plt.scatter(track[0], track[2], c="k")
# plt.scatter(*zip(*my_car1.vision_tensor.reshape((200, 2))))
# plt.plot(*zip(*my_car1.vision_tensor[1]))
plt.show()
