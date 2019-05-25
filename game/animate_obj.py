import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
#plt.style.use('seaborn-pastel')

# lambda function to scale an array to [min,max]
scale = lambda arr,min,max: min + (max-min)*(arr-np.min(arr))/(np.max(arr)-np.min(arr))

def read_track(in_file):
    x = []
    y1 = []
    y2 = []
    with open(in_file,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(float(row[0]))
            y1.append(float(row[1]))
            y2.append(float(row[2]))
    return np.array(x), np.array(y1), np.array(y2)

x,yu,yd = read_track('sample_path_track.csv')
x_new = np.linspace(0,100,x.shape[0])
car_pos = [50]*x_new.shape[0]
fig = plt.figure()
ax = plt.axes(xlim=(0, 100))

car = patches.Rectangle((0,50), 5, 10, color='b', alpha=1)
car_pos = [(yu[i]+yd[i])/2 for i in range(x_new.shape[0])]
y_u = ax.scatter(x_new,yu,color='k')
y_d = ax.scatter(x_new,yd,color='k')
plots = [car]

def init():
    ax.add_patch(car)
    return plots

def animate(t,x,plots,car_pos):
    car = plots[0]
    car.set_xy((x[t],car_pos[t]))
    return plots

anim = FuncAnimation(fig, animate, init_func=init, fargs=(x_new,plots,car_pos),
                               frames=x_new.shape[0], interval=10, blit=False)
plt.show()
