import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
#plt.style.use('seaborn-pastel')

# lambda function to scale an array to [min,max]
scale = lambda arr,min,max: min + (max-min)*(arr-np.min(arr))/(np.max(arr)-np.min(arr))

def derivative(y,x):
    y = np.insert(y,0,0)
    x = np.insert(x,0,0)
    return np.diff(y)/np.diff(x)

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

x,yu,yd = read_track('test_4_track.csv')
x_new = np.linspace(0,100,x.shape[0])

car_pos = (yu+yd)/2
max_vel = 30
car_vel_y = derivative(car_pos,x_new)
car_vel_x = np.ones(x_new.shape)
car_vel_y[np.isnan(car_vel_y)] = max_vel
car_vel_y[np.isinf(car_vel_y)] = max_vel
car_vel = np.sqrt(np.square(car_vel_y)+np.square(car_vel_x))
car_vel = scale(car_vel,0,1)
#car_acc = derivative(car_vel,x_new)
print(np.min(car_vel),np.max(car_vel))
colors = { (0,0.2): (0,1,0), (0.2,0.4): (0.5,1,0), (0.4,0.6): (1,1,0), (0.6,0.8): (1,0.5,0), (0.8,1): (1,0,0) }

fig = plt.figure()
ax = plt.axes(xlim=(0, 100))
car = patches.Rectangle((0,(yu[0]+yd[0])/2), 1, 5, color='b', alpha=1)
y_u = ax.scatter(x_new,yu,color='k')
y_d = ax.scatter(x_new,yd,color='k')
plots = [car]

def init():
    ax.add_patch(car)
    return plots

def animate(t,x,plots,car_attr,color):
    car = plots[0]
    car_pos,car_vel = car_attr
    car.set_xy((x[t],car_pos[t]))
    for r,c in color.items():
        if r[0] <= car_vel[t] <= r[1]:
            car.set_color(c)
            break
    return plots

anim = FuncAnimation(fig, animate, init_func=init, fargs=(x_new,plots,[car_pos,car_vel],colors),
                               frames=x_new.shape[0], interval=10, blit=False)
plt.show()
