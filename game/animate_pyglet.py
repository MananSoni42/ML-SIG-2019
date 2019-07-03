import pyglet
import time
import csv
import numpy as np
from scipy.interpolate import griddata

class car_animation(pyglet.window.Window):
    """
    animate_car:
        Class to display an animation of cars

    Arguements:
        cars (list): list of objects of class car defined in car.py
        car_paths (list)(None): list of paths to image to use for each car
        track_path (list)(sample_path.csv): path to track file to use(csv)
                    generated using get_track.py
    """
    def __init__(self, cars, track_path='sample_path.csv',car_paths=None):

        pyglet.window.Window.__init__(self, width=1000, height=600, resizable = True)

        # get number of cars
        num_cars = len(cars)

        self.num_cars = num_cars

        # assign default car for all cars if not specified
        if car_paths is None:
            car_paths = []
            for i in range(self.num_cars):
                car_paths.append('cars/car_yellow.png')

        self.total_time = 10
        self.fps = 100
        self.time = 0
        self.width = 1000
        self.height = 600

        self.drawableObjects = []
        self.track = None
        self.carSprite = []
        self.get_pos(cars)
        self.createDrawableObjects(car_paths,track_path)

    # interpolate given positions to get proper fps
    def get_pos(self, cars):
        self.pos_x = np.zeros((self.num_cars,self.total_time*self.fps))
        self.pos_y = np.zeros((self.num_cars,self.total_time*self.fps))
        for i in range(self.num_cars):
            x = np.array(cars[i].pos_history)[:,0]
            y = np.array(cars[i].pos_history)[:,1]

            self.pos_x[i] = griddata(np.linspace(1,x.shape[0],x.shape[0]), x,
                                    np.linspace(1,x.shape[0],self.total_time*self.fps),
                                    method='linear')

            self.pos_y[i] = griddata(np.linspace(1,y.shape[0],y.shape[0]), y,
                                    np.linspace(1,y.shape[0],self.total_time*self.fps),
                                    method='linear')


    # utility to read csv track files into numpy arrays
    def read_track(self,in_file='tracks/ample_path.csv',scale=[1000,600]):
        x = []
        y1 = []
        y2 = []
        with open(in_file,'r') as f:
            reader = csv.reader(f)
            for row in reader:
                x.append(float(row[0]))
                y1.append(float(row[1]))
                y2.append(float(row[2]))
        return scale[0]*np.array(x), scale[1]*np.array(y1), scale[1]*np.array(y2)

    # create the track object in pyglet
    def get_track(self,track_path):
        x,y1,y2 = self.read_track(track_path)
        self.track = pyglet.graphics.Batch()
        for i in range(x.shape[0]-1):
            self.track.add(2, pyglet.gl.GL_LINES, None,
                                ('v2f', (x[i],y1[i],x[i+1],y1[i+1])),
                                ('c3B', (0, 0, 0, 0, 0, 0)))
            self.track.add(2, pyglet.gl.GL_LINES, None,
                                ('v2f', (x[i],y2[i],x[i+1],y2[i+1])),
                                ('c3B', (0, 0, 0, 0, 0, 0)))

    # create the car(s) and add track to draw on every frame
    def createDrawableObjects(self,car_paths,track_path):
        """
        Create objects that will be drawn within the
        window.
        """
        # Add track
        self.get_track(track_path)
        self.drawableObjects.append(self.track)

        # Add cars
        for i in range(self.num_cars):
            car_img = pyglet.image.load(car_paths[i])
            car_img.anchor_x = car_img.width // 2
            car_img.anchor_y = car_img.height // 2
            carSprite = pyglet.sprite.Sprite(car_img)
            carSprite.scale_x = float(50/car_img.width)
            carSprite.scale_y = float(50/car_img.height)
            carSprite.position = (0,100)
            self.carSprite.append(carSprite)
            self.drawableObjects.append(carSprite)

    # draw any frame
    def on_draw(self):
        self.clear()
        for d in self.drawableObjects:
            d.draw()

    # update frames
    def update(self,dt):
        if self.time < self.fps*self.total_time:
            for i in range(self.num_cars):
                self.carSprite[i].x = self.pos_x[i,self.time]
                self.carSprite[i].y = self.pos_y[i,self.time]
            self.time += 1

def animate_cars(cars,track_path=None,car_paths=None):
    win = car_animation(cars,track_path=track_path,car_paths=car_paths)
    pyglet.gl.glClearColor(1, 1, 1, 1)

    pyglet.clock.schedule_interval(win.update, 1/win.fps)
    pyglet.app.run()

#animate_cars(num_cars=5,acc_x=4*(np.random.rand(5,100)-0.5),acc_y=0.5*(np.random.rand(5,100)-0.5))
