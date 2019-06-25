import pyglet
import time
import csv
import numpy as np

class car_animation(pyglet.window.Window):
    def __init__(self, width=1000, height=600):
        pyglet.window.Window.__init__(self,
                        width=width,
                        height=height,
                        resizable = True)
        self.drawableObjects = []
        self.track = None
        self.carSprite = None
        self.createDrawableObjects()
        #self.adjustWindowSize()

    # utility to read csv track files into numpy arrays
    def read_track(self,in_file='sample_path.csv'):
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

    # create the track object in pyglet
    def get_track(self,track_path='sample_path.csv'):
        x,y1,y2 = self.read_track(track_path)
        x = (1000*x).astype(int)
        y1 = (600*y1).astype(int)
        y2 = (600*y2).astype(int)
        self.track = pyglet.graphics.Batch()
        for i in range(x.shape[0]-1):
            self.track.add(2, pyglet.gl.GL_LINES, None,
                                ('v2f', (x[i],y1[i],x[i+1],y1[i+1])),
                                ('c3B', (0, 0, 0, 0, 0, 0)))
            self.track.add(2, pyglet.gl.GL_LINES, None,
                                ('v2f', (x[i],y2[i],x[i+1],y2[i+1])),
                                ('c3B', (0, 0, 0, 0, 0, 0)))

    def createDrawableObjects(self):
        """
        Create objects that will be drawn within the
        window.
        """
        # Add track
        self.get_track()
        self.drawableObjects.append(self.track)

        # Add car
        car_img = pyglet.image.load('car.png')
        car_img.anchor_x = car_img.width // 2
        car_img.anchor_y = car_img.height // 2

        self.carSprite = pyglet.sprite.Sprite(car_img)
        self.carSprite.position = ( self.carSprite.width,
                                    self.carSprite.height)
        self.drawableObjects.append(self.carSprite)

    def on_draw(self):
        self.clear()
        for d in self.drawableObjects:
            d.draw()

    def moveObjects(self, t):
        self.carSprite.x += 5

win = car_animation()
pyglet.gl.glClearColor(1, 1, 1, 1)
pyglet.clock.schedule_interval(win.moveObjects, 1.0/20)
pyglet.app.run()
