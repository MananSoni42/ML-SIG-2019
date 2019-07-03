from car import car,read_track
from animate_pyglet import animate_cars

track_path = 'sample_path.csv'

track = read_track(track_path)
my_car = car(track)

for i in range(500):
    my_car.run()

"""
plt.plot(track[0], track[1], c="k")
plt.plot(track[0], track[2], c="k")
plt.plot(*zip(*my_car1.pos_history))

plt.gca().set_xlim(-10,1010)
plt.gca().set_ylim(-10,1010)
plt.show()
"""
animate_cars([my_car],track_path=track_path)
