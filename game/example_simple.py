from car import Car, read_track
import numpy as np
from animate_pyglet import animate_cars

track_path = "tracks/test_3.csv"

track = read_track(track_path)


def example_accl_function(params, **kwargs):
    distances = params[:4]
    last_dist = params[4:8]
    last_vel = params[8:10]
    du, dd, _, _ = distances
    accl = [0, 0]
    accl[0] = 0.1  # Accel along x axis

    # Accel along y axis which takes into account the distances from the tracks
    accl[1] = (1 + du) / (1 + dd) - 1

    if distances[0] < 200 or distances[2] < 200:
        accl[1] -= 0.2

    if distances[1] < 200 or distances[3] < 200:
        accl[1] += 0.2

    return np.array(accl)


my_car = Car(track, example_accl_function)

for i in range(2000):
    my_car.run()

my_car.plot_history()
# print(my_car.utility())

animate_cars([my_car], track_path=track_path)
