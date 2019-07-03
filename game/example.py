from car import car,read_track
from animate_pyglet import animate_cars

track_path = 'tracks/test_3.csv'

track = read_track(track_path)
my_car = car(track)

for i in range(2000):
    my_car.run()

my_car.plot_history()
#print(my_car.utility())

animate_cars([my_car],track_path=track_path)
