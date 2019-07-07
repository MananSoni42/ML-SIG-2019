from car import car, read_track
import numpy as np
from animate_pyglet import animate_cars
import matplotlib.pyplot as plt

track_path = "tracks/test_3.csv"
track = read_track(track_path)
epochs = 5
iter = 500
num_cars = 9
fig_size = (3,3)

def example_accl_function(params, **kwargs):
    weights = kwargs['w']
    acc =  np.dot(weights,params).reshape((2,))
    return acc

def update_weights(weights,utilities,mut_prob=0.01,cross_prob=0.5):
    # checkout these pages:
    # https://ai.stackexchange.com/questions/3428/mutation-and-crossover-in-a-genetic-algorithm-with-real-numbers
    # https://www.geeksforgeeks.org/mutation-algorithms-for-real-valued-parameters-ga/
    utilities = np.array(utilities)
    new_weights = np.zeros(weights.shape)

    # selection
    probs = utilities/np.sum(utilities)
    ind_w = np.random.choice(np.arange(num_cars),size=num_cars,p=probs,replace=True)
    new_weights = weights[ind_w]

    # crossover
    num_sites = 4
    for t in range(num_cars//2):
        i,j = np.random.choice(num_cars, size=2, replace=False)
        sites = np.random.choice(10, size=num_sites, replace=False)
        for site in sites:
            if np.random.rand() <= cross_prob:
                new_weights[i,:,site], new_weights[j,:,site] = np.mean((weights[i,:,site], weights[j,:,site]), axis=0)

    # mutation -TODO

    return new_weights

# initialize weights in range -1 to 1
weights =2*np.random.rand(num_cars,2,10) -1

for e in range(epochs):
    #print(weights[0,:])
    print(e)
    # create n_cars
    my_cars = []
    utilities = []
    for n in range(num_cars):
        my_cars.append(car(track, example_accl_function))

    # Test run the cars and get their utilities
    for n in range(num_cars):
        for i in range(iter):
            my_cars[n].run(w=weights[n])
        utilities.append(my_cars[n].utility()[0])

    # update the weights
    weights = update_weights(weights,utilities)

    # plot results
    for n in range(num_cars):
        plt.subplot(fig_size[0],fig_size[1],n+1)
        my_cars[n].plot_history()
    plt.title(f'epoch - {e}')
    plt.gca().set_xlim(-10, 1010)
    plt.gca().set_ylim(-10, 610)
    plt.show()
    print(f'top 5 performers:')
    print(sorted(utilities)[:5])
