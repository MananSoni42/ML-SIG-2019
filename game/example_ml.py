from car import Car, read_track
import numpy as np
from animate_pyglet import animate_cars
import matplotlib.pyplot as plt
from tqdm import tqdm

track_path = "tracks/test_4.csv"
track = read_track(track_path)
epochs = 15
iter = 1000
num_cars = 36
fig_size = (6, 6)


def example_accl_function(params, **kwargs):
    weights = kwargs["w"]
    acc = np.dot(weights, params).reshape((2,))
    return acc


def update_weights(weights, utilities, mut_prob=0.01, cross_prob=0.5, top_n=0.15):
    # checkout these pages:
    # https://ai.stackexchange.com/questions/3428/mutation-and-crossover-in-a-genetic-algorithm-with-real-numbers
    # https://www.geeksforgeeks.org/mutation-algorithms-for-real-valued-parameters-ga/
    utilities = np.array(utilities)
    new_weights = np.zeros(weights.shape)

    # selection
    probs = utilities / np.sum(utilities)
    ind_w = np.random.choice(np.arange(num_cars), size=num_cars, p=probs, replace=True)

    # save top_n% of cars as it is
    new_weights = weights[ind_w]

    # crossover
    num_sites = 4
    for t in range(num_cars // 2):
        # select 2 random weight
        i, j = np.random.choice(num_cars, size=2, replace=False)
        # select sites within weights
        sites = np.random.choice(10, size=num_sites, replace=False)
        for site in sites:
            # average the genes at the sites across the 2 genes
            if np.random.rand() <= cross_prob:
                new_weights[i, :, site], new_weights[j, :, site] = np.mean(
                    (weights[i, :, site], weights[j, :, site]), axis=0
                )

    # save top_n % weights
    n = int(top_n * utilities.shape[0])
    ind_best_n = ind = np.argpartition(utilities, -n)[-n:]
    ind_worst_n = ind = np.argpartition(utilities, -n)[:n]
    new_weights[ind_worst_n] = weights[ind_best_n]

    # mutation

    return new_weights


# initialize weights in range -1 to 1
weights = 2 * np.random.rand(num_cars, 2, 12) - 1

for e in range(epochs + 1):
    print(f"Epoch - {e} / {epochs}")

    # create n_cars
    my_cars = []
    for n in range(num_cars):
        my_cars.append(Car(track, example_accl_function))

    # Test run the cars and get their utilities
    utilities = []
    for n in tqdm(range(len(my_cars)), desc="cars"):
        car = my_cars[n]
        for i in range(iter):
            car.run(w=weights[n])
        u = car.utility()
        utilities.append(u[0])

    # update the weights
    weights = update_weights(weights, utilities)

    # display relevant info
    print(
        f"top 5 performers: {sorted(utilities)[-5:]} | epoch average: {np.mean(utilities)}"
    )
    if e % 5 == 0:
        for n in range(len(my_cars)):
            plt.subplot(fig_size[0], fig_size[1], n + 1)
            my_cars[n].plot_history()
        plt.title(f"epoch - {e}")
        plt.show()

print([car.utility() for car in my_cars])
animate_cars(my_cars, track_path=track_path)
