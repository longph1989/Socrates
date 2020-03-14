import numpy as np
import csv
import matplotlib.pyplot as plt

with open('data.csv', 'r') as csv_file:
    fig, ax = plt.subplots(1, 10)
    i = 0

    for data in csv.reader(csv_file):
        # The first column is the label
        label = data[0]

        # The rest of columns are pixels
        pixels = data[1:]

        # Make those columns into a array of 8-bits pixels
        # This array will be of 1D with length 784
        # The pixel intensity values are integers from 0 to 255
        pixels = np.array(pixels, dtype='uint8')

        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))

        # Plot
        ax[i].set(title='Label is {label}'.format(label=label))
        ax[i].imshow(pixels, cmap='gray')

        i = i + 1

    plt.show()
