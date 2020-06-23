import numpy as np
import matplotlib.pyplot as plt

class Display:
    def __init__(self, mean, std, resolution):
        self.mean = mean
        self.std = std
        self.resolution = resolution


    def __denormalize(self, x):
        step = int(x.size / len(self.mean))

        for i in range(len(self.mean)):
            x[i * step : (i + 1) * step] = x[i * step : (i + 1) * step] * self.std[i] + self.mean[i]

        x = (x * 255).astype('uint8')

        if np.prod(self.resolution) == x.size:
            x = x.reshape(self.resolution)
        else:
            new_shape = (x.size / np.prod(self.resolution), *self.resolution)
            x = x.reshape(new_shape).transpose(1, 2, 0)

        return x


    def show(self, model, x0, y0, x, y):
        x0 = self.__denormalize(x0)
        x = self.__denormalize(x)

        fig, ax = plt.subplots(1, 2)

        ax[0].set(title='Original. Label is {}'.format(y0))
        ax[1].set(title='Adv. sample. Label is {}'.format(y))

        if len(x.shape) == 2:
            ax[0].imshow(x0, cmap='gray')
            ax[1].imshow(x, cmap='gray')
        else:
            ax[0].imshow(x0)
            ax[1].imshow(x)

        plt.show()
