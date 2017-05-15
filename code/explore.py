import pylab
import numpy as np

from code.util import getDataFromFile, get_data


def showImage(img, tags):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    fig = pylab.gcf()
    fig.canvas.set_window_title(tags)
    plt.show()



x_train, x_valid, y_train, y_valid = get_data()

for img, tag in zip(x_train, y_valid):
    showImage(np.array(img, np.byte), tag)