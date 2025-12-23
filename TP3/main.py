#!/bin/python3

from utils import *


if __name__ == "__main__":
    learning_rate = 0.05
    epochs = 5000
    shape = (500, 2)

    x, y = createData(shape[0], shape[1])

    x = normalizeDate(x)

    plotData(x, y)

    w, b = train(x, y, shape[1], learning_rate, epochs)

    plot_decision_boundary(x, y, w, b)
    
    