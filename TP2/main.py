#!/bin/python3

from utils import *

if __name__ == "__main__":
    learning_rate = 0.01
    epochs = 500000

    x, y = createData(100)

    plotBinaryData(x, y)

    w, b = train(x, y, learning_rate, epochs)
    plotDecisionBoundary(x, y, w, b)
    