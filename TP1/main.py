#!/bin/python3 

from utils import *



if __name__ == "__main__":
    SIZE_DATA = 1000
    learning_rate = 0.0001
    epoch = 100000
    
    (x, y) = createData(SIZE_DATA)

    
    



    w,b = train(x, y, learning_rate, epoch)

    print(f"Found Equation = {w} * x + {b}")

    plotDataWithCurve(x, y, w, b)
