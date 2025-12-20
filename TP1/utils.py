import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Tuple

def createData(size : int) -> Tuple[np.ndarray, np.ndarray] :
    a = random.random()*10
    c = random.random()*10

    print(f"Equation = {a} * x + {c}")

    f = lambda x : a*x + c + np.random.normal(0, 10, size=x.shape[0])

    x = np.random.randint(-100, 100, size )
    y = f(x)

    return (x, y)

def plotData(x : np.ndarray, y : np.ndarray) -> None:
    plt.plot(x, y, 'ro')
    plt.show()


def predict(x: np.ndarray, w: float, b: float) -> np.ndarray:
    return w*x + b

def loss(y: np.ndarray, y_predict : np.ndarray) -> float :
    if(y.shape != y_predict.shape):
        raise Exception("The shape of the array should be the same")

    return np.mean((y - y_predict)**2)


def gradients(x: np.ndarray, y: np.ndarray, y_predict: np.ndarray) -> Tuple[float, float]:
    dw = 2 * np.mean((y_predict-y)*x)
    db = 2* np.mean((y_predict-y))
    print(dw)

    return (dw, db)

def gradientDescent(val : float, val_derivative: float, learning_rate : float) -> float :
    return val - learning_rate * val_derivative

def updateParam(w : float, dw : float, b : float, db : float, learning_rate : float) -> Tuple[float, float] :
    return (gradientDescent(w, dw, learning_rate), gradientDescent(b, db, learning_rate))

def plotDataWithCurve(x : np.ndarray, y : np.ndarray, w : float, b : float) -> None:
    min_x = x.min()
    max_x = x.max()
    x_line = np.linspace(min_x, max_x, 100)  # plus de points pour une droite lisse
    curve = w * x_line + b
    plt.plot(x, y, 'ro', label='Data')
    plt.plot(x_line, curve, 'b-', label='Fitted line')
    # plt.legend()
    plt.show()



def train(x: np.ndarray, y: np.ndarray, learning_rate: float = 0.001, epochs: int = 1000) -> Tuple[float, float]:
    w = random.random()
    b = random.random()
    for i in range(epochs):
        y_predict = predict(x, w, b)
        MSE = loss(y, y_predict)
        dw, db = gradients(x, y, y_predict)
        print(dw)
        w, b = updateParam(w, dw, b, db, learning_rate)
        print(f"[{i}]/{epochs} : MSE : {MSE}")
    return w, b