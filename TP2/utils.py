import numpy as np 
from typing import Tuple
import matplotlib.pyplot as plt
import random

def sigmoid(x : np.ndarray) -> np.ndarray :
    return 1 / (1 + np.exp(-x))

def createData(size: int) -> Tuple[np.ndarray, np.ndarray]:
    a = random.random() * 10
    c = random.random() * 10

    x = np.random.randint(-100, 100, size)

    f = lambda x: a*x + c + np.random.normal(0, 100, size=x.shape[0])
    y = (f(x) > 0).astype(int)

    return x, y

def plotBinaryData(x: np.ndarray, y: np.ndarray) -> None:
    """
    Affiche les points x,y avec deux couleurs selon les classes 0 ou 1.
    """
    # On met tous les points à y=0 mais avec des couleurs différentes
    plt.scatter(x[y==0], np.zeros(np.sum(y==0)), color='red', label='Classe 0', alpha=0.6)
    plt.scatter(x[y==1], np.zeros(np.sum(y==1)), color='blue', label='Classe 1', alpha=0.6)
    plt.xlabel("x")
    plt.ylabel("Classe")
    plt.title("Dataset binaire")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def predict(x : np.ndarray, w : float, b : float) -> np.ndarray :
    return sigmoid(w*x+b)
    
def loss(y : np.ndarray, y_pred : np.ndarray) -> np.ndarray :
    if(y.shape != y_pred.shape):
        raise Exception("The shape of the array should be the same")
    
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Pour s'assurer que rien est a 0
    return -np.mean(y * np.log(y_pred) + ((1-y)*np.log(1-y_pred)))


def gradients(x : np.ndarray, y : np.ndarray, y_predict : np.ndarray) -> Tuple[float, float]:
    epsilon = 1e-15
    y_predict = np.clip(y_predict, epsilon, 1 - epsilon)
    
    dw = np.mean((y_predict - y) * x)
    db = np.mean(y_predict - y)
    
    return (dw, db)

def gradientDescent(val : float, val_derivative: float, learning_rate : float) -> float :
    return val - learning_rate * val_derivative

def updateParam(w : float, dw : float, b : float, db : float, learning_rate : float) -> Tuple[float, float] :
    return (gradientDescent(w, dw, learning_rate), gradientDescent(b, db, learning_rate))

def accuracy(y : np.ndarray, y_predict : np.ndarray) -> float : 
    y_pred_tmp = (y_predict > 0.5).astype(int)

    return np.mean(y == y_pred_tmp)


def train(x: np.ndarray, y: np.ndarray, learning_rate: float = 0.001, epochs: int = 1000) -> Tuple[float, float]:
    w = random.random()
    b = random.random()
    for i in range(epochs):
        y_predict = predict(x, w, b)
        BCE = loss(y, y_predict)
        dw, db = gradients(x, y, y_predict)
        w, b = updateParam(w, dw, b, db, learning_rate)
        if(i % (epochs / 10) == 0):
            print(f"[{i}]/{epochs} : BCE={BCE} ; acc={accuracy(y, y_predict)} ")
    return w, b


def plotDecisionBoundary(x: np.ndarray, y: np.ndarray, w: float, b: float) -> None:
    """
    Affiche le dataset binaire et la frontière de décision
    """
    # Afficher les données
    plt.scatter(x[y==0], np.zeros(np.sum(y==0)), color='red', label='Classe 0', alpha=0.6, s=100)
    plt.scatter(x[y==1], np.ones(np.sum(y==1)), color='blue', label='Classe 1', alpha=0.6, s=100)
    
    # Créer une grille pour la sigmoid
    x_line = np.linspace(x.min(), x.max(), 300)
    y_probs = sigmoid(w * x_line + b)
    
    # Tracer la courbe sigmoid
    plt.plot(x_line, y_probs, color='green', linewidth=2, label='P(y=1|x) = σ(wx+b)')
    
    # Ligne de seuil à 0.5
    plt.axhline(0.5, color='black', linestyle='--', linewidth=1, label='Seuil de décision (0.5)')
    
    # Frontière de décision verticale (où sigmoid = 0.5, donc wx+b = 0)
    if w != 0:
        x_boundary = -b / w
        plt.axvline(x_boundary, color='purple', linestyle=':', linewidth=2, 
                   label=f'Frontière: x={x_boundary:.2f}')
    
    plt.xlabel("x")
    plt.ylabel("P(y=1|x)")
    plt.title("Classification binaire avec frontière de décision")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.show()