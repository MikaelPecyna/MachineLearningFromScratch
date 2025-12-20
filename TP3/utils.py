import numpy as np
import random
import matplotlib.pyplot as plt 
from typing import Tuple



def createData(row: int, col: int) -> Tuple[np.ndarray, np.ndarray]:
    """Version avec vrai hyperplan de séparation"""
    # Génération des features
    X = np.random.randn(row, col)
    
    W = np.random.randn(col, 1)
    b = np.random.randn()
    
    # Scores + bruit
    scores = X @ W + b + np.random.normal(1, 1, (row, 1))
    
    # Labels binaires
    y = (scores > 0).astype(int).flatten()
    
    return X, y

def plotData(x: np.ndarray, y: np.ndarray, W: np.ndarray = None, b: float = None):
    """
    Visualise les données de classification binaire.
    
    Args:
        x: données d'entrée (N, d)
        y: labels binaires (N,)
        W: poids du classifieur (d, 1) - optionnel
        b: biais - optionnel
    """
    
    # Cas 1D : x a une seule feature
    if x.shape[1] == 1:
        plt.figure(figsize=(10, 6))
        
        # Afficher les points
        plt.scatter(x[y == 0], y[y == 0], c='red', label='Classe 0', alpha=0.6, s=50)
        plt.scatter(x[y == 1], y[y == 1], c='blue', label='Classe 1', alpha=0.6, s=50)
        
        # Afficher la frontière de décision si W et b fournis
        if W is not None and b is not None:
            x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
            decision_boundary = -(W[0, 0] * x_line + b) / W[0, 0]
            plt.axvline(x=-b/W[0, 0], color='green', linestyle='--', 
                       linewidth=2, label='Frontière de décision')
        
        plt.xlabel('Feature x', fontsize=12)
        plt.ylabel('Classe y', fontsize=12)
        plt.title('Classification binaire (1D)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    # Cas 2D : x a deux features
    elif x.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        
        # Afficher les points
        plt.scatter(x[y == 0, 0], x[y == 0, 1], c='red', 
                   label='Classe 0', alpha=0.6, s=50, edgecolors='black')
        plt.scatter(x[y == 1, 0], x[y == 1, 1], c='blue', 
                   label='Classe 1', alpha=0.6, s=50, edgecolors='black')
        
        # Afficher la frontière de décision si W et b fournis
        if W is not None and b is not None:
            x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
            y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
            
            # Créer une grille
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                np.linspace(y_min, y_max, 200))
            
            # Calculer les prédictions sur la grille
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = (grid @ W + b).reshape(xx.shape)
            
            # Afficher la frontière (où Z = 0)
            plt.contour(xx, yy, Z, levels=[0], colors='green', 
                       linewidths=2, linestyles='--')
            plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], 
                        colors=['red', 'blue'], alpha=0.1)
        
        plt.xlabel('Feature x₁', fontsize=12)
        plt.ylabel('Feature x₂', fontsize=12)
        plt.title('Classification binaire (2D)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    # Cas 3D : x a trois features
    elif x.shape[1] == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Afficher les points
        ax.scatter(x[y == 0, 0], x[y == 0, 1], x[y == 0, 2], 
                  c='red', label='Classe 0', alpha=0.6, s=50)
        ax.scatter(x[y == 1, 0], x[y == 1, 1], x[y == 1, 2], 
                  c='blue', label='Classe 1', alpha=0.6, s=50)
        
        # Afficher le plan de décision si W et b fournis
        if W is not None and b is not None:
            x_range = np.linspace(x[:, 0].min(), x[:, 0].max(), 20)
            y_range = np.linspace(x[:, 1].min(), x[:, 1].max(), 20)
            xx, yy = np.meshgrid(x_range, y_range)
            
            # Plan : W[0]*x + W[1]*y + W[2]*z + b = 0
            # => z = -(W[0]*x + W[1]*y + b) / W[2]
            if abs(W[2, 0]) > 1e-6:  # Éviter division par zéro
                zz = -(W[0, 0] * xx + W[1, 0] * yy + b) / W[2, 0]
                ax.plot_surface(xx, yy, zz, alpha=0.3, color='green')
        
        ax.set_xlabel('Feature x₁', fontsize=12)
        ax.set_ylabel('Feature x₂', fontsize=12)
        ax.set_zlabel('Feature x₃', fontsize=12)
        ax.set_title('Classification binaire (3D)', fontsize=14)
        ax.legend()
        
    else:
        print(f"Visualisation non supportée pour {x.shape[1]} dimensions")
        print("Affichage des 2 premières dimensions uniquement")
        plotData(x[:, :2], y, W[:2] if W is not None else None, b)
        return
    
    plt.tight_layout()
    plt.show()



