import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from tqdm import tqdm


# =========================
# Utilitaires NumPy / Maths
# =========================

def normalize(x: np.ndarray) -> np.ndarray:
    """Standardise les features colonne par colonne (moyenne 0, variance 1)."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0  # eviter / 0 
    return (x - mean) / std


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoïde logistique numériquement stable.
    Voir techniques de stabilisation pour grandes valeurs de x. [web:3]
    """
    x = np.clip(x, -500, 500)  # simple stabilisation
    return 1.0 / (1.0 + np.exp(-x))


# =========================
# Génération de données
# =========================

def create_data(n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Génère des données synthétiques linéairement séparables avec un vrai hyperplan.

    Retourne :
        X : (N, d), features normalisées
        y : (N,), labels binaires {0,1}
        w_true : (d, 1), poids "vrais"
        b_true : float, biais "vrai"
    """
    x = np.random.randn(n_samples, n_features)
    x = normalize(x)

    w_true = np.random.randn(n_features, 1)
    b_true = float(np.random.randn())

    scores = x @ w_true + b_true + np.random.normal(0, 0.005, (n_samples, 1))
    y = (scores > 0).astype(int).flatten()

    return x, y, w_true, b_true


# =========================
# Modèle / Métriques
# =========================

def predict(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Retourne p(y=1|x) pour chaque échantillon.
    x : (N, d), w : (d, 1), b : scalaire
    -> sortie : (N, 1)
    """
    logits = x @ w + b          # (N, 1)
    return sigmoid(logits)      # (N, 1)


def loss(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Binary Cross-Entropy (BCE) moyenne.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    # Assurer des formes compatibles : (N,) pour les deux
    if y_pred.ndim == 2:
        y_pred = y_pred.ravel()
    return float(-np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))


def accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy binaire avec seuil 0.5.
    """
    # Assurer des formes compatibles : (N,) pour comparaison
    if y_pred.ndim == 2:
        y_pred = y_pred.ravel()
    y_pred = (y_pred > 0.5).astype(int)
    return float(np.mean(y == y_pred))


def compute_gradients(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Gradients de la BCE par rapport à w et b.
    x : (N, d)
    y : (N,)
    y_pred : (N, 1)
    """
    n_samples = x.shape[0]

    # Assure que y_pred est (N, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Assure que y est (N, 1) pour la soustraction, puis re-flatten si besoin
    if y.ndim == 1:
        y_col = y.reshape(-1, 1)
    else:
        y_col = y

    error = y_pred - y_col           # (N, 1)
    dw = (x.T @ error) / n_samples    # (d, 1)
    db = float(error.mean())

    return dw, db



def update_parameters(
    w: np.ndarray,
    dw: np.ndarray,
    b: float,
    db: float,
    learning_rate: float
) -> Tuple[np.ndarray, float]:
    """
    Mise à jour simple des paramètres par descente de gradient. [web:1][web:7]
    """
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b


# =========================
# Entraînement
# =========================

def train(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 1e-3,
    epochs: int = 1000,
    early_stopping: bool = True,
    early_stopping_window: int = 20,
    early_stopping_tol: float = 1e-6,
) -> Tuple[np.ndarray, float, Dict[str, List[float]]]:
    """
    Entraîne un modèle de régression logistique via descente de gradient pleine.

    Retourne :
        w : (d,1)
        b : float
        history : dict contenant les listes 'loss' et 'accuracy'.
    """
    n_samples, n_features = x.shape

    # Initialisation : distribution normale centrée
    w = np.random.randn(n_features, 1)
    b = float(np.random.randn())

    history = {"loss": [], "accuracy": []}

    progress_bar = tqdm(range(epochs), desc="Training", ncols=100)

    for epoch in progress_bar:
        y_pred = predict(x, w, b)
        loss_value = loss(y, y_pred)
        acc_value = accuracy(y, y_pred)

        dw, db = compute_gradients(x, y, y_pred)
        w, b = update_parameters(w, dw, b, db, learning_rate)

        history["loss"].append(loss_value)
        history["accuracy"].append(acc_value)

        if epoch % max(1, epochs // 10) == 0:
            progress_bar.set_postfix({"loss": f"{loss_value:.4f}", "acc": f"{acc_value:.4f}"})

        # Early stopping simple sur la convergence de la loss
        if early_stopping and epoch >= early_stopping_window:
            recent = history["loss"][-early_stopping_window:]
            if max(recent) - min(recent) < early_stopping_tol:
                break

    progress_bar.close()
    return w, b, history


# =========================
# Visualisation des données
# =========================

def _scatter_binary_2d(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    label0: str = "Classe 0",
    label1: str = "Classe 1"
) -> None:
    """
    Fonction interne pour factoriser les scatter 2D. [web:1]
    """
    ax.scatter(
        x[y == 0, 0],
        x[y == 0, 1],
        c="red",
        label=label0,
        alpha=0.6,
        s=50,
        edgecolors="black",
    )
    ax.scatter(
        x[y == 1, 0],
        x[y == 1, 1],
        c="blue",
        label=label1,
        alpha=0.6,
        s=50,
        edgecolors="black",
    )


def plot_data(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
    b: float | None = None,
    grid_resolution: int = 200,
) -> None:
    """
    Visualise les données (1D, 2D, 3D) avec éventuellement la frontière de décision.
    """
    n_features = x.shape[1]

    if n_features == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x[y == 0], y[y == 0], c="red", label="Classe 0", alpha=0.6, s=50)
        ax.scatter(x[y == 1], y[y == 1], c="blue", label="Classe 1", alpha=0.6, s=50)

        if w is not None and b is not None:
            boundary = -b / float(w[0, 0])
            ax.axvline(x=boundary, color="green", linestyle="--", linewidth=2, label="Frontière de décision")

        ax.set_xlabel("Feature x")
        ax.set_ylabel("Classe y")
        ax.set_title("Classification binaire (1D)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif n_features == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        _scatter_binary_2d(x, y, ax)

        if w is not None and b is not None:
            x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
            y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1

            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, grid_resolution),
                np.linspace(y_min, y_max, grid_resolution),
            )

            grid = np.c_[xx.ravel(), yy.ravel()]
            z = (grid @ w + b).reshape(xx.shape)

            ax.contour(xx, yy, z, levels=[0.0], colors="green", linewidths=2, linestyles="--")
            ax.contourf(xx, yy, z, levels=[-np.inf, 0.0, np.inf], colors=["red", "blue"], alpha=0.1)

        ax.set_xlabel("Feature x₁")
        ax.set_ylabel("Feature x₂")
        ax.set_title("Classification binaire (2D)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif n_features == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(x[y == 0, 0], x[y == 0, 1], x[y == 0, 2], c="red", label="Classe 0", alpha=0.6, s=50)
        ax.scatter(x[y == 1, 0], x[y == 1, 1], x[y == 1, 2], c="blue", label="Classe 1", alpha=0.6, s=50)

        if w is not None and b is not None:
            x_range = np.linspace(x[:, 0].min(), x[:, 0].max(), 20)
            y_range = np.linspace(x[:, 1].min(), x[:, 1].max(), 20)
            xx, yy = np.meshgrid(x_range, y_range)

            if abs(w[2, 0]) > 1e-6:
                zz = -(w[0, 0] * xx + w[1, 0] * yy + b) / w[2, 0]
                ax.plot_surface(xx, yy, zz, alpha=0.3, color="green")

        ax.set_xlabel("Feature x₁")
        ax.set_ylabel("Feature x₂")
        ax.set_zlabel("Feature x₃")
        ax.set_title("Classification binaire (3D)")
        ax.legend()

    else:
        print(f"Visualisation non supportée pour {n_features} dimensions.")
        print("Affichage des 2 premières dimensions uniquement.")
        plot_data(x[:, :2], y, w[:2] if w is not None else None, b, grid_resolution=grid_resolution)
        return

    plt.tight_layout()
    plt.show()

