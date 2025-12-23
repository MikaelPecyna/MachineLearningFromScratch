#!/usr/bin/env python3

from utils import (
    create_data,
    normalize,
    train,
    plot_data,
)


def main() -> None:
    learning_rate = 0.001
    epochs = 50000
    n_samples, n_features = 1000, 3

    x, y, w_true, b_true = create_data(n_samples, n_features)


    plot_data(x, y, w_true, b_true)

    w, b, history = train(
        x,
        y,
        learning_rate=learning_rate,
        epochs=epochs,
        early_stopping=True,
    )


    plot_data(x, y, w, b)

    


if __name__ == "__main__":
    main()
