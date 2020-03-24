def import_MNIST():
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    return (X[:60000]/255, y[:60000]), (X[60000:]/255, y[60000:])
