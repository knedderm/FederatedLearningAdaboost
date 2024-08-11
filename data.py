import openml

# Loads the dataset and splits it into training, test, and local (which is the data all clients get access to)
def load_mnist():
    """
    Loads the MNIST dataset using OpenML
    Dataset link: https://www.openml.org/d/554
    """
    mnist_openml = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    X = Xy[:, :-1]  # the last column contains labels
    y = Xy[:, -1]
    # First 60000 samples consist of the train set
    x_train, y_train = X[:50000], y[:50000]
    x_local, y_local = X[50000:60000], y[50000:60000]
    x_test, y_test = X[60000:], y[60000:]
    return (x_train, y_train), (x_test, y_test), (x_local, y_local)