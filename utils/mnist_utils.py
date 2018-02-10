import gzip, pickle
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def load_mnist():
    with gzip.open('data/mnist.pkl.gz', 'rb') as ff:
        u = pickle._Unpickler(ff)
        u.encoding = 'latin1'
        train, val, test = u.load()

        train_X = np.reshape(train[0], (-1, 28, 28, 1))
        train_y = train[1]
        val_X = np.reshape(val[0], (-1, 28, 28, 1))
        val_y = val[1]
        test_X = np.reshape(test[0], (-1, 28, 28, 1))
        test_y = test[1]

    return train_X, train_y, val_X, val_y, test_X, test_y

def sample(train_X, train_y):
    # Plot a random image
    sample_number = 5111
    plt.imshow(train_X[sample_number,:, :, -1].reshape(28,28), cmap="gray_r")
    plt.axis('off')
    print("Image Label: ", train_y[sample_number])
    plt.show()

def random_batches(X, y, batch=100):
    pass

if __name__ == "__main__":
    train_X, train_y, val_X, val_y, test_X, test_y = load_mnist()    
    print(train_X.shape, val_X.shape)
    sample(train_X, train_y)