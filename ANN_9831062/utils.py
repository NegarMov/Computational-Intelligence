import numpy as np
import matplotlib.pyplot as plt

# A function to plot images
def show_image(item):
    image = item[0].reshape((28, 28))
    plt.imshow(image, 'gray')

    index = np.where(item[1] == 1)
    plt.title(f"label: {index[0][0]}")

    plt.show()


# Reading The Train Set
def get_train_set():
    train_images_file = open('train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)

    train_labels_file = open('train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    train_set = []
    for n in range(num_of_train_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256
        
        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1
        
        train_set.append((image, label))

    return train_set


# Reading The Test Set
def get_test_set():
    test_images_file = open('t10k-images.idx3-ubyte', 'rb')
    test_images_file.seek(4)

    test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
    test_labels_file.seek(8)

    num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
    test_images_file.seek(16)

    test_set = []
    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256
        
        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1
        
        test_set.append((image, label))
    
    return test_set

# Load the data from datasets
def load_data():
    return get_train_set(), get_test_set()

# Compute the sigmoid function of the input x
def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

# Compute the derivative of the sigmoid function
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))