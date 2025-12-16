import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image

def load_mice_protein(one_hot=False):
    filling_value = -100000
    X = np.genfromtxt(
        "../data/mice_protein/Data_Cortex_Nuclear.csv",
        delimiter=",",
        skip_header=1,
        usecols=(1, 78),
        filling_values=filling_value,
        encoding="UTF-8",
    )
    classes = np.genfromtxt(
        "../data/mice_protein/Data_Cortex_Nuclear.csv",
        delimiter=",",
        skip_header=1,
        usecols=range(78, 81),
        dtype=None,
        encoding="UTF-8",
    )
    for i, row in enumerate(X):
        for j, val in enumerate(row):
            if val == filling_value:
                X[i, j] = np.mean(
                    [
                        X[k, j]
                        for k in range(classes.shape[0])
                        if np.all(classes[i] == classes[k])
                    ]
                )
    X = MinMaxScaler().fit_transform(X)
    DY = np.zeros((classes.shape[0]), dtype=np.uint8)
    for i, row in enumerate(classes):
        for j, (val, label) in enumerate(zip(row, ["Control", "Memantine", "C/S"])):
            DY[i] += (2**j) * (val == label)
    Y = OneHotEncoder().fit_transform(DY.reshape(-1, 1)).toarray()
    if not one_hot:
        Y = DY
    indices = np.arrange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
    return (train_X, train_Y), (test_X, test_Y)

def load_data(fashion=False, digit=None, normalize=False):
    if fashion:
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if digit is not None and 0 <= digit and digit <= 9:
        train = test = {y: [] for y in range(10)}
        for x, y in zip(x_train, y_train):
            train[y].append(x)
        for x, y in zip(x_test, y_test):
            test[y].append(x)

        for y in range(10):

            train[y] = np.asarray(train[y])
            test[y] = np.asarray(test[y])

        x_train = train[digit]
        x_test = test[digit]

    x_train = x_train.reshape((-1, x_train.shape[1] * x_train.shape[2])).astype(
        np.float32
    )
    x_test = x_test.reshape((-1, x_test.shape[1] * x_test.shape[2])).astype(np.float32)

    if normalize:
        X = np.concatenate((x_train, x_test))
        X = (X - X.min()) / (X.max() - X.min())
        x_train = X[: len(y_train)]
        x_test = X[len(y_train) :]

    #     print(x_train.shape, y_train.shape)
    #     print(x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)


def load_mnist():
    train, test = load_data(fashion=False, normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(test[0], test[1], test_size=0.2)
    return (x_train, y_train), (x_test, y_test)


def load_fashion():
    train, test = load_data(fashion=True, normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(test[0], test[1], test_size=0.2)
    return (x_train, y_train), (x_test, y_test)


def load_mnist_two_digits(digit1, digit2):
    train_digit_1, _ = load_data(digit=digit1)
    train_digit_2, _ = load_data(digit=digit2)

    X_train_1, X_test_1 = train_test_split(train_digit_1[0], test_size=0.6)
    X_train_2, X_test_2 = train_test_split(train_digit_2[0], test_size=0.6)

    X_train = np.concatenate((X_train_1, X_train_2))
    y_train = np.array([0] * X_train_1.shape[0] + [1] * X_train_2.shape[0])
    shuffled_idx = np.random.permutation(X_train.shape[0])
    np.take(X_train, shuffled_idx, axis=0, out=X_train)
    np.take(y_train, shuffled_idx, axis=0, out=y_train)

    X_test = np.concatenate((X_test_1, X_test_2))
    y_test = np.array([0] * X_test_1.shape[0] + [1] * X_test_2.shape[0])
    shuffled_idx = np.random.permutation(X_test.shape[0])
    np.take(X_test, shuffled_idx, axis=0, out=X_test)
    np.take(y_test, shuffled_idx, axis=0, out=y_test)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    return (X_train, y_train), (X_test, y_test)

def load_isolet():
    train_X = np.genfromtxt(
        "../data/isolet/isolet1234.data",
        delimiter=",",
        usecols=range(0, 617),
        encoding="UTF-8",
    )
    train_Y = np.genfromtxt(
        "../data/isolet/isolet1234.data",
        delimiter=",",
        usecols=[617],
        encoding="UTF-8",
    )
    test_X = np.genfromtxt(
        "../data/isolet/isolet5.data",
        delimiter=",",
        usecols=range(0, 617),
        encoding="UTF-8",
    )
    test_Y = np.genfromtxt(
        "../data/isolet/isolet5.data",
        delimiter=",",
        usecols=[617],
        encoding="UTF-8",
    )
    X = MinMaxScaler().fit_transform(np.concatenate((train_X, test_X)))
    train_Y -= 1
    test_Y -= 1
    train_X = X[: len(train_Y)]
    test_X = X[len(train_Y) :]
    return (train_X, train_Y), (test_X, test_Y)

def load_coil_20():
    data = np.zeros((1440, 400))
    targets = np.zeros(1440)
    for i in range(1, 21):
        for j in range(72):
            obj_img = Image.open(
                f"../data/coil-20/coil-20-proc/obj{i}_{j}.png"
            )
            rescaled = obj_img.resize((20, 20))
            data[(i - 1) * 72 + j] = np.array(rescaled).reshape(400)
            targets[(i - 1) * 72 + j] = i - 1
    data = MinMaxScaler().fit_transform(data)
    indices = np.arrange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    targets = targets[indices]
    targets = targets.astype(np.int64)
    train_X, test_X, train_Y, test_Y = train_test_split(data, targets, test_size=0.2)
    return (train_X, train_Y), (test_X, test_Y)

def load_activity():
    train_X = np.genfromtxt(
        "../data/activity/final_X_train.txt",
        delimiter=",",
        encoding="UTF-8",
    )
    test_X = np.genfromtxt(
        "../data/activity/final_X_test.txt",
        delimiter=",",
        encoding="UTF-8",
    )
    train_Y = np.genfromtxt(
        "../data/activity/final_y_train.txt",
        delimiter=",",
        encoding="UTF-8",
    )
    test_Y = np.genfromtxt(
        "../data/activity/final_y_test.txt",
        delimiter=",",
        encoding="UTF-8",
    )
    X = MinMaxScaler().fit_transform(np.concatenate((train_X, test_X)))
    train_Y -= 1
    test_Y -= 1
    train_X = X[: len(train_Y)]
    test_X = X[len(train_Y) :]
    return (train_X, train_Y), (test_X, test_Y)

def load_dataset(dataset):
    if dataset == "MICE":
        return load_mice_protein()
    elif dataset == "MNIST":
        return load_mnist()
    elif dataset == "MNIST-Fashion":
        return load_fashion()
    elif dataset == "ISOLET":
        return load_isolet()
    elif dataset == "COIL":
        return load_coil_20()
    elif dataset == "Activity":
        return load_activity()
    else:
        print("Please specify a valid dataset")
        return None