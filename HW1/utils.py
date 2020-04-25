import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def train_validation_test_split(X, y, train_size=0.5, val_size=0.2, test_size=0.3, shuffle=True, random_state=None):
    if (train_size + test_size + val_size != 1.0):
        raise ValueError("The parameters train_size, test_size, val_size do not sum to 1.0")
    if train_size == 0. or test_size == 0. or val_size == 0.:
        raise ValueError("One of parameters train_size, test_size, val_size is equal to 0.0")

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size= test_size,
                                                        shuffle=shuffle,
                                                        random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      train_size= train_size/(train_size+val_size),
													  shuffle=shuffle, 
													  random_state=random_state,
                                                      #shuffle=False,  # the data is already shuffled from the previous train_test_split
                                                                      # TODO: is it necessary to shuffle again?
                                                      #random_state=random_state
                                                    )
    return X_train, X_val, X_test, y_train, y_val, y_test



def plot_decision_function(X, y, clf, ax, title=None, x_label=None, y_label=None):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k') # TODO: add cmap to modify

    if title is not None:
        ax.title.set_text(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

def GridSearch_():
    return

def plot_gridsearch_():
    return
