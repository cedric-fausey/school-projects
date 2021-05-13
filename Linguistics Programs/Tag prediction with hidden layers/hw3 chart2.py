import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

labelsFile = open("hw3_labels.txt", "r")

if __name__ == "__main__":
    X = np.loadtxt("hw3_hiddenlayers.txt")
    Y = labelsFile.readlines()
    X2 = TSNE(perplexity=20.0).fit_transform(X)
    plt.scatter(X2[:, 0], X2[:, 1], 20, label=Y)
    for i in range(200):
        plt.text(X2[i,0], X2[i,1], str(Y[i]))
    plt.show()
