import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


cmap_bg='Pastel1'
cmap_fg='Set1'


def plot_2d(X, y):
    if X.shape[1] > 2:
        X_2d = PCA(n_components=2).fit_transform(X)
    else:
        X_2d = X
    
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap_fg)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()


def plot_classifier_boundary(model, X, y, sc=None, h=.05):
    assert X.shape[1] == 2, 'The dataset needs to be 2 dimentional'
    # this function can be used with any sklearn classifier
    # ready for two classes but can be easily extended]
    x_min, x_max = X[:, 0].min()-.2, X[:, 0].max()+.2
    y_min, y_max = X[:, 1].min()-.2, X[:, 1].max()+.2
    # generate a grid with step h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # the method ravel flattens xx and yy
    predict_input = np.c_[xx.ravel(), yy.ravel()]
    if sc is not None:
        predict_input = sc.transform(predict_input)
    
    Z = model.predict(predict_input)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_bg)
    plt.xlim((x_min,x_max))
    plt.ylim((y_min,y_max))
    
    plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_fg)
    #plt.title('test')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
    plt.show()
