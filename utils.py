import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import colors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from tensorflow import keras as k


cmap_bg='Pastel1'
cmap_fg='Set1'


def plot_2d(X, y, title=''):
    assert X.shape[1] >= 2
    if X.shape[1] > 2:
        print('Reducing the dataset dimension to 2D...')
        X_2d = PCA(n_components=2).fit_transform(X)
    else:
        X_2d = X
    
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap_fg)
    plt.title(title)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()


def plot_classifier_boundary(model, X, y, sc=None, lib='skl', h=.05, title=''):
    assert X.shape[1] == 2, 'The dataset needs to be 2 dimentional'
    # this function can be used with any sklearn classifier
    # ready for two classes but can be easily extended
    x_min, x_max = X[:, 0].min()-.2, X[:, 0].max()+.2
    y_min, y_max = X[:, 1].min()-.2, X[:, 1].max()+.2
    # generate a grid with step h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # the method ravel flattens xx and yy
    predict_input = np.c_[xx.ravel(), yy.ravel()]
    if sc is not None:
        predict_input = sc.transform(predict_input)
    
    if lib == 'skl':
        Z = model.predict(predict_input)
    elif lib == 'keras':
        turn_binary = lambda x: 1 if x > .5 else 0 
        Z = model(predict_input).numpy()
        Z = np.asarray([turn_binary(x) for x in Z[:, 0]])
    else:
        raise Exception('Invalid lib type')
        
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_bg)
    plt.xlim((x_min,x_max))
    plt.ylim((y_min,y_max))
    
    plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_fg)
    plt.title(title)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
    plt.show()


# plot boundaries for 3 class problems?
def plot_classifier_boundary_3(model,X,y,h = .05):
    # this function can be used with any sklearn classifier
    # ready for two classes but can be easily extended
    cmap = colors.ListedColormap(['blue','orange','green'])
    cmap_light = colors.ListedColormap(['lightsteelblue', 'peachpuff','lightgreen'])
    x_min, x_max = X[:, 0].min()-.2, X[:, 0].max()+.2
    y_min, y_max = X[:, 1].min()-.2, X[:, 1].max()+.2
    # generate a grid with step h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # the method ravel flattens xx and yy
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.xlim((x_min,x_max))
    plt.ylim((y_min,y_max))
    plt.scatter(X[:,0],X[:,1],color=cmap(y))
    plt.show()


def test_model(model, X, y, n_tests=10):
    result_sum = 0
    auc_list = []
    for _ in range(n_tests):
        auc_score = 0
        while True: # gambiarra
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
                model.fit(X_train, y_train)
                auc_score = roc_auc_score(y_test, model.predict(X_test))
                break
            except ValueError:
                print('ValueError due to lack of samples for class y. Iteration is repeated.')

        result_sum += auc_score
        auc_list.append(auc_score)

    print('Mean AUC score: %.3f' % (result_sum / n_tests))
    if X.shape[1] == 2:
        plot_classifier_boundary(model, X, y)
    else:
        print('''The classifier boundary can't be plotted because the dataset has more than 2 dimensions''')


def test_model_min(model, X, y, n_tests=10):
    auc_list = []
    
    for _ in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
        model.fit(X_train, y_train)
        auc_score = roc_auc_score(y_test, model.predict(X_test))
        auc_list.append(auc_score)

    return np.mean(auc_list)


def test_model_with_standard_scaler(model, X, y, n_tests=10):
    result_sum = 0
    for _ in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
        
        sc = StandardScaler().fit(X_train)
        sc_X_train = sc.transform(X_train)
        sc_X_test = sc.transform(X_test)
        
        model.fit(sc_X_train, y_train)
        result_sum += roc_auc_score(y_test, model.predict(sc_X_test))

    print('Mean AUC score: %.3f' % (result_sum / n_tests))
    if X.shape[1] == 2:
        plot_classifier_boundary(model, X, y, sc)
    else:
        print('''The classifier boundary can't be plotted because the dataset has more than 2 dimensions''')


def test_keras_model(model, X, y, n_tests=10):
    opt = k.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[k.metrics.AUC(from_logits=True)])

    results = 0
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
        
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, verbose=0)
        for key in history.history.keys():
            if key.startswith('val_auc'):
                auc_key = key
                break
        results += history.history[auc_key][-1]

    print('Mean AUC score: %.3f' % (results / 10))
    plot_classifier_boundary(model, X, y, lib='keras')

