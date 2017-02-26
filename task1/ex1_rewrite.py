import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets, metrics, cross_validation
from matplotlib.colors import ListedColormap

k_max = 30
N = 1000

Data = datasets.make_classification(n_samples=N, n_features =2, n_informative = 2,
                                                      n_classes = 3, n_redundant=0,
                                                      n_clusters_per_class=1, random_state=3)

train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(Data[0],
                                                                                     Data[1],
                                                                                     test_size = 0.3,
                                                                                     random_state = 1)
colors = ListedColormap(['red', 'blue', 'yellow'])
light_colors = ListedColormap(['lightcoral', 'lightblue', 'lightyellow'])
plt.figure(figsize=(8,6))
plt.scatter(map(lambda x: x[0], Data[0]), map(lambda x: x[1], Data[0]),
              c=Data[1], cmap=colors, s=100, alpha = 0.4)
plt.show()


def get_meshgrid(data, step=.05, border=.5,):
    x_min, x_max = data[:, 0].min() - border, data[:, 0].max() + border
    y_min, y_max = data[:, 1].min() - border, data[:, 1].max() + border
    return np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))


def plot_decision_surface(estimator, train_data, train_labels, test_data, test_labels,
                          colors=colors, light_colors=light_colors):
    # fit model
    estimator.fit(train_data, train_labels)

    # set figure size
    plt.figure(figsize=(16, 6))

    # plot decision surface on the train data
    plt.subplot(1, 2, 1)
    xx, yy = get_meshgrid(train_data)
    mesh_predictions = np.array(estimator.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=100, cmap=colors)
    plt.title(
        'Train data, accuracy={:.2f}'.format(metrics.accuracy_score(train_labels, estimator.predict(train_data))))

    # plot decision surface on the test data
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors)
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, s=100, cmap=colors)
    plt.title('Test data, accuracy={:.2f}'.format(metrics.accuracy_score(test_labels, estimator.predict(test_data))))

    plt.show()

for i in [1, 3, 5, 10, 20]:
    plot_decision_surface(KNeighborsClassifier(), train_data,  train_labels, test_data, test_labels)
'''
k_acc = []
for i in range(k_max+1)[1:]:
    print i
    Classifier = KNeighborsClassifier(i)
    k_acc.append(np.mean(cross_val_score(Classifier, Data, klasses, cv=5)))


plt.subplot(222)
plt.plot(range(k_max+1)[1:], k_acc)
plt.xlabel('K nearest neighbours', fontsize=20)
plt.ylabel('accuracy', fontsize=20)
plt.show()
'''