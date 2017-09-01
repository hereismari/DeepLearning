from scipy.spatial import distance

class myKNN():

    def closest(self, row):
        best_dist = distance.euclidean(row, self.x[0])
        best_index = 0
        for i in xrange(1, len(self.x)):
            dist = distance.euclidean(row, self.x[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.y[best_index]

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, test):
        pred = []
        for row in test:
            label = self.closest(row)
            pred.append(label)
        return pred


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

# classifier
neigh_clf = KNeighborsClassifier()
neigh_clf.fit(x_train, y_train)

my_clf = myKNN()
my_clf.fit(x_train, y_train)

# pretidiction
neigh_pred = neigh_clf.predict(x_test)
my_pred = my_clf.predict(x_test)

# accuracy
print accuracy_score(y_test, neigh_pred)
print accuracy_score(y_test, my_pred)
