import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBCustomRegressor:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self._estimators = []
        self.f0 = 0

    def fit(self, x, y):
        f_x = np.full_like(y, np.mean(y))
        self.f0 = np.mean(y)
        for t in range(self.n_estimators):
            r_t = y - f_x
            tree_t = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state)
            tree_t.fit(x, r_t)
            f_x = f_x + self.learning_rate * tree_t.predict(x)
            self._estimators.append(tree_t)

    def predict(self, x):
        result = self.f0
        for tree in self._estimators:
            result += tree.predict(x) * self.learning_rate
        return result

    @property
    def estimators_(self):
        return self._estimators


class GBCustomClassifier:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self._estimators = []
        self.f0 = 0

    @staticmethod
    def sigmoida(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        self.f0 = np.log(np.count_nonzero(y) / np.sum(y == 0))\
            if np.sum(y == 0) != 0 else 100
        f_x = np.full_like(y, self.f0)
        for t in range(self.n_estimators):
            r_t = y - self.sigmoida(f_x)
            tree_t = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state)
            tree_t.fit(x, r_t)
            f_x = f_x + self.learning_rate * tree_t.predict(x)
            self._estimators.append(tree_t)

    def predict_proba(self, x):
        f_x = self.f0
        for tree in self._estimators:
            f_x += tree.predict(x) * self.learning_rate
        proba1 = self.sigmoida(f_x)
        return np.stack((1 - proba1, proba1), axis=1)

    def predict(self, x):
        classes = np.array([0, 1])
        return classes[np.argmax(self.predict_proba(x), axis=1)]

    @property
    def estimators_(self):
        return self._estimators
