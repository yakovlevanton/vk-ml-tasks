import numpy as np


class SoftmaxRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=100,
            tol=0.001,
            random_state=None,
            eta0=0.01,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            shuffle=True,
            batch_size=32
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = int(max_iter)
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = int(n_iter_no_change)
        self.shuffle = shuffle
        self.batch_size = int(batch_size)
        self._coef = None
        self._intercept = None
        self.best_loss = np.inf
        self.no_improvement_count = 0
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def get_penalty_grad(self):
        if self.penalty == "l2":
            return 2 * self.alpha * self._coef
        elif self.penalty == "l1":
            return self.alpha * np.sign(self._coef)

    def split_into_batches(self, x, y):
        N = len(y)
        indices = np.arange(N)
        if self.shuffle:
            np.random.shuffle(indices)
        x_batches, y_batches = [], []
        for start in range(0, N, self.batch_size):
            end = min(N, start + self.batch_size)
            x_batch = x[indices[start:end]]
            y_batch = y[indices[start:end]]
            x_batches.append(x_batch)
            y_batches.append(y_batch)
        return x_batches, y_batches

    def gradient(self, x_batch, y_batch):
        logits = x_batch @ self._coef.T + self._intercept
        y_predict = self.softmax(logits)
        y_batch_encoded = np.eye(len(self.classes_))[y_batch]
        error = y_predict - y_batch_encoded
        grad_weights = ((error.T @ x_batch) / self.batch_size +
                        self.get_penalty_grad())
        grad_intercept = np.sum(error, axis=0) / self.batch_size
        return grad_weights, grad_intercept

    def fit(self, x, y):
        N = x.shape[0]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = x.shape[1]

        self._coef = np.random.randn(len(self.classes_), n_features)
        self._intercept = np.random.randn(n_classes)

        if self.early_stopping:
            validation_size = int(self.validation_fraction * N)
            x_train, y_train = x[:-validation_size], y[:-validation_size]
            x_valid, y_valid = x[-validation_size:], y[-validation_size:]
        else:
            x_train, y_train = x, y
        train_loss_prev = np.inf
        for epoch in range(self.max_iter):
            x_batches, y_batches = self.split_into_batches(x_train, y_train)
            for x_batch, y_batch in zip(x_batches, y_batches):
                grad_weights, grad_intercept = self.gradient(x_batch, y_batch)
                self._coef -= self.eta0 * grad_weights
                self._intercept -= self.eta0 * grad_intercept

            if self.early_stopping:
                logits_valid = x_valid @ self._coef.T + self._intercept
                valid_predict = self.softmax(logits_valid)
                y_valid_encoded = np.eye(len(self.classes_))[y_valid]
                valid_loss = -np.sum(y_valid_encoded *
                                     np.log(valid_predict
                                            + 1e-15)).sum() / x_valid.shape[0]
                if valid_loss >= self.best_loss - self.tol:
                    self.no_improvement_count += 1
                else:
                    self.best_loss = np.minimum(valid_loss, self.best_loss)
                    self.no_improvement_count = 0

                if self.no_improvement_count >= self.n_iter_no_change:
                    break
            else:
                logits_train = x_train @ self._coef.T + self._intercept
                train_predict = self.softmax(logits_train)
                y_train_encoded = np.eye(len(self.classes_))[y_train]
                train_loss = -np.sum(y_train_encoded
                                     * np.log(train_predict +
                                              1e-15)).sum() / x_train.shape[0]
                if (epoch != 0 and
                        np.abs(train_loss - train_loss_prev) < self.tol):
                    break
                train_loss_prev = train_loss

        return self

    def predict_proba(self, x):
        return self.softmax(x @ self._coef.T + self._intercept)

    def predict(self, x):
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]

    @staticmethod
    def softmax(z):
        max_z = np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z - max_z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @coef_.setter
    def coef_(self, value):
        self._coef = value

    @intercept_.setter
    def intercept_(self, value):
        self._intercept = value
