import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class NeuralNetwork:

    def __init__(self, n_hidden=10, activation="tanh", learning_rate=0.01):
        self.n_hidden = n_hidden
        self.activation = activation
        self.lr = learning_rate
        self.losses_train = []
        self.losses_test = []

    #weight initialization
    def _init_weights(self):
        if self.activation == "relu":
            #he initialization
            s1 = np.sqrt(2.0 / 1)
            s2 = np.sqrt(2.0 / self.n_hidden)
        else:
            #xavier initialization
            s1 = np.sqrt(1.0 / 1)
            s2 = np.sqrt(1.0 / self.n_hidden)

        self.W1 = np.random.randn(1, self.n_hidden) * s1
        self.b1 = np.zeros((1, self.n_hidden))
        self.W2 = np.random.randn(self.n_hidden, 1) * s2
        self.b2 = np.zeros((1, 1))

    #activation functions
    def _activate(self, s):
        if self.activation == "tanh":
            return np.tanh(s)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))
        elif self.activation == "relu":
            return np.maximum(0, s)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activate_deriv(self, s):
        if self.activation == "tanh":
            t = np.tanh(s)
            return 1.0 - t ** 2
        elif self.activation == "sigmoid":
            sig = 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))
            return sig * (1.0 - sig)
        elif self.activation == "relu":
            return (s > 0).astype(float)

    #forward pass
    def forward(self, X):
        self.X = X
        self.s1 = X @ self.W1 + self.b1        #pre-activation
        self.h = self._activate(self.s1)        #hidden output
        self.y_hat = self.h @ self.W2 + self.b2 #network output
        return self.y_hat

    def predict(self, X):
        s1 = X @ self.W1 + self.b1
        h = self._activate(s1)
        return h @ self.W2 + self.b2

    #backward pass
    def backward(self, Y):
        n = Y.shape[0]
        #dE/dy_hat = 2/n * (y_hat - y)  (MSE derivative)
        d_out = (2.0 / n) * (self.y_hat - Y)            #(n, 1)

        #output layer gradients
        self.dW2 = self.h.T @ d_out                      # (n_hidden, 1)
        self.db2 = np.sum(d_out, axis=0, keepdims=True)  # (1, 1)

        #propagate error to hidden layer
        d_h = d_out @ self.W2.T                          # (n, n_hidden)

        #apply activation derivative
        d_h *= self._activate_deriv(self.s1)             # (n, n_hidden)

        #hidden layer gradients
        self.dW1 = self.X.T @ d_h                        # (1, n_hidden)
        self.db1 = np.sum(d_h, axis=0, keepdims=True)    # (1, n_hidden)

    def _update(self):
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

    #training
    def fit(self, X_tr, Y_tr, X_te=None, Y_te=None,
            epochs=5000, mode="batch", verbose=True):
        self._init_weights()
        self.losses_train = []
        self.losses_test = []

        for ep in range(epochs):
            if mode == "batch":
                self.forward(X_tr)
                self.backward(Y_tr)
                self._update()
            elif mode == "online":
                order = np.random.permutation(len(X_tr))
                for i in order:
                    xi = X_tr[i : i + 1]
                    yi = Y_tr[i : i + 1]
                    self.forward(xi)
                    self.backward(yi)
                    self._update()

            y_pred_train = self.predict(X_tr)
            mse_tr = np.mean((Y_tr - self.predict(X_tr)) ** 2)
            self.losses_train.append(mse_tr)

            if X_te is not None:
                y_pred_test = self.predict(X_te)
                mse_te = np.mean((Y_te - self.predict(X_te)) ** 2)
                self.losses_test.append(mse_te)

            if verbose and (ep % (epochs // 5) == 0 or ep == epochs - 1):
                msg = f"  Epoch {ep:5d} | Train MSE: {mse_tr:.6f}"
                if X_te is not None:
                    msg += f" | Test MSE: {mse_te:.6f}"
                print(msg)


def plot_results(nn, X_tr, Y_tr, X_te, Y_te, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    xp = np.linspace(X_tr.min() - 0.3, X_tr.max() + 0.3, 300).reshape(-1, 1)
    yp = nn.predict(xp)

    ax1.scatter(X_tr, Y_tr, c="blue", s=20, alpha=0.6, label="Train")
    ax1.scatter(X_te, Y_te, c="red", marker="x", s=50, label="Test")
    ax1.plot(xp, yp, "g-", lw=2, label="NN prediction")
    ax1.set_xlabel("x");
    ax1.set_ylabel("y")
    ax1.set_title(title);
    ax1.legend();
    ax1.grid(True, alpha=0.3)

    ax2.plot(nn.losses_train, label="Train MSE")
    if nn.losses_test:
        ax2.plot(nn.losses_test, label="Test MSE")
    ax2.set_xlabel("Epoch");
    ax2.set_ylabel("MSE")
    ax2.set_title("Learning curve");
    ax2.legend()
    ax2.set_yscale("log");
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"  Final Train MSE: {nn.losses_train[-1]:.6f}")
    if nn.losses_test:
        print(f"  Final Test  MSE: {nn.losses_test[-1]:.6f}")


#Task 1
print("1")

data = np.loadtxt("dane/dane2.txt")
x_all = data[:, 0]
y_all = data[:, 1]

#normalise inputs to [-1, 1]
x_mean, x_std = x_all.mean(), x_all.std()
y_mean, y_std = y_all.mean(), y_all.std()
x_norm = (x_all - x_mean) / x_std
y_norm = (y_all - y_mean) / y_std

#shuffle and split 80/20
idx = np.arange(len(x_all))
np.random.shuffle(idx)

split = int(0.8 * len(idx))
tr_idx, te_idx = idx[:split], idx[split:]

X_train = x_norm[tr_idx].reshape(-1, 1)
Y_train = y_norm[tr_idx].reshape(-1, 1)
X_test  = x_norm[te_idx].reshape(-1, 1)
Y_test  = y_norm[te_idx].reshape(-1, 1)

print(f"Total samples : {len(x_all)}")
print(f"Training      : {len(X_train)}")
print(f"Test          : {len(X_test)}")
print(f"x range (raw) : [{x_all.min():.1f}, {x_all.max():.1f}]")
print(f"y range (raw) : [{y_all.min():.2f}, {y_all.max():.2f}]")

plt.figure(figsize=(8, 4))
plt.scatter(x_all[tr_idx], y_all[tr_idx], c="blue", label=f"Train ({len(tr_idx)})")
plt.scatter(x_all[te_idx], y_all[te_idx], c="red", marker="x", s=70, label=f"Test ({len(te_idx)})")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Data split (raw scale)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()


#Task 2
print("2")

nn_tanh_batch = NeuralNetwork(n_hidden=10, activation="tanh", learning_rate=0.01)
nn_tanh_batch.fit(X_train, Y_train, X_test, Y_test, epochs=5000, mode="batch")
plot_results(nn_tanh_batch, X_train, Y_train, X_test, Y_test,
             "Tanh + Batch (10 hidden neurons)")

#Task 3
print("3")

print("Underfitting")
nn_under = NeuralNetwork(n_hidden=2, activation="tanh", learning_rate=0.01)
nn_under.fit(X_train, Y_train, X_test, Y_test, epochs=3000, mode="batch")
plot_results(nn_under, X_train, Y_train, X_test, Y_test,
             "Underfitting (2 neurons)")


print("Optimal")

nn_opt = NeuralNetwork(n_hidden=10, activation="tanh", learning_rate=0.01)
nn_opt.fit(X_train, Y_train, X_test, Y_test, epochs=5000, mode="batch")
plot_results(nn_opt, X_train, Y_train, X_test, Y_test,
             "Optimal (10 neurons)")

print("Overfitting")
nn_over = NeuralNetwork(n_hidden=50, activation="tanh", learning_rate=0.01)
nn_over.fit(X_train, Y_train, X_test, Y_test, epochs=20000, mode="batch")
plot_results(nn_over, X_train, Y_train, X_test, Y_test,
             "Overfitting (50 neurons, 20k epochs)")

print("\nSummary")
print(f"{'Model':<35} {'Train MSE':>11} {'Test MSE':>11}  Diagnosis")
print(f"{'2 neurons':<35} {nn_under.losses_train[-1]:>11.6f} {nn_under.losses_test[-1]:>11.6f}  UNDERFITTING")
print(f"{'10 neurons':<35} {nn_opt.losses_train[-1]:>11.6f} {nn_opt.losses_test[-1]:>11.6f}  OPTIMAL")
print(f"{'50 neurons, 20k epochs':<35} {nn_over.losses_train[-1]:>11.6f} {nn_over.losses_test[-1]:>11.6f}  OVERFITTING")

#Task 4
print("4")

nn_tanh_online = NeuralNetwork(n_hidden=10, activation="tanh", learning_rate=0.005)
nn_tanh_online.fit(X_train, Y_train, X_test, Y_test, epochs=2000, mode="online")
plot_results(nn_tanh_online, X_train, Y_train, X_test, Y_test,
             "Tanh + Online (10 neurons)")

print("\n  Batch vs Online (same 10-neuron tanh):")
print(f"Batch  → Train: {nn_tanh_batch.losses_train[-1]:.6f}, Test: {nn_tanh_batch.losses_test[-1]:.6f}")
print(f"Online → Train: {nn_tanh_online.losses_train[-1]:.6f}, Test: {nn_tanh_online.losses_test[-1]:.6f}")


#Task 5
print("5")

nn_relu = NeuralNetwork(n_hidden=15, activation="relu", learning_rate=0.005)
nn_relu.fit(X_train, Y_train, X_test, Y_test, epochs=5000, mode="batch")
plot_results(nn_relu, X_train, Y_train, X_test, Y_test,
             "ReLU + Batch (15 neurons)")


#Final comparison
print("Final summary")
print(f"{'Model':<45} {'Train MSE':>11} {'Test MSE':>11}")
all_models = [
    ("Tanh, Batch, 10 neurons",      nn_tanh_batch),
    ("Tanh, Batch, 2 (underfit)",    nn_under),
    ("Tanh, Batch, 10 (optimal)",    nn_opt),
    ("Tanh, Batch, 50 (overfit)",    nn_over),
    ("Tanh, Online, 10 neurons",      nn_tanh_online),
    ("ReLU, Batch, 15 neurons",       nn_relu),
]
for name, m in all_models:
    print(f"{name:<45} {m.losses_train[-1]:>11.6f} {m.losses_test[-1]:>11.6f}")

#all predictions in one figure
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
xp = np.linspace(X_train.min() - 0.3, X_train.max() + 0.3, 300).reshape(-1, 1)

titles = [n for n, _ in all_models]
nets   = [m for _, m in all_models]

for ax, nn, t in zip(axes.flat, nets, titles):
    ax.scatter(X_train, Y_train, c="blue", s=12, alpha=0.5, label="Train")
    ax.scatter(X_test, Y_test, c="red", marker="x", s=40, label="Test")
    ax.plot(xp, nn.predict(xp), "g-", lw=2, label="Prediction")
    ax.set_title(t, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("All Models Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()