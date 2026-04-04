import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor

# Define a four-objective facility location problem in 2D
# x represents the facility location
# a1, a2, a3, a4 are four demand centers

a1 = np.array([0.0, 0.0])
a2 = np.array([1.0, 0.0])
a3 = np.array([0.5, 1.0])
a4 = np.array([1.0, 1.0])


def f1(x):
    return np.sum((x - a1) ** 2)


def f2(x):
    return np.sum((x - a2) ** 2)


def f3(x):
    return np.sum((x - a3) ** 2)


def f4(x):
    return np.sum((x - a4) ** 2)


def f(x, w):
    return w[0] * f1(x) + w[1] * f2(x) + w[2] * f3(x) + w[3] * f4(x)

# Sample 10 weight vectors on the 4-simplex
w_list = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.25, 0.25, 0.25, 0.25],
    [0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.0, 0.5, 0.0],
    [0.0, 0.5, 0.0, 0.5],
    [0.33, 0.33, 0.34, 0.0],
])

optimals = []
for w in w_list:
    result = minimize(lambda x: f(x, w), x0=np.array([0.5, 0.5]), method="L-BFGS-B")
    x_opt = result.x
    optimals.append((w, x_opt, f1(x_opt), f2(x_opt), f3(x_opt), f4(x_opt)))

# Prepare training data for Bézier simplex fitting
X = np.array([p[0] for p in optimals])
y = np.array([[p[2], p[3], p[4], p[5]] for p in optimals])

# Fit a degree-3 Bézier simplex to the Pareto front
regressor = BezierSimplexRegressor(degree=3)
regressor.fit(X, y)

# For visualization, project to 3D by dropping one objective (e.g., f4)
# Plot f1, f2, f3

# Compute approximation error on training points
pred_train = regressor.predict(X)
max_error = np.max(np.linalg.norm(pred_train - y, axis=1))
print(f"Max training approximation error: {max_error:.4f}")

# Visualize in 3D (f1, f2, f3)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(y[:, 0], y[:, 1], y[:, 2], color="blue", label="Optimization-derived Pareto points", s=40)
ax.scatter(pred_train[:, 0], pred_train[:, 1], pred_train[:, 2], color="red", alpha=0.7, s=20, label="Bézier simplex approximation")
ax.set_xlabel("Distance to demand center A (f₁)")
ax.set_ylabel("Distance to demand center B (f₂)")
ax.set_zlabel("Distance to demand center C (f₃)")
ax.set_title("Four-Objective Facility Location Pareto Front (Projected): Optimization vs Bézier Simplex")
ax.legend()

plt.tight_layout()
plt.savefig("docs/_static/facility_location_pareto_4obj.png", dpi=150, bbox_inches="tight")
print("Plot saved.")