import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor

# Define objective functions for communication routing example
# x = [x1, x2] representing routing allocations
def f1(x):
    # Congestion excess: penalize deviation from target capacity
    return x[0]**2 + x[1]**2

def f2(x):
    # Route volatility: penalize deviation from previous allocation (assume previous was [0.5, 0.5])
    prev = np.array([0.5, 0.5])
    return np.sum((x - prev)**2)

def f(x, w):
    # Weighted sum: w[0]*f1 + w[1]*f2
    return w[0] * f1(x) + w[1] * f2(x)

# Sample weights from w=(1,0) to w=(0,1)
n_samples = 10
weights = np.linspace(0, 1, n_samples)
w_list = [(1-w, w) for w in weights]

# For each weight, find optimal x
optimal_points = []
for w in w_list:
    # Initial guess
    x0 = np.array([0.5, 0.5])
    # Minimize f(x, w)
    result = minimize(lambda x: f(x, w), x0, method='L-BFGS-B')
    x_opt = result.x
    f1_val = f1(x_opt)
    f2_val = f2(x_opt)
    optimal_points.append((w, x_opt, f1_val, f2_val))

# Extract data for fitting
w1_vals = np.array([p[0][0] for p in optimal_points])
w2_vals = np.array([p[0][1] for p in optimal_points])
f1_targets = np.array([p[2] for p in optimal_points])
f2_targets = np.array([p[3] for p in optimal_points])

# Fit Bezier simplex (multi-output: predict f1 and f2)
X = np.column_stack([w1_vals, w2_vals])
y = np.column_stack([f1_targets, f2_targets])
regressor = BezierSimplexRegressor(degree=3)
regressor.fit(X, y)

# Generate smooth curve for visualization
t_smooth = np.linspace(0, 1, 100)
w1_smooth = 1 - t_smooth
w2_smooth = t_smooth
X_smooth = np.column_stack([w1_smooth, w2_smooth])
y_smooth = regressor.predict(X_smooth)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(f1_targets, f2_targets, color='blue', label='Optimal points from optimization')
ax.plot(y_smooth[:, 0], y_smooth[:, 1], 'r-', label='Fitted Bezier simplex')
ax.set_xlabel('Congestion Excess (f₁)')
ax.set_ylabel('Route Volatility (f₂)')
ax.set_title('Pareto Front: Optimization vs Bezier Simplex Fitting')
ax.legend()
ax.grid(True, alpha=0.3)

# Calculate approximation error
errors = np.sqrt((y_smooth[:, 0] - np.interp(t_smooth, weights, f1_targets))**2 +
                 (y_smooth[:, 1] - np.interp(t_smooth, weights, f2_targets))**2)
max_error = np.max(errors)
print(f'Max approximation error: {max_error:.4f}')

plt.tight_layout()
plt.savefig('docs/_static/communication_pareto_optimization.png', dpi=150, bbox_inches='tight')
plt.show()