"""
Test Case 2: Cylinder Wake (2D Unsteady)
=========================================
Quick validation with synthetic data (no .mat file needed).
Original: [3,50,50,50,50,3], 110K Adam iters + LBFGS
Test:     [3,20,20,3],     100 Adam iters, no LBFGS
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from nsfnet_module import NSFnet2DUnsteady, relative_l2_error

np.random.seed(1234)
print("=" * 60)
print("TEST CASE 2: Cylinder Wake (2D Unsteady) [Synthetic Data]")
print("=" * 60)

# --- Generate synthetic training data ---
# Domain: x∈[1,8], y∈[-2,2], t∈[0,7]
# For testing, use small datasets

# Initial condition (t=0): random points with simple analytic pattern
N_ini = 200
x0 = np.random.uniform(1, 8, (N_ini, 1)).astype(np.float32)
y0 = np.random.uniform(-2, 2, (N_ini, 1)).astype(np.float32)
t0 = np.zeros((N_ini, 1), dtype=np.float32)
# Simple wake-like velocity pattern
u0 = (1.0 - 0.3 * np.exp(-((y0)/0.5)**2)).astype(np.float32)
v0 = (0.1 * np.sin(x0/2) * np.exp(-((y0)/0.5)**2)).astype(np.float32)

# Boundary points (x=1, x=8, y=-2, y=2)
N_bnd = 100
xb1 = np.ones((N_bnd//4, 1), dtype=np.float32) * 1.0
xb8 = np.ones((N_bnd//4, 1), dtype=np.float32) * 8.0
xb_lo = np.random.uniform(1, 8, (N_bnd//4, 1)).astype(np.float32)
xb_hi = np.random.uniform(1, 8, (N_bnd//4, 1)).astype(np.float32)
xb = np.concatenate([xb1, xb8, xb_lo, xb_hi], 0)
yb = np.concatenate([
    np.random.uniform(-2, 2, (N_bnd//4, 1)).astype(np.float32),
    np.random.uniform(-2, 2, (N_bnd//4, 1)).astype(np.float32),
    np.ones((N_bnd//4, 1), dtype=np.float32) * (-2),
    np.ones((N_bnd//4, 1), dtype=np.float32) * 2,
], 0)
tb = np.random.uniform(0, 7, (N_bnd, 1)).astype(np.float32)
ub = (1.0 - 0.3 * np.exp(-((yb)/0.5)**2) * np.cos(xb)).astype(np.float32)
vb = (0.1 * np.sin(xb/2) * np.exp(-((yb)/0.5)**2)).astype(np.float32)

# Interior collocation points
N_int = 500
x = np.random.uniform(1, 8, (N_int, 1)).astype(np.float32)
y = np.random.uniform(-2, 2, (N_int, 1)).astype(np.float32)
t = np.random.uniform(0, 7, (N_int, 1)).astype(np.float32)

print(f"Initial: {N_ini}, Boundary: {N_bnd}, Interior: {N_int}")

# --- Build reduced model ---
layers = [3, 20, 20, 3]
model = NSFnet2DUnsteady(
    x0, y0, t0, u0, v0,
    xb, yb, tb, ub, vb,
    x, y, t, layers,
    nu=0.01, alpha=100.0, beta=100.0
)
n_params = sum(int(np.prod(p.shape)) for p in model.net.trainable_params())
print(f"Parameters: {n_params}")

# --- Quick training ---
loss0 = float(model.loss_fn().asnumpy())
print(f"Initial loss: {loss0:.4f}")

t0 = time.time()
model.adam_train(nIter=100, learning_rate=1e-3, print_every=50)
elapsed = time.time() - t0

loss1 = float(model.loss_fn().asnumpy())
print(f"Final loss:   {loss1:.4f}")
print(f"Loss reduction: {(loss0 - loss1) / loss0 * 100:.1f}%")
print(f"Training time: {elapsed:.1f}s")

# --- Quick test prediction ---
x_test = np.random.uniform(1, 8, (50, 1)).astype(np.float32)
y_test = np.random.uniform(-2, 2, (50, 1)).astype(np.float32)
t_test = np.ones((50, 1), dtype=np.float32) * 3.5
u_pred, v_pred, p_pred = model.predict(x_test, y_test, t_test)
print(f"Prediction shapes: u={u_pred.shape}, v={v_pred.shape}, p={p_pred.shape}")

# --- Result ---
passed = loss1 < loss0 * 0.95
print()
print(f"RESULT: {'PASSED' if passed else 'FAILED'} (loss decrease {'sufficient' if passed else 'insufficient'})")
print("=" * 60)
sys.exit(0 if passed else 1)
