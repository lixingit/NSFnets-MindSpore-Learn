"""
Test Case 1: Kovasznay Flow (2D Steady)
=========================================
Quick validation with reduced network and iterations.
Original: [2,50,50,50,50,3], 110K Adam iters + LBFGS
Test:     [2,20,20,3],     100 Adam iters, no LBFGS
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from nsfnet_module import NSFnet2D, relative_l2_error

np.random.seed(1234)
print("=" * 60)
print("TEST CASE 1: Kovasznay Flow (2D Steady)")
print("=" * 60)

# --- Data (same domain as original) ---
Re = 40
lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

x = np.linspace(-0.5, 1.0, 101)
y = np.linspace(-0.5, 1.5, 101)

yb1 = np.array([-0.5]*100); yb2 = np.array([1.5]*100)
xb1 = np.array([-0.5]*100); xb2 = np.array([1.0]*100)

y_train1 = np.concatenate([y[1:101], y[0:100], xb1, xb2], 0)
x_train1 = np.concatenate([yb1, yb2, x[0:100], x[1:101]], 0)

xb_train = x_train1.reshape(-1,1).astype(np.float32)
yb_train = y_train1.reshape(-1,1).astype(np.float32)
ub_train = (1 - np.exp(lam * xb_train) * np.cos(2*np.pi*yb_train)).astype(np.float32)
vb_train = (lam/(2*np.pi) * np.exp(lam*xb_train) * np.sin(2*np.pi*yb_train)).astype(np.float32)

x_train = ((np.random.rand(500, 1) - 1/3) * 3/2).astype(np.float32)
y_train = ((np.random.rand(500, 1) - 1/4) * 2).astype(np.float32)

# --- Build reduced model ---
layers = [2, 20, 20, 3]  # Smaller: 2 hidden layers, 20 neurons
model = NSFnet2D(xb_train, yb_train, ub_train, vb_train,
                 x_train, y_train, layers,
                 Re=40.0, alpha=1.0)
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

# --- Quick evaluation ---
x_star = ((np.random.rand(100, 1) - 1/3) * 3/2).astype(np.float32)
y_star = ((np.random.rand(100, 1) - 1/4) * 2).astype(np.float32)
u_true = (1 - np.exp(lam * x_star) * np.cos(2*np.pi*y_star)).astype(np.float32)
v_true = (lam/(2*np.pi) * np.exp(lam*x_star) * np.sin(2*np.pi*y_star)).astype(np.float32)

u_pred, v_pred, p_pred = model.predict(x_star, y_star)
err_u = relative_l2_error(u_pred, u_true)
err_v = relative_l2_error(v_pred, v_true)

print(f"Error u: {err_u:.4f}, Error v: {err_v:.4f}")

# --- Result ---
passed = loss1 < loss0 * 0.95  # Loss should decrease by at least 5%
print()
print(f"RESULT: {'PASSED' if passed else 'FAILED'} (loss decrease {'sufficient' if passed else 'insufficient'})")
print("=" * 60)
sys.exit(0 if passed else 1)
