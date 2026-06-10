"""
Test Case 4: Turbulent Channel Flow (3D Unsteady)
==================================================
Quick validation using pre-generated .npy data (if available).
Original: [4,100x10,4], ~825K minibatch Adam iters + LBFGS
Test:     [4,20,20,20,4],   100 Adam iters, no LBFGS
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from nsfnet_module import NSFnet3DUnsteady, relative_l2_error

np.random.seed(1234)
print("=" * 60)
print("TEST CASE 4: Turbulent Channel Flow (3D Unsteady)")
print("=" * 60)

# --- Try to load .npy data ---
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'npy data')

use_real_data = os.path.exists(data_dir)
if use_real_data:
    print(f"Using .npy data from: {data_dir}")
    suffix = '1'
    train_ini = np.load(f'{data_dir}/train_ini{suffix}.npy')
    train_iniv = np.load(f'{data_dir}/train_iniv{suffix}.npy')
    train_xb = np.load(f'{data_dir}/train_xb{suffix}.npy')
    train_vb = np.load(f'{data_dir}/train_vb{suffix}.npy')

    # Use subset of data for speed
    n_ini = min(500, len(train_ini))
    n_bnd = min(500, len(train_xb))

    idx_ini = np.random.choice(len(train_ini), n_ini, replace=False)
    idx_bnd = np.random.choice(len(train_xb), n_bnd, replace=False)

    x0 = train_ini[idx_ini, 0:1].astype(np.float32)
    y0 = train_ini[idx_ini, 1:2].astype(np.float32)
    z0 = train_ini[idx_ini, 2:3].astype(np.float32)
    t0 = np.zeros_like(x0, dtype=np.float32)
    u0 = train_iniv[idx_ini, 0:1].astype(np.float32)
    v0 = train_iniv[idx_ini, 1:2].astype(np.float32)
    w0 = train_iniv[idx_ini, 2:3].astype(np.float32)

    xb = train_xb[idx_bnd, 0:1].astype(np.float32)
    yb = train_xb[idx_bnd, 1:2].astype(np.float32)
    zb = train_xb[idx_bnd, 2:3].astype(np.float32)
    tb = train_xb[idx_bnd, 3:4].astype(np.float32)
    ub = train_vb[idx_bnd, 0:1].astype(np.float32)
    vb = train_vb[idx_bnd, 1:2].astype(np.float32)
    wb = train_vb[idx_bnd, 2:3].astype(np.float32)

    print(f"Initial: {n_ini}, Boundary: {n_bnd}")
else:
    print("WARNING: .npy data not found, using synthetic data for structural test")
    print("(Install pyJHTDB and run DATA/ notebooks to generate real data)")

    n_ini, n_bnd = 200, 200
    x0 = np.random.uniform(12.47, 12.66, (n_ini, 1)).astype(np.float32)
    y0 = np.random.uniform(-0.9, -0.7, (n_ini, 1)).astype(np.float32)
    z0 = np.random.uniform(4.61, 4.82, (n_ini, 1)).astype(np.float32)
    t0 = np.zeros((n_ini, 1), dtype=np.float32)
    u0 = np.random.randn(n_ini, 1).astype(np.float32) * 0.01
    v0 = np.random.randn(n_ini, 1).astype(np.float32) * 0.01
    w0 = np.random.randn(n_ini, 1).astype(np.float32) * 0.01

    xb = np.random.uniform(12.47, 12.66, (n_bnd, 1)).astype(np.float32)
    yb = np.random.uniform(-0.9, -0.7, (n_bnd, 1)).astype(np.float32)
    zb = np.random.uniform(4.61, 4.82, (n_bnd, 1)).astype(np.float32)
    tb = np.random.uniform(0, 0.1, (n_bnd, 1)).astype(np.float32)
    ub = np.random.randn(n_bnd, 1).astype(np.float32) * 0.01
    vb = np.random.randn(n_bnd, 1).astype(np.float32) * 0.01
    wb = np.random.randn(n_bnd, 1).astype(np.float32) * 0.01
    print(f"Initial: {n_ini}, Boundary: {n_bnd}")

# Interior points
n_int = 500
xi = np.random.uniform(12.47, 12.66, (n_int, 1)).astype(np.float32)
yi = np.random.uniform(-0.9, -0.7, (n_int, 1)).astype(np.float32)
zi = np.random.uniform(4.61, 4.82, (n_int, 1)).astype(np.float32)
ti = np.random.uniform(0, 0.1, (n_int, 1)).astype(np.float32)
print(f"Interior: {n_int}")

# --- Build reduced model ---
layers = [4, 20, 20, 20, 4]
model = NSFnet3DUnsteady(
    x0, y0, z0, t0, u0, v0, w0,
    xb, yb, zb, tb, ub, vb, wb,
    xi, yi, zi, ti, layers,
    Re=999.35, alpha=100.0, beta=100.0
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
x_test = np.random.uniform(12.47, 12.66, (50, 1)).astype(np.float32)
y_test = np.random.uniform(-0.9, -0.7, (50, 1)).astype(np.float32)
z_test = np.random.uniform(4.61, 4.82, (50, 1)).astype(np.float32)
t_test = np.zeros((50, 1), dtype=np.float32)
u_pred, v_pred, w_pred, p_pred = model.predict(x_test, y_test, z_test, t_test)
print(f"Prediction shapes: u={u_pred.shape}, v={v_pred.shape}, w={w_pred.shape}, p={p_pred.shape}")

# --- Result ---
passed = loss1 < loss0 * 0.95
print()
print(f"RESULT: {'PASSED' if passed else 'FAILED'} (loss decrease {'sufficient' if passed else 'insufficient'})")
print("=" * 60)
sys.exit(0 if passed else 1)
