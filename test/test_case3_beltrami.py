"""
Test Case 3: Beltrami Flow (3D Unsteady)
=========================================
Quick validation with reduced network and iterations.
Original: [4,100x10,4], 110K Adam iters + LBFGS
Test:     [4,20,20,20,4],   100 Adam iters, no LBFGS
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from nsfnet_module import NSFnet3DUnsteady, relative_l2_error

np.random.seed(1234)
print("=" * 60)
print("TEST CASE 3: Beltrami Flow (3D Unsteady)")
print("=" * 60)

# --- Analytic solution ---
def beltrami_solution(x, y, z, t, a=1.0, d=1.0):
    ex = np.exp(a*x); ey = np.exp(a*y); ez = np.exp(a*z)
    e2x = np.exp(2*a*x); e2y = np.exp(2*a*y); e2z = np.exp(2*a*z)
    edt = np.exp(-d**2*t); e2dt = np.exp(-2*d**2*t)
    sx = np.sin(a*x+d*y); cx = np.cos(a*x+d*y)
    sy = np.sin(a*y+d*z); cy = np.cos(a*y+d*z)
    sz = np.sin(a*z+d*x); cz = np.cos(a*z+d*x)

    u = -a * (ex * sy + ez * cx) * edt
    v = -a * (ey * sz + ex * cy) * edt
    w = -a * (ez * sx + ey * cz) * edt
    p = -0.5*a**2 * (e2x+e2y+e2z +
         2*sx*cz*np.exp(a*(y+z)) + 2*sy*cx*np.exp(a*(z+x)) + 2*sz*cy*np.exp(a*(x+y))) * e2dt
    return u.astype(np.float32), v.astype(np.float32), w.astype(np.float32), p.astype(np.float32)

print("Analytic solution: defined")

# --- Generate training data (small scale) ---
# Boundary: 6 faces, 100 pts each, 5 time steps
b0 = np.zeros(100, dtype=np.float32) - 1
b1 = np.ones(100, dtype=np.float32)

# 6 faces
xb = np.concatenate([b1, b0, np.random.uniform(-1,1,400).astype(np.float32)])
yb = np.concatenate([np.random.uniform(-1,1,200).astype(np.float32), b1, b0, np.random.uniform(-1,1,200).astype(np.float32)])
zb = np.concatenate([np.random.uniform(-1,1,400).astype(np.float32), b1, b0])
tb = np.zeros(600, dtype=np.float32)  # single time step for test

ub, vb, wb, _ = beltrami_solution(xb, yb, zb, tb)

# Initial condition
N_ini = 500
x0 = np.random.uniform(-1, 1, N_ini).astype(np.float32)
y0 = np.random.uniform(-1, 1, N_ini).astype(np.float32)
z0 = np.random.uniform(-1, 1, N_ini).astype(np.float32)
t0 = np.zeros(N_ini, dtype=np.float32)
u0, v0, w0, _ = beltrami_solution(x0, y0, z0, t0)

# Interior
N_int = 500
xi = np.random.uniform(-1, 1, N_int).astype(np.float32)
yi = np.random.uniform(-1, 1, N_int).astype(np.float32)
zi = np.random.uniform(-1, 1, N_int).astype(np.float32)
ti = np.random.uniform(0, 1, N_int).astype(np.float32)

def col(a): return a.reshape(-1,1).astype(np.float32)

print(f"Initial: {N_ini}, Boundary: {len(xb)}, Interior: {N_int}")

# --- Build reduced model ---
layers = [4, 20, 20, 20, 4]  # 3 hidden layers x 20 (vs 10 x 100)
model = NSFnet3DUnsteady(
    col(x0), col(y0), col(z0), col(t0), col(u0), col(v0), col(w0),
    col(xb), col(yb), col(zb), col(tb), col(ub), col(vb), col(wb),
    col(xi), col(yi), col(zi), col(ti), layers,
    Re=1.0, alpha=100.0, beta=100.0
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

# --- Quick evaluation ---
x_test = np.random.uniform(-1, 1, 100).astype(np.float32)
y_test = np.random.uniform(-1, 1, 100).astype(np.float32)
z_test = np.random.uniform(-1, 1, 100).astype(np.float32)
t_test = np.random.uniform(0, 1, 100).astype(np.float32)
u_true, v_true, w_true, _ = beltrami_solution(x_test, y_test, z_test, t_test)

u_pred, v_pred, w_pred, p_pred = model.predict(
    col(x_test), col(y_test), col(z_test), col(t_test))

err_u = relative_l2_error(u_pred, col(u_true))
err_v = relative_l2_error(v_pred, col(v_true))
err_w = relative_l2_error(w_pred, col(w_true))
print(f"Error u: {err_u:.4f}, Error v: {err_v:.4f}, Error w: {err_w:.4f}")

# --- Result ---
passed = loss1 < loss0 * 0.95
print()
print(f"RESULT: {'PASSED' if passed else 'FAILED'} (loss decrease {'sufficient' if passed else 'insufficient'})")
print("=" * 60)
sys.exit(0 if passed else 1)
