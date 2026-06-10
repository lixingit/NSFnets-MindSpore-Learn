"""
NSFnets MindSpore Module
=========================
Physics-Informed Neural Network for solving incompressible Navier-Stokes equations.
Supports 2D steady, 2D unsteady, 3D unsteady cases.

Original TF1 code by Zihao Hu (2020), ported to MindSpore.

Key architectural difference from TF1:
  TF1:  tf.gradients(u, x) works on tensors in-graph
  MindSpore: ms.grad(fn, grad_position) creates gradient functions
"""

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore import dtype as mstype
import numpy as np
import time

# PyNative mode — easier for research/development
# Device target: set via environment variable MS_DEVICE_TARGET (default: CPU)
#   CPU: export MS_DEVICE_TARGET=CPU
#   GPU: export MS_DEVICE_TARGET=GPU
#   Ascend: export MS_DEVICE_TARGET=Ascend
import os as _os
_device = _os.environ.get("MS_DEVICE_TARGET", "CPU")
ms.set_context(mode=ms.PYNATIVE_MODE, device_target=_device)


# ============================================================
# 1. MLP Network
# ============================================================
class MLP(nn.Cell):
    """Fully-connected network with tanh activation.

    Input is normalized to [-1, 1] using domain bounds.
    Xavier uniform initialization for all layers.
    No activation on output layer.
    """
    def __init__(self, layers, lb, ub):
        super().__init__()
        self.lb = Parameter(Tensor(lb.astype(np.float32)), requires_grad=False)
        self.ub = Parameter(Tensor(ub.astype(np.float32)), requires_grad=False)
        self.n_hidden = len(layers) - 2

        self.dense_layers = nn.CellList()
        for i in range(len(layers) - 1):
            self.dense_layers.append(
                nn.Dense(layers[i], layers[i+1],
                         weight_init='xavier_uniform',
                         bias_init='zeros')
            )

    def construct(self, x):
        h = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        for i in range(self.n_hidden):
            h = ops.tanh(self.dense_layers[i](h))
        return self.dense_layers[-1](h)  # linear output


# ============================================================
# 2. NSFnets 2D Steady — Kovasznay Flow
# ============================================================
class NSFnet2D:
    """VP-NSFnet for 2D steady incompressible Navier-Stokes.

    Loss = alpha * L_boundary + L_residual
    where L_residual = |f_u|^2 + |f_v|^2 + |f_e|^2

    PDE residuals:
      f_u = u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)
      f_v = u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy)
      f_e = u_x + v_y

    Uses ms.grad to compute 1st and 2nd order spatial derivatives of
    network outputs w.r.t. inputs.
    """

    def __init__(self, xb, yb, ub, vb, x, y, layers,
                 Re=40.0, alpha=1.0):
        """
        Args:
            xb, yb:  boundary coords, shape (Nb, 1)
            ub, vb:  boundary velocities, shape (Nb, 1)
            x, y:    interior collocation points, shape (Ni, 1)
            layers:  [input_dim, ...hidden..., output_dim]
            Re:      Reynolds number
            alpha:   boundary loss weight
        """
        # Domain bounds
        self.lb = np.array([xb.min(), yb.min()], dtype=np.float32)
        self.ub = np.array([xb.max(), yb.max()], dtype=np.float32)

        # Build network
        self.net = MLP(layers, self.lb, self.ub)

        # Store data as MindSpore Tensors
        self.xb = Tensor(xb.astype(np.float32))
        self.yb = Tensor(yb.astype(np.float32))
        self.ub = Tensor(ub.astype(np.float32))
        self.vb = Tensor(vb.astype(np.float32))
        self.x  = Tensor(x.astype(np.float32))
        self.y  = Tensor(y.astype(np.float32))
        self.alpha = alpha
        self.Re = Re

        # ---- Build gradient functions via ms.grad ----
        # ms.grad(fn, grad_position) computes ∂(fn_output)/∂(input_at_position)
        # fn must return a scalar; MindSpore auto-sums if non-scalar.
        # For batch inputs, each output only depends on its own input,
        # so d(Σu_i)/dx_j = ∂u_j/∂x_j (the per-element derivative).

        # Helper: component extractors that return a scalar (summed)
        def u_sum(x, y):
            return ops.sum(self._forward(x, y)[:, 0:1])

        def v_sum(x, y):
            return ops.sum(self._forward(x, y)[:, 1:2])

        # First-order derivatives
        self._du_dx = ms.grad(u_sum, grad_position=0)       # ∂u/∂x
        self._du_dy = ms.grad(u_sum, grad_position=1)       # ∂u/∂y
        self._dv_dx = ms.grad(v_sum, grad_position=0)       # ∂v/∂x
        self._dv_dy = ms.grad(v_sum, grad_position=1)       # ∂v/∂y

        # Pressure gradients
        def p_sum(x, y):
            return ops.sum(self._forward(x, y)[:, 2:3])
        self._dp_dx = ms.grad(p_sum, grad_position=0)       # ∂p/∂x
        self._dp_dy = ms.grad(p_sum, grad_position=1)       # ∂p/∂y

        # Second-order derivatives (grad of grad)
        def du_dx_sum(x, y):
            return ops.sum(self._du_dx(x, y))
        def du_dy_sum(x, y):
            return ops.sum(self._du_dy(x, y))
        def dv_dx_sum(x, y):
            return ops.sum(self._dv_dx(x, y))
        def dv_dy_sum(x, y):
            return ops.sum(self._dv_dy(x, y))

        self._d2u_dx2 = ms.grad(du_dx_sum, grad_position=0)  # ∂²u/∂x²
        self._d2u_dy2 = ms.grad(du_dy_sum, grad_position=1)  # ∂²u/∂y²
        self._d2v_dx2 = ms.grad(dv_dx_sum, grad_position=0)  # ∂²v/∂x²
        self._d2v_dy2 = ms.grad(dv_dy_sum, grad_position=1)  # ∂²v/∂y²

    def _forward(self, x, y):
        """Network forward pass: concat and run MLP."""
        inp = ops.concat([x, y], axis=1)
        return self.net(inp)

    def loss_fn(self):
        """Compute total loss. Called by the training loop.

        Returns:
            total_loss: scalar MindSpore Tensor
        """
        # --- Boundary loss (supervised) ---
        b_out = self._forward(self.xb, self.yb)
        u_b_pred = b_out[:, 0:1]
        v_b_pred = b_out[:, 1:2]
        loss_b_u = ops.mean(ops.square(self.ub - u_b_pred))
        loss_b_v = ops.mean(ops.square(self.vb - v_b_pred))

        # --- Interior PDE residual (unsupervised) ---
        i_out = self._forward(self.x, self.y)
        u = i_out[:, 0:1]
        v = i_out[:, 1:2]

        # First-order spatial derivatives
        u_x = self._du_dx(self.x, self.y)
        u_y = self._du_dy(self.x, self.y)
        v_x = self._dv_dx(self.x, self.y)
        v_y = self._dv_dy(self.x, self.y)
        p_x = self._dp_dx(self.x, self.y)
        p_y = self._dp_dy(self.x, self.y)

        # Second-order spatial derivatives
        u_xx = self._d2u_dx2(self.x, self.y)
        u_yy = self._d2u_dy2(self.x, self.y)
        v_xx = self._d2v_dx2(self.x, self.y)
        v_yy = self._d2v_dy2(self.x, self.y)

        # NS residuals
        f_u = (u * u_x + v * u_y) + p_x - (1.0 / self.Re) * (u_xx + u_yy)
        f_v = (u * v_x + v * v_y) + p_y - (1.0 / self.Re) * (v_xx + v_yy)
        f_e = u_x + v_y

        loss_f_u = ops.mean(ops.square(f_u))
        loss_f_v = ops.mean(ops.square(f_v))
        loss_f_e = ops.mean(ops.square(f_e))

        total = (self.alpha * loss_b_u + self.alpha * loss_b_v +
                 loss_f_u + loss_f_v + loss_f_e)
        return total

    # ---- Training ----
    def adam_train(self, nIter=5000, learning_rate=1e-3, print_every=10):
        """Adam optimizer training with progressive learning rate schedule."""
        optimizer = nn.Adam(self.net.trainable_params(),
                           learning_rate=learning_rate)

        def forward_fn():
            return self.loss_fn()

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

        def train_step():
            loss, grads = grad_fn()
            optimizer(grads)
            return loss

        history = []
        t0 = time.time()
        for it in range(nIter):
            loss = train_step()
            if it % print_every == 0:
                elapsed = time.time() - t0
                val = float(loss.asnumpy())
                history.append((it, val, elapsed))
                print(f'It: {it}, Loss: {val:.3e}, Time: {elapsed:.2f}s')
                t0 = time.time()
        return history

    def lbfgs_train(self, maxiter=50000):
        """Fine-tune with L-BFGS (via mindspore.scipy.optimize.minimize).

        Flattens all network parameters into a single vector, defines
        loss and gradient functions on that vector, and calls LBFGS.
        """
        from mindspore.scipy.optimize import minimize

        params = list(self.net.trainable_params())
        shapes = [p.shape for p in params]
        sizes  = [int(np.prod(s)) for s in shapes]
        total  = sum(sizes)

        def get_flat():
            arr = np.empty(total, dtype=np.float32)
            off = 0
            for p in params:
                sz = int(np.prod(p.shape))
                arr[off:off+sz] = p.asnumpy().ravel()
                off += sz
            return arr

        def set_flat(arr):
            off = 0
            for p in params:
                sz = int(np.prod(p.shape))
                p.set_data(Tensor(arr[off:off+sz].reshape(p.shape), mstype.float32))
                off += sz

        # Build grad function (w.r.t. network parameters)
        fwd = lambda: self.loss_fn()
        grad_fn = ms.value_and_grad(fwd, None, params)

        def loss_flat(arr):
            set_flat(arr)
            return float(self.loss_fn().asnumpy())

        def grad_flat(arr):
            set_flat(arr)
            _, grads = grad_fn()
            g = np.empty(total, dtype=np.float32)
            off = 0
            for grad in grads:
                sz = int(np.prod(grad.shape))
                g[off:off+sz] = grad.asnumpy().ravel()
                off += sz
            return g

        x0 = get_flat()
        print(f'LBFGS: {len(x0)} parameters, maxiter={maxiter}')

        res = minimize(loss_flat, x0, method='LBFGS',
                       jac=grad_flat,
                       options={'maxiter': maxiter})
        set_flat(res)
        print(f'LBFGS done. Loss: {loss_flat(res):.3e}')
        return res

    # ---- Prediction ----
    def predict(self, x_star, y_star):
        """Evaluate (u, v, p) at given test points.

        Args:
            x_star, y_star: numpy arrays, shape (N, 1)
        Returns:
            u_pred, v_pred, p_pred: numpy arrays, shape (N, 1)
        """
        tx = Tensor(x_star.astype(np.float32))
        ty = Tensor(y_star.astype(np.float32))
        out = self._forward(tx, ty)
        return (out[:, 0:1].asnumpy(),
                out[:, 1:2].asnumpy(),
                out[:, 2:3].asnumpy())


# ============================================================
# 3. NSFnets 2D Unsteady — Cylinder Wake
# ============================================================
class NSFnet2DUnsteady:
    """VP-NSFnet for 2D unsteady incompressible Navier-Stokes.

    Input: (x, y, t) -> Output: (u, v, p)

    Loss = alpha * L_initial + beta * L_boundary + L_residual

    PDE residuals include time derivative:
      f_u = u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
      f_v = v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
      f_e = u_x + v_y
    """

    def __init__(self, x0, y0, t0, u0, v0,
                 xb, yb, tb, ub, vb,
                 x, y, t, layers,
                 nu=0.01, alpha=100.0, beta=100.0):
        # Domain bounds (use overall domain for normalization)
        all_x = np.concatenate([x0, xb, x])
        all_y = np.concatenate([y0, yb, y])
        all_t = np.concatenate([t0, tb, t])
        self.lb = np.array([all_x.min(), all_y.min(), all_t.min()], dtype=np.float32)
        self.ub = np.array([all_x.max(), all_y.max(), all_t.max()], dtype=np.float32)

        self.net = MLP(layers, self.lb, self.ub)

        # Store data
        self.x0 = Tensor(x0.astype(np.float32)); self.y0 = Tensor(y0.astype(np.float32))
        self.t0 = Tensor(t0.astype(np.float32))
        self.u0 = Tensor(u0.astype(np.float32)); self.v0 = Tensor(v0.astype(np.float32))
        self.xb = Tensor(xb.astype(np.float32)); self.yb = Tensor(yb.astype(np.float32))
        self.tb = Tensor(tb.astype(np.float32))
        self.ub = Tensor(ub.astype(np.float32)); self.vb = Tensor(vb.astype(np.float32))
        self.x  = Tensor(x.astype(np.float32));  self.y  = Tensor(y.astype(np.float32))
        self.t  = Tensor(t.astype(np.float32))
        self.alpha = alpha; self.beta = beta; self.nu = nu

        # Build gradient functions
        def u_sum(x, y, t):
            return ops.sum(self._forward(x, y, t)[:, 0:1])
        def v_sum(x, y, t):
            return ops.sum(self._forward(x, y, t)[:, 1:2])
        def p_sum(x, y, t):
            return ops.sum(self._forward(x, y, t)[:, 2:3])

        # First-order spatial
        self._du_dx = ms.grad(u_sum, grad_position=0)
        self._du_dy = ms.grad(u_sum, grad_position=1)
        self._dv_dx = ms.grad(v_sum, grad_position=0)
        self._dv_dy = ms.grad(v_sum, grad_position=1)
        self._dp_dx = ms.grad(p_sum, grad_position=0)
        self._dp_dy = ms.grad(p_sum, grad_position=1)
        # First-order temporal
        self._du_dt = ms.grad(u_sum, grad_position=2)
        self._dv_dt = ms.grad(v_sum, grad_position=2)

        # Second-order spatial
        du_dx_s = lambda x, y, t: ops.sum(self._du_dx(x, y, t))
        du_dy_s = lambda x, y, t: ops.sum(self._du_dy(x, y, t))
        dv_dx_s = lambda x, y, t: ops.sum(self._dv_dx(x, y, t))
        dv_dy_s = lambda x, y, t: ops.sum(self._dv_dy(x, y, t))
        self._d2u_dx2 = ms.grad(du_dx_s, grad_position=0)
        self._d2u_dy2 = ms.grad(du_dy_s, grad_position=1)
        self._d2v_dx2 = ms.grad(dv_dx_s, grad_position=0)
        self._d2v_dy2 = ms.grad(dv_dy_s, grad_position=1)

    def _forward(self, x, y, t):
        inp = ops.concat([x, y, t], axis=1)
        return self.net(inp)

    def loss_fn(self):
        # --- Initial condition loss ---
        i0_out = self._forward(self.x0, self.y0, self.t0)
        u0_pred = i0_out[:, 0:1]; v0_pred = i0_out[:, 1:2]
        loss_i = (ops.mean(ops.square(self.u0 - u0_pred)) +
                  ops.mean(ops.square(self.v0 - v0_pred)))

        # --- Boundary loss ---
        b_out = self._forward(self.xb, self.yb, self.tb)
        ub_pred = b_out[:, 0:1]; vb_pred = b_out[:, 1:2]
        loss_b = (ops.mean(ops.square(self.ub - ub_pred)) +
                  ops.mean(ops.square(self.vb - vb_pred)))

        # --- PDE residual ---
        i_out = self._forward(self.x, self.y, self.t)
        u = i_out[:, 0:1]; v = i_out[:, 1:2]

        u_t = self._du_dt(self.x, self.y, self.t)
        u_x = self._du_dx(self.x, self.y, self.t)
        u_y = self._du_dy(self.x, self.y, self.t)
        u_xx = self._d2u_dx2(self.x, self.y, self.t)
        u_yy = self._d2u_dy2(self.x, self.y, self.t)
        v_t = self._dv_dt(self.x, self.y, self.t)
        v_x = self._dv_dx(self.x, self.y, self.t)
        v_y = self._dv_dy(self.x, self.y, self.t)
        v_xx = self._d2v_dx2(self.x, self.y, self.t)
        v_yy = self._d2v_dy2(self.x, self.y, self.t)
        p_x = self._dp_dx(self.x, self.y, self.t)
        p_y = self._dp_dy(self.x, self.y, self.t)

        f_u = u_t + (u * u_x + v * u_y) + p_x - self.nu * (u_xx + u_yy)
        f_v = v_t + (u * v_x + v * v_y) + p_y - self.nu * (v_xx + v_yy)
        f_e = u_x + v_y

        loss_r = (ops.mean(ops.square(f_u)) +
                  ops.mean(ops.square(f_v)) +
                  ops.mean(ops.square(f_e)))

        return self.alpha * loss_i + self.beta * loss_b + loss_r

    def adam_train(self, nIter=5000, learning_rate=1e-3, print_every=10):
        optimizer = nn.Adam(self.net.trainable_params(), learning_rate=learning_rate)
        fwd = lambda: self.loss_fn()
        gf = ms.value_and_grad(fwd, None, optimizer.parameters)

        history = []; t0 = time.time()
        for it in range(nIter):
            loss, grads = gf()
            optimizer(grads)
            if it % print_every == 0:
                elapsed = time.time() - t0; t0 = time.time()
                val = float(loss.asnumpy())
                history.append((it, val, elapsed))
                print(f'It: {it}, Loss: {val:.3e}, Time: {elapsed:.2f}s')
        return history

    def lbfgs_train(self, maxiter=50000):
        from mindspore.scipy.optimize import minimize
        params = list(self.net.trainable_params())
        shapes = [p.shape for p in params]
        sizes  = [int(np.prod(s)) for s in shapes]
        total  = sum(sizes)

        def get_flat():
            arr = np.empty(total, dtype=np.float32); off = 0
            for p in params:
                sz = int(np.prod(p.shape)); arr[off:off+sz] = p.asnumpy().ravel(); off += sz
            return arr

        def set_flat(arr):
            off = 0
            for p in params:
                sz = int(np.prod(p.shape))
                p.set_data(Tensor(arr[off:off+sz].reshape(p.shape), mstype.float32)); off += sz

        fwd = lambda: self.loss_fn()
        gf = ms.value_and_grad(fwd, None, params)

        def loss_flat(arr):
            set_flat(arr); return float(self.loss_fn().asnumpy())

        def grad_flat(arr):
            set_flat(arr); _, grads = gf()
            g = np.empty(total, dtype=np.float32); off = 0
            for grad in grads:
                sz = int(np.prod(grad.shape)); g[off:off+sz] = grad.asnumpy().ravel(); off += sz
            return g

        x0 = get_flat()
        print(f'LBFGS: {len(x0)} params, maxiter={maxiter}')
        res = minimize(loss_flat, x0, method='LBFGS', jac=grad_flat,
                       options={'maxiter': maxiter})
        set_flat(res)
        return res

    def predict(self, x_star, y_star, t_star):
        tx = Tensor(x_star.astype(np.float32))
        ty = Tensor(y_star.astype(np.float32))
        tt = Tensor(t_star.astype(np.float32))
        out = self._forward(tx, ty, tt)
        return (out[:, 0:1].asnumpy(),
                out[:, 1:2].asnumpy(),
                out[:, 2:3].asnumpy())


# ============================================================
# 4. NSFnets 3D Unsteady — Beltrami Flow / Turbulent Channel
# ============================================================
class NSFnet3DUnsteady:
    """VP-NSFnet for 3D unsteady incompressible Navier-Stokes.

    Input: (x, y, z, t) -> Output: (u, v, w, p)

    Loss = alpha * L_initial + beta * L_boundary + L_residual

    PDE residuals:
      f_u = u_t + (u*u_x + v*u_y + w*u_z) + p_x - (1/Re)*(u_xx + u_yy + u_zz)
      f_v = v_t + (u*v_x + v*v_y + w*v_z) + p_y - (1/Re)*(v_xx + v_yy + v_zz)
      f_w = w_t + (u*w_x + v*w_y + w*w_z) + p_z - (1/Re)*(w_xx + w_yy + w_zz)
      f_e = u_x + v_y + w_z
    """

    def __init__(self, x0, y0, z0, t0, u0, v0, w0,
                 xb, yb, zb, tb, ub, vb, wb,
                 x, y, z, t, layers,
                 Re=1.0, alpha=100.0, beta=100.0):
        # Domain bounds
        all_x = np.concatenate([x0, xb, x]); all_y = np.concatenate([y0, yb, y])
        all_z = np.concatenate([z0, zb, z]); all_t = np.concatenate([t0, tb, t])
        self.lb = np.array([all_x.min(), all_y.min(), all_z.min(), all_t.min()], dtype=np.float32)
        self.ub = np.array([all_x.max(), all_y.max(), all_z.max(), all_t.max()], dtype=np.float32)

        self.net = MLP(layers, self.lb, self.ub)

        # Store data
        self.x0=Tensor(x0.astype(np.float32)); self.y0=Tensor(y0.astype(np.float32))
        self.z0=Tensor(z0.astype(np.float32)); self.t0=Tensor(t0.astype(np.float32))
        self.u0=Tensor(u0.astype(np.float32)); self.v0=Tensor(v0.astype(np.float32))
        self.w0=Tensor(w0.astype(np.float32))
        self.xb=Tensor(xb.astype(np.float32)); self.yb=Tensor(yb.astype(np.float32))
        self.zb=Tensor(zb.astype(np.float32)); self.tb=Tensor(tb.astype(np.float32))
        self.ub=Tensor(ub.astype(np.float32)); self.vb=Tensor(vb.astype(np.float32))
        self.wb=Tensor(wb.astype(np.float32))
        self.x =Tensor(x.astype(np.float32));  self.y =Tensor(y.astype(np.float32))
        self.z =Tensor(z.astype(np.float32));  self.t =Tensor(t.astype(np.float32))
        self.alpha=alpha; self.beta=beta; self.Re=Re

        # Build gradient functions (spatial + temporal, 1st + 2nd order)
        def _us(x,y,z,t): return ops.sum(self._forward(x,y,z,t)[:,0:1])
        def _vs(x,y,z,t): return ops.sum(self._forward(x,y,z,t)[:,1:2])
        def _ws(x,y,z,t): return ops.sum(self._forward(x,y,z,t)[:,2:3])
        def _ps(x,y,z,t): return ops.sum(self._forward(x,y,z,t)[:,3:4])

        # 1st order spatial
        self._du_dx=ms.grad(_us,0); self._du_dy=ms.grad(_us,1); self._du_dz=ms.grad(_us,2)
        self._dv_dx=ms.grad(_vs,0); self._dv_dy=ms.grad(_vs,1); self._dv_dz=ms.grad(_vs,2)
        self._dw_dx=ms.grad(_ws,0); self._dw_dy=ms.grad(_ws,1); self._dw_dz=ms.grad(_ws,2)
        self._dp_dx=ms.grad(_ps,0); self._dp_dy=ms.grad(_ps,1); self._dp_dz=ms.grad(_ps,2)
        # 1st order temporal
        self._du_dt=ms.grad(_us,3); self._dv_dt=ms.grad(_vs,3); self._dw_dt=ms.grad(_ws,3)

        # 2nd order spatial
        ux_s = lambda x,y,z,t: ops.sum(self._du_dx(x,y,z,t))
        uy_s = lambda x,y,z,t: ops.sum(self._du_dy(x,y,z,t))
        uz_s = lambda x,y,z,t: ops.sum(self._du_dz(x,y,z,t))
        vx_s = lambda x,y,z,t: ops.sum(self._dv_dx(x,y,z,t))
        vy_s = lambda x,y,z,t: ops.sum(self._dv_dy(x,y,z,t))
        vz_s = lambda x,y,z,t: ops.sum(self._dv_dz(x,y,z,t))
        wx_s = lambda x,y,z,t: ops.sum(self._dw_dx(x,y,z,t))
        wy_s = lambda x,y,z,t: ops.sum(self._dw_dy(x,y,z,t))
        wz_s = lambda x,y,z,t: ops.sum(self._dw_dz(x,y,z,t))

        self._d2u_dx2=ms.grad(ux_s,0); self._d2u_dy2=ms.grad(uy_s,1); self._d2u_dz2=ms.grad(uz_s,2)
        self._d2v_dx2=ms.grad(vx_s,0); self._d2v_dy2=ms.grad(vy_s,1); self._d2v_dz2=ms.grad(vz_s,2)
        self._d2w_dx2=ms.grad(wx_s,0); self._d2w_dy2=ms.grad(wy_s,1); self._d2w_dz2=ms.grad(wz_s,2)

    def _forward(self, x, y, z, t):
        inp = ops.concat([x, y, z, t], axis=1)
        return self.net(inp)

    def loss_fn(self):
        # Initial condition
        i0_out = self._forward(self.x0, self.y0, self.z0, self.t0)
        u0p=i0_out[:,0:1]; v0p=i0_out[:,1:2]; w0p=i0_out[:,2:3]
        loss_i = (ops.mean(ops.square(self.u0-u0p)) +
                  ops.mean(ops.square(self.v0-v0p)) +
                  ops.mean(ops.square(self.w0-w0p)))

        # Boundary
        b_out = self._forward(self.xb, self.yb, self.zb, self.tb)
        ubp=b_out[:,0:1]; vbp=b_out[:,1:2]; wbp=b_out[:,2:3]
        loss_b = (ops.mean(ops.square(self.ub-ubp)) +
                  ops.mean(ops.square(self.vb-vbp)) +
                  ops.mean(ops.square(self.wb-wbp)))

        # PDE residual
        out = self._forward(self.x, self.y, self.z, self.t)
        u=out[:,0:1]; v=out[:,1:2]; w=out[:,2:3]

        args = (self.x, self.y, self.z, self.t)
        u_t=self._du_dt(*args); u_x=self._du_dx(*args); u_y=self._du_dy(*args); u_z=self._du_dz(*args)
        u_xx=self._d2u_dx2(*args); u_yy=self._d2u_dy2(*args); u_zz=self._d2u_dz2(*args)
        v_t=self._dv_dt(*args); v_x=self._dv_dx(*args); v_y=self._dv_dy(*args); v_z=self._dv_dz(*args)
        v_xx=self._d2v_dx2(*args); v_yy=self._d2v_dy2(*args); v_zz=self._d2v_dz2(*args)
        w_t=self._dw_dt(*args); w_x=self._dw_dx(*args); w_y=self._dw_dy(*args); w_z=self._dw_dz(*args)
        w_xx=self._d2w_dx2(*args); w_yy=self._d2w_dy2(*args); w_zz=self._d2w_dz2(*args)
        p_x=self._dp_dx(*args); p_y=self._dp_dy(*args); p_z=self._dp_dz(*args)

        invRe = 1.0/self.Re
        f_u = u_t + (u*u_x+v*u_y+w*u_z) + p_x - invRe*(u_xx+u_yy+u_zz)
        f_v = v_t + (u*v_x+v*v_y+w*v_z) + p_y - invRe*(v_xx+v_yy+v_zz)
        f_w = w_t + (u*w_x+v*w_y+w*w_z) + p_z - invRe*(w_xx+w_yy+w_zz)
        f_e = u_x + v_y + w_z

        loss_r = (ops.mean(ops.square(f_u)) + ops.mean(ops.square(f_v)) +
                  ops.mean(ops.square(f_w)) + ops.mean(ops.square(f_e)))

        return self.alpha*loss_i + self.beta*loss_b + loss_r

    def adam_train(self, nIter=5000, learning_rate=1e-3, print_every=10):
        optimizer = nn.Adam(self.net.trainable_params(), learning_rate=learning_rate)
        fwd = lambda: self.loss_fn()
        gf = ms.value_and_grad(fwd, None, optimizer.parameters)
        history = []; t0 = time.time()
        for it in range(nIter):
            loss, grads = gf()
            optimizer(grads)
            if it % print_every == 0:
                elapsed = time.time() - t0; t0 = time.time()
                val = float(loss.asnumpy())
                history.append((it, val, elapsed))
                print(f'It: {it}, Loss: {val:.3e}, Time: {elapsed:.2f}s')
        return history

    def lbfgs_train(self, maxiter=50000):
        from mindspore.scipy.optimize import minimize
        params = list(self.net.trainable_params())
        shapes = [p.shape for p in params]
        sizes  = [int(np.prod(s)) for s in shapes]
        total  = sum(sizes)

        def get_flat():
            arr = np.empty(total, dtype=np.float32); off = 0
            for p in params:
                sz = int(np.prod(p.shape)); arr[off:off+sz] = p.asnumpy().ravel(); off += sz
            return arr
        def set_flat(arr):
            off = 0
            for p in params:
                sz = int(np.prod(p.shape))
                p.set_data(Tensor(arr[off:off+sz].reshape(p.shape), mstype.float32)); off += sz

        gf = ms.value_and_grad(lambda: self.loss_fn(), None, params)
        def loss_flat(arr):
            set_flat(arr); return float(self.loss_fn().asnumpy())
        def grad_flat(arr):
            set_flat(arr); _, grads = gf()
            g = np.empty(total, dtype=np.float32); off = 0
            for grad in grads:
                sz = int(np.prod(grad.shape)); g[off:off+sz] = grad.asnumpy().ravel(); off += sz
            return g

        x0 = get_flat()
        print(f'LBFGS: {len(x0)} params, maxiter={maxiter}')
        res = minimize(loss_flat, x0, method='LBFGS', jac=grad_flat,
                       options={'maxiter': maxiter})
        set_flat(res)
        return res

    def predict(self, x_star, y_star, z_star, t_star):
        tx=Tensor(x_star.astype(np.float32)); ty=Tensor(y_star.astype(np.float32))
        tz=Tensor(z_star.astype(np.float32)); tt=Tensor(t_star.astype(np.float32))
        out = self._forward(tx, ty, tz, tt)
        return (out[:,0:1].asnumpy(), out[:,1:2].asnumpy(),
                out[:,2:3].asnumpy(), out[:,3:4].asnumpy())


# ============================================================
# Utility: Relative L2 error
# ============================================================
def relative_l2_error(u_pred, u_true):
    """||u_pred - u_true||_2 / ||u_true||_2"""
    return np.linalg.norm(u_true - u_pred, 2) / np.linalg.norm(u_true, 2)
