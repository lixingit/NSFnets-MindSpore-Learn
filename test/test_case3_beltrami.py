"""
测试案例 3: Beltrami 流（3D 非定常不可压 Navier-Stokes）
========================================================
快速验证脚本 —— 使用解析解生成数据和精简网络验证代码可运行性。

原版参数：[4,100×10,4], 110K Adam 迭代 + LBFGS
测试参数：[4,20,20,20,4],  100 Adam 迭代, 无 LBFGS
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from nsfnet_module import NSFnet3DUnsteady, relative_l2_error

np.random.seed(1234)
print("=" * 60)
print("测试案例 3: Beltrami 流 (3D 非定常)")
print("=" * 60)

# ---- 1. Beltrami 流解析解 ----
# 这是 NS 方程少数已知的 3D 精确解之一（a=d=1, Re=1）
def beltrami_solution(x, y, z, t, a=1.0, d=1.0):
    """Beltrami 流解析解，返回 (u, v, w, p)。"""
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

print("Beltrami 解析解: 已定义")

# ---- 2. 生成训练数据 ----
# 空间域: x,y,z ∈ [-1,1], 时间域: t ∈ [0,1]
# 边界配点（6个面，每个面100点，共600点）
b0 = np.zeros(100, dtype=np.float32) - 1  # 面 x=-1, y=-1, z=-1
b1 = np.ones(100, dtype=np.float32)       # 面 x=1, y=1, z=1

# 构造 6 个面上的坐标
xb = np.concatenate([b1, b0, np.random.uniform(-1,1,400).astype(np.float32)])
yb = np.concatenate([np.random.uniform(-1,1,200).astype(np.float32), b1, b0, np.random.uniform(-1,1,200).astype(np.float32)])
zb = np.concatenate([np.random.uniform(-1,1,400).astype(np.float32), b1, b0])
tb = np.zeros(600, dtype=np.float32)  # 单时间步用于测试
ub, vb, wb, _ = beltrami_solution(xb, yb, zb, tb)

# 初始条件配点（t=0，随机采样500点）
N_ini = 500
x0 = np.random.uniform(-1, 1, N_ini).astype(np.float32)
y0 = np.random.uniform(-1, 1, N_ini).astype(np.float32)
z0 = np.random.uniform(-1, 1, N_ini).astype(np.float32)
t0 = np.zeros(N_ini, dtype=np.float32)
u0, v0, w0, _ = beltrami_solution(x0, y0, z0, t0)

# 内部配点（用于计算 PDE 残差）
N_int = 500
xi = np.random.uniform(-1, 1, N_int).astype(np.float32)
yi = np.random.uniform(-1, 1, N_int).astype(np.float32)
zi = np.random.uniform(-1, 1, N_int).astype(np.float32)
ti = np.random.uniform(0, 1, N_int).astype(np.float32)

# 辅助函数：转为 (N,1) 列向量
def col(a): return a.reshape(-1,1).astype(np.float32)

print(f"数据量：初始 {N_ini} 点, 边界 {len(xb)} 点, 内部 {N_int} 点")

# ---- 3. 构建精简模型 ----
layers = [4, 20, 20, 20, 4]  # 输入(x,y,z,t) → 3层各20神经元 → 输出(u,v,w,p)
model = NSFnet3DUnsteady(
    col(x0), col(y0), col(z0), col(t0), col(u0), col(v0), col(w0),
    col(xb), col(yb), col(zb), col(tb), col(ub), col(vb), col(wb),
    col(xi), col(yi), col(zi), col(ti), layers,
    Re=1.0, alpha=100.0, beta=100.0
)
n_params = sum(int(np.prod(p.shape)) for p in model.net.trainable_params())
print(f"可训练参数: {n_params}")

# ---- 4. 快速训练 ----
loss0 = float(model.loss_fn().asnumpy())
print(f"初始 Loss: {loss0:.4f}")

t0 = time.time()
model.adam_train(nIter=100, learning_rate=1e-3, print_every=50)
elapsed = time.time() - t0

loss1 = float(model.loss_fn().asnumpy())
print(f"最终 Loss:   {loss1:.4f}")
print(f"Loss 下降: {(loss0 - loss1) / loss0 * 100:.1f}%")
print(f"训练耗时: {elapsed:.1f}s")

# ---- 5. 快速评估 ----
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
print(f"误差 — u: {err_u:.4f}, v: {err_v:.4f}, w: {err_w:.4f}")

# ---- 6. 判断测试结果 ----
passed = loss1 < loss0 * 0.95
print()
print(f"结果: {'通过' if passed else '未通过'} ({'Loss 下降充分' if passed else 'Loss 下降不足'})")
print("=" * 60)
sys.exit(0 if passed else 1)
