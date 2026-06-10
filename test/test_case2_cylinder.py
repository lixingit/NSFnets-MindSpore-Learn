"""
测试案例 2: 圆柱绕流（2D 非定常不可压 Navier-Stokes）
====================================================
快速验证脚本 —— 使用合成数据（无需 .mat 文件）和精简网络验证代码可运行性。

原版参数：[3,50,50,50,50,3], 110K Adam 迭代 + LBFGS, 需 CFD 数据
测试参数：[3,20,20,3],       100 Adam 迭代, 无 LBFGS, 合成数据
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from nsfnet_module import NSFnet2DUnsteady, relative_l2_error

np.random.seed(1234)
print("=" * 60)
print("测试案例 2: 圆柱绕流 (2D 非定常) [合成数据]")
print("=" * 60)

# ---- 1. 生成合成训练数据 ----
# 模拟圆柱绕流尾迹的简化速度分布（仅用于代码验证，非真实物理）
# 空间域：x∈[1,8], y∈[-2,2], 时间域：t∈[0,7]

# 初始条件（t=0）
N_ini = 200
x0 = np.random.uniform(1, 8, (N_ini, 1)).astype(np.float32)
y0 = np.random.uniform(-2, 2, (N_ini, 1)).astype(np.float32)
t0 = np.zeros((N_ini, 1), dtype=np.float32)
# 简化的尾迹速度分布：中间快、两侧慢，略带上洗
u0 = (1.0 - 0.3 * np.exp(-((y0)/0.5)**2)).astype(np.float32)
v0 = (0.1 * np.sin(x0/2) * np.exp(-((y0)/0.5)**2)).astype(np.float32)

# 边界条件（四条边界：x=1, x=8, y=-2, y=2）
N_bnd = 100
xb1 = np.ones((N_bnd//4, 1), dtype=np.float32) * 1.0
xb8 = np.ones((N_bnd//4, 1), dtype=np.float32) * 8.0
xb_lo = np.random.uniform(1, 8, (N_bnd//4, 1)).astype(np.float32)
xb_hi = np.random.uniform(1, 8, (N_bnd//4, 1)).astype(np.float32)
xb = np.concatenate([xb1, xb8, xb_lo, xb_hi], 0)
yb = np.concatenate([
    np.random.uniform(-2, 2, (N_bnd//4, 1)).astype(np.float32),   # x=1 边上的 y
    np.random.uniform(-2, 2, (N_bnd//4, 1)).astype(np.float32),   # x=8 边上的 y
    np.ones((N_bnd//4, 1), dtype=np.float32) * (-2),              # y=-2 边上的 x
    np.ones((N_bnd//4, 1), dtype=np.float32) * 2,                 # y=2 边上的 x
], 0)
tb = np.random.uniform(0, 7, (N_bnd, 1)).astype(np.float32)
ub = (1.0 - 0.3 * np.exp(-((yb)/0.5)**2) * np.cos(xb)).astype(np.float32)
vb = (0.1 * np.sin(xb/2) * np.exp(-((yb)/0.5)**2)).astype(np.float32)

# 内部配点（用于计算 PDE 残差）
N_int = 500
x = np.random.uniform(1, 8, (N_int, 1)).astype(np.float32)
y = np.random.uniform(-2, 2, (N_int, 1)).astype(np.float32)
t = np.random.uniform(0, 7, (N_int, 1)).astype(np.float32)

print(f"数据量：初始 {N_ini} 点, 边界 {N_bnd} 点, 内部 {N_int} 点")

# ---- 2. 构建精简模型 ----
layers = [3, 20, 20, 3]  # 输入 (x,y,t) → 2层各20神经元 → 输出 (u,v,p)
model = NSFnet2DUnsteady(
    x0, y0, t0, u0, v0,
    xb, yb, tb, ub, vb,
    x, y, t, layers,
    nu=0.01, alpha=100.0, beta=100.0
)
n_params = sum(int(np.prod(p.shape)) for p in model.net.trainable_params())
print(f"可训练参数: {n_params}")

# ---- 3. 快速训练 ----
loss0 = float(model.loss_fn().asnumpy())
print(f"初始 Loss: {loss0:.4f}")

t0 = time.time()
model.adam_train(nIter=100, learning_rate=1e-3, print_every=50)
elapsed = time.time() - t0

loss1 = float(model.loss_fn().asnumpy())
print(f"最终 Loss:   {loss1:.4f}")
print(f"Loss 下降: {(loss0 - loss1) / loss0 * 100:.1f}%")
print(f"训练耗时: {elapsed:.1f}s")

# ---- 4. 验证推理功能 ----
x_test = np.random.uniform(1, 8, (50, 1)).astype(np.float32)
y_test = np.random.uniform(-2, 2, (50, 1)).astype(np.float32)
t_test = np.ones((50, 1), dtype=np.float32) * 3.5  # t=3.5 时刻
u_pred, v_pred, p_pred = model.predict(x_test, y_test, t_test)
print(f"预测输出形状: u={u_pred.shape}, v={v_pred.shape}, p={p_pred.shape}")

# ---- 5. 判断测试结果 ----
passed = loss1 < loss0 * 0.95
print()
print(f"结果: {'通过' if passed else '未通过'} ({'Loss 下降充分' if passed else 'Loss 下降不足'})")
print("=" * 60)
sys.exit(0 if passed else 1)
