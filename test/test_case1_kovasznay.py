"""
测试案例 1: Kovasznay 流（2D 定常不可压 Navier-Stokes）
======================================================
快速验证脚本 —— 使用精简网络和少量迭代验证代码可运行性。

原版参数：[2,50,50,50,50,3], 110K Adam 迭代 + LBFGS
测试参数：[2,20,20,3],       100 Adam 迭代, 无 LBFGS
"""
import sys, os
# 将父目录加入模块搜索路径，以便导入 nsfnet_module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from nsfnet_module import NSFnet2D, relative_l2_error

np.random.seed(1234)
print("=" * 60)
print("测试案例 1: Kovasznay 流 (2D 定常)")
print("=" * 60)

# ---- 1. 生成训练数据 ----
# Kovasznay 流解析解参数
Re = 40
lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

# 空间域离散化：x ∈ [-0.5, 1.0], y ∈ [-0.5, 1.5]
x = np.linspace(-0.5, 1.0, 101)
y = np.linspace(-0.5, 1.5, 101)

# 构造四条边上的边界配点
yb1 = np.array([-0.5]*100); yb2 = np.array([1.5]*100)
xb1 = np.array([-0.5]*100); xb2 = np.array([1.0]*100)

y_train1 = np.concatenate([y[1:101], y[0:100], xb1, xb2], 0)
x_train1 = np.concatenate([yb1, yb2, x[0:100], x[1:101]], 0)

xb_train = x_train1.reshape(-1,1).astype(np.float32)   # 边界点 x 坐标
yb_train = y_train1.reshape(-1,1).astype(np.float32)   # 边界点 y 坐标
# 边界上的真实速度（由 Kovasznay 解析解给出）
ub_train = (1 - np.exp(lam * xb_train) * np.cos(2*np.pi*yb_train)).astype(np.float32)
vb_train = (lam/(2*np.pi) * np.exp(lam*xb_train) * np.sin(2*np.pi*yb_train)).astype(np.float32)

# 内部配点（随机采样，用于计算 PDE 残差）
x_train = ((np.random.rand(500, 1) - 1/3) * 3/2).astype(np.float32)
y_train = ((np.random.rand(500, 1) - 1/4) * 2).astype(np.float32)

# ---- 2. 构建精简模型 ----
layers = [2, 20, 20, 3]  # 2个隐藏层，各20个神经元（原版4层50神经元）
model = NSFnet2D(xb_train, yb_train, ub_train, vb_train,
                 x_train, y_train, layers,
                 Re=40.0, alpha=1.0)
n_params = sum(int(np.prod(p.shape)) for p in model.net.trainable_params())
print(f"可训练参数: {n_params}")

# ---- 3. 快速训练 ----
loss0 = float(model.loss_fn().asnumpy())
print(f"初始 Loss: {loss0:.4f}")

t0 = time.time()
model.adam_train(nIter=100, learning_rate=1e-3, print_every=50)
elapsed = time.time() - t0

loss1 = float(model.loss_fn().asnumpy())
print(f"最终 Loss: {loss1:.4f}")
print(f"Loss 下降: {(loss0 - loss1) / loss0 * 100:.1f}%")
print(f"训练耗时: {elapsed:.1f}s")

# ---- 4. 快速评估 ----
# 在随机测试点上比较预测值与解析解
x_star = ((np.random.rand(100, 1) - 1/3) * 3/2).astype(np.float32)
y_star = ((np.random.rand(100, 1) - 1/4) * 2).astype(np.float32)
u_true = (1 - np.exp(lam * x_star) * np.cos(2*np.pi*y_star)).astype(np.float32)
v_true = (lam/(2*np.pi) * np.exp(lam*x_star) * np.sin(2*np.pi*y_star)).astype(np.float32)

u_pred, v_pred, p_pred = model.predict(x_star, y_star)
err_u = relative_l2_error(u_pred, u_true)
err_v = relative_l2_error(v_pred, v_true)
print(f"误差 — u: {err_u:.4f}, v: {err_v:.4f}")

# ---- 5. 判断测试结果 ----
passed = loss1 < loss0 * 0.95  # Loss 至少下降 5% 视为通过
print()
print(f"结果: {'通过' if passed else '未通过'} ({'Loss 下降充分' if passed else 'Loss 下降不足'})")
print("=" * 60)
sys.exit(0 if passed else 1)
