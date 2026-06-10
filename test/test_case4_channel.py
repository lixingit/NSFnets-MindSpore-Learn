"""
测试案例 4: 湍流槽道流（3D 非定常不可压 Navier-Stokes）
======================================================
快速验证脚本 —— 使用预生成的 .npy DNS 数据（或合成数据）验证代码可运行性。

原版参数：[4,100×10,4], ~825K minibatch Adam 迭代 + LBFGS
测试参数：[4,20,20,20,4],  100 Adam 迭代, 无 LBFGS

数据来源：
  - 优先使用 npy data/ 中的 JHU 湍流数据库 DNS 数据
  - 如果 .npy 文件不存在，使用随机合成数据进行结构验证
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from nsfnet_module import NSFnet3DUnsteady, relative_l2_error

np.random.seed(1234)
print("=" * 60)
print("测试案例 4: 湍流槽道流 (3D 非定常)")
print("=" * 60)

# ---- 1. 加载数据 ----
# .npy 数据位于项目根目录的 npy data/ 下
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'npy data')
use_real_data = os.path.exists(data_dir)

if use_real_data:
    print(f"数据来源: {data_dir} (JHU 湍流数据库 DNS 数据)")
    suffix = '1'  # 使用版本1（较小域，129时间步）
    train_ini = np.load(f'{data_dir}/train_ini{suffix}.npy')
    train_iniv = np.load(f'{data_dir}/train_iniv{suffix}.npy')
    train_xb = np.load(f'{data_dir}/train_xb{suffix}.npy')
    train_vb = np.load(f'{data_dir}/train_vb{suffix}.npy')

    # 取子集以加速测试
    n_ini = min(500, len(train_ini))
    n_bnd = min(500, len(train_xb))
    idx_ini = np.random.choice(len(train_ini), n_ini, replace=False)
    idx_bnd = np.random.choice(len(train_xb), n_bnd, replace=False)

    # 初始条件
    x0 = train_ini[idx_ini, 0:1].astype(np.float32)
    y0 = train_ini[idx_ini, 1:2].astype(np.float32)
    z0 = train_ini[idx_ini, 2:3].astype(np.float32)
    t0 = np.zeros_like(x0, dtype=np.float32)
    u0 = train_iniv[idx_ini, 0:1].astype(np.float32)
    v0 = train_iniv[idx_ini, 1:2].astype(np.float32)
    w0 = train_iniv[idx_ini, 2:3].astype(np.float32)

    # 边界条件
    xb = train_xb[idx_bnd, 0:1].astype(np.float32)
    yb = train_xb[idx_bnd, 1:2].astype(np.float32)
    zb = train_xb[idx_bnd, 2:3].astype(np.float32)
    tb = train_xb[idx_bnd, 3:4].astype(np.float32)
    ub = train_vb[idx_bnd, 0:1].astype(np.float32)
    vb = train_vb[idx_bnd, 1:2].astype(np.float32)
    wb = train_vb[idx_bnd, 2:3].astype(np.float32)
else:
    # 回退：使用随机合成数据（仅用于代码结构验证）
    print("警告: .npy 数据未找到，使用合成数据进行结构测试")
    print("(如需真实训练数据，请确保 npy data/ 目录存在)")

    n_ini, n_bnd = 200, 200
    # 槽道流空间域：x∈[12.47,12.66], y∈[-0.9,-0.7], z∈[4.61,4.82]
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

# 内部配点（用于计算 PDE 残差）
n_int = 500
xi = np.random.uniform(12.47, 12.66, (n_int, 1)).astype(np.float32)
yi = np.random.uniform(-0.9, -0.7, (n_int, 1)).astype(np.float32)
zi = np.random.uniform(4.61, 4.82, (n_int, 1)).astype(np.float32)
ti = np.random.uniform(0, 0.1, (n_int, 1)).astype(np.float32)

print(f"数据量：初始 {n_ini} 点, 边界 {n_bnd} 点, 内部 {n_int} 点")

# ---- 2. 构建精简模型 ----
layers = [4, 20, 20, 20, 4]  # 输入(x,y,z,t) → 3层各20神经元 → 输出(u,v,w,p)
model = NSFnet3DUnsteady(
    x0, y0, z0, t0, u0, v0, w0,
    xb, yb, zb, tb, ub, vb, wb,
    xi, yi, zi, ti, layers,
    Re=999.35, alpha=100.0, beta=100.0
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
x_test = np.random.uniform(12.47, 12.66, (50, 1)).astype(np.float32)
y_test = np.random.uniform(-0.9, -0.7, (50, 1)).astype(np.float32)
z_test = np.random.uniform(4.61, 4.82, (50, 1)).astype(np.float32)
t_test = np.zeros((50, 1), dtype=np.float32)
u_pred, v_pred, w_pred, p_pred = model.predict(x_test, y_test, z_test, t_test)
print(f"预测输出形状: u={u_pred.shape}, v={v_pred.shape}, w={w_pred.shape}, p={p_pred.shape}")

# ---- 5. 判断测试结果 ----
passed = loss1 < loss0 * 0.95
print()
print(f"结果: {'通过' if passed else '未通过'} ({'Loss 下降充分' if passed else 'Loss 下降不足'})")
print("=" * 60)
sys.exit(0 if passed else 1)
