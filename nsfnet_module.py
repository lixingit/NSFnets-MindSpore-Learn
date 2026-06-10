"""
NSFnets MindSpore 核心模块
===========================
基于物理信息神经网络（PINN）求解不可压 Navier-Stokes 方程。
支持 2D 定常、2D 非定常、3D 非定常三种情况。

原 TF1 代码 by Zihao Hu (2020)，迁移至 MindSpore 框架。

TF1 与 MindSpore 的关键架构差异：
  TF1:       tf.gradients(u, x) 在图内直接对张量求导
  MindSpore: ms.grad(fn, grad_position) 对函数创建梯度算子
"""

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore import dtype as mstype
import numpy as np
import time

# ============================================================
# 运行模式与设备配置
# ============================================================
# PyNative 动态图模式，便于科研调试（生产环境可切换为 GRAPH_MODE）
# 设备目标通过环境变量 MS_DEVICE_TARGET 控制：
#   CPU:    export MS_DEVICE_TARGET=CPU
#   GPU:    export MS_DEVICE_TARGET=GPU
#   Ascend: export MS_DEVICE_TARGET=Ascend
import os as _os
_device = _os.environ.get("MS_DEVICE_TARGET", "CPU")
ms.set_context(mode=ms.PYNATIVE_MODE, device_target=_device)


# ============================================================
# 1. MLP 多层感知机网络
# ============================================================
class MLP(nn.Cell):
    """全连接神经网络（多层感知机），tanh 激活函数。

    功能：
      - 将输入归一化到 [-1, 1] 区间
      - 隐藏层使用 tanh 激活，输出层为线性（无激活函数）
      - 权重采用 Xavier 均匀分布初始化

    参数：
      layers: 每层神经元数量，如 [2, 50, 50, 50, 50, 3] 表示
              输入2维 → 4个隐藏层各50神经元 → 输出3维
      lb: 输入域下界 (input_dim,)
      ub: 输入域上界 (input_dim,)
    """
    def __init__(self, layers, lb, ub):
        super().__init__()
        # 输入归一化参数（不可训练，仅用于前向计算时的数据缩放）
        self.lb = Parameter(Tensor(lb.astype(np.float32)), requires_grad=False)
        self.ub = Parameter(Tensor(ub.astype(np.float32)), requires_grad=False)
        self.n_hidden = len(layers) - 2  # 隐藏层数量（不含输入层和输出层）

        # 逐层构建全连接层
        self.dense_layers = nn.CellList()
        for i in range(len(layers) - 1):
            self.dense_layers.append(
                nn.Dense(layers[i], layers[i+1],
                         weight_init='xavier_uniform',  # Xavier 均匀初始化
                         bias_init='zeros')              # 偏置初始化为0
            )

    def construct(self, x):
        """前向传播。

        输入:
          x: (N, input_dim) 张量
        返回:
          (N, output_dim) 张量，经归一化→隐藏层(tanh)→输出层(线性)
        """
        # 步骤1：输入归一化到 [-1, 1]
        h = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        # 步骤2：通过隐藏层，每层后接 tanh 激活
        for i in range(self.n_hidden):
            h = ops.tanh(self.dense_layers[i](h))
        # 步骤3：输出层（线性，无激活函数）
        return self.dense_layers[-1](h)


# ============================================================
# 2. 2D 定常 NSFnet —— Kovasznay 流
# ============================================================
class NSFnet2D:
    """2D 定常不可压 Navier-Stokes 方程的物理信息神经网络。

    适用场景：Kovasznay 流（有解析解，Re=40）

    网络结构：输入 (x, y) → 隐藏层 → 输出 (u, v, p)

    损失函数组成：
      Loss = α · L_boundary（边界条件损失，有监督）
           + L_residual（PDE 残差损失，无监督）

    PDE 残差（不可压 NS 方程）：
      f_u = u·∂u/∂x + v·∂u/∂y + ∂p/∂x - (1/Re)(∂²u/∂x² + ∂²u/∂y²) = 0
      f_v = u·∂v/∂x + v·∂v/∂y + ∂p/∂y - (1/Re)(∂²v/∂x² + ∂²v/∂y²) = 0
      f_e = ∂u/∂x + ∂v/∂y = 0（连续性方程）

    关键实现：
      使用 ms.grad 计算网络输出对输入的 1 阶和 2 阶空间偏导数。
      ms.grad(fn, grad_position=N) 对函数 fn 的第 N 个位置参数求导。
      对于批量输入，各样本相互独立，因此 d(Σu_i)/dx_j = ∂u_j/∂x_j。
    """

    def __init__(self, xb, yb, ub, vb, x, y, layers,
                 Re=40.0, alpha=1.0):
        """初始化 2D 定常 NSFnet。

        参数：
          xb, yb: 边界配点坐标，形状 (Nb, 1)
          ub, vb: 边界上的真实速度值，形状 (Nb, 1)
          x, y:   内部配点坐标，形状 (Ni, 1)
          layers: 网络结构列表，如 [2, 50, 50, 50, 50, 3]
          Re:     雷诺数（默认 40）
          alpha:  边界损失权重（默认 1.0）
        """
        # --- 计算输入域边界（用于归一化） ---
        self.lb = np.array([xb.min(), yb.min()], dtype=np.float32)
        self.ub = np.array([xb.max(), yb.max()], dtype=np.float32)

        # --- 构建 MLP 网络 ---
        self.net = MLP(layers, self.lb, self.ub)

        # --- 将训练数据存储为 MindSpore 张量 ---
        self.xb = Tensor(xb.astype(np.float32))  # 边界点 x 坐标
        self.yb = Tensor(yb.astype(np.float32))  # 边界点 y 坐标
        self.ub = Tensor(ub.astype(np.float32))  # 边界真实 u 速度
        self.vb = Tensor(vb.astype(np.float32))  # 边界真实 v 速度
        self.x  = Tensor(x.astype(np.float32))   # 内部点 x 坐标
        self.y  = Tensor(y.astype(np.float32))   # 内部点 y 坐标
        self.alpha = alpha  # 边界损失权重
        self.Re = Re         # 雷诺数

        # --- 通过 ms.grad 构建各阶偏导数函数 ---
        # 原理：ms.grad(标量输出函数, grad_position=位置索引)
        # 返回一个新函数，计算输出对指定位置参数的梯度

        # 定义分量求和函数（用于构造标量输出，供 ms.grad 使用）
        def u_sum(x, y):
            """返回 u 分量的求和（标量），用于计算 ∂u/∂x, ∂u/∂y"""
            return ops.sum(self._forward(x, y)[:, 0:1])

        def v_sum(x, y):
            """返回 v 分量的求和（标量），用于计算 ∂v/∂x, ∂v/∂y"""
            return ops.sum(self._forward(x, y)[:, 1:2])

        # 一阶空间导数：∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y
        self._du_dx = ms.grad(u_sum, grad_position=0)   # ∂u/∂x
        self._du_dy = ms.grad(u_sum, grad_position=1)   # ∂u/∂y
        self._dv_dx = ms.grad(v_sum, grad_position=0)   # ∂v/∂x
        self._dv_dy = ms.grad(v_sum, grad_position=1)   # ∂v/∂y

        # 压力梯度：∂p/∂x, ∂p/∂y
        def p_sum(x, y):
            """返回 p 分量的求和（标量），用于计算 ∂p/∂x, ∂p/∂y"""
            return ops.sum(self._forward(x, y)[:, 2:3])
        self._dp_dx = ms.grad(p_sum, grad_position=0)   # ∂p/∂x
        self._dp_dy = ms.grad(p_sum, grad_position=1)   # ∂p/∂y

        # 二阶空间导数：对一阶导数再次求导
        # 先将一阶导数结果求和转为标量，再用 ms.grad 求导
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
        """网络前向传播：拼接输入坐标 → 通过 MLP → 输出 (u, v, p)。

        输入:
          x, y: (N, 1) 空间坐标张量
        返回:
          (N, 3) 张量，列为 [u, v, p]
        """
        inp = ops.concat([x, y], axis=1)
        return self.net(inp)

    # ---- 损失函数 ----
    def loss_fn(self):
        """计算总损失函数。

        损失 = α · 边界MSE + PDE残差MSE

        返回:
          total_loss: 标量 MindSpore 张量
        """
        # === 第一部分：边界条件损失（有监督学习） ===
        # 网络在边界点上的预测值应与真实值一致
        b_out = self._forward(self.xb, self.yb)
        u_b_pred = b_out[:, 0:1]  # 边界上预测的 u
        v_b_pred = b_out[:, 1:2]  # 边界上预测的 v
        loss_b_u = ops.mean(ops.square(self.ub - u_b_pred))  # u 边界 MSE
        loss_b_v = ops.mean(ops.square(self.vb - v_b_pred))  # v 边界 MSE

        # === 第二部分：PDE 残差损失（无监督学习） ===
        # 计算内部配点上的 NS 方程残差，网络需学习满足物理规律
        i_out = self._forward(self.x, self.y)
        u = i_out[:, 0:1]  # 预测的 u
        v = i_out[:, 1:2]  # 预测的 v

        # 调用预构建的梯度函数，计算一阶偏导数
        u_x = self._du_dx(self.x, self.y)      # ∂u/∂x
        u_y = self._du_dy(self.x, self.y)      # ∂u/∂y
        v_x = self._dv_dx(self.x, self.y)      # ∂v/∂x
        v_y = self._dv_dy(self.x, self.y)      # ∂v/∂y
        p_x = self._dp_dx(self.x, self.y)      # ∂p/∂x
        p_y = self._dp_dy(self.x, self.y)      # ∂p/∂y

        # 调用预构建的梯度函数，计算二阶偏导数
        u_xx = self._d2u_dx2(self.x, self.y)   # ∂²u/∂x²
        u_yy = self._d2u_dy2(self.x, self.y)   # ∂²u/∂y²
        v_xx = self._d2v_dx2(self.x, self.y)   # ∂²v/∂x²
        v_yy = self._d2v_dy2(self.x, self.y)   # ∂²v/∂y²

        # 构建 NS 方程残差
        f_u = (u * u_x + v * u_y) + p_x - (1.0 / self.Re) * (u_xx + u_yy)  # x 动量
        f_v = (u * v_x + v * v_y) + p_y - (1.0 / self.Re) * (v_xx + v_yy)  # y 动量
        f_e = u_x + v_y                                                      # 连续性

        # 残差的均方误差
        loss_f_u = ops.mean(ops.square(f_u))
        loss_f_v = ops.mean(ops.square(f_v))
        loss_f_e = ops.mean(ops.square(f_e))

        # 总损失 = 边界损失 + PDE 残差损失
        total = (self.alpha * loss_b_u + self.alpha * loss_b_v +
                 loss_f_u + loss_f_v + loss_f_e)
        return total

    # ---- Adam 优化器训练 ----
    def adam_train(self, nIter=5000, learning_rate=1e-3, print_every=10):
        """使用 Adam 优化器训练网络。

        Adam 是一种自适应学习率的一阶梯度优化算法，适合大规模参数和高噪声梯度场景。
        训练采用渐进降低学习率的策略：先用较大学习率快速收敛，再逐步降低做精细调整。

        参数：
          nIter:        迭代次数
          learning_rate: 学习率
          print_every:  每隔多少步打印一次 loss
        返回：
          history: [(迭代号, loss值, 耗时), ...] 训练历史记录
        """
        # 创建 Adam 优化器，传入网络可训练参数
        optimizer = nn.Adam(self.net.trainable_params(),
                           learning_rate=learning_rate)

        # 定义前向计算函数
        def forward_fn():
            return self.loss_fn()

        # 使用 value_and_grad 同时获取损失值和梯度
        # 参数说明：forward_fn=前向函数, None=不对额外参数求导,
        #           optimizer.parameters=对网络参数求导
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

        # 定义单步训练：计算梯度 → 更新参数
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

    # ---- L-BFGS 二阶优化精调 ----
    def lbfgs_train(self, maxiter=50000):
        """使用 L-BFGS 拟牛顿法对网络进行精调。

        L-BFGS 是二阶优化算法，利用历史梯度信息近似 Hessian 矩阵的逆，
        在接近最优解时收敛速度远快于 Adam。但每次迭代计算量较大，
        通常在 Adam 训练后作为精调步骤使用。

        实现方式：
          1. 将所有网络参数展平为一个一维向量
          2. 定义该向量上的损失函数和梯度函数
          3. 调用 MindSpore 的 SciPy 兼容接口 minimize(method='LBFGS')

        参数：
          maxiter: L-BFGS 最大迭代次数
        返回：
          minimize 的优化结果对象
        """
        from mindspore.scipy.optimize import minimize

        # 获取所有可训练参数及其形状
        params = list(self.net.trainable_params())
        shapes = [p.shape for p in params]
        sizes  = [int(np.prod(s)) for s in shapes]  # 每个参数的标量元素数量
        total  = sum(sizes)                           # 参数总数量

        # 将参数展平为一维 numpy 数组
        def get_flat():
            arr = np.empty(total, dtype=np.float32)
            off = 0
            for p in params:
                sz = int(np.prod(p.shape))
                arr[off:off+sz] = p.asnumpy().ravel()
                off += sz
            return arr

        # 将一维数组恢复为网络参数
        def set_flat(arr):
            off = 0
            for p in params:
                sz = int(np.prod(p.shape))
                p.set_data(Tensor(arr[off:off+sz].reshape(p.shape), mstype.float32))
                off += sz

        # 构建梯度计算函数（对网络参数求导）
        fwd = lambda: self.loss_fn()
        grad_fn = ms.value_and_grad(fwd, None, params)

        # 展平版本的损失函数
        def loss_flat(arr):
            set_flat(arr)
            return float(self.loss_fn().asnumpy())

        # 展平版本的梯度函数
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
        print(f'LBFGS: {len(x0)} 个参数, 最大迭代={maxiter}')

        res = minimize(loss_flat, x0, method='LBFGS',
                       jac=grad_flat,
                       options={'maxiter': maxiter})
        set_flat(res)  # 将最优参数写回网络
        print(f'LBFGS 完成. 最终 Loss: {loss_flat(res):.3e}')
        return res

    # ---- 模型推理 ----
    def predict(self, x_star, y_star):
        """在给定测试点上预测速度场和压力场。

        参数：
          x_star, y_star: numpy 数组，形状 (N, 1)，测试点坐标
        返回：
          (u_pred, v_pred, p_pred): numpy 数组，各形状 (N, 1)
        """
        tx = Tensor(x_star.astype(np.float32))
        ty = Tensor(y_star.astype(np.float32))
        out = self._forward(tx, ty)
        return (out[:, 0:1].asnumpy(),   # u 分量
                out[:, 1:2].asnumpy(),   # v 分量
                out[:, 2:3].asnumpy())   # p 分量


# ============================================================
# 3. 2D 非定常 NSFnet —— 圆柱绕流
# ============================================================
class NSFnet2DUnsteady:
    """2D 非定常不可压 Navier-Stokes 方程的物理信息神经网络。

    适用场景：圆柱绕流尾迹（Re=100，ν=0.01）

    网络结构：输入 (x, y, t) → 隐藏层 → 输出 (u, v, p)

    损失函数组成：
      Loss = α · L_initial（初始条件损失，t=0 时刻）
           + β · L_boundary（边界条件损失）
           + L_residual（PDE 残差损失，含时间导数项）

    PDE 残差（含时间导数）：
      f_u = ∂u/∂t + u·∂u/∂x + v·∂u/∂y + ∂p/∂x - ν(∂²u/∂x² + ∂²u/∂y²) = 0
      f_v = ∂v/∂t + u·∂v/∂x + v·∂v/∂y + ∂p/∂y - ν(∂²v/∂x² + ∂²v/∂y²) = 0
      f_e = ∂u/∂x + ∂v/∂y = 0
    """

    def __init__(self, x0, y0, t0, u0, v0,
                 xb, yb, tb, ub, vb,
                 x, y, t, layers,
                 nu=0.01, alpha=100.0, beta=100.0):
        """初始化 2D 非定常 NSFnet。

        参数：
          x0,y0,t0: 初始条件配点坐标（t=0 时刻），形状 (N0, 1)
          u0,v0:    初始条件真实速度，形状 (N0, 1)
          xb,yb,tb: 边界配点坐标，形状 (Nb, 1)
          ub,vb:    边界真实速度，形状 (Nb, 1)
          x,y,t:    内部配点坐标，形状 (Ni, 1)
          layers:   网络结构，如 [3, 50, 50, 50, 50, 3]
          nu:       运动粘度（0.01 对应 Re≈100）
          alpha:    初始条件损失权重
          beta:     边界条件损失权重
        """
        # 用全部数据（初始 + 边界 + 内部）的极值确定归一化范围
        all_x = np.concatenate([x0, xb, x])
        all_y = np.concatenate([y0, yb, y])
        all_t = np.concatenate([t0, tb, t])
        self.lb = np.array([all_x.min(), all_y.min(), all_t.min()], dtype=np.float32)
        self.ub = np.array([all_x.max(), all_y.max(), all_t.max()], dtype=np.float32)

        self.net = MLP(layers, self.lb, self.ub)

        # 存储初始条件数据
        self.x0 = Tensor(x0.astype(np.float32)); self.y0 = Tensor(y0.astype(np.float32))
        self.t0 = Tensor(t0.astype(np.float32))
        self.u0 = Tensor(u0.astype(np.float32)); self.v0 = Tensor(v0.astype(np.float32))

        # 存储边界条件数据
        self.xb = Tensor(xb.astype(np.float32)); self.yb = Tensor(yb.astype(np.float32))
        self.tb = Tensor(tb.astype(np.float32))
        self.ub = Tensor(ub.astype(np.float32)); self.vb = Tensor(vb.astype(np.float32))

        # 存储内部配点数据
        self.x  = Tensor(x.astype(np.float32));  self.y  = Tensor(y.astype(np.float32))
        self.t  = Tensor(t.astype(np.float32))

        self.alpha = alpha; self.beta = beta; self.nu = nu

        # --- 构建梯度函数（空间 + 时间，一阶 + 二阶） ---
        # 分量求和函数（三个输入：x, y, t）
        def u_sum(x, y, t):
            return ops.sum(self._forward(x, y, t)[:, 0:1])
        def v_sum(x, y, t):
            return ops.sum(self._forward(x, y, t)[:, 1:2])
        def p_sum(x, y, t):
            return ops.sum(self._forward(x, y, t)[:, 2:3])

        # 一阶空间导数
        self._du_dx = ms.grad(u_sum, grad_position=0)  # ∂u/∂x
        self._du_dy = ms.grad(u_sum, grad_position=1)  # ∂u/∂y
        self._dv_dx = ms.grad(v_sum, grad_position=0)  # ∂v/∂x
        self._dv_dy = ms.grad(v_sum, grad_position=1)  # ∂v/∂y
        self._dp_dx = ms.grad(p_sum, grad_position=0)  # ∂p/∂x
        self._dp_dy = ms.grad(p_sum, grad_position=1)  # ∂p/∂y
        # 一阶时间导数
        self._du_dt = ms.grad(u_sum, grad_position=2)  # ∂u/∂t
        self._dv_dt = ms.grad(v_sum, grad_position=2)  # ∂v/∂t

        # 二阶空间导数
        du_dx_s = lambda x, y, t: ops.sum(self._du_dx(x, y, t))
        du_dy_s = lambda x, y, t: ops.sum(self._du_dy(x, y, t))
        dv_dx_s = lambda x, y, t: ops.sum(self._dv_dx(x, y, t))
        dv_dy_s = lambda x, y, t: ops.sum(self._dv_dy(x, y, t))
        self._d2u_dx2 = ms.grad(du_dx_s, grad_position=0)  # ∂²u/∂x²
        self._d2u_dy2 = ms.grad(du_dy_s, grad_position=1)  # ∂²u/∂y²
        self._d2v_dx2 = ms.grad(dv_dx_s, grad_position=0)  # ∂²v/∂x²
        self._d2v_dy2 = ms.grad(dv_dy_s, grad_position=1)  # ∂²v/∂y²

    def _forward(self, x, y, t):
        """网络前向传播：拼接 (x,y,t) → MLP → (u,v,p)。"""
        inp = ops.concat([x, y, t], axis=1)
        return self.net(inp)

    def loss_fn(self):
        """计算总损失：α·初始条件损失 + β·边界损失 + PDE残差损失。"""
        # === 初始条件损失（t=0 时刻的速度场匹配） ===
        i0_out = self._forward(self.x0, self.y0, self.t0)
        u0_pred = i0_out[:, 0:1]; v0_pred = i0_out[:, 1:2]
        loss_i = (ops.mean(ops.square(self.u0 - u0_pred)) +
                  ops.mean(ops.square(self.v0 - v0_pred)))

        # === 边界条件损失 ===
        b_out = self._forward(self.xb, self.yb, self.tb)
        ub_pred = b_out[:, 0:1]; vb_pred = b_out[:, 1:2]
        loss_b = (ops.mean(ops.square(self.ub - ub_pred)) +
                  ops.mean(ops.square(self.vb - vb_pred)))

        # === PDE 残差损失（含时间导数项） ===
        i_out = self._forward(self.x, self.y, self.t)
        u = i_out[:, 0:1]; v = i_out[:, 1:2]

        # 计算所有一阶和二阶偏导数
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

        # 非定常 NS 残差（含 ∂u/∂t, ∂v/∂t）
        f_u = u_t + (u * u_x + v * u_y) + p_x - self.nu * (u_xx + u_yy)
        f_v = v_t + (u * v_x + v * v_y) + p_y - self.nu * (v_xx + v_yy)
        f_e = u_x + v_y

        loss_r = (ops.mean(ops.square(f_u)) +
                  ops.mean(ops.square(f_v)) +
                  ops.mean(ops.square(f_e)))

        return self.alpha * loss_i + self.beta * loss_b + loss_r

    def adam_train(self, nIter=5000, learning_rate=1e-3, print_every=10):
        """Adam 优化器训练（与 2D 定常版本逻辑一致）。"""
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
        """L-BFGS 精调（与 2D 定常版本逻辑一致）。"""
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
        print(f'LBFGS: {len(x0)} 个参数, 最大迭代={maxiter}')
        res = minimize(loss_flat, x0, method='LBFGS', jac=grad_flat,
                       options={'maxiter': maxiter})
        set_flat(res)
        return res

    def predict(self, x_star, y_star, t_star):
        """在给定时空测试点上预测速度场和压力场。

        参数：
          x_star,y_star,t_star: numpy 数组，各形状 (N, 1)
        返回：
          (u_pred, v_pred, p_pred): numpy 数组
        """
        tx = Tensor(x_star.astype(np.float32))
        ty = Tensor(y_star.astype(np.float32))
        tt = Tensor(t_star.astype(np.float32))
        out = self._forward(tx, ty, tt)
        return (out[:, 0:1].asnumpy(),
                out[:, 1:2].asnumpy(),
                out[:, 2:3].asnumpy())


# ============================================================
# 4. 3D 非定常 NSFnet —— Beltrami 流 / 湍流槽道流
# ============================================================
class NSFnet3DUnsteady:
    """3D 非定常不可压 Navier-Stokes 方程的物理信息神经网络。

    适用场景：
      - Beltrami 流（有解析解，Re=1）
      - 湍流槽道流（DNS 数据驱动，Re=999.35）

    网络结构：输入 (x, y, z, t) → 隐藏层 → 输出 (u, v, w, p)

    损失函数组成：
      Loss = α · L_initial + β · L_boundary + L_residual

    PDE 残差（完整 3D 非定常 NS 方程）：
      f_u = ∂u/∂t + (u·∇)u + ∂p/∂x - (1/Re)∇²u = 0
      f_v = ∂v/∂t + (u·∇)v + ∂p/∂y - (1/Re)∇²v = 0
      f_w = ∂w/∂t + (u·∇)w + ∂p/∂z - (1/Re)∇²w = 0
      f_e = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
    """

    def __init__(self, x0, y0, z0, t0, u0, v0, w0,
                 xb, yb, zb, tb, ub, vb, wb,
                 x, y, z, t, layers,
                 Re=1.0, alpha=100.0, beta=100.0):
        """初始化 3D 非定常 NSFnet。

        参数：
          x0,y0,z0,t0: 初始条件配点（t=0），形状 (N0, 1)
          u0,v0,w0:    初始条件真实速度，形状 (N0, 1)
          xb,yb,zb,tb: 边界配点，形状 (Nb, 1)
          ub,vb,wb:    边界真实速度，形状 (Nb, 1)
          x,y,z,t:     内部配点，形状 (Ni, 1)
          layers:      网络结构，如 [4, 100, ..., 100, 4]（10个隐藏层）
          Re:          雷诺数
          alpha:       初始条件损失权重
          beta:        边界条件损失权重
        """
        # 计算全域归一化范围
        all_x = np.concatenate([x0, xb, x]); all_y = np.concatenate([y0, yb, y])
        all_z = np.concatenate([z0, zb, z]); all_t = np.concatenate([t0, tb, t])
        self.lb = np.array([all_x.min(), all_y.min(), all_z.min(), all_t.min()], dtype=np.float32)
        self.ub = np.array([all_x.max(), all_y.max(), all_z.max(), all_t.max()], dtype=np.float32)

        self.net = MLP(layers, self.lb, self.ub)

        # 存储初始条件数据
        self.x0=Tensor(x0.astype(np.float32)); self.y0=Tensor(y0.astype(np.float32))
        self.z0=Tensor(z0.astype(np.float32)); self.t0=Tensor(t0.astype(np.float32))
        self.u0=Tensor(u0.astype(np.float32)); self.v0=Tensor(v0.astype(np.float32))
        self.w0=Tensor(w0.astype(np.float32))
        # 存储边界条件数据
        self.xb=Tensor(xb.astype(np.float32)); self.yb=Tensor(yb.astype(np.float32))
        self.zb=Tensor(zb.astype(np.float32)); self.tb=Tensor(tb.astype(np.float32))
        self.ub=Tensor(ub.astype(np.float32)); self.vb=Tensor(vb.astype(np.float32))
        self.wb=Tensor(wb.astype(np.float32))
        # 存储内部配点数据
        self.x =Tensor(x.astype(np.float32));  self.y =Tensor(y.astype(np.float32))
        self.z =Tensor(z.astype(np.float32));  self.t =Tensor(t.astype(np.float32))
        self.alpha=alpha; self.beta=beta; self.Re=Re

        # --- 构建梯度函数（四维输入：x,y,z,t，一阶 + 二阶导数） ---
        # 分量求和函数（转为标量供 ms.grad 使用）
        def _us(x,y,z,t): return ops.sum(self._forward(x,y,z,t)[:,0:1])  # u 分量和
        def _vs(x,y,z,t): return ops.sum(self._forward(x,y,z,t)[:,1:2])  # v 分量和
        def _ws(x,y,z,t): return ops.sum(self._forward(x,y,z,t)[:,2:3])  # w 分量和
        def _ps(x,y,z,t): return ops.sum(self._forward(x,y,z,t)[:,3:4])  # p 分量和

        # 一阶空间导数（grad_position: 0=x, 1=y, 2=z）
        self._du_dx=ms.grad(_us,0); self._du_dy=ms.grad(_us,1); self._du_dz=ms.grad(_us,2)
        self._dv_dx=ms.grad(_vs,0); self._dv_dy=ms.grad(_vs,1); self._dv_dz=ms.grad(_vs,2)
        self._dw_dx=ms.grad(_ws,0); self._dw_dy=ms.grad(_ws,1); self._dw_dz=ms.grad(_ws,2)
        self._dp_dx=ms.grad(_ps,0); self._dp_dy=ms.grad(_ps,1); self._dp_dz=ms.grad(_ps,2)
        # 一阶时间导数（grad_position: 3=t）
        self._du_dt=ms.grad(_us,3); self._dv_dt=ms.grad(_vs,3); self._dw_dt=ms.grad(_ws,3)

        # 二阶空间导数（对一阶导数再求导）
        ux_s=lambda x,y,z,t:ops.sum(self._du_dx(x,y,z,t))
        uy_s=lambda x,y,z,t:ops.sum(self._du_dy(x,y,z,t))
        uz_s=lambda x,y,z,t:ops.sum(self._du_dz(x,y,z,t))
        vx_s=lambda x,y,z,t:ops.sum(self._dv_dx(x,y,z,t))
        vy_s=lambda x,y,z,t:ops.sum(self._dv_dy(x,y,z,t))
        vz_s=lambda x,y,z,t:ops.sum(self._dv_dz(x,y,z,t))
        wx_s=lambda x,y,z,t:ops.sum(self._dw_dx(x,y,z,t))
        wy_s=lambda x,y,z,t:ops.sum(self._dw_dy(x,y,z,t))
        wz_s=lambda x,y,z,t:ops.sum(self._dw_dz(x,y,z,t))

        self._d2u_dx2=ms.grad(ux_s,0); self._d2u_dy2=ms.grad(uy_s,1); self._d2u_dz2=ms.grad(uz_s,2)
        self._d2v_dx2=ms.grad(vx_s,0); self._d2v_dy2=ms.grad(vy_s,1); self._d2v_dz2=ms.grad(vz_s,2)
        self._d2w_dx2=ms.grad(wx_s,0); self._d2w_dy2=ms.grad(wy_s,1); self._d2w_dz2=ms.grad(wz_s,2)

    def _forward(self, x, y, z, t):
        """网络前向传播：拼接 (x,y,z,t) → MLP → (u,v,w,p)。"""
        inp = ops.concat([x, y, z, t], axis=1)
        return self.net(inp)

    def loss_fn(self):
        """计算总损失：α·初始条件 + β·边界条件 + PDE残差（含时间导数 + 3D空间导数）。"""
        # === 初始条件损失 ===
        i0_out = self._forward(self.x0, self.y0, self.z0, self.t0)
        u0p=i0_out[:,0:1]; v0p=i0_out[:,1:2]; w0p=i0_out[:,2:3]
        loss_i = (ops.mean(ops.square(self.u0-u0p)) +
                  ops.mean(ops.square(self.v0-v0p)) +
                  ops.mean(ops.square(self.w0-w0p)))

        # === 边界条件损失 ===
        b_out = self._forward(self.xb, self.yb, self.zb, self.tb)
        ubp=b_out[:,0:1]; vbp=b_out[:,1:2]; wbp=b_out[:,2:3]
        loss_b = (ops.mean(ops.square(self.ub-ubp)) +
                  ops.mean(ops.square(self.vb-vbp)) +
                  ops.mean(ops.square(self.wb-wbp)))

        # === PDE 残差损失（3D 非定常 NS 方程） ===
        out = self._forward(self.x, self.y, self.z, self.t)
        u=out[:,0:1]; v=out[:,1:2]; w=out[:,2:3]

        args = (self.x, self.y, self.z, self.t)
        # 一阶时间导数
        u_t=self._du_dt(*args); v_t=self._dv_dt(*args); w_t=self._dw_dt(*args)
        # 一阶空间导数
        u_x=self._du_dx(*args); u_y=self._du_dy(*args); u_z=self._du_dz(*args)
        v_x=self._dv_dx(*args); v_y=self._dv_dy(*args); v_z=self._dv_dz(*args)
        w_x=self._dw_dx(*args); w_y=self._dw_dy(*args); w_z=self._dw_dz(*args)
        p_x=self._dp_dx(*args); p_y=self._dp_dy(*args); p_z=self._dp_dz(*args)
        # 二阶空间导数
        u_xx=self._d2u_dx2(*args); u_yy=self._d2u_dy2(*args); u_zz=self._d2u_dz2(*args)
        v_xx=self._d2v_dx2(*args); v_yy=self._d2v_dy2(*args); v_zz=self._d2v_dz2(*args)
        w_xx=self._d2w_dx2(*args); w_yy=self._d2w_dy2(*args); w_zz=self._d2w_dz2(*args)

        invRe = 1.0/self.Re
        # x 方向动量残差
        f_u = u_t + (u*u_x+v*u_y+w*u_z) + p_x - invRe*(u_xx+u_yy+u_zz)
        # y 方向动量残差
        f_v = v_t + (u*v_x+v*v_y+w*v_z) + p_y - invRe*(v_xx+v_yy+v_zz)
        # z 方向动量残差
        f_w = w_t + (u*w_x+v*w_y+w*w_z) + p_z - invRe*(w_xx+w_yy+w_zz)
        # 连续性方程残差
        f_e = u_x + v_y + w_z

        loss_r = (ops.mean(ops.square(f_u)) + ops.mean(ops.square(f_v)) +
                  ops.mean(ops.square(f_w)) + ops.mean(ops.square(f_e)))

        return self.alpha*loss_i + self.beta*loss_b + loss_r

    def adam_train(self, nIter=5000, learning_rate=1e-3, print_every=10):
        """Adam 优化器训练（与其他版本逻辑一致）。"""
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
        """L-BFGS 精调（与其他版本逻辑一致）。"""
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
        print(f'LBFGS: {len(x0)} 个参数, 最大迭代={maxiter}')
        res = minimize(loss_flat, x0, method='LBFGS', jac=grad_flat,
                       options={'maxiter': maxiter})
        set_flat(res)
        return res

    def predict(self, x_star, y_star, z_star, t_star):
        """在给定 4D 测试点上预测速度场和压力场。

        参数：
          x_star,y_star,z_star,t_star: numpy 数组，各形状 (N, 1)
        返回：
          (u_pred, v_pred, w_pred, p_pred): numpy 数组
        """
        tx=Tensor(x_star.astype(np.float32)); ty=Tensor(y_star.astype(np.float32))
        tz=Tensor(z_star.astype(np.float32)); tt=Tensor(t_star.astype(np.float32))
        out = self._forward(tx, ty, tz, tt)
        return (out[:,0:1].asnumpy(), out[:,1:2].asnumpy(),
                out[:,2:3].asnumpy(), out[:,3:4].asnumpy())


# ============================================================
# 工具函数：相对 L2 误差
# ============================================================
def relative_l2_error(u_pred, u_true):
    """计算相对 L2 误差：||u_pred - u_true||₂ / ||u_true||₂

    用于评估 PINN 预测精度。值越小说明预测越准确。

    参数：
      u_pred: 预测值 (numpy 数组)
      u_true: 真实值 (numpy 数组)
    返回：
      标量相对误差
    """
    return np.linalg.norm(u_true - u_pred, 2) / np.linalg.norm(u_true, 2)
