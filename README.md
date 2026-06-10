# NSFnets — MindSpore 实现

基于物理信息神经网络（PINN）求解不可压 Navier-Stokes 方程，MindSpore 框架实现。

原论文：NSFnets (Navier-Stokes Flow nets)，TF1 原版代码 by Zihao Hu (2020)

## 四个案例

| Notebook | 物理问题 | 维度 | Re |
|----------|---------|------|-----|
| `01_kovasznay_flow.ipynb` | Kovasznay 流 | 2D 定常 | 40 |
| `02_cylinder_wake.ipynb` | 圆柱绕流 | 2D 非定常 | 100 |
| `03_beltrami_flow.ipynb` | Beltrami 流 | 3D 非定常 | 1 |
| `04_turbulent_channel.ipynb` | 湍流槽道流 | 3D 非定常 | 999.35 |

## 快速开始

### 环境要求

- Python 3.9+
- MindSpore 2.3+
- NumPy < 2.0, SciPy

详细配置见 [REQUIREMENTS.md](REQUIREMENTS.md)。

### 安装

```bash
pip install mindspore numpy scipy
```

### 设备配置

```bash
# CPU（默认）
export MS_DEVICE_TARGET=CPU

# GPU
export MS_DEVICE_TARGET=GPU

# Ascend NPU
export MS_DEVICE_TARGET=Ascend
```

### 运行测试

```bash
cd test
python test_case1_kovasznay.py   # 2D 定常
python test_case2_cylinder.py    # 2D 非定常
python test_case3_beltrami.py    # 3D 非定常
python test_case4_channel.py     # 3D 湍流（需 npy data/）

# 或一键运行
bash run_all_tests.sh
```

### 完整训练

在 Jupyter Notebook 中打开对应的 `.ipynb` 文件，按顺序执行所有 Cell。

## 文件结构

```
├── nsfnet_module.py           # 核心模块（MLP + PDE残差 + 训练）
├── 01_kovasznay_flow.ipynb    # Case 1 Notebook
├── 02_cylinder_wake.ipynb     # Case 2 Notebook
├── 03_beltrami_flow.ipynb     # Case 3 Notebook
├── 04_turbulent_channel.ipynb # Case 4 Notebook
├── REQUIREMENTS.md            # 环境配置指南
├── npy data/                  # Case 4 训练数据（12个 .npy）
└── test/                      # 快速验证脚本
```

## 数据说明

- Case 1、3 使用解析解生成训练数据，无需外部文件
- Case 2 需要 `cylinder_nektar_wake.mat`（约 200MB），从 [Raissi/PINNs](https://github.com/maziarraissi/PINNs) 下载
- Case 4 的 .npy 数据已包含在本仓库 `npy data/` 中
