# 华为云 ModelArts Notebook 环境配置指南

## 1. 创建 Notebook

在 ModelArts 控制台创建 Notebook 实例：

| 配置项 | 选择 |
|--------|------|
| **镜像** | `mindspore2.3.0-cann8.0.rc1-py3.9-euler2.10.7` 或更高版本 |
| **类型** | GPU（推荐）或 Ascend NPU |
| **规格** | Case 1-2 用 `GPU: 1*V100(32GB)`；Case 3-4 用 `GPU: 1*A100(40GB)` 或以上 |
| **存储** | 50GB+（含 .npy 数据约需 30MB + checkpoints） |

> **推荐镜像版本**：MindSpore **2.3.0+**，Python **3.9+**

---

## 2. Python 环境依赖

### 已测试通过的环境（本地 macOS ARM）

| 包 | 版本 | 说明 |
|----|------|------|
| Python | 3.12.2 | 3.9+ 均可 |
| MindSpore | 2.9.0 | 最低 2.3.0 |
| NumPy | 1.26.4 | **必须 < 2.0**，MindSpore 约束 `numpy>=1.20,<2.0` |
| SciPy | 1.13.1 | ≥1.5.4 即可 |
| protobuf | 4.25.3 | ≥3.13.0 |
| pillow | 10.4.0 | ≥6.2.0 |
| asttokens | 2.0.5 | ≥2.0.4 |
| astunparse | 1.6.3 | ≥1.6.3 |
| safetensors | 0.7.0 | ≥0.4.0 |
| dill | 0.3.8 | ≥0.3.7 |
| psutil | 5.9.0 | ≥5.7.0 |
| packaging | 24.1 | ≥20.0 |

### ModelArts 安装命令

```bash
# 如果 ModelArts 镜像已预装 MindSpore，只需补充：
pip install astunparse safetensors dill

# 如果需要完整安装：
pip install mindspore==2.3.0
pip install numpy==1.26.4 scipy==1.13.1
```

### 验证环境

```python
import mindspore as ms
print(f"MindSpore: {ms.__version__}")
print(f"Device:    {ms.get_context('device_target')}")

import numpy as np
print(f"NumPy:     {np.__version__}")

import scipy
print(f"SciPy:     {scipy.__version__}")
```

---

## 3. 部署代码

将 `NSFnets-MindSpore/` 文件夹上传到 Notebook 的 `/home/ma-user/work/` 目录：

```
/home/ma-user/work/NSFnets-MindSpore/
├── nsfnet_module.py
├── 01_kovasznay_flow.ipynb
├── 02_cylinder_wake.ipynb
├── 03_beltrami_flow.ipynb
├── 04_turbulent_channel.ipynb
└── test/
    ├── test_case1_kovasznay.py
    ├── test_case2_cylinder.py
    ├── test_case3_beltrami.py
    ├── test_case4_channel.py
    └── run_all_tests.sh
```

如果运行 Case 4，还需上传 `.npy` 数据文件到 `npy data/` 目录。

---

## 4. 适配 ModelArts 运行环境

修改 `nsfnet_module.py` 第 18 行，根据硬件类型切换设备：

```python
# ModelArts GPU 环境：
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")

# ModelArts Ascend NPU 环境：
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

# CPU 环境（仅用于调试）：
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
```

> ⚠️ **注意**：MindSpore 2.9.0 的 `set_context(device_target=...)` 已标记为 deprecated，未来版本需改为 `ms.set_device("GPU")`。如果使用 2.9+ 且有警告可忽略。

---

## 5. 运行测试

```bash
# 进入工作目录
cd /home/ma-user/work/NSFnets-MindSpore

# 一键运行 4 个快速测试
cd test
python test_case1_kovasznay.py
python test_case2_cylinder.py
python test_case3_beltrami.py
python test_case4_channel.py
```

---

## 6. 已知兼容性注意

| 问题 | 说明 |
|------|------|
| `numpy>=2.0` 不兼容 | MindSpore 依赖 `numpy<2.0`，如误升级需降级：`pip install 'numpy<2.0'` |
| `FusedSparseAdam` deprecated 警告 | MindSpore 2.8+ 的已知警告，不影响运行 |
| `set_context` deprecated 警告 | 2.9 版本起建议改用 `ms.set_device()`，当前代码仍可用 |
| LBFGS 不可用 | 极少数精简版镜像可能未包含 `mindspore.scipy`，改用纯 Adam 训练即可 |
