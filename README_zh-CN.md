# ROC-Calib

[English](README.md) · [论文求解器](docs/paper-solver.md) · [结果](docs/results.md)

现已提供 **2026-10-26 论文采用的纯 C++ 求解器**：全点 PnP 初始化、粗阶段区域对齐、精阶段区域与采样感知有界轮廓优化、边界保持精修。无梯度采样延续，无内部时间截断；不使用强度或外部外参初值。

## 构建和运行（Linux / WSL）

```sh
sudo apt-get install build-essential cmake libeigen3-dev libopencv-dev libssl-dev libblas-dev liblapack-dev
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j4
python3 tools/download_demo.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 build/native/autocalib_native --root demo/carla --group 6 --output outputs/paper-carla.json
```

Python仅用于已有的示例文件下载工具，标定运行链路全部为C++。输出 `solve.matrix` 是LiDAR到相机的4×4变换，平移单位为米。输出文件必须是新文件。

论文十组均值：RRE **0.4062°**、RTE **0.04655 m**。RRE为相对旋转矩阵的内禀ZYX欧拉角绝对值之和，不能与旧版旋转测地角混用。

## 网页工作台

运行 `./start.sh`（CPU 模式用 `./start.sh --cpu`），打开 http://localhost:3000 。用 SAM2、点云聚类或画笔保存配对后，选择 **无初值 PnP**，点击 **计算外参**，即可运行与论文命令行相同的 C++ 区域＋轮廓求解器。选择 **基于初始外参精调** 时，求解器从所选外参开始优化。已有安装可拉取 `main` 后重新运行原启动命令。独立的 Python 命令 `roc-calib` 仍用于旧版区域算法。

[详细说明与验证](docs/paper-solver.md) · [旧版结果](docs/legacy-results.md)
