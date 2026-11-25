# ROC-Calib

[English](README.md) · [使用方法](docs/usage.md) · [数据格式](docs/data.md) · [结果](docs/results.md)

在图像和点云中选中同一物体，计算相机—雷达外参。无需标定板，也无需提供初始外参。图像标注可使用 SAM2 辅助。

![标定工作台](docs/assets/workspace.png)

## 运行示例

需要 Python 3.12。示例在 CPU 上运行，不需要启动网页或下载 SAM2。

```bash
git clone https://github.com/zhangtingyu11/ROC-Calib.git
cd ROC-Calib
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock -e .
python tools/download_demo.py
roc-calib demo/carla/request.json --initial demo/carla/initial.json \
  --output outputs/carla.json --verify demo/carla/expected.json
```

示例包含 19 帧完整点云和 20 对已保存标注，使用冻结的物体对应 PnP 初值复现结果。删除 `--initial` 可重新计算 PnP 初值。

输出为 `camera_from_lidar` 的 4×4 矩阵，平移单位为米：`p_camera = T @ p_lidar`。

## 启动网页

安装 Docker 和 NVIDIA Container Toolkit，然后运行：

```bash
./start.sh
```

打开 **http://localhost:3000**，新建组、添加数据、保存物体配对、计算外参。完整步骤见[使用方法](docs/usage.md)。

无 GPU 时使用 `./start.sh --cpu`。数据默认保存在 `./data`，更换端口可使用 `ROC_CALIB_PORT=3002 ./start.sh`。

## 效果

10 组冻结标注、220 对物体的平均误差为 **0.284° / 4.90 cm**。这里统计的是各组结果的算术平均，不代表重新人工标注后的精度保证。[分组结果](docs/results.md)。

代码使用 [MIT 许可证](LICENSE)。引用格式见英文 README。
