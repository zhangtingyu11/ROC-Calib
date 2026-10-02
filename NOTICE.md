# Third-party components

ROC-Calib code is licensed under MIT. The following components are installed separately and keep their original licenses:

| Component | Source |
| --- | --- |
| SAM2 code and checkpoint | https://github.com/facebookresearch/sam2 |
| PyTorch and TorchVision | https://github.com/pytorch/pytorch |
| OpenCV | https://github.com/opencv/opencv |
| NumPy / SciPy | https://numpy.org / https://scipy.org |
| React / Three.js / vinext | https://github.com/facebook/react / https://github.com/mrdoob/three.js / https://github.com/cloudflare/vinext |
| CARLA simulator and content | https://github.com/carla-simulator/carla |

The CARLA demo contains generated sensor observations and saved annotations. It includes no KITTI, nuScenes, PandaSet, Waymo or mine-site raw data. Obtain external datasets from their providers under their respective terms.

Repository organization was informed by OpenMMLab's MMDetection and RTMDet documentation. No OpenMMLab implementation is included.
