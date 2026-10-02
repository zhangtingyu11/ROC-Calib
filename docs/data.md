# Data format

Use a ROS2 SQLite `.db3` bag or a directory of paired images and point clouds. For a directory, put `autocalib.json` beside the data:

```text
scene/
  autocalib.json
  images/000001.jpg
  lidar/000001.bin
```

```json
{
  "formatVersion": 1,
  "id": "scene",
  "rigId": "rig-1",
  "name": "Scene 1",
  "cameras": [{
    "id": "front",
    "intrinsic": [1000, 0, 960, 0, 1000, 540, 0, 0, 1],
    "distortion": [0, 0, 0, 0, 0],
    "distortionModel": "radtan",
    "frames": [{"label": "000001", "timestampNs": "1000000000", "path": "images/000001.jpg"}]
  }],
  "lidars": [{
    "id": "lidar",
    "pointFormat": "xyzi-f32",
    "frames": [{"label": "000001", "timestampNs": "1000000000", "path": "lidar/000001.bin"}]
  }]
}
```

Replace the example intrinsics with the parameters for your camera. Point coordinates are in meters. Intrinsics are row-major. Paths are relative to the manifest; matching frames have the same label and order. Timestamps are nanoseconds stored as strings.

| Point format | Record |
| --- | --- |
| `xyz-f32` | x, y, z as little-endian float32 |
| `xyzi-f32` | x, y, z, intensity |
| `xyzring-f32` | x, y, z, ring |
| `xyziring-f32` | x, y, z, intensity, ring |
| `autocalib-acp1` | Prepared cache format used by the bundled demo |

For same-name files, generate the manifest with:

```bash
python tools/build_paired_manifest.py /path/to/scene \
  --id scene --rig-id rig-1 --name 'Scene 1' \
  --camera front=images --lidar lidar=lidar --point-format lidar=xyzi-f32 \
  --camera-calibration front=/path/to/camera.json
```

The camera JSON contains `intrinsic`, `distortion` and `distortionModel`. The generator writes the manifest without copying or thinning point clouds. Run `--help` for timestamp and checksum options. The [JSON schema](paired-dataset.schema.json) describes additional fields.

For Docker, place the scene in `data/paired/scene`, then choose it under **添加数据**. Put server-side bags in `data/bags/` and YAML intrinsics in `data/groups/intrinsics/`.

Use static observations or correctly synchronized and compensated moving data. A matching frame label alone does not establish synchronization.
