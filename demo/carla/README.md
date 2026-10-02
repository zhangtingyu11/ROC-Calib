# CARLA example

19 complete point clouds, 19 images and 20 saved object pairs. Download the raw inputs from the repository root:

```bash
python tools/download_demo.py
roc-calib demo/carla/request.json --initial demo/carla/initial.json \
  --output outputs/carla.json --verify demo/carla/expected.json
```

`initial.json` is the frozen object-representative PnP estimate. `expected.json` is read only after fitting. It checks the computed matrix; its error metrics come from the original evaluation.

The point indices, masks and all cloud records are unchanged. `checksums.json` identifies every input. The camera coordinate axes are right, down, forward; cloud coordinates follow the supplied CARLA sensor frame.
