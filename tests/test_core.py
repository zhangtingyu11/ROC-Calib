"""Small checks for geometry and saved-state safety."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

temporary = tempfile.TemporaryDirectory()
os.environ["AUTOCALIB_DATA_ROOT"] = temporary.name

from roc_calib.pixel_region import PixelRegion
from roc_calib import groups
from roc_calib.calibration import _resolve_cloud


class CoreChecks(unittest.TestCase):
    def test_pixel_cell_boundary_and_hole(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        mask[2, 2] = False
        region = PixelRegion(mask)
        np.testing.assert_allclose(region.outside_distance(np.array([
            [.5, 1], [.25, 1], [2, 2], [4, 4], [1, 1],
        ])), [0, .25, .5, np.sqrt(.5), 0])

    def test_cloud_path_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            _resolve_cloud("/data/cache/../../outside.bin")

    def test_stale_save_preserves_newer_annotation(self):
        group = groups.create_group(groups.GroupCreate(name="conflict", rig_id="test"))
        groups.write_camera_state(group["id"], groups.CameraGroupState(annotations=[{"id": "a", "value": "new"}]))
        with self.assertRaises(groups.StateConflict):
            groups.write_camera_state(group["id"], groups.CameraGroupState(annotations=[{"id": "a", "value": "old"}]))
        self.assertEqual(groups.read_camera_state(group["id"]).annotations[0]["value"], "new")

    def test_deleted_group_is_recoverable(self):
        group = groups.create_group(groups.GroupCreate(name="recover", rig_id="test"))
        groups.delete_group(group["id"])
        self.assertFalse((groups.GROUP_ROOT / group["id"]).exists())
        archived = list((Path(temporary.name) / "trash").glob("recover-*/group.json"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(json.loads(archived[0].read_text())["id"], "recover")


if __name__ == "__main__":
    unittest.main()
