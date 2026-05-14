import numpy as np


def test_bbox_percentile_picks_top_activations():
    from backend.grad_cam_utils import bbox_from_heatmap_percentile
    heatmap = np.full((512, 512), 0.1, dtype=np.float32)
    heatmap[100:150, 200:250] = 0.9
    bbox = bbox_from_heatmap_percentile(heatmap, percentile=90.0, blur_sigma=0.0, size=512)
    assert bbox is not None
    cx = bbox["x"] + bbox["width"] // 2
    cy = bbox["y"] + bbox["height"] // 2
    assert 200 - 15 <= cx <= 250 + 15
    assert 100 - 15 <= cy <= 150 + 15


def test_bbox_percentile_returns_none_for_flat_heatmap():
    from backend.grad_cam_utils import bbox_from_heatmap_percentile
    heatmap = np.zeros((512, 512), dtype=np.float32)
    assert bbox_from_heatmap_percentile(heatmap, percentile=90.0) is None


def test_bbox_percentile_blur_widens_bbox():
    from backend.grad_cam_utils import bbox_from_heatmap_percentile
    heatmap = np.full((512, 512), 0.1, dtype=np.float32)
    heatmap[250:260, 250:260] = 1.0
    bbox_sharp = bbox_from_heatmap_percentile(heatmap, percentile=90.0, blur_sigma=0.0)
    bbox_blurred = bbox_from_heatmap_percentile(heatmap, percentile=90.0, blur_sigma=8.0)
    assert bbox_blurred["width"] > bbox_sharp["width"]
