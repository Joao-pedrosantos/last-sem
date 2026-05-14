import io
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian


def _make_dicom(pixels: np.ndarray, photometric: str = "MONOCHROME2",
                window_center: float | None = None, window_width: float | None = None) -> bytes:
    """Build a minimal in-memory DICOM with the given pixel array."""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("test.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    if window_center is not None:
        ds.WindowCenter = window_center
        ds.WindowWidth = window_width
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def test_read_dicom_inverts_monochrome1():
    from backend.cxr_pipeline import read_dicom
    pixels = np.tile(np.linspace(0, 65535, 16, dtype=np.uint16), (16, 1)).T
    raw = _make_dicom(pixels, photometric="MONOCHROME1")
    img = read_dicom(raw)
    assert img.dtype == np.uint8
    assert img.shape == (16, 16)
    assert img[0, 0] > img[-1, 0]


def test_read_dicom_monochrome2_is_not_inverted():
    from backend.cxr_pipeline import read_dicom
    pixels = np.tile(np.linspace(0, 65535, 16, dtype=np.uint16), (16, 1)).T
    raw = _make_dicom(pixels, photometric="MONOCHROME2")
    img = read_dicom(raw)
    assert img[0, 0] < img[-1, 0]


def test_read_dicom_returns_uint8_in_full_range():
    from backend.cxr_pipeline import read_dicom
    pixels = np.tile(np.linspace(0, 65535, 32, dtype=np.uint16), (32, 1)).T
    raw = _make_dicom(pixels, photometric="MONOCHROME2")
    img = read_dicom(raw)
    assert img.min() <= 5
    assert img.max() >= 250


def test_lung_roi_crop_strips_black_border():
    from backend.cxr_pipeline import lung_roi_crop
    img = np.zeros((1024, 1024), dtype=np.uint8)
    img[100:924, 100:924] = 200
    cropped = lung_roi_crop(img)
    assert cropped.shape[0] == cropped.shape[1]
    assert 800 <= cropped.shape[0] <= 850


def test_lung_roi_crop_falls_back_on_uniform_image():
    from backend.cxr_pipeline import lung_roi_crop
    img = np.full((1024, 1024), 128, dtype=np.uint8)
    cropped = lung_roi_crop(img)
    assert cropped.shape[0] == cropped.shape[1]
    assert 850 <= cropped.shape[0] <= 870


def test_lung_roi_crop_falls_back_on_thin_blob():
    from backend.cxr_pipeline import lung_roi_crop
    img = np.zeros((1024, 1024), dtype=np.uint8)
    img[200:824, 500:520] = 255
    cropped = lung_roi_crop(img)
    assert cropped.shape[1] > 100


def test_to_rgb_shape_and_dtype():
    from backend.cxr_pipeline import to_rgb
    img = np.full((824, 824), 128, dtype=np.uint8)
    out = to_rgb(img, size=640)
    assert out.shape == (640, 640, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])


def test_make_val_transforms_produces_tensor():
    import torch
    from backend.cxr_pipeline import make_val_transforms
    img = np.full((640, 640, 3), 128, dtype=np.uint8)
    out = make_val_transforms(640)(image=img)["image"]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 640, 640)
    assert out.dtype == torch.float32


def test_make_train_transforms_is_stochastic():
    import torch
    from backend.cxr_pipeline import make_train_transforms
    img = np.full((640, 640, 3), 128, dtype=np.uint8)
    tf = make_train_transforms(640)
    out1 = tf(image=img)["image"]
    differs = any(not torch.equal(tf(image=img)["image"], out1) for _ in range(20))
    assert differs


def test_tta_predict_averages_original_and_flip():
    import torch
    import torch.nn as nn
    from backend.cxr_pipeline import tta_predict

    class FlipSensitive(nn.Module):
        def forward(self, x):
            left = x[:, :, :, :x.shape[-1] // 2].sum()
            return left.view(1, 1)

    model = FlipSensitive().eval()
    t = torch.zeros(1, 3, 8, 8)
    t[:, :, :, :4] = 1.0
    p = tta_predict(model, t)
    assert 0.7 < p < 0.8
