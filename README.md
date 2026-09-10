# raw-image-opener-app

A small desktop viewer for headerless `.raw` sensor dumps (e.g. from a
machine-vision camera). Pick a folder, and step through the images with
Next / Previous (or the arrow keys).

These `.raw` files have no header, so each one needs a matching
`<name>.json` sidecar next to it describing its width, height, pixel
format, and Bayer pattern — files without one can't be opened. The
"Capture metadata" panel shows the sidecar's capture settings
(`exposure_us`, `gain_db`, `light_current_ma`) for the currently displayed
image.

That panel also shows the working distance and the aperture, which the rig
records only in the name of the capture folder — `wd600f4_Capture_…` means
600 mm working distance at f/4. They are read from whichever ancestor folder
of the image matches that pattern, so they show up whether you open the
capture folder itself or its `FullSize_RAW_Images` subfolder, and they stay
visible even if an image's sidecar is missing. Folders named some other way
just show `—`.

The Load Data tab lists the same two values as sortable `WD (mm)` and
`f-number` columns, so you can order a whole sweep of trials by working
distance or aperture before opening one.

Supports 8-bit and 16-bit source data, and optional Bayer demosaicing
(RGGB / BGGR / GRBG / GBRG) to a full-color image, all driven by the
sidecar.

## Run from source

Requires [uv](https://docs.astral.sh/uv/).

```
uv run main.py
```

## OCR

In the Viewer tab, enable **OCR** and drag a box around the code. The
selected, displayed pixels (including rotation and image adjustments) are
processed in a background thread. Multiple recognized lines appear in the
OCR panel. Confidence is the mean model score, not a guarantee of accuracy.
Check critical part identifiers against the image, especially for faint,
rotated, or low-contrast dot-peen marks.

Clear, navigation, and image adjustments discard outdated results. New
selection is paused while OCR is busy. Closing during OCR waits for the
current request to finish; inference cannot currently be cancelled.

The current dependency configuration targets an NVIDIA GPU with a driver
compatible with CUDA 12.6. OCR was verified on Windows with an RTX 3090,
PaddleOCR 3.4.1, PaddlePaddle GPU 3.3.1, and cuDNN 9.9. CPU-only deployment
and standalone executable OCR have not been validated.

### Offline models

The app never downloads OCR models. It requires these complete model
directories in `%USERPROFILE%\.paddlex\official_models`:

- `PP-LCNet_x1_0_doc_ori`
- `UVDoc`
- `PP-LCNet_x1_0_textline_ori`
- `PP-OCRv5_server_det`
- `en_PP-OCRv5_mobile_rec`

Each directory must contain `inference.json`, `inference.pdiparams`, and
`inference.yml`. Copy the model directories from a provisioned machine, or
explicitly download them once on a connected machine using this PowerShell
command from the project directory:

```powershell
@'
import os
from raw_viewer.ocr_processing import _register_cuda_dll_directories
_register_cuda_dll_directories()
os.environ.pop("HF_HUB_OFFLINE", None)
from paddleocr import PaddleOCR
PaddleOCR(use_textline_orientation=True, lang="en")
'@ | uv run python -
```

Set `PADDLE_PDX_CACHE_HOME` before launching to use another cache root
(models remain under its `official_models` subdirectory). Missing or
incomplete models produce an OCR error without a download attempt. Restart
the app after replacing loaded models.

### Tests

Fast preprocessing, offline-model validation, and headless Qt regression tests:

```powershell
uv run python -m unittest discover -s tests -v
```

Real multiline and blank-image inference using cached models, with socket
connections blocked by the test:

```powershell
$env:OCR_REAL_TESTS = "1"
uv run python -m unittest discover -s tests -p test_ocr_integration.py -v
Remove-Item Env:OCR_REAL_TESTS
```

The real-model test uses synthetic text, not a photographed DPM fixture;
it verifies pipeline operation but does not establish DPM accuracy across
different parts, lighting conditions, or arbitrary angles.

## Build a standalone Windows executable

```
./build.ps1
```

This produces `dist/RawImageViewer.exe` — a single portable file with no
installer and no Python runtime required on the target machine.
