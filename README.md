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

Supports 8-bit and 16-bit source data, and optional Bayer demosaicing
(RGGB / BGGR / GRBG / GBRG) to a full-color image, all driven by the
sidecar.

## Run from source

Requires [uv](https://docs.astral.sh/uv/).

```
uv run main.py
```

## Build a standalone Windows executable

```
./build.ps1
```

This produces `dist/RawImageViewer.exe` — a single portable file with no
installer and no Python runtime required on the target machine.
