# raw-image-opener-app

A small desktop viewer for headerless `.raw` sensor dumps (e.g. from a
machine-vision camera). Pick a folder, and step through the images with
Next / Previous (or the arrow keys).

These `.raw` files have no header, so the app can't detect width, height,
bit depth, or Bayer pattern on its own — you set them in the "Raw format"
panel and the preview updates live. A wrong width typically shows up as
diagonal tearing; nudge it until the image lines up, then leave "Auto
(from file size)" checked and the height will be computed for you. Your
settings are remembered between runs.

Supports 8-bit and 16-bit (little/big-endian) source data, and optional
Bayer demosaicing (RGGB / BGGR / GRBG / GBRG) to a full-color image.

## Run from source

```
pip install -r requirements.txt
python main.py
```

## Build a standalone Windows executable

```
pip install -r requirements.txt pyinstaller
./build.ps1
```

This produces `dist/RawImageViewer.exe` — a single portable file with no
installer and no Python runtime required on the target machine.
