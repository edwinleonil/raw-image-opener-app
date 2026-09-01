# Build a standalone Windows executable with PyInstaller.
# Usage: ./build.ps1
# Output: dist/RawImageViewer.exe (single file, no installer needed)

uv run --group dev pyinstaller --noconfirm --onefile --windowed --name "RawImageViewer" main.py

Write-Host "Executable created at dist\RawImageViewer.exe"
