#!/usr/bin/env python3
"""Generate macOS .icns file from PNG image."""

from PIL import Image
import os
import subprocess

# Load source image
source_path = "/Users/markpinnuck/Dev/GitHub/AusSuperPredictor/resources/asx200predictor.png"
img = Image.open(source_path)

# Create iconset directory
iconset_dir = "/Users/markpinnuck/Dev/GitHub/AusSuperPredictor/resources/asx200predictor.iconset"
os.makedirs(iconset_dir, exist_ok=True)

# Icon sizes: (size, filename)
sizes = [
    (16, "icon_16x16.png"),
    (32, "icon_32x32.png"),
    (64, "icon_64x64.png"),
    (128, "icon_128x128.png"),
    (256, "icon_256x256.png"),
    (512, "icon_512x512.png"),
    (1024, "icon_1024x1024.png"),
]

# Generate icons
print("Generating icon sizes...")
for size, filename in sizes:
    resized = img.resize((size, size), Image.Resampling.LANCZOS)
    resized.save(os.path.join(iconset_dir, filename))
    print(f"✓ {filename} ({size}x{size})")

# Generate retina versions (@2x)
retina_sizes = [
    (32, "icon_16x16@2x.png"),
    (64, "icon_32x32@2x.png"),
    (256, "icon_128x128@2x.png"),
    (512, "icon_256x256@2x.png"),
    (1024, "icon_512x512@2x.png"),
]

for size, filename in retina_sizes:
    resized = img.resize((size, size), Image.Resampling.LANCZOS)
    resized.save(os.path.join(iconset_dir, filename))
    print(f"✓ {filename} ({size}x{size}@2x)")

# Convert iconset to .icns using iconutil
output_icns = "/Users/markpinnuck/Dev/GitHub/AusSuperPredictor/resources/asx200predictor.icns"
print(f"\nConverting to .icns...")
result = subprocess.run(
    ["iconutil", "-c", "icns", "-o", output_icns, iconset_dir],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print(f"✓ Created: {output_icns}")
    # Get file size
    size_mb = os.path.getsize(output_icns) / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")
else:
    print(f"✗ Error: {result.stderr}")
    exit(1)

print("\n✓ Done! .icns file ready for use in Xcode/PyInstaller")
