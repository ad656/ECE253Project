#!/usr/bin/env python3
"""Create synthetic overexposed patches on images in a directory.

Saves augmented images and binary/soft masks to the output directory.

Usage:
  python tools/create_overexposed_patches.py --input_dir example_images --output_dir tmp_overex --patches_per_image 2

Options include patch radius range, intensity range, and randomness seed.
"""
import argparse
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image


def make_circular_gaussian_mask(h, w, cx, cy, radius, sigma_scale=0.5):
    """Return HxW float mask in [0,1] with a gaussian falloff from center.
    radius controls approximate size (where mask ~ 0.6); sigma_scale scales sigma relative to radius.
    """
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    sigma = max(1.0, radius * sigma_scale)
    mask = np.exp(-d2 / (2.0 * sigma * sigma))
    # Optional: ramp so mask has ~0.01 at r=3*sigma
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


def apply_overexposure(img_arr, mask, intensity):
    """Apply overexposure to img_arr (H x W x 3 float in [0,1]).
    intensity: positive float controlling brightness addition. Larger -> more saturated whites.
    mask: HxW in [0,1].
    Returns new image array clipped to [0,1].
    """
    # Ensure float
    img = img_arr.astype('float32')
    # Additive exposure: push toward white by adding mask*intensity
    # Use broadcast for 3 channels
    add = mask[..., None] * float(intensity)
    out = img + add
    # Clip to [0,1] to simulate sensor saturation
    out = np.clip(out, 0.0, 1.0)
    return out


def process_image(path, out_dir, patches_per_image=2, radius_min=20, radius_max=120, intensity_min=0.6, intensity_max=1.2, seed=None):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype('float32') / 255.0
    h, w = arr.shape[:2]

    base_name = Path(path).stem


    # Prepare subfolders under out_dir (no more rgb values/)
    seeds_dir = out_dir / 'seeds'
    processed_dir = out_dir / 'processed'
    for d in (seeds_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Note: To extract per-pixel RGB values later, just load the image and use numpy:
    #   arr = np.asarray(Image.open(path).convert('RGB')).astype('float32') / 255.0
    #   # arr shape: (H, W, 3), values in [0,1]

    # Seed RNGs for reproducibility for this image
    random.seed(seed)
    np.random.seed(seed if seed is not None else random.randint(0, 2**31 - 1))

    # Also save the seed used for this image
    try:
        seed_file = seeds_dir / f"{base_name}_seed.txt"
        with open(seed_file, 'w') as f:
            f.write(str(int(seed) if seed is not None else 'None'))
    except Exception as e:
        print('Warning: failed to write seed file for', path, 'error:', e)

    # For reproducibility we create specified number of augmented images each with several patches
    for i in range(patches_per_image):
        out = arr.copy()
        combined_mask = np.zeros((h, w), dtype='float32')
        # place 1-3 blobs per augmented image (random)
        blobs = random.randint(1, 3)
        for b in range(blobs):
            radius = random.randint(radius_min, min(radius_max, int(min(h, w) * 0.5)))
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            mask = make_circular_gaussian_mask(h, w, cx, cy, radius, sigma_scale=0.6)
            intensity = random.uniform(intensity_min, intensity_max)
            out = apply_overexposure(out, mask, intensity)
            combined_mask = np.maximum(combined_mask, mask)

        out_img = Image.fromarray((out * 255.0).astype('uint8'))
        mask_img = Image.fromarray((np.clip(combined_mask, 0.0, 1.0) * 255.0).astype('uint8'))

        out_path = processed_dir / f"{base_name}_overexp_{i+1}.png"
        mask_path = processed_dir / f"{base_name}_overexp_{i+1}_mask.png"
        out_img.save(out_path)
        mask_img.save(mask_path)


def main():
    parser = argparse.ArgumentParser(description='Create synthetic overexposed patches on images')
    parser.add_argument('--input_dir', required=True, help='Directory with input images')
    parser.add_argument('--output_dir', required=True, help='Directory to save augmented images')
    parser.add_argument('--patches_per_image', type=int, default=2)
    parser.add_argument('--radius_min', type=int, default=20)
    parser.add_argument('--radius_max', type=int, default=120)
    parser.add_argument('--intensity_min', type=float, default=0.6)
    parser.add_argument('--intensity_max', type=float, default=1.2)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png', '.tif', '.bmp'}
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    if not files:
        print('No images found in', input_dir)
        return

    for p in files:
        try:
            process_image(p, output_dir, patches_per_image=args.patches_per_image,
                          radius_min=args.radius_min, radius_max=args.radius_max,
                          intensity_min=args.intensity_min, intensity_max=args.intensity_max,
                          seed=(args.seed if args.seed is not None else None))
            print('Processed', p.name)
        except Exception as e:
            print('Failed', p.name, 'error:', e)

if __name__ == '__main__':
    main()
