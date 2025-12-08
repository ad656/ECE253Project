#!/usr/bin/env python3
"""Preprocess images to repair overexposed pixels (glare detection + inpainting).

Simplified version: outputs only the corrected image, no diagnostic masks.

Algorithm:
- Compute Rec.709 luminance Y and saturation S
- Detect overexposed pixels: Y > 0.95 and S < S_TH(Y) where S_TH(Y) = exp(2.4*(Y-1))
- Classify overexposed regions into glare vs. white objects using:
  * Size (small = glare, large = white)
  * Halo detection (radial brightness falloff = glare)
  * Chromaticity (color-cast = glare, balanced = white)
- Inpaint only glare using OpenCV Telea algorithm
- Preserve intentional white objects

Usage:
  python tools/preprocess_overexposure_simple.py --input_dir example_images --output_dir tmp_preproc_simple

Options:
  --radius: neighborhood radius for reconstruction (default 7, unused in this version)
  --smooth_sigma: gaussian sigma for final blending (default 2.0)
  --min_glare_size: min fraction of image area to classify as glare (default 0.005)
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import cv2


def compute_Y_S(arr):
    """arr: HxWx3 uint8 or float (0..255 or 0..1). Return Y (0..1), S (0..1), with arr in 0..255 scale."""
    a = arr.astype('float32')
    if a.max() <= 1.1:
        a = a * 255.0
    R = a[..., 0]
    G = a[..., 1]
    B = a[..., 2]
    Y = (0.2126 * R + 0.7152 * G + 0.0722 * B) / 255.0
    mx = np.maximum.reduce([R, G, B])
    mn = np.minimum.reduce([R, G, B])
    S = np.zeros_like(mx)
    nonzero = mx > 1e-6
    S[nonzero] = (mx[nonzero] - mn[nonzero]) / mx[nonzero]
    return Y, S


def classify_glare_vs_white(arr, over_mask, min_glare_size=0.005):
    """Classify saturated regions into glare (small, halo-like) vs white objects (large, flat).
    
    Returns:
    - glare_mask: boolean HxW, True where likely glare
    - white_mask: boolean HxW, True where likely intentional white
    """
    h, w = over_mask.shape
    total_pixels = h * w
    min_size = max(50, int(min_glare_size * total_pixels))
    
    labeled, num_features = ndimage.label(over_mask)
    
    glare_mask = np.zeros_like(over_mask)
    white_mask = np.zeros_like(over_mask)
    
    arr_f = arr.astype('float32')
    Y, S = compute_Y_S(arr_f)
    
    for comp_id in range(1, num_features + 1):
        comp_mask = labeled == comp_id
        comp_size = np.sum(comp_mask)
        
        if comp_size < min_size:
            glare_mask[comp_mask] = True
            continue

        ys, xs = np.nonzero(comp_mask)
        if len(ys) == 0:
            continue
        
        cy, cx = np.mean(ys), np.mean(xs)
        dists = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        
        if len(dists) > 10:
            sort_idx = np.argsort(dists)
            Y_vals = Y[ys[sort_idx], xs[sort_idx]]

            window_size = max(1, len(Y_vals) // 5)
            Y_smooth = np.convolve(Y_vals, np.ones(window_size) / window_size, mode='same')
            diffs = np.diff(Y_smooth)
            decreasing_ratio = np.sum(diffs < 0) / len(diffs)
            
            if decreasing_ratio > 0.6:
                glare_mask[comp_mask] = True
                continue

        R = arr_f[comp_mask, 0]
        G = arr_f[comp_mask, 1]
        B = arr_f[comp_mask, 2]
        max_c = np.maximum.reduce([R, G, B])
        safe = max_c > 1e-6
        if np.sum(safe) > 0:
            r_norm = R[safe] / max_c[safe]
            g_norm = G[safe] / max_c[safe]
            b_norm = B[safe] / max_c[safe]
            channel_var = np.var([r_norm, g_norm, b_norm], axis=0).mean()
            if channel_var > 0.05:
                glare_mask[comp_mask] = True
                continue
        
        white_mask[comp_mask] = True
    
    return glare_mask, white_mask


def smooth_blend(original_arr, reconstructed_arr, over_mask, smooth_sigma=2.0):
    """Blend reconstructed pixels with a gaussian-smoothed version while preserving edges."""
    sm = np.zeros_like(reconstructed_arr)
    for c in range(3):
        sm[..., c] = ndimage.gaussian_filter(reconstructed_arr[..., c], sigma=smooth_sigma)

    Y, _ = compute_Y_S(original_arr)
    gx = ndimage.sobel(Y, axis=1)
    gy = ndimage.sobel(Y, axis=0)
    grad = np.hypot(gx, gy)
    mean_grad = np.mean(grad) + 1e-8
    alpha = 1.0 - np.exp(-grad / mean_grad)
    alpha = np.clip(alpha, 0.0, 1.0)

    out = original_arr.copy()
    am = alpha[..., None]
    blended = am * reconstructed_arr + (1.0 - am) * sm
    out[over_mask] = blended[over_mask]
    return out


def process_image(path, out_dir, smooth_sigma=2.0, min_glare_size=0.005):
    """Process a single image: detect glare, inpaint, save corrected image only."""
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype('float32')
    Y, S = compute_Y_S(arr)
    S_TH = np.exp(2.4 * (Y - 1.0))
    over_mask = (Y > 0.95) & (S < S_TH)

    glare_mask, white_mask = classify_glare_vs_white(arr, over_mask, min_glare_size=min_glare_size)
    
    if np.any(glare_mask):
        glare_mask_uint8 = (glare_mask.astype('uint8') * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask_dilated = cv2.dilate(glare_mask_uint8, kernel, iterations=1)
        
        arr_uint8 = np.clip(arr, 0, 255).astype('uint8')
        inpainted = cv2.inpaint(arr_uint8, glare_mask_dilated, 3, cv2.INPAINT_TELEA)
        
        final = smooth_blend(arr, inpainted.astype('float32'), glare_mask, smooth_sigma=smooth_sigma)
    else:
        final = arr.copy()
    
    final[white_mask] = arr[white_mask]

    processed_dir = Path(out_dir) / 'processed_preproc'
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_name = Path(path).stem + '_preproc.png'
    Image.fromarray(np.clip(final, 0, 255).astype('uint8')).save(processed_dir / out_name)


def main():
    parser = argparse.ArgumentParser(description='Preprocess images to repair overexposure (output corrected image only)')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--smooth_sigma', type=float, default=2.0, help='Gaussian sigma for final smoothing')
    parser.add_argument('--min_glare_size', type=float, default=0.005, help='Min fraction of image area to classify as glare (vs white object)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.bmp', '.JPG', '.JPEG'}
    files = [p for p in input_dir.iterdir() if p.suffix in exts]
    if not files:
        print('No images found in', input_dir)
        return

    for p in files:
        try:
            process_image(p, output_dir, smooth_sigma=args.smooth_sigma, min_glare_size=args.min_glare_size)
            print('Preprocessed', p.name)
        except Exception as e:
            print('Failed', p.name, 'error:', e)


if __name__ == '__main__':
    main()
