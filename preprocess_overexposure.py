#!/usr/bin/env python3
"""Preprocess images to repair overexposed pixels using a perceptual rule.

Algorithm: Detect overexposed/glare regions and darken them by 10% while preserving color.
For blown-out whites, sample nearby colors. Apply bilateral filtering for smooth blending.

Usage:
  python tools/preprocess_overexposure.py --input_dir example_images --output_dir tmp_preproc --mode darken
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


def reconstruct_overexposed(arr, over_mask, radius=7, sigma=None):
    """Reconstruct overexposed pixels using distance-weighted average of nearby non-over pixels."""
    h, w = over_mask.shape
    out = arr.copy()
    if sigma is None:
        sigma = max(1.0, radius / 2.0)

    ys, xs = np.mgrid[-radius:radius+1, -radius:radius+1]
    dist2 = xs**2 + ys**2
    kernel = np.exp(-dist2 / (2.0 * sigma * sigma)).astype('float32')

    inv_mask = (~over_mask).astype('float32')
    denom = ndimage.convolve(inv_mask, kernel, mode='constant', cval=0.0)

    numerator = np.zeros_like(arr, dtype='float32')
    for c in range(3):
        numerator[..., c] = ndimage.convolve(arr[..., c] * inv_mask, kernel, mode='constant', cval=0.0)

    valid = denom > 1e-8
    mask_use = over_mask & valid
    if np.any(mask_use):
        for c in range(3):
            out_chan = out[..., c]
            out_chan[mask_use] = (numerator[..., c][mask_use] / denom[mask_use])
            out[..., c] = out_chan

    mask_fallback = over_mask & (~valid)
    if np.any(mask_fallback):
        inv_bool = (~over_mask)
        if inv_bool.any():
            _, inds = ndimage.distance_transform_edt(inv_bool == 0, return_indices=True)
            nearest_y = inds[0]
            nearest_x = inds[1]
            ys_f, xs_f = np.nonzero(mask_fallback)
            out[ys_f, xs_f] = arr[nearest_y[ys_f, xs_f], nearest_x[ys_f, xs_f]]

    return out


def color_propagate_inpaint(arr, over_mask, radius=10):
    """Inpaint overexposed regions using color propagation from non-saturated neighbors."""
    arr_out = arr.copy()
    inv_mask = ~over_mask
    
    for c in range(3):
        chan = arr[..., c]
        valid_chan = np.where(inv_mask, chan, 0).astype('float32')
        kernel_size = 2 * radius + 1
        chan_sum = cv2.boxFilter(valid_chan, -1, (kernel_size, kernel_size), normalize=False)
        valid_count = cv2.boxFilter(inv_mask.astype('float32'), -1, (kernel_size, kernel_size), normalize=False)
        valid_count = np.maximum(valid_count, 1e-8)
        chan_avg = chan_sum / valid_count
        arr_out[..., c][over_mask] = chan_avg[over_mask]
    
    return arr_out


def classify_glare_vs_white(arr, over_mask, min_glare_size=0.005):
    """Classify saturated regions into glare (small, halo-like) vs white objects (large, flat).
    
    Key insight: Regions with high clipping (>= 250) are overexposure, not white objects.
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
        
        # NEW: Check if region is clipped (saturated channels at 250+)
        # This distinguishes overexposure from intentional white objects
        R = arr_f[comp_mask, 0]
        G = arr_f[comp_mask, 1]
        B = arr_f[comp_mask, 2]
        max_c = np.maximum.reduce([R, G, B])
        
        clipped_ratio = np.sum(max_c >= 250) / comp_size
        if clipped_ratio >= 0.3:
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
            diffs = np.diff(Y_vals)
            decreasing_ratio = np.sum(diffs < 0) / len(diffs)
            
            if decreasing_ratio > 0.6:
                glare_mask[comp_mask] = True
                continue
        
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


def smooth_blend(original_arr, reconstructed_arr, over_mask, smooth_sigma=2.0, feather_sigma=5.0):
    """Blend reconstructed pixels with gaussian smoothing and edge-aware feathering."""
    sm = np.zeros_like(reconstructed_arr)
    for c in range(3):
        sm[..., c] = ndimage.gaussian_filter(reconstructed_arr[..., c], sigma=smooth_sigma)

    Y_recon, _ = compute_Y_S(reconstructed_arr)
    gx = ndimage.sobel(Y_recon, axis=1)
    gy = ndimage.sobel(Y_recon, axis=0)
    grad = np.hypot(gx, gy)
    grad_mean = np.mean(grad[over_mask]) + 1e-8
    
    over_mask_uint8 = over_mask.astype('uint8')
    dist_from_edge, _ = ndimage.distance_transform_edt(over_mask, return_indices=True)
    
    max_dist = np.max(dist_from_edge[over_mask]) + 1e-8
    feather_weight = 1.0 - np.exp(-(dist_from_edge / feather_sigma) ** 2)
    feather_weight = np.clip(feather_weight, 0.0, 1.0)
    
    grad_norm = grad / (grad_mean + 1e-8)
    grad_weight = 1.0 - np.exp(-grad_norm)
    grad_weight = np.clip(grad_weight, 0.0, 1.0)
    
    alpha = feather_weight * (0.5 + 0.5 * grad_weight)
    alpha = np.clip(alpha, 0.0, 1.0)

    out = original_arr.copy()
    am = alpha[..., None]
    blended = am * reconstructed_arr + (1.0 - am) * sm
    out[over_mask] = blended[over_mask]
    return out


def pyramid_inpaint(arr_uint8, mask_uint8, levels=3, base_radius=3):
    """Coarse-to-fine inpainting using OpenCV inpaint at multiple scales."""
    if levels <= 1:
        return cv2.inpaint(arr_uint8, mask_uint8, base_radius, cv2.INPAINT_TELEA)

    imgs = [arr_uint8]
    masks = [mask_uint8]
    for l in range(1, levels):
        imgs.append(cv2.pyrDown(imgs[-1]))
        masks.append(cv2.pyrDown(masks[-1]))
        masks[-1] = (masks[-1] > 127).astype('uint8') * 255

    coarsest_idx = levels - 1
    radius = max(3, base_radius * (2 ** (levels - 1)))
    inpaint_coarse = cv2.inpaint(imgs[coarsest_idx], masks[coarsest_idx], radius, cv2.INPAINT_TELEA)

    cur = inpaint_coarse
    for l in range(coarsest_idx - 1, -1, -1):
        up = cv2.pyrUp(cur)
        if up.shape[0] != imgs[l].shape[0] or up.shape[1] != imgs[l].shape[1]:
            up = cv2.resize(up, (imgs[l].shape[1], imgs[l].shape[0]), interpolation=cv2.INTER_LINEAR)

        mask_l = masks[l]
        img_l = imgs[l].copy()
        if len(mask_l.shape) == 3:
            mask_chan = mask_l[..., 0]
        else:
            mask_chan = mask_l
        fill_idx = mask_chan > 127
        img_l[fill_idx] = up[fill_idx]

        radius_l = max(1, base_radius * (2 ** l))
        cur = cv2.inpaint(img_l, (mask_chan.astype('uint8') * 255), radius_l, cv2.INPAINT_TELEA)

    return cur


def apply_clahe(img_uint8):
    """Apply Contrast Limited Adaptive Histogram Equalization in LAB space.
    Redistributes brightness within glare regions while preserving local contrast."""
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_multi_scale_retinex(img_uint8, scales=None):
    """Multi-Scale Retinex: separates illumination from reflectance.
    Helps recover detail in bright areas without creating uniform gray patches.
    Uses faster, smaller scales for reasonable processing time."""
    if scales is None:
        scales = [15, 80]  # Reduced from [15, 80, 250] for speed
    
    img_float = img_uint8.astype('float32') + 1.0
    msr = np.zeros_like(img_float)
    
    for scale in scales:
        blurred = cv2.GaussianBlur(img_float, (0, 0), scale)
        msr += np.log10(img_float) - np.log10(blurred + 1e-8)
    
    msr = msr / len(scales)
    msr_normalized = (msr - msr.min()) / (msr.max() - msr.min() + 1e-8)
    return np.clip(msr_normalized * 255, 0, 255).astype('uint8')


def blend_enhancement(original, enhanced, mask_soft_norm, blend_factor=0.6):
    """Blend original image with enhanced version using soft mask.
    This prevents hard edges at mask boundaries."""
    blend_weight = blend_factor * mask_soft_norm[..., None]
    blended = original * (1.0 - blend_weight) + enhanced.astype('float32') * blend_weight
    return np.clip(blended, 0, 255).astype('float32')


def process_image(path, out_dir, radius=7, smooth_sigma=2.0, feather_sigma=5.0, min_glare_size=0.005, pyramid_levels=3, mode='inpaint', darken_frac=0.35,
                  base_reduction=0.12, extra_reduction=0.20, severity_start=220.0, severity_range=35.0, blend_factor=0.85, gamma_exponent=1.2, reduction_cap=0.75):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype('float32')
    Y, S = compute_Y_S(arr)
    S_TH = np.exp(2.4 * (Y - 1.0))
    # Base perceptual overexposure detection (luminance + low saturation)
    over_mask = (Y > 0.95) & (S < S_TH)
    # Also include very high channel values (near clipping) or extremely high luminance
    max_chan = np.maximum.reduce([arr[..., 0], arr[..., 1], arr[..., 2]])
    over_mask = over_mask | (max_chan >= 245) | (Y > 0.98)
    
    # Detect large bright regions (suns, sunsets) via morphology and local luminance
    total_pixels = arr.shape[0] * arr.shape[1]
    bright_mask = max_chan >= 235
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bright_opened = cv2.morphologyEx((bright_mask.astype('uint8') * 255), cv2.MORPH_OPEN, kernel, iterations=1)
    bright_labeled, ncomp = ndimage.label((bright_opened > 0))
    
    # For each large bright component, check if local mean Y is high (likely sun/highlight)
    for comp_id in range(1, ncomp + 1):
        comp_mask = bright_labeled == comp_id
        comp_area = np.sum(comp_mask)
        # Lower threshold: detect components >= 0.1% of image
        if comp_area / float(total_pixels) >= 0.001:
            mean_y = np.mean(Y[comp_mask])
            mean_max = np.mean(max_chan[comp_mask])
            # If component is very bright, mark as overexposed
            if mean_y > 0.90 or mean_max > 230:
                over_mask[comp_mask] = True
    
    # Morphological dilation to grow overexposed regions and catch halos
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    over_mask = cv2.dilate((over_mask.astype('uint8') * 255), kernel_dilate, iterations=2) > 0

    glare_mask, white_mask = classify_glare_vs_white(arr, over_mask, min_glare_size=min_glare_size)
    
    if np.any(glare_mask):
        glare_mask_uint8 = (glare_mask.astype('uint8') * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask_dilated = cv2.dilate(glare_mask_uint8, kernel, iterations=1)
        glare_mask_soft = cv2.GaussianBlur(glare_mask_dilated, (7, 7), sigmaX=2.0, sigmaY=2.0)
        glare_mask_soft_norm = glare_mask_soft.astype('float32') / 255.0

        if mode == 'darken':
            # Aggressive darkening strategy:
            # 1) Apply CLAHE to bring out local contrast
            # 2) Blend enhanced image with original using soft mask
            # 3) Apply stronger per-pixel HSV-based Value reduction proportional to clipping severity
            # 4) Apply an additional gamma compression on very bright pixels and bilateral smoothing
            arr_uint8 = np.clip(arr, 0, 255).astype('uint8')
            final = arr_uint8.copy().astype('float32')
            pixel_changes = []

            # 1) CLAHE enhancement
            clahe_img = apply_clahe(arr_uint8)

            # 2) Blend CLAHE with original using soft mask to avoid hard edges
            final = blend_enhancement(final, clahe_img, glare_mask_soft_norm, blend_factor=blend_factor)

            # 3) Compute clipping severity from original image (0..1)
            R_o = arr_uint8[..., 0].astype('float32')
            G_o = arr_uint8[..., 1].astype('float32')
            B_o = arr_uint8[..., 2].astype('float32')
            max_val = np.maximum.reduce([R_o, G_o, B_o])
            # severity: start at severity_start -> 0, severity_start+severity_range -> 1
            severity = np.clip((max_val - severity_start) / severity_range, 0.0, 1.0)

            mask_strength = glare_mask_soft_norm.astype('float32')

            # reduction fraction per-pixel: base + extra*severity scaled by mask strength
            reduction = (base_reduction + extra_reduction * severity) * mask_strength
            reduction = np.clip(reduction, 0.0, reduction_cap)

            # Convert blended result to HSV and reduce Value channel only
            final_uint8 = np.clip(final, 0, 255).astype('uint8')
            hsv = cv2.cvtColor(final_uint8, cv2.COLOR_RGB2HSV).astype('float32')
            V = hsv[..., 2]

            # Apply per-pixel multiplicative reduction on V
            V_new = V * (1.0 - reduction)

            # Adaptive gamma compression on very bright values to further tame highlights (milder)
            gamma_mask = V_new > 235
            if np.any(gamma_mask):
                # milder gamma for very bright pixels
                V_new[gamma_mask] = 255.0 * np.power((V_new[gamma_mask] / 255.0), float(gamma_exponent))

            hsv[..., 2] = np.clip(V_new, 0, 255)
            final_rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype('uint8'), cv2.COLOR_HSV2RGB).astype('float32')

            # 4) STRICT per-channel cap: ensure no channel reduces by more than 10% of original
            orig_rgb = arr_uint8.astype('float32')
            
            # Compute minimum allowed value per channel: 90% of original
            min_allowed = 0.90 * orig_rgb
            
            # Apply hard floor: never go below 90% of original
            final_capped = np.maximum(final_rgb, min_allowed)
            final = final_capped

            # Apply bilateral smoothing BEFORE final clamping to preserve the cap
            final_uint8 = np.clip(final, 0, 255).astype('uint8')
            final_filtered = cv2.bilateralFilter(final_uint8, d=9, sigmaColor=75, sigmaSpace=75)
            final = final_filtered.astype('float32')
            
            # RE-APPLY cap after bilateral filter to ensure it's not violated by filter artifacts
            final = np.maximum(final, min_allowed)

            # Log sample pixel changes for analysis
            mask_idx = mask_strength > 0.01
            if np.any(mask_idx):
                ys, xs = np.nonzero(mask_idx)
                for i in range(0, len(ys), max(1, len(ys) // 200)):
                    y, x = ys[i], xs[i]
                    orig = tuple(np.round(arr_uint8[y, x]).astype(int))
                    new = tuple(np.round(final[y, x]).astype(int))
                    pixel_changes.append({'y': y, 'x': x, 'orig_rgb': orig, 'new_rgb': new, 'type': 'aggressive_darken'})
            
            # Save pixel change log
            if pixel_changes:
                log_path = Path(out_dir) / 'pixel_changes' / (Path(path).stem + '_changes.txt')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, 'w') as f:
                    f.write('y,x,orig_r,orig_g,orig_b,new_r,new_g,new_b,type\n')
                    for change in pixel_changes:
                        orig = change['orig_rgb']
                        new = change['new_rgb']
                        f.write(f"{change['y']},{change['x']},{orig[0]},{orig[1]},{orig[2]},{new[0]},{new[1]},{new[2]},{change['type']}\n")
        else:
            # Default: use color propagation inpaint
            inpainted = color_propagate_inpaint(arr, glare_mask_soft_norm > 0.5, radius=10)
            final = smooth_blend(arr, inpainted.astype('float32'), glare_mask_soft_norm > 0.5, smooth_sigma=smooth_sigma, feather_sigma=feather_sigma)
    else:
        final = arr.copy()

    images_dir = Path(out_dir) / 'images'
    masks_dir = Path(out_dir) / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    out_name = Path(path).stem + '_preproc.png'
    glare_mask_name = Path(path).stem + '_glare_mask.png'
    white_mask_name = Path(path).stem + '_white_mask.png'
    
    Image.fromarray(np.clip(final, 0, 255).astype('uint8')).save(images_dir / out_name)
    Image.fromarray((glare_mask.astype('uint8') * 255)).save(masks_dir / glare_mask_name)
    Image.fromarray((white_mask.astype('uint8') * 255)).save(masks_dir / white_mask_name)


def main():
    parser = argparse.ArgumentParser(description='Preprocess images to repair overexposure')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--radius', type=int, default=7, help='Neighborhood radius for reconstruction')
    parser.add_argument('--smooth_sigma', type=float, default=2.0, help='Gaussian sigma for final smoothing')
    parser.add_argument('--feather_sigma', type=float, default=5.0, help='Distance sigma for feathering blend at patch boundaries')
    parser.add_argument('--min_glare_size', type=float, default=0.005, help='Min fraction of image area to classify as glare')
    parser.add_argument('--pyramid_levels', type=int, default=3, help='Number of pyramid levels for coarse-to-fine inpainting')
    parser.add_argument('--mode', type=str, choices=['inpaint','darken'], default='inpaint', help='Processing mode: inpaint (default) or darken')
    parser.add_argument('--darken_frac', type=float, default=0.35, help='Fraction to darken overexposed areas (0..1)')
    # Darkening tuning parameters
    parser.add_argument('--base_reduction', type=float, default=0.12, help='Base per-pixel reduction fraction (0..1)')
    parser.add_argument('--extra_reduction', type=float, default=0.20, help='Extra per-pixel reduction scaled by severity (0..1)')
    parser.add_argument('--severity_start', type=float, default=220.0, help='Start of severity mapping (pixel value)')
    parser.add_argument('--severity_range', type=float, default=35.0, help='Range over which severity maps to 0..1')
    parser.add_argument('--blend_factor', type=float, default=0.85, help='Blend factor for CLAHE/enhanced image')
    parser.add_argument('--gamma_exponent', type=float, default=1.2, help='Gamma exponent for very bright value compression')
    parser.add_argument('--reduction_cap', type=float, default=0.75, help='Maximum allowed reduction fraction')
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
            process_image(p, output_dir, radius=args.radius, smooth_sigma=args.smooth_sigma, feather_sigma=args.feather_sigma, min_glare_size=args.min_glare_size, pyramid_levels=args.pyramid_levels, mode=args.mode, darken_frac=args.darken_frac,
                      base_reduction=args.base_reduction, extra_reduction=args.extra_reduction, severity_start=args.severity_start, severity_range=args.severity_range, blend_factor=args.blend_factor, gamma_exponent=args.gamma_exponent, reduction_cap=args.reduction_cap)
            print('Preprocessed', p.name)
        except Exception as e:
            print('Failed', p.name, 'error:', e)


if __name__ == '__main__':
    main()
