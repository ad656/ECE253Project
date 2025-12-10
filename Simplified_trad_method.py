#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import cv2


def compute_Y_S(arr):
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


def poisson_blend(source, target, mask, method='mixed'):
    """
    Poisson blending for seamless integration.
    method: 'mixed' (default) uses mixed gradients, 'normal' uses source gradients only
    """
    if not np.any(mask):
        return target
    
    # Convert mask to uint8
    mask_uint8 = (mask > 0.5).astype('uint8') * 255
    
    # Find center of mask for seamlessClone
    ys, xs = np.nonzero(mask > 0.5)
    if len(ys) == 0:
        return target
    
    center = (int(np.mean(xs)), int(np.mean(ys)))
    
    # Ensure proper data types
    source_uint8 = np.clip(source, 0, 255).astype('uint8')
    target_uint8 = np.clip(target, 0, 255).astype('uint8')
    
    try:
        if method == 'mixed':
            result = cv2.seamlessClone(source_uint8, target_uint8, mask_uint8, center, cv2.MIXED_CLONE)
        else:
            result = cv2.seamlessClone(source_uint8, target_uint8, mask_uint8, center, cv2.NORMAL_CLONE)
        return result.astype('float32')
    except:
        # Fallback to regular blending if seamlessClone fails
        return target


def multiscale_blend(source, target, mask, levels=3):
    """
    Multi-scale Laplacian pyramid blending for natural texture preservation.
    """
    if not np.any(mask):
        return target
    
    source_uint8 = np.clip(source, 0, 255).astype('uint8')
    target_uint8 = np.clip(target, 0, 255).astype('uint8')
    mask_float = mask.astype('float32')
    
    # Build Gaussian pyramids for source and target
    src_pyr = [source_uint8.astype('float32')]
    tgt_pyr = [target_uint8.astype('float32')]
    mask_pyr = [mask_float]
    
    for i in range(levels - 1):
        src_pyr.append(cv2.pyrDown(src_pyr[-1]))
        tgt_pyr.append(cv2.pyrDown(tgt_pyr[-1]))
        mask_down = cv2.pyrDown(mask_pyr[-1])
        mask_pyr.append(mask_down)
    
    # Build Laplacian pyramids
    src_lap = []
    tgt_lap = []
    
    for i in range(levels - 1):
        expanded = cv2.pyrUp(src_pyr[i + 1])
        if expanded.shape[:2] != src_pyr[i].shape[:2]:
            expanded = cv2.resize(expanded, (src_pyr[i].shape[1], src_pyr[i].shape[0]))
        src_lap.append(src_pyr[i] - expanded)
        
        expanded = cv2.pyrUp(tgt_pyr[i + 1])
        if expanded.shape[:2] != tgt_pyr[i].shape[:2]:
            expanded = cv2.resize(expanded, (tgt_pyr[i].shape[1], tgt_pyr[i].shape[0]))
        tgt_lap.append(tgt_pyr[i] - expanded)
    
    src_lap.append(src_pyr[-1])
    tgt_lap.append(tgt_pyr[-1])
    
    # Blend Laplacian pyramids
    blended_pyr = []
    for i in range(levels):
        mask_3c = mask_pyr[i]
        if len(mask_3c.shape) == 2:
            mask_3c = mask_3c[..., np.newaxis]
        blended = src_lap[i] * mask_3c + tgt_lap[i] * (1.0 - mask_3c)
        blended_pyr.append(blended)
    
    # Reconstruct from pyramid
    result = blended_pyr[-1]
    for i in range(levels - 2, -1, -1):
        expanded = cv2.pyrUp(result)
        if expanded.shape[:2] != blended_pyr[i].shape[:2]:
            expanded = cv2.resize(expanded, (blended_pyr[i].shape[1], blended_pyr[i].shape[0]))
        result = expanded + blended_pyr[i]
    
    return np.clip(result, 0, 255).astype('float32')


def adaptive_feather_mask(mask, source, target, min_sigma=2.0, max_sigma=15.0):
    """
    Create adaptive feathering based on local texture complexity.
    Areas with more texture get less feathering, flat areas get more.
    """
    # Compute texture complexity using Laplacian variance
    source_gray = cv2.cvtColor(np.clip(source, 0, 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(np.clip(target, 0, 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    
    lap_src = cv2.Laplacian(source_gray, cv2.CV_64F)
    lap_tgt = cv2.Laplacian(target_gray, cv2.CV_64F)
    
    # Compute local variance
    texture_map = np.abs(lap_src) + np.abs(lap_tgt)
    texture_map = cv2.GaussianBlur(texture_map, (15, 15), 3.0)
    
    # Normalize to [0, 1]
    texture_norm = (texture_map - texture_map.min()) / (texture_map.max() - texture_map.min() + 1e-8)
    
    # Less texture -> more feathering (higher sigma)
    adaptive_sigma = max_sigma - (max_sigma - min_sigma) * texture_norm
    
    # Create distance transform
    mask_bool = mask > 0.5
    dist_transform = ndimage.distance_transform_edt(mask_bool)
    
    # Apply adaptive feathering
    feathered = np.zeros_like(mask, dtype='float32')
    
    # Vectorized adaptive gaussian falloff
    ys, xs = np.nonzero(mask_bool)
    for y, x in zip(ys, xs):
        d = dist_transform[y, x]
        sigma = adaptive_sigma[y, x]
        feathered[y, x] = np.exp(-(d ** 2) / (2 * sigma ** 2))
    
    # Smooth the feathered mask
    feathered = cv2.GaussianBlur(feathered, (15, 15), 3.0)
    
    return np.clip(feathered, 0, 1)


def gradient_domain_blend(source, target, mask, bandwidth=10):
    """
    Gradient-domain blending that preserves gradients from surrounding areas.
    """
    mask_bool = mask > 0.5
    if not np.any(mask_bool):
        return target
    
    result = target.copy()
    
    # Erode mask to get interior region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bandwidth*2+1, bandwidth*2+1))
    mask_eroded = cv2.erode((mask_bool.astype('uint8') * 255), kernel, iterations=1) > 0
    
    # Boundary is the difference
    boundary = mask_bool & (~mask_eroded)
    
    if not np.any(boundary):
        return target
    
    # For each channel, blend gradients
    for c in range(3):
        src_chan = source[..., c]
        tgt_chan = target[..., c]
        
        # Compute gradients
        src_gx = cv2.Sobel(src_chan.astype('float32'), cv2.CV_32F, 1, 0, ksize=3)
        src_gy = cv2.Sobel(src_chan.astype('float32'), cv2.CV_32F, 0, 1, ksize=3)
        tgt_gx = cv2.Sobel(tgt_chan.astype('float32'), cv2.CV_32F, 1, 0, ksize=3)
        tgt_gy = cv2.Sobel(tgt_chan.astype('float32'), cv2.CV_32F, 0, 1, ksize=3)
        
        # Use stronger gradients (mixed gradients approach)
        mixed_gx = np.where(np.abs(src_gx) > np.abs(tgt_gx), src_gx, tgt_gx)
        mixed_gy = np.where(np.abs(src_gy) > np.abs(tgt_gy), src_gy, tgt_gy)
        
        # Smooth transition at boundary
        boundary_float = boundary.astype('float32')
        boundary_smooth = cv2.GaussianBlur(boundary_float, (bandwidth*2+1, bandwidth*2+1), bandwidth/2.0)
        
        # Blend gradients
        final_gx = mixed_gx * mask_bool.astype('float32') + tgt_gx * (1.0 - mask_bool.astype('float32'))
        final_gy = mixed_gy * mask_bool.astype('float32') + tgt_gy * (1.0 - mask_bool.astype('float32'))
        
        # Reconstruct from gradients (simplified - using source values in interior)
        result[..., c][mask_eroded] = src_chan[mask_eroded]
    
    # Smooth boundaries
    result = cv2.bilateralFilter(np.clip(result, 0, 255).astype('uint8'), 9, 75, 75).astype('float32')
    
    return result


def enhanced_smooth_blend(original, reconstructed, mask, mode='auto', smooth_sigma=2.0, feather_sigma=5.0):
    """
    Enhanced blending with multiple techniques.
    mode: 'poisson', 'multiscale', 'gradient', 'adaptive', or 'auto' (tries best method)
    """
    if not np.any(mask):
        return original
    
    mask_bool = mask > 0.5
    
    if mode == 'auto':
        # Automatically choose best method based on mask characteristics
        mask_area = np.sum(mask_bool) / float(mask_bool.size)
        
        if mask_area < 0.01:  # Small regions - use Poisson
            mode = 'poisson'
        elif mask_area > 0.3:  # Large regions - use multiscale
            mode = 'multiscale'
        else:  # Medium regions - use adaptive
            mode = 'adaptive'
    
    if mode == 'poisson':
        # Try mixed clone first, fallback to normal clone
        result = poisson_blend(reconstructed, original, mask_bool, method='mixed')
        if np.array_equal(result, original):  # If failed
            result = poisson_blend(reconstructed, original, mask_bool, method='normal')
    
    elif mode == 'multiscale':
        # Multi-scale Laplacian blending
        result = multiscale_blend(reconstructed, original, mask_bool, levels=4)
    
    elif mode == 'gradient':
        # Gradient-domain blending
        result = gradient_domain_blend(reconstructed, original, mask_bool, bandwidth=12)
    
    elif mode == 'adaptive':
        # Adaptive feathering + alpha blending
        feather_mask = adaptive_feather_mask(mask_bool.astype('float32'), reconstructed, original)
        
        # Smooth reconstructed
        smooth_recon = np.zeros_like(reconstructed)
        for c in range(3):
            smooth_recon[..., c] = ndimage.gaussian_filter(reconstructed[..., c], sigma=smooth_sigma)
        
        # Alpha blend with adaptive mask
        alpha = feather_mask[..., np.newaxis]
        result = smooth_recon * alpha + original * (1.0 - alpha)
    
    else:
        # Fallback to original simple method
        smooth_recon = np.zeros_like(reconstructed)
        for c in range(3):
            smooth_recon[..., c] = ndimage.gaussian_filter(reconstructed[..., c], sigma=smooth_sigma)
        
        dist_transform = ndimage.distance_transform_edt(mask_bool)
        max_dist = np.max(dist_transform[mask_bool]) if np.any(mask_bool) else 1.0
        feather = 1.0 - np.exp(-(dist_transform / feather_sigma) ** 2)
        feather = np.clip(feather, 0, 1)
        
        alpha = feather[..., np.newaxis]
        result = smooth_recon * alpha + original * (1.0 - alpha)
    
    return np.clip(result, 0, 255)


def reconstruct_overexposed(arr, over_mask, radius=7, sigma=None):
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
        
        R = arr_f[comp_mask, 0]
        G = arr_f[comp_mask, 1]
        B = arr_f[comp_mask, 2]
        max_c = np.maximum.reduce([R, G, B])
        
        clipped_ratio = np.sum(max_c >= 252) / comp_size
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


def robust_clip_repair(arr, mask):
    mask_bool = (mask > 0.5)
    if not np.any(mask_bool):
        return arr.astype('uint8')

    arr_uint8 = np.clip(arr, 0, 255).astype('uint8')
    valid_boundary_found = False
    result = arr_uint8.copy()

    search_radii = [10, 20, 30, 40, 50]
    mask_uint8 = mask_bool.astype('uint8')

    for radius in search_radii:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        boundary = (mask_dilated > 0) & (mask_uint8 == 0)

        if not np.any(boundary):
            continue

        boundary_pixels = arr[boundary]
        if boundary_pixels.size == 0:
            continue

        max_vals = np.max(boundary_pixels, axis=1)
        non_white_ratio = np.sum(max_vals < 248) / float(len(max_vals))

        if non_white_ratio > 0.2:
            try:
                inpaint_mask = (mask_uint8 * 255).astype('uint8')
                inpainted = cv2.inpaint(arr_uint8, inpaint_mask, inpaintRadius=min(radius, 25), flags=cv2.INPAINT_NS)
                result = inpainted
                valid_boundary_found = True
                break
            except Exception:
                valid_boundary_found = False
                continue

    if not valid_boundary_found:
        hsv = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2HSV).astype('float32')
        dist_mask = ndimage.distance_transform_edt(mask_bool)
        max_dist = np.max(dist_mask[mask_bool]) if np.any(mask_bool) else 1.0
        dist_norm = np.clip(dist_mask / (max_dist + 1e-8), 0, 1)
        reduction = 0.25 + 0.15 * dist_norm
        V = hsv[..., 2]
        V_new = V.copy()
        V_new[mask_bool] = V[mask_bool] * (1.0 - reduction[mask_bool])
        hsv[..., 2] = np.clip(V_new, 0, 255)
        result = cv2.cvtColor(np.clip(hsv, 0, 255).astype('uint8'), cv2.COLOR_HSV2RGB)

    try:
        result = cv2.bilateralFilter(result, d=7, sigmaColor=40, sigmaSpace=40)
    except Exception:
        pass

    return result.astype('uint8')


def pyramid_inpaint(arr_uint8, mask_uint8, levels=3, base_radius=3):
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
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_multi_scale_retinex(img_uint8, scales=None):
    if scales is None:
        scales = [15, 80]
    
    img_float = img_uint8.astype('float32') + 1.0
    msr = np.zeros_like(img_float)
    
    for scale in scales:
        blurred = cv2.GaussianBlur(img_float, (0, 0), scale)
        msr += np.log10(img_float) - np.log10(blurred + 1e-8)
    
    msr = msr / len(scales)
    msr_normalized = (msr - msr.min()) / (msr.max() - msr.min() + 1e-8)
    return np.clip(msr_normalized * 255, 0, 255).astype('uint8')


def blend_enhancement(original, enhanced, mask_soft_norm, blend_factor=0.6):
    blend_weight = blend_factor * mask_soft_norm[..., None]
    blended = original * (1.0 - blend_weight) + enhanced.astype('float32') * blend_weight
    return np.clip(blended, 0, 255).astype('float32')


def smooth_blend(original_arr, reconstructed_arr, over_mask, smooth_sigma=2.0, feather_sigma=5.0, blend_method='auto'):
    """Enhanced version using smart blending techniques"""
    # Smooth the reconstruction first
    smoothed = np.zeros_like(reconstructed_arr)
    for c in range(3):
        smoothed[..., c] = ndimage.gaussian_filter(reconstructed_arr[..., c], sigma=smooth_sigma)
    
    # Apply smart blending
    result = smart_blend(original_arr, smoothed, over_mask, method=blend_method)
    
    return result


def simple_inpaint_reconstruction(arr, mask, radius=20):
    arr_uint8 = np.clip(arr, 0, 255).astype('uint8')
    mask_uint8 = (mask.astype('uint8') * 255)
    
    inpainted = cv2.inpaint(arr_uint8, mask_uint8, radius, cv2.INPAINT_TELEA)
    
    return inpainted


def smart_blend(original, reconstructed, mask, method='auto'):
    mask_bool = mask > 0.5
    if not np.any(mask_bool):
        return original
    
    if method == 'auto':
        mask_area = np.sum(mask_bool) / float(mask_bool.size)
        labeled, n_regions = ndimage.label(mask_bool)
        
        if n_regions > 5 or mask_area < 0.01:
            method = 'poisson'
        elif mask_area > 0.3:
            method = 'multiscale'
        else:
            method = 'adaptive'
    
    if method == 'poisson':
        result = poisson_blend(reconstructed, original, mask_bool)
        if np.array_equal(result, original):
            result = adaptive_feather_blend(reconstructed, original, mask_bool)
    elif method == 'multiscale':
        result = multiscale_blend(reconstructed, original, mask_bool, levels=4)
    elif method == 'adaptive':
        result = adaptive_feather_blend(reconstructed, original, mask_bool)
    elif method == 'gradient':
        result = gradient_domain_blend(reconstructed, original, mask_bool)
    else:
        result = adaptive_feather_blend(reconstructed, original, mask_bool)
    
    return result


def classify_glare_type(arr, mask):
    """
    Classify overexposed pixels as:
    - Pure white clipping (no hue, essentially colorless)
    - Colored glare (has detectable hue through RGB imbalance)
    Returns two masks: white_clip, colored_glare
    """
    arr_f = arr.astype('float32')
    R = arr_f[..., 0]
    G = arr_f[..., 1]
    B = arr_f[..., 2]
    
    max_c = np.maximum.reduce([R, G, B])
    min_c = np.minimum.reduce([R, G, B])
    
    rgb_diff = max_c - min_c
    hue_strength = rgb_diff / (max_c + 1e-8)
    
    white_clip = mask & (hue_strength < 0.08) & (max_c >= 245)
    colored_glare = mask & (hue_strength >= 0.08)
    
    return white_clip, colored_glare


def adaptive_feather_blend(source, target, mask, min_feather=2, max_feather=15):
    mask_bool = mask > 0.5
    if not np.any(mask_bool):
        return target
    
    src_gray = cv2.cvtColor(np.clip(source, 0, 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(np.clip(target, 0, 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    
    lap_src = np.abs(cv2.Laplacian(src_gray, cv2.CV_64F))
    lap_tgt = np.abs(cv2.Laplacian(tgt_gray, cv2.CV_64F))
    texture_map = lap_src + lap_tgt
    texture_map = cv2.GaussianBlur(texture_map, (15, 15), 3.0)
    
    texture_norm = (texture_map - texture_map.min()) / (texture_map.max() - texture_map.min() + 1e-8)
    
    dist_map = ndimage.distance_transform_edt(mask_bool)
    max_dist = np.max(dist_map[mask_bool]) if np.any(mask_bool) else 1.0
    
    sigma_map = max_feather - (max_feather - min_feather) * texture_norm
    
    alpha = np.zeros_like(mask, dtype='float32')
    alpha[mask_bool] = np.exp(-(dist_map[mask_bool] ** 2) / (2 * sigma_map[mask_bool] ** 2))
    alpha = cv2.GaussianBlur(alpha, (15, 15), 3.0)
    alpha = np.clip(alpha, 0, 1)
    
    result = source * alpha[..., np.newaxis] + target * (1 - alpha[..., np.newaxis])
    return np.clip(result, 0, 255).astype('float32')


def process_overexposed_hybrid(arr, mask, mode_per_pixel):
    """
    Hybrid processing: inpaint colored glare, darken pure white clipping.
    mode_per_pixel: 0=no change, 1=inpaint colored, 2=darken white
    """
    arr_uint8 = np.clip(arr, 0, 255).astype('uint8')
    result = arr.copy()
    
    inpaint_mask = (mode_per_pixel == 1).astype('float32')
    darken_mask = (mode_per_pixel == 2).astype('float32')
    
    if np.any(mode_per_pixel == 1):
        inpaint_bool = mode_per_pixel == 1
        inpaint_mask_uint8 = (inpaint_bool.astype('uint8') * 255)
        
        if inpaint_bool.any():
            radius = 20
            inpainted = cv2.inpaint(arr_uint8, inpaint_mask_uint8, radius, cv2.INPAINT_TELEA)
            result[inpaint_bool] = inpainted.astype('float32')[inpaint_bool]
    
    if np.any(mode_per_pixel == 2):
        darken_bool = mode_per_pixel == 2
        
        darken_f = arr_uint8.astype('float32')
        hsv = cv2.cvtColor(np.clip(darken_f, 0, 255).astype('uint8'), cv2.COLOR_RGB2HSV).astype('float32')
        
        V = hsv[..., 2]
        V_new = V * 0.7
        
        hsv[..., 2] = np.clip(V_new, 0, 255)
        darkened_rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype('uint8'), cv2.COLOR_HSV2RGB).astype('float32')
        
        result[darken_bool] = darkened_rgb[darken_bool]
    
    return result
    mask_bool = mask > 0.5
    if not np.any(mask_bool):
        return target
    
    src_gray = cv2.cvtColor(np.clip(source, 0, 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(np.clip(target, 0, 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    
    lap_src = np.abs(cv2.Laplacian(src_gray, cv2.CV_64F))
    lap_tgt = np.abs(cv2.Laplacian(tgt_gray, cv2.CV_64F))
    texture_map = lap_src + lap_tgt
    texture_map = cv2.GaussianBlur(texture_map, (15, 15), 3.0)
    
    texture_norm = (texture_map - texture_map.min()) / (texture_map.max() - texture_map.min() + 1e-8)
    
    dist_map = ndimage.distance_transform_edt(mask_bool)
    max_dist = np.max(dist_map[mask_bool]) if np.any(mask_bool) else 1.0
    
    sigma_map = max_feather - (max_feather - min_feather) * texture_norm
    
    alpha = np.zeros_like(mask, dtype='float32')
    alpha[mask_bool] = np.exp(-(dist_map[mask_bool] ** 2) / (2 * sigma_map[mask_bool] ** 2))
    alpha = cv2.GaussianBlur(alpha, (15, 15), 3.0)
    alpha = np.clip(alpha, 0, 1)
    
    result = source * alpha[..., np.newaxis] + target * (1 - alpha[..., np.newaxis])
    return np.clip(result, 0, 255).astype('float32')


def process_image(path, out_dir, radius=7, smooth_sigma=2.0, feather_sigma=5.0, min_glare_size=0.005, 
                  pyramid_levels=3, mode='hybrid', blend_method='auto', darken_frac=0.35,
                  base_reduction=0.12, extra_reduction=0.20, severity_start=220.0, severity_range=35.0, 
                  blend_factor=0.85, gamma_exponent=1.2, reduction_cap=0.75):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype('float32')
    Y, S = compute_Y_S(arr)
    S_TH = np.exp(2.4 * (Y - 1.0))
    over_mask = (Y > 0.95) & (S < S_TH)
    max_chan = np.maximum.reduce([arr[..., 0], arr[..., 1], arr[..., 2]])
    over_mask = over_mask | (max_chan >= 250) | (Y > 0.99)
    
    total_pixels = arr.shape[0] * arr.shape[1]
    bright_mask = max_chan >= 245
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bright_opened = cv2.morphologyEx((bright_mask.astype('uint8') * 255), cv2.MORPH_OPEN, kernel, iterations=1)
    bright_labeled, ncomp = ndimage.label((bright_opened > 0))
    
    for comp_id in range(1, ncomp + 1):
        comp_mask = bright_labeled == comp_id
        comp_area = np.sum(comp_mask)
        if comp_area / float(total_pixels) >= 0.001:
            mean_y = np.mean(Y[comp_mask])
            mean_max = np.mean(max_chan[comp_mask])
            if mean_y > 0.95 or mean_max > 240:
                over_mask[comp_mask] = True
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    over_mask = cv2.dilate((over_mask.astype('uint8') * 255), kernel_dilate, iterations=2) > 0

    glare_mask, white_mask = classify_glare_vs_white(arr, over_mask, min_glare_size=min_glare_size)
    
    if np.any(glare_mask):
        glare_mask_uint8 = (glare_mask.astype('uint8') * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask_dilated = cv2.dilate(glare_mask_uint8, kernel, iterations=2)
        glare_mask_soft = cv2.GaussianBlur(glare_mask_dilated, (15, 15), sigmaX=5.0, sigmaY=5.0)
        glare_mask_soft_norm = glare_mask_soft.astype('float32') / 255.0

        if mode == 'darken':
            arr_uint8 = np.clip(arr, 0, 255).astype('uint8')
            final = arr_uint8.copy().astype('float32')
            pixel_changes = []

            clahe_img = apply_clahe(arr_uint8)
            final = blend_enhancement(final, clahe_img, glare_mask_soft_norm, blend_factor=blend_factor)

            R_o = arr_uint8[..., 0].astype('float32')
            G_o = arr_uint8[..., 1].astype('float32')
            B_o = arr_uint8[..., 2].astype('float32')
            max_val = np.maximum.reduce([R_o, G_o, B_o])
            severity = np.clip((max_val - severity_start) / severity_range, 0.0, 1.0)

            mask_strength = glare_mask_soft_norm.astype('float32')
            reduction = (base_reduction + extra_reduction * severity) * mask_strength
            reduction = np.clip(reduction, 0.0, reduction_cap)

            final_uint8 = np.clip(final, 0, 255).astype('uint8')
            orig_rgb = arr_uint8.astype('float32')
            
            hsv = cv2.cvtColor(final_uint8, cv2.COLOR_RGB2HSV).astype('float32')
            H = hsv[..., 0]
            S = hsv[..., 1]
            V = hsv[..., 2]
            
            V_new = V * (1.0 - reduction)
            gamma_mask = V_new > 235
            if np.any(gamma_mask):
                V_new[gamma_mask] = 255.0 * np.power((V_new[gamma_mask] / 255.0), float(gamma_exponent))
            
            S_enhanced = np.clip(S * (1.0 + reduction * 0.3), 0, 255)
            
            hsv_darkened = np.stack([H, S_enhanced, V_new], axis=-1)
            final_rgb = cv2.cvtColor(np.clip(hsv_darkened, 0, 255).astype('uint8'), cv2.COLOR_HSV2RGB).astype('float32')
            
            min_allowed = 0.85 * orig_rgb
            final_rgb = np.maximum(final_rgb, min_allowed)
            
            final_uint8 = np.clip(final_rgb, 0, 255).astype('uint8')
            final_filtered = cv2.bilateralFilter(final_uint8, d=11, sigmaColor=100, sigmaSpace=100)
            final_float_smooth = final_filtered.astype('float32')
            
            transition_zone = (mask_strength > 0.05) & (mask_strength < 0.95)
            blending_weight = np.where(transition_zone, 0.5, 0.85)
            final = final_float_smooth * blending_weight[..., None] + final_rgb * (1.0 - blending_weight[..., None])
            final = np.clip(final, 0, 255)

            mask_idx = mask_strength > 0.01
            if np.any(mask_idx):
                ys, xs = np.nonzero(mask_idx)
                for i in range(0, len(ys), max(1, len(ys) // 200)):
                    y, x = ys[i], xs[i]
                    orig = tuple(np.round(arr_uint8[y, x]).astype(int))
                    new = tuple(np.round(final[y, x]).astype(int))
                    pixel_changes.append({'y': y, 'x': x, 'orig_rgb': orig, 'new_rgb': new, 'type': 'aggressive_darken'})
            
            if pixel_changes:
                log_path = Path(out_dir) / 'pixel_changes' / (Path(path).stem + '_changes.txt')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, 'w') as f:
                    f.write('y,x,orig_r,orig_g,orig_b,new_r,new_g,new_b,type\n')
                    for change in pixel_changes:
                        orig = change['orig_rgb']
                        new = change['new_rgb']
                        f.write(f"{change['y']},{change['x']},{orig[0]},{orig[1]},{orig[2]},{new[0]},{new[1]},{new[2]},{change['type']}\n")
        elif mode == 'hybrid':
            white_clip, colored_glare = classify_glare_type(arr, glare_mask_soft_norm > 0.5)
            
            mode_map = np.zeros(arr.shape[:2], dtype='uint8')
            mode_map[colored_glare] = 1
            mode_map[white_clip] = 2
            
            final = process_overexposed_hybrid(arr, glare_mask_soft_norm > 0.5, mode_map)
            
            if np.any(colored_glare):
                colored_mask = colored_glare.astype('float32')
                colored_smooth = cv2.GaussianBlur((colored_mask * 255).astype('uint8'), (15, 15), 5.0)
                colored_smooth_norm = colored_smooth.astype('float32') / 255.0
                final = final * colored_smooth_norm[..., None] + arr * (1.0 - colored_smooth_norm[..., None])
            
            if np.any(white_clip):
                white_smooth = cv2.GaussianBlur((white_clip.astype('uint8') * 255), (15, 15), 5.0)
                white_smooth_norm = white_smooth.astype('float32') / 255.0
                final = final * white_smooth_norm[..., None] + arr * (1.0 - white_smooth_norm[..., None])
        else:
            # Use enhanced blending with robust repair
            repaired = robust_clip_repair(arr, glare_mask_soft_norm > 0.5)
            final = smooth_blend(arr, repaired.astype('float32'), glare_mask_soft_norm > 0.5, 
                               smooth_sigma=smooth_sigma, feather_sigma=feather_sigma, 
                               blend_method=blend_method)
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
    parser.add_argument('--mode', type=str, choices=['inpaint','darken','hybrid'], default='hybrid', help='Processing mode: inpaint, darken, or hybrid (default)')
    parser.add_argument('--blend_method', type=str, choices=['auto','poisson','multiscale','adaptive','gradient'], 
                        default='auto', help='Blending method: auto (default), poisson, multiscale, adaptive, or gradient')
    parser.add_argument('--darken_frac', type=float, default=0.35, help='Fraction to darken overexposed areas (0..1)')
    parser.add_argument('--base_reduction', type=float, default=0.06, help='Base per-pixel reduction fraction (0..1)')
    parser.add_argument('--extra_reduction', type=float, default=0.10, help='Extra per-pixel reduction scaled by severity (0..1)')
    parser.add_argument('--severity_start', type=float, default=235.0, help='Start of severity mapping (pixel value)')
    parser.add_argument('--severity_range', type=float, default=20.0, help='Range over which severity maps to 0..1')
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
            process_image(p, output_dir, radius=args.radius, smooth_sigma=args.smooth_sigma, 
                        feather_sigma=args.feather_sigma, min_glare_size=args.min_glare_size, 
                        pyramid_levels=args.pyramid_levels, mode=args.mode, blend_method=args.blend_method,
                        darken_frac=args.darken_frac, base_reduction=args.base_reduction, 
                        extra_reduction=args.extra_reduction, severity_start=args.severity_start, 
                        severity_range=args.severity_range, blend_factor=args.blend_factor, 
                        gamma_exponent=args.gamma_exponent, reduction_cap=args.reduction_cap)
            print('Preprocessed', p.name)
        except Exception as e:
            print('Failed', p.name, 'error:', e)


if __name__ == '__main__':
    main()
