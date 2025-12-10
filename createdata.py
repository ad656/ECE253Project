#!/usr/bin/env python3
"""
Complete pipeline for glare/overexposure detection and reconstruction:
1. Synthetic glare generation
2. Train/test split
3. Traditional ML model for detection + reconstruction
Uses: Random Forest, Gradient Boosting, and classical CV techniques
"""

import argparse
import os
import random
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import json
import pickle

# ============================================================================
# PART 1: SYNTHETIC GLARE GENERATION
# ============================================================================

def add_circular_glare(img_arr, num_glares=None, min_radius=20, max_radius=100, intensity_range=(0.7, 1.0)):
    """Add circular glare spots with realistic falloff"""
    h, w = img_arr.shape[:2]
    result = img_arr.copy().astype('float32')
    
    if num_glares is None:
        num_glares = random.randint(1, 4)
    
    glare_mask = np.zeros((h, w), dtype='float32')
    
    for _ in range(num_glares):
        cx = random.randint(max_radius, w - max_radius)
        cy = random.randint(max_radius, h - max_radius)
        radius = random.randint(min_radius, max_radius)
        
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        sigma = radius / 2.5
        glare = np.exp(-(dist**2) / (2 * sigma**2))
        
        intensity = random.uniform(*intensity_range)
        glare = glare * intensity
        
        glare_mask = np.maximum(glare_mask, glare)
    
    for c in range(3):
        result[..., c] = result[..., c] * (1.0 - glare_mask) + 255.0 * glare_mask
    
    return np.clip(result, 0, 255).astype('uint8'), (glare_mask > 0.3).astype('uint8')


def add_elliptical_glare(img_arr, num_glares=None):
    """Add elliptical/oval glare patterns"""
    h, w = img_arr.shape[:2]
    result = img_arr.copy().astype('float32')
    
    if num_glares is None:
        num_glares = random.randint(1, 3)
    
    glare_mask = np.zeros((h, w), dtype='float32')
    
    for _ in range(num_glares):
        cx = random.randint(100, w - 100)
        cy = random.randint(100, h - 100)
        
        major_axis = random.randint(60, 150)
        minor_axis = random.randint(40, 100)
        angle = random.uniform(0, 180)
        
        y, x = np.ogrid[:h, :w]
        x_rot = (x - cx) * np.cos(np.radians(angle)) + (y - cy) * np.sin(np.radians(angle))
        y_rot = -(x - cx) * np.sin(np.radians(angle)) + (y - cy) * np.cos(np.radians(angle))
        
        dist = np.sqrt((x_rot / major_axis)**2 + (y_rot / minor_axis)**2)
        glare = np.exp(-dist**2 * 4)
        
        intensity = random.uniform(0.6, 0.95)
        glare = glare * intensity
        
        glare_mask = np.maximum(glare_mask, glare)
    
    for c in range(3):
        result[..., c] = result[..., c] * (1.0 - glare_mask) + 255.0 * glare_mask
    
    return np.clip(result, 0, 255).astype('uint8'), (glare_mask > 0.3).astype('uint8')


def add_lens_flare(img_arr):
    """Add realistic lens flare effects"""
    h, w = img_arr.shape[:2]
    result = img_arr.copy().astype('float32')
    
    cx = random.randint(w//4, 3*w//4)
    cy = random.randint(h//4, 3*h//4)
    
    glare_mask = np.zeros((h, w), dtype='float32')
    
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    central_glare = np.exp(-(dist**2) / (2 * (h/10)**2)) * 0.9
    glare_mask = np.maximum(glare_mask, central_glare)
    
    num_spots = random.randint(2, 5)
    for i in range(num_spots):
        offset_x = random.randint(-w//3, w//3)
        offset_y = random.randint(-h//3, h//3)
        spot_x = np.clip(cx + offset_x, 0, w-1)
        spot_y = np.clip(cy + offset_y, 0, h-1)
        
        dist = np.sqrt((x - spot_x)**2 + (y - spot_y)**2)
        spot_size = random.randint(20, 60)
        spot = np.exp(-(dist**2) / (2 * spot_size**2)) * random.uniform(0.3, 0.6)
        glare_mask = np.maximum(glare_mask, spot)
    
    tint = np.array([random.uniform(0.9, 1.0), random.uniform(0.85, 1.0), random.uniform(0.8, 0.95)])
    for c in range(3):
        result[..., c] = result[..., c] * (1.0 - glare_mask) + 255.0 * glare_mask * tint[c]
    
    return np.clip(result, 0, 255).astype('uint8'), (glare_mask > 0.3).astype('uint8')


def add_overexposure(img_arr, regions=None):
    """Simulate overexposed regions"""
    h, w = img_arr.shape[:2]
    result = img_arr.copy().astype('float32')
    
    if regions is None:
        regions = random.randint(1, 3)
    
    glare_mask = np.zeros((h, w), dtype='float32')
    
    for _ in range(regions):
        x1 = random.randint(0, w - 100)
        y1 = random.randint(0, h - 100)
        width = random.randint(80, 200)
        height = random.randint(80, 200)
        x2 = min(x1 + width, w)
        y2 = min(y1 + height, h)
        
        region_mask = np.zeros((h, w), dtype='float32')
        region_mask[y1:y2, x1:x2] = 1.0
        
        region_mask = cv2.GaussianBlur(region_mask, (51, 51), 15)
        
        intensity = random.uniform(0.7, 0.95)
        region_mask = region_mask * intensity
        
        glare_mask = np.maximum(glare_mask, region_mask)
    
    for c in range(3):
        result[..., c] = np.where(glare_mask > 0.3, 255, result[..., c])
    
    return np.clip(result, 0, 255).astype('uint8'), (glare_mask > 0.3).astype('uint8')


def generate_synthetic_glare(image_path, output_dir, glare_types=['circular', 'elliptical', 'lens_flare', 'overexposure']):
    """Generate synthetic glare for one image"""
    img = Image.open(image_path).convert('RGB')
    img_arr = np.array(img)
    
    stem = Path(image_path).stem
    
    glare_type = random.choice(glare_types)
    
    if glare_type == 'circular':
        glared, mask = add_circular_glare(img_arr)
    elif glare_type == 'elliptical':
        glared, mask = add_elliptical_glare(img_arr)
    elif glare_type == 'lens_flare':
        glared, mask = add_lens_flare(img_arr)
    else:
        glared, mask = add_overexposure(img_arr)
    
    glared_path = Path(output_dir) / 'glared' / f"{stem}_glared.png"
    glared_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(glared).save(glared_path)
    
    mask_path = Path(output_dir) / 'masks' / f"{stem}_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask * 255).save(mask_path)
    
    original_path = Path(output_dir) / 'original' / f"{stem}_original.png"
    original_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_arr).save(original_path)
    
    return {
        'original': str(original_path),
        'glared': str(glared_path),
        'mask': str(mask_path),
        'glare_type': glare_type
    }


# ============================================================================
# PART 2: DATASET PREPARATION
# ============================================================================

def prepare_dataset(data_dir, output_dir, test_split=0.2):
    """Generate synthetic data and split into train/test"""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}
    images = [p for p in data_path.iterdir() if p.suffix.lower() in image_exts]
    
    print(f"Found {len(images)} images")
    
    print("Generating synthetic glare...")
    all_data = []
    for i, img_path in enumerate(images):
        try:
            result = generate_synthetic_glare(img_path, output_path)
            all_data.append(result)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(images)}")
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    
    train_data, test_data = train_test_split(all_data, test_size=test_split, random_state=42)
    
    for split_name, split_data in [('train', train_data), ('test', test_data)]:
        split_dir = output_path / split_name
        
        for data_type in ['glared', 'masks', 'original']:
            (split_dir / data_type).mkdir(parents=True, exist_ok=True)
        
        for item in split_data:
            for key in ['glared', 'mask', 'original']:
                src = Path(item[key if key != 'mask' else 'mask'])
                dst_type = key if key != 'mask' else 'masks'
                dst = split_dir / dst_type / src.name
                shutil.copy(src, dst)
    
    metadata = {
        'train_size': len(train_data),
        'test_size': len(test_data),
        'total_size': len(all_data)
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset prepared:")
    print(f"  Train: {len(train_data)} images")
    print(f"  Test: {len(test_data)} images")
    print(f"  Saved to: {output_path}")
    
    return train_data, test_data


# ============================================================================
# PART 3: FEATURE EXTRACTION & TRADITIONAL ML
# ============================================================================

def extract_pixel_features(img_arr, patch_size=5):
    """Extract features for each pixel"""
    h, w = img_arr.shape[:2]
    pad = patch_size // 2
    
    # Convert to LAB for better color features
    lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB).astype('float32')
    
    # Pad image
    img_padded = np.pad(img_arr, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    lab_padded = np.pad(lab, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    features_list = []
    
    for y in range(h):
        for x in range(w):
            # RGB values
            r, g, b = img_arr[y, x]
            
            # LAB values
            l, a, b_lab = lab[y, x]
            
            # Local patch statistics
            patch_rgb = img_padded[y:y+patch_size, x:x+patch_size]
            patch_lab = lab_padded[y:y+patch_size, x:x+patch_size]
            
            # Basic stats
            mean_rgb = np.mean(patch_rgb, axis=(0, 1))
            std_rgb = np.std(patch_rgb, axis=(0, 1))
            max_rgb = np.max(patch_rgb, axis=(0, 1))
            min_rgb = np.min(patch_rgb, axis=(0, 1))
            
            # Brightness and saturation
            brightness = l
            saturation = np.sqrt(a**2 + b_lab**2)
            
            # Create feature vector
            features = [
                r, g, b,  # RGB
                l, a, b_lab,  # LAB
                brightness, saturation,  # Color properties
                *mean_rgb, *std_rgb,  # Local stats
                max_rgb[0], max_rgb[1], max_rgb[2],  # Max values
                min_rgb[0] - min_rgb[2],  # Color difference
            ]
            
            features_list.append(features)
    
    return np.array(features_list)


def extract_features_fast(img_arr, sample_rate=10, use_local_context=True):
    """Enhanced feature extraction with local context and glare-specific features"""
    h, w = img_arr.shape[:2]
    
    # Sample pixels
    ys = np.arange(0, h, sample_rate)
    xs = np.arange(0, w, sample_rate)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    
    sampled_pixels = img_arr[yy, xx]
    
    # RGB features
    r = sampled_pixels[..., 0].flatten().astype('float32')
    g = sampled_pixels[..., 1].flatten().astype('float32')
    b = sampled_pixels[..., 2].flatten().astype('float32')
    
    # Basic color features
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    max_val = np.maximum.reduce([r, g, b])
    min_val = np.minimum.reduce([r, g, b])
    saturation = (max_val - min_val) / (max_val + 1e-6)
    
    # Glare-specific features
    # 1. How close to white (all channels high)
    whiteness = (r + g + b) / (3 * 255)
    near_white = ((r > 240) & (g > 240) & (b > 240)).astype('float32')
    
    # 2. Channel clipping indicators
    r_clipped = (r >= 250).astype('float32')
    g_clipped = (g >= 250).astype('float32')
    b_clipped = (b >= 250).astype('float32')
    any_clipped = (r_clipped + g_clipped + b_clipped) / 3.0
    
    # 3. Color uniformity (glare is often uniform)
    color_range = max_val - min_val
    uniformity = 1.0 - (color_range / 255.0)
    
    # 4. LAB color space features (better for perceptual analysis)
    # Convert sampled region to LAB
    sampled_rgb = img_arr[yy, xx].reshape(-1, 3)
    lab_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
    sampled_lab = lab_img[yy, xx].reshape(-1, 3)
    
    L = sampled_lab[:, 0].astype('float32')  # Lightness
    A = sampled_lab[:, 1].astype('float32')  # Green-Red
    B_lab = sampled_lab[:, 2].astype('float32')  # Blue-Yellow
    
    # High L with low A/B indicates desaturated bright (typical glare)
    lab_saturation = np.sqrt(A**2 + B_lab**2)
    high_L_low_sat = ((L > 200) & (lab_saturation < 30)).astype('float32')
    
    features = [
        r, g, b,  # 0-2: RGB
        brightness,  # 3: Overall brightness
        saturation,  # 4: HSV saturation
        max_val, min_val,  # 5-6: Range
        r - g, r - b, g - b,  # 7-9: Color differences
        whiteness,  # 10: How white
        near_white,  # 11: Binary near-white flag
        r_clipped, g_clipped, b_clipped,  # 12-14: Clipping indicators
        any_clipped,  # 15: Any channel clipped
        color_range,  # 16: Color variation
        uniformity,  # 17: Color uniformity
        L, A, B_lab,  # 18-20: LAB color space
        lab_saturation,  # 21: LAB-based saturation
        high_L_low_sat,  # 22: Bright desaturated indicator
    ]
    
    # Add local context features if requested
    if use_local_context:
        # Compute local gradients (edge detection)
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY).astype('float32')
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        sampled_grad = gradient_mag[yy, xx].flatten()
        low_gradient = (sampled_grad < 10).astype('float32')  # Flat regions (typical in glare)
        
        # Local brightness variance (glare has low variance)
        kernel_size = sample_rate * 2 + 1
        brightness_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY).astype('float32')
        local_mean = cv2.blur(brightness_img, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(brightness_img**2, (kernel_size, kernel_size))
        local_var = local_sq_mean - local_mean**2
        
        sampled_var = local_var[yy, xx].flatten()
        low_variance = (sampled_var < 50).astype('float32')
        
        features.extend([
            sampled_grad,  # 23: Gradient magnitude
            low_gradient,  # 24: Low gradient flag
            sampled_var,  # 25: Local variance
            low_variance,  # 26: Low variance flag
        ])
    
    features_array = np.column_stack(features)
    
    return features_array, (yy.flatten(), xx.flatten())


def train_glare_detector(train_dir, model_save_path, sample_rate=10, max_images=100):
    """Train Random Forest classifier for glare detection"""
    print("Training glare detector...")
    
    train_glared_dir = Path(train_dir) / 'glared'
    train_mask_dir = Path(train_dir) / 'masks'
    
    glared_files = list(train_glared_dir.glob('*.png'))[:max_images]
    
    all_features = []
    all_labels = []
    
    for i, gf in enumerate(glared_files):
        stem = gf.stem.replace('_glared', '')
        
        # Load images
        img = np.array(Image.open(gf).convert('RGB'))
        mask_file = train_mask_dir / f"{stem}_mask.png"
        mask = np.array(Image.open(mask_file).convert('L')) > 127
        
        # Extract features
        features, (ys, xs) = extract_features_fast(img, sample_rate=sample_rate)
        labels = mask[ys, xs].astype(int)
        
        all_features.append(features)
        all_labels.append(labels)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(glared_files)} images")
    
    # Concatenate all data
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    print(f"Training on {len(X)} samples...")
    print(f"  Glare pixels: {np.sum(y == 1)}")
    print(f"  Normal pixels: {np.sum(y == 0)}")
    
    # Train Random Forest with better hyperparameters
    clf = RandomForestClassifier(
        n_estimators=100,  # More trees
        max_depth=20,  # Deeper trees
        min_samples_split=50,  # Less strict
        min_samples_leaf=20,
        class_weight='balanced',  # Handle class imbalance
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    clf.fit(X, y)
    
    # Save model
    with open(model_save_path, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"Model saved to: {model_save_path}")
    
    return clf


def train_reconstruction_model(train_dir, model_save_path, sample_rate=10, max_images=100):
    """
    Train model to predict ORIGINAL RGB values from glared RGB values.
    This learns the inverse transformation.
    """
    print("Training reconstruction model...")
    
    train_glared_dir = Path(train_dir) / 'glared'
    train_original_dir = Path(train_dir) / 'original'
    train_mask_dir = Path(train_dir) / 'masks'
    
    glared_files = list(train_glared_dir.glob('*.png'))[:max_images]
    
    all_features = []
    all_targets_r = []
    all_targets_g = []
    all_targets_b = []
    
    for i, gf in enumerate(glared_files):
        stem = gf.stem.replace('_glared', '')
        
        # Load glared image
        img_glared = np.array(Image.open(gf).convert('RGB'))
        
        # Load original image
        orig_file = train_original_dir / f"{stem}_original.png"
        img_original = np.array(Image.open(orig_file).convert('RGB'))
        
        # Load mask (only train on glare pixels)
        mask_file = train_mask_dir / f"{stem}_mask.png"
        mask = np.array(Image.open(mask_file).convert('L')) > 127
        
        # Extract features from glared pixels
        features, (ys, xs) = extract_features_fast(img_glared, sample_rate=sample_rate)
        
        # Get original RGB values at those positions (targets)
        orig_r = img_original[ys, xs, 0]
        orig_g = img_original[ys, xs, 1]
        orig_b = img_original[ys, xs, 2]
        
        # Only keep samples where mask indicates glare
        mask_sampled = mask[ys, xs]
        
        all_features.append(features[mask_sampled])
        all_targets_r.append(orig_r[mask_sampled])
        all_targets_g.append(orig_g[mask_sampled])
        all_targets_b.append(orig_b[mask_sampled])
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(glared_files)} images")
    
    # Concatenate all data
    X = np.vstack(all_features)
    y_r = np.concatenate(all_targets_r)
    y_g = np.concatenate(all_targets_g)
    y_b = np.concatenate(all_targets_b)
    
    print(f"Training on {len(X)} glare pixels...")
    
    # Train separate regressors for R, G, B channels with better hyperparameters
    print("  Training R channel regressor...")
    reg_r = GradientBoostingRegressor(
        n_estimators=100,  # More estimators
        max_depth=7,  # Deeper trees
        learning_rate=0.05,  # Lower learning rate
        subsample=0.8,  # Stochastic gradient boosting
        random_state=42,
        verbose=1
    )
    reg_r.fit(X, y_r)
    
    print("  Training G channel regressor...")
    reg_g = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=1
    )
    reg_g.fit(X, y_g)
    
    print("  Training B channel regressor...")
    reg_b = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=1
    )
    reg_b.fit(X, y_b)
    
    # Save all models
    models = {
        'r': reg_r,
        'g': reg_g,
        'b': reg_b
    }
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(models, f)
    
    print(f"Reconstruction models saved to: {model_save_path}")
    
    return models


def intelligent_color_recovery(arr, mask, threshold=50):
    """Recover colors for detected glare regions"""
    # Use lower threshold for mask
    mask_bool = mask > threshold
    
    print(f"  Mask stats: min={mask.min()}, max={mask.max()}, pixels_detected={np.sum(mask_bool)}")
    
    if not np.any(mask_bool):
        print("  WARNING: No glare pixels detected!")
        return arr
    
    result = arr.copy().astype('float32')
    
    R = result[..., 0]
    G = result[..., 1]
    B = result[..., 2]
    
    max_val = np.maximum.reduce([R, G, B])
    min_val = np.minimum.reduce([R, G, B])
    color_diff = max_val - min_val
    
    pixels_processed = 0
    white_pixels = 0
    colored_pixels = 0
    gray_pixels = 0
    
    for y, x in zip(*np.nonzero(mask_bool)):
        r, g, b = R[y, x], G[y, x], B[y, x]
        max_c = max(r, g, b)
        color_d = color_diff[y, x]
        
        if r >= 250 and g >= 250 and b >= 250:
            # Pure white -> light gray
            R[y, x] = 235
            G[y, x] = 235
            B[y, x] = 235
            white_pixels += 1
        elif color_d >= 10:
            # Detectable color -> reduce by 12%
            R[y, x] = r * 0.88
            G[y, x] = g * 0.88
            B[y, x] = b * 0.88
            colored_pixels += 1
        else:
            # Light gray -> darken
            if min_c >= 245:
                R[y, x] = 232
                G[y, x] = 232
                B[y, x] = 232
            else:
                R[y, x] = r * 0.88
                G[y, x] = g * 0.88
                B[y, x] = b * 0.88
            gray_pixels += 1
        
        pixels_processed += 1
    
    print(f"  Processed {pixels_processed} pixels: white={white_pixels}, colored={colored_pixels}, gray={gray_pixels}")
    
    result[..., 0] = R
    result[..., 1] = G
    result[..., 2] = B
    
    return np.clip(result, 0, 255).astype('uint8')


def test_glare_detector(model_path_detect, model_path_recon, test_dir, output_dir, num_samples=10, sample_rate=10, use_gt_mask=False):
    """Test the glare detector and reconstruction"""
    print("Testing glare detection and reconstruction...")
    if use_gt_mask:
        print("  Using ground truth masks for testing reconstruction quality")
    
    # Load detection model
    with open(model_path_detect, 'rb') as f:
        clf = pickle.load(f)
    
    # Load reconstruction models
    with open(model_path_recon, 'rb') as f:
        recon_models = pickle.load(f)
    
    test_glared_dir = Path(test_dir) / 'glared'
    test_mask_dir = Path(test_dir) / 'masks'
    test_original_dir = Path(test_dir) / 'original'
    test_files = list(test_glared_dir.glob('*.png'))[:num_samples]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create change log directory
    changes_dir = output_path / 'pixel_changes'
    changes_dir.mkdir(exist_ok=True)
    
    for i, test_file in enumerate(test_files):
        print(f"\nProcessing image {i+1}/{len(test_files)}: {test_file.name}")
        
        # Load glared image
        img_glared = np.array(Image.open(test_file).convert('RGB'))
        h, w = img_glared.shape[:2]
        
        # Load ground truth
        stem = test_file.stem.replace('_glared', '')
        gt_mask_file = test_mask_dir / f"{stem}_mask.png"
        gt_original_file = test_original_dir / f"{stem}_original.png"
        
        gt_mask = np.array(Image.open(gt_mask_file).convert('L')) if gt_mask_file.exists() else None
        gt_original = np.array(Image.open(gt_original_file).convert('RGB')) if gt_original_file.exists() else None
        
        # STEP 1: Detect glare regions (or use ground truth)
        if use_gt_mask and gt_mask is not None:
            print("  Using ground truth mask")
            pred_mask_binary = gt_mask
        else:
            print("  Extracting features...")
            features, (ys, xs) = extract_features_fast(img_glared, sample_rate=sample_rate)
            
            print("  Predicting glare regions...")
            pred_labels = clf.predict(features)
            pred_proba = clf.predict_proba(features)[:, 1]
            
            print(f"  Predicted {np.sum(pred_labels)} glare pixels out of {len(pred_labels)} sampled")
            print(f"  Probability range: {pred_proba.min():.3f} to {pred_proba.max():.3f}")
            
            # Create full-size mask using direct assignment (no interpolation yet)
            pred_mask_sparse = np.zeros((h, w), dtype='uint8')
            pred_mask_sparse[ys, xs] = (pred_labels * 255).astype('uint8')
            
            # Dilate to fill gaps from sampling
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sample_rate*2+1, sample_rate*2+1))
            pred_mask_filled = cv2.dilate(pred_mask_sparse, kernel_dilate, iterations=1)
            
            # Smooth the mask
            pred_mask_binary = cv2.GaussianBlur(pred_mask_filled, (21, 21), 5)
            
            print(f"  Mask after processing: min={pred_mask_binary.min()}, max={pred_mask_binary.max()}, "
                  f"pixels>{50}: {np.sum(pred_mask_binary > 50)}")
        
        # STEP 2: Reconstruct using learned model
        print("  Reconstructing image using learned model...")
        reconstructed = img_glared.copy()
        
        # Get all pixels where mask > threshold
        glare_pixels = pred_mask_binary > 50
        num_glare_pixels = np.sum(glare_pixels)
        print(f"  Reconstructing {num_glare_pixels} pixels...")
        
        # Track pixel changes
        pixel_changes = []
        
        if num_glare_pixels > 0:
            # Extract features for glare pixels
            glare_ys, glare_xs = np.nonzero(glare_pixels)
            
            # Sample features (can't extract for every pixel, too slow)
            step = max(1, len(glare_ys) // 10000)  # Limit to 10k pixels
            sample_idx = np.arange(0, len(glare_ys), step)
            glare_ys_sample = glare_ys[sample_idx]
            glare_xs_sample = glare_xs[sample_idx]
            
            print(f"    Extracting features for {len(glare_ys_sample)} sampled pixels...")
            
            # Extract features for sampled glare pixels (using enhanced features)
            glare_features = []
            for y, x in zip(glare_ys_sample, glare_xs_sample):
                # Enhanced feature extraction matching training
                r, g, b = img_glared[y, x]
                
                # Basic features
                brightness = 0.299 * r + 0.587 * g + 0.114 * b
                max_val = max(r, g, b)
                min_val = min(r, g, b)
                saturation = (max_val - min_val) / (max_val + 1e-6)
                
                # Glare-specific features
                whiteness = (r + g + b) / (3 * 255)
                near_white = float((r > 240) and (g > 240) and (b > 240))
                r_clipped = float(r >= 250)
                g_clipped = float(g >= 250)
                b_clipped = float(b >= 250)
                any_clipped = (r_clipped + g_clipped + b_clipped) / 3.0
                color_range = max_val - min_val
                uniformity = 1.0 - (color_range / 255.0)
                
                # LAB features
                pixel_rgb = np.array([[[r, g, b]]], dtype='uint8')
                pixel_lab = cv2.cvtColor(pixel_rgb, cv2.COLOR_RGB2LAB)[0, 0]
                L, A, B_lab = pixel_lab
                lab_saturation = np.sqrt(A**2 + B_lab**2)
                high_L_low_sat = float((L > 200) and (lab_saturation < 30))
                
                # Local context (simplified for single pixel)
                grad_val = 0  # Would need neighborhood
                low_gradient = 0
                local_variance = 0
                low_variance = 0
                
                features = [
                    r, g, b, brightness, saturation, max_val, min_val,
                    r-g, r-b, g-b, whiteness, near_white,
                    r_clipped, g_clipped, b_clipped, any_clipped,
                    color_range, uniformity, L, A, B_lab, lab_saturation, high_L_low_sat,
                    grad_val, low_gradient, local_variance, low_variance
                ]
                glare_features.append(features)
            
            glare_features = np.array(glare_features)
            
            # Predict original RGB values
            print("    Predicting original RGB values...")
            pred_r = recon_models['r'].predict(glare_features)
            pred_g = recon_models['g'].predict(glare_features)
            pred_b = recon_models['b'].predict(glare_features)
            
            # Apply predictions
            for idx, (y, x) in enumerate(zip(glare_ys_sample, glare_xs_sample)):
                orig_rgb = img_glared[y, x]
                new_r = np.clip(pred_r[idx], 0, 255)
                new_g = np.clip(pred_g[idx], 0, 255)
                new_b = np.clip(pred_b[idx], 0, 255)
                
                reconstructed[y, x] = [new_r, new_g, new_b]
                
                # Track changes for first 200 pixels
                if len(pixel_changes) < 200:
                    rgb_change = np.abs(np.array([new_r, new_g, new_b]) - orig_rgb)
                    pixel_changes.append({
                        'y': y, 'x': x,
                        'orig_r': int(orig_rgb[0]), 'orig_g': int(orig_rgb[1]), 'orig_b': int(orig_rgb[2]),
                        'new_r': int(new_r), 'new_g': int(new_g), 'new_b': int(new_b),
                        'change_r': float(rgb_change[0]), 'change_g': float(rgb_change[1]), 'change_b': float(rgb_change[2]),
                        'total_change': float(np.sum(rgb_change))
                    })
            
            # Interpolate to fill all glare pixels
            print("    Interpolating to fill all glare regions...")
            reconstructed = cv2.inpaint(reconstructed, (glare_pixels).astype('uint8') * 255, 3, cv2.INPAINT_NS)
        
        # Save pixel changes to CSV
        if pixel_changes:
            change_file = changes_dir / f"{stem}_changes.csv"
            with open(change_file, 'w') as f:
                f.write('y,x,orig_r,orig_g,orig_b,new_r,new_g,new_b,change_r,change_g,change_b,total_change\n')
                for change in pixel_changes:
                    f.write(f"{change['y']},{change['x']},{change['orig_r']},{change['orig_g']},{change['orig_b']},"
                           f"{change['new_r']},{change['new_g']},{change['new_b']},{change['change_r']:.2f},"
                           f"{change['change_g']:.2f},{change['change_b']:.2f},{change['total_change']:.2f}\n")
            
            # Print statistics
            avg_change_r = np.mean([c['change_r'] for c in pixel_changes])
            avg_change_g = np.mean([c['change_g'] for c in pixel_changes])
            avg_change_b = np.mean([c['change_b'] for c in pixel_changes])
            avg_total = np.mean([c['total_change'] for c in pixel_changes])
            
            print(f"  Average RGB changes: R={avg_change_r:.2f}, G={avg_change_g:.2f}, B={avg_change_b:.2f}")
            print(f"  Average total change: {avg_total:.2f}")
        
        # Create visualizations
        # Side-by-side comparison
        comparison = np.hstack([img_glared, reconstructed])
        
        # If we have ground truth, add it and compute error
        if gt_original is not None:
            comparison = np.hstack([comparison, gt_original])
            
            # Compute reconstruction error (only on glare pixels)
            if gt_mask is not None:
                gt_mask_bool = gt_mask > 127
                error = np.abs(reconstructed.astype('float32') - gt_original.astype('float32'))
                error_on_glare = error[gt_mask_bool]
                avg_error = np.mean(error_on_glare)
                print(f"  Average reconstruction error on glare pixels: {avg_error:.2f}")
                
                # Create error heatmap
                error_map = np.mean(error, axis=2).astype('uint8')
                error_heatmap = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
                Image.fromarray(cv2.cvtColor(error_heatmap, cv2.COLOR_BGR2RGB)).save(
                    output_path / f"{stem}_06_error_map.png")
        
        # Create change visualization
        change_vis = np.zeros_like(img_glared)
        if pixel_changes:
            for change in pixel_changes:
                y, x = change['y'], change['x']
                intensity = min(255, int(change['total_change']))
                change_vis[y, x] = [intensity, 0, 255 - intensity]  # Red = big change, Blue = small change
        
        # Save results
        Image.fromarray(img_glared).save(output_path / f"{stem}_01_glared.png")
        Image.fromarray(pred_mask_binary).save(output_path / f"{stem}_02_pred_mask.png")
        Image.fromarray(reconstructed).save(output_path / f"{stem}_03_reconstructed.png")
        Image.fromarray(comparison).save(output_path / f"{stem}_04_comparison.png")
        Image.fromarray(change_vis).save(output_path / f"{stem}_05_change_vis.png")
        
        if gt_mask is not None:
            Image.fromarray(gt_mask).save(output_path / f"{stem}_07_gt_mask.png")
        if gt_original is not None:
            Image.fromarray(gt_original).save(output_path / f"{stem}_08_gt_original.png")
    
    print(f"\nTest results saved to: {output_path}")
    print("Files saved:")
    print("  *_01_glared.png - Original glared image")
    print("  *_02_pred_mask.png - Predicted glare mask")
    print("  *_03_reconstructed.png - Reconstructed image")
    print("  *_04_comparison.png - Side-by-side: glared | reconstructed | ground truth")
    print("  *_05_change_vis.png - Heatmap of pixel changes")
    print("  *_06_error_map.png - Error compared to ground truth")
    print("  pixel_changes/*_changes.csv - Detailed pixel-by-pixel changes")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Glare detection and reconstruction pipeline')
    parser.add_argument('--data_dir', default=r'C:\Users\allan\Downloads\data', help='Input images directory')
    parser.add_argument('--output_dir', default=r'C:\Users\allan\Downloads\glare_dataset', help='Output directory')
    parser.add_argument('--mode', choices=['prepare', 'train', 'test', 'all'], default='all')
    parser.add_argument('--detect_model_path', default=r'C:\Users\allan\Downloads\glare_detect_model.pkl', 
                       help='Detection model path')
    parser.add_argument('--recon_model_path', default=r'C:\Users\allan\Downloads\glare_recon_model.pkl',
                       help='Reconstruction model path')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--sample_rate', type=int, default=10, help='Pixel sampling rate for training')
    parser.add_argument('--max_train_images', type=int, default=200, help='Max images for training')
    parser.add_argument('--use_gt_mask', action='store_true', 
                       help='Use ground truth masks during testing (to test reconstruction only)')
    
    args = parser.parse_args()
    
    if args.mode in ['prepare', 'all']:
        print("=" * 70)
        print("STEP 1: PREPARING DATASET")
        print("=" * 70)
        prepare_dataset(args.data_dir, args.output_dir, test_split=args.test_split)
    
    if args.mode in ['train', 'all']:
        print("\n" + "=" * 70)
        print("STEP 2A: TRAINING DETECTION MODEL")
        print("=" * 70)
        train_dir = Path(args.output_dir) / 'train'
        train_glare_detector(train_dir, args.detect_model_path, 
                           sample_rate=args.sample_rate, 
                           max_images=args.max_train_images)
        
        print("\n" + "=" * 70)
        print("STEP 2B: TRAINING RECONSTRUCTION MODEL")
        print("=" * 70)
        train_reconstruction_model(train_dir, args.recon_model_path,
                                  sample_rate=args.sample_rate,
                                  max_images=args.max_train_images)
    
    if args.mode in ['test', 'all']:
        print("\n" + "=" * 70)
        print("STEP 3: TESTING MODELS")
        print("=" * 70)
        test_dir = Path(args.output_dir) / 'test'
        test_output_dir = Path(args.output_dir) / 'test_results'
        test_glare_detector(args.detect_model_path, args.recon_model_path, 
                          test_dir, test_output_dir, 
                          num_samples=10, sample_rate=args.sample_rate,
                          use_gt_mask=args.use_gt_mask)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()