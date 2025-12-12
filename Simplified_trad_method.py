import numpy as np
import cv2
from scipy import ndimage
from PIL import Image
from glob import glob
import os
from tqdm import tqdm
def compute_Y_S(arr):
    """Compute luminance (Y) and saturation (S) from RGB array."""
    Y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    max_chan = np.maximum.reduce([arr[..., 0], arr[..., 1], arr[..., 2]])
    min_chan = np.minimum.reduce([arr[..., 0], arr[..., 1], arr[..., 2]])
    S = (max_chan - min_chan) / (max_chan + 1e-10)
    return Y, S

def classify_glare_vs_white(arr, over_mask, min_glare_size=0.005):
    """Classify overexposed regions into glare and white areas."""
    total_pixels = arr.shape[0] * arr.shape[1]
    
    if not np.any(over_mask):
        return np.zeros_like(over_mask, dtype=bool), np.zeros_like(over_mask, dtype=bool)
    

    std_per_channel = np.std(arr, axis=2)
    color_uniformity = std_per_channel / (np.mean(arr, axis=2) + 1e-10)
  
    glare_mask = over_mask & (color_uniformity > 0.05)
    white_mask = over_mask & (color_uniformity <= 0.05)
    
    glare_labeled, n_glare = ndimage.label(glare_mask)
    for i in range(1, n_glare + 1):
        comp_mask = glare_labeled == i
        if np.sum(comp_mask) / total_pixels < min_glare_size:
            glare_mask[comp_mask] = False
            white_mask[comp_mask] = True
    
    white_labeled, n_white = ndimage.label(white_mask)
    for i in range(1, n_white + 1):
        comp_mask = white_labeled == i
        if np.sum(comp_mask) / total_pixels < min_glare_size:
            white_mask[comp_mask] = False
    
    return glare_mask, white_mask

def classify_glare_type(arr, glare_mask):
    """Classify glare regions into white clip and colored glare."""
    if not np.any(glare_mask):
        return np.zeros_like(glare_mask, dtype=bool), np.zeros_like(glare_mask, dtype=bool)

    R = arr[..., 0][glare_mask]
    G = arr[..., 1][glare_mask]
    B = arr[..., 2][glare_mask]
 
    max_vals = np.maximum.reduce([R, G, B])
    min_vals = np.minimum.reduce([R, G, B])
    white_ratio = min_vals / (max_vals + 1e-10)

    white_clip_mask = np.zeros_like(glare_mask, dtype=bool)
    colored_glare_mask = np.zeros_like(glare_mask, dtype=bool)

    white_threshold = 0.9
    white_pixels = white_ratio > white_threshold

    glare_indices = np.where(glare_mask)
    if len(glare_indices[0]) > 0:
       
        white_indices = np.where(white_pixels)[0]
        white_coords = (glare_indices[0][white_indices], glare_indices[1][white_indices])
        white_clip_mask[white_coords] = True
        
        colored_indices = np.where(~white_pixels)[0]
        colored_coords = (glare_indices[0][colored_indices], glare_indices[1][colored_indices])
        colored_glare_mask[colored_coords] = True
    
    return white_clip_mask, colored_glare_mask

def apply_clahe(img_rgb, clip_limit=2.0, grid_size=8):
    """Apply CLAHE to RGB image in LAB space."""
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_clahe = clahe.apply(l)
    
    img_lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)

def blend_enhancement(base, enhanced, mask, blend_factor=0.7):
    """Blend enhancement based on mask."""
    blend = base * (1 - mask[..., None]) + enhanced * mask[..., None]
    return blend

def process_overexposed_hybrid(arr, mask, mode_map):
  
    if mask.dtype != bool:
        mask = mask > 0.5

    masked_region = np.where(mask)
    
    if len(masked_region[0]) == 0:
        return arr
  
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    result = method5_hybrid(arr_uint8).astype(np.float32)
    mask_float = mask.astype(np.float32)
    final = arr * (1 - mask_float[..., None]) + result * mask_float[..., None]
    
    return final

def method2_adaptive_gamma(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    avg_luminance = np.mean(gray) / 255.0
    
    
    if avg_luminance > 0.5:
        gamma = 1.5 + (avg_luminance - 0.5) * 3.0  
    else:
        gamma = 1.0

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    result = cv2.LUT(image, table)
    
    return result

def method4_tone_mapping(image):
 
    img_float = image.astype(np.float32) / 255.0

    luminance = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
    epsilon = 1e-5
    log_avg_lum = np.exp(np.mean(np.log(luminance + epsilon)))
    
    key = 0.10  
    scaled_lum = (key / log_avg_lum) * luminance
    white_point = 1.0  
    tone_mapped_lum = (scaled_lum * (1 + scaled_lum / (white_point ** 2))) / (1 + scaled_lum)

    result = np.zeros_like(img_float)
    for c in range(3):
        result[:, :, c] = img_float[:, :, c] * (tone_mapped_lum / (luminance + epsilon))
    
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)
    
    return result


def method5_hybrid(image):

    img_float = image.astype(np.float32) / 255.0
    original_img = img_float.copy() 

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    avg_brightness = np.mean(gray) / 255.0
    img_float = image.astype(np.float32) / 255.0
    original_img = img_float.copy() 
    

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    avg_brightness = np.mean(gray) / 255.0

    print(f"  Original average brightness: {avg_brightness:.3f}")
    
    if avg_brightness > 0.6:  
        if avg_brightness > 0.8:  
            darken_factor = 0.6  
        elif avg_brightness > 0.7: 
            darken_factor = 0.7  
        elif avg_brightness > 0.6:  
            darken_factor = 0.8  
        else:
            darken_factor = 1.0 
        
        print(f"  Applying global darkening factor: {darken_factor:.2f}")
        img_float = img_float * darken_factor
    luminance = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
    
    highlight_threshold = 0.7  # Increased from 0.6 (only affect >70% brightness)
    highlight_mask = np.clip((luminance - highlight_threshold) / (1.0 - highlight_threshold), 0, 1)
    
    if np.max(highlight_mask) > 0:
        # More gradual compression for fewer artifacts
        compression_strength = highlight_mask ** 0.8 * 0.4  
        
        # Blend compression gradually to avoid harsh transitions
        for c in range(3):
            img_float[:, :, c] = img_float[:, :, c] * (1 - compression_strength) + \
                                 img_float[:, :, c] * 0.1 * compression_strength  # Preserve some brightness
    
    # Step 3: MODERATE ADAPTIVE GAMMA CORRECTION
    current_avg = np.mean(luminance)
    if current_avg > 0.5:  # Apply gamma only if still quite bright
        # More moderate gamma values
        if current_avg > 0.75:
            gamma = 1.4  # Reduced from 1.8
        elif current_avg > 0.65:
            gamma = 1.3  # Reduced from 1.6
        elif current_avg > 0.55:
            gamma = 1.2  # Reduced from 1.4
        else:
            gamma = 1.1  # Reduced from 1.2
        
        inv_gamma = 1.0 / gamma
        img_float = np.power(img_float, inv_gamma)
    
    # Convert to uint8 for next steps
    intermediate = np.clip(img_float * 255, 0, 255).astype(np.uint8)
    
    # Step 4: GENTLE MULTI-SCALE RETINEX with artifact prevention
    retinex_result = gentle_retinex(intermediate)
    
    # Blend with moderate weight
    blend_ratio = 0.4  # Reduced from 0.6 (less Retinex influence)
    result = cv2.addWeighted(intermediate, 1 - blend_ratio, retinex_result, blend_ratio, 0)
    
    # Step 5: NATURAL COLOR RESTORATION (less aggressive)
    result_float = result.astype(np.float32) / 255.0
    
    # Calculate luminance after processing
    luminance_result = 0.299 * result_float[:, :, 0] + 0.587 * result_float[:, :, 1] + 0.114 * result_float[:, :, 2]
    
    # More natural saturation boost - only boost where it was lost
    # Calculate original saturation
    original_max = np.maximum.reduce([original_img[:, :, 0], original_img[:, :, 1], original_img[:, :, 2]])
    original_min = np.minimum.reduce([original_img[:, :, 0], original_img[:, :, 1], original_img[:, :, 2]])
    original_saturation = (original_max - original_min) / (original_max + 1e-10)
    
    # Calculate current saturation
    current_max = np.maximum.reduce([result_float[:, :, 0], result_float[:, :, 1], result_float[:, :, 2]])
    current_min = np.minimum.reduce([result_float[:, :, 0], result_float[:, :, 1], result_float[:, :, 2]])
    current_saturation = (current_max - current_min) / (current_max + 1e-10)
    
    # Only boost saturation where it was lost
    saturation_loss = np.maximum(0, original_saturation - current_saturation)
    saturation_boost = saturation_loss * 0.5  # Max 50% of lost saturation (was 80%)
    
    # Convert to HSV and apply gentle saturation adjustment
    result_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
    result_hsv[:, :, 1] = np.clip(result_hsv[:, :, 1] * (1 + saturation_boost), 0, 255)
    result = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Step 6: GENTLE LOCAL CONTRAST ENHANCEMENT with artifact reduction
    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Use milder CLAHE to avoid artifacts
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Reduced from 3.0
    l_enhanced = clahe.apply(l)
    
    # Apply gentle contrast stretching only if needed
    l_min, l_max = l_enhanced.min(), l_enhanced.max()
    if l_max - l_min > 60 and l_max - l_min < 200:  # Only if reasonable dynamic range
        # Gentle normalization, preserving some original contrast
        l_enhanced = cv2.normalize(l_enhanced, None, l_min * 0.8, l_max * 1.2, cv2.NORM_MINMAX)
    
    result = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    
    # Step 7: ARTIFACT REDUCTION AND DETAIL PRESERVATION
    # Apply mild bilateral filtering to reduce artifacts while preserving edges
    result_float = result.astype(np.float32) / 255.0
    
    # Calculate edge mask to protect edges from smoothing
    gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_result, 50, 150)
    edge_mask = edges > 0
    edge_mask_dilated = cv2.dilate(edge_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    
    # Apply bilateral filter only to non-edge areas
    for c in range(3):
        channel = result_float[:, :, c]
        smoothed = cv2.bilateralFilter((channel * 255).astype(np.uint8), 5, 25, 25).astype(np.float32) / 255.0
        # Blend: smooth non-edges, keep edges sharp
        result_float[:, :, c] = smoothed * (1 - edge_mask_dilated.astype(np.float32)) + \
                                channel * edge_mask_dilated.astype(np.float32)
    
    result = np.clip(result_float * 255, 0, 255).astype(np.uint8)
    
    # Step 8: FINAL BRIGHTNESS ADJUSTMENT AND COLOR BALANCE
    # Calculate final brightness
    final_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    final_avg = np.mean(final_gray) / 255.0
    
    # Adjust brightness to natural range (0.4-0.6 is ideal for most images)
    if final_avg < 0.35:  # Too dark
        brighten_factor = 0.4 / final_avg  # Target 0.4 brightness
        result = np.clip(result.astype(np.float32) * brighten_factor, 0, 255).astype(np.uint8)
        print(f"  Brightness adjustment: ×{brighten_factor:.2f} (was too dark)")
    elif final_avg > 0.65:  # Still too bright
        darken_factor = 0.6 / final_avg  # Target 0.6 brightness
        result = np.clip(result.astype(np.float32) * darken_factor, 0, 255).astype(np.uint8)
        print(f"  Brightness adjustment: ×{darken_factor:.2f} (was too bright)")
    
    # Final gentle sharpening for detail (very mild)
    blurred = cv2.GaussianBlur(result, (0, 0), 0.8)
    sharpened = cv2.addWeighted(result, 1.2, blurred, -0.2, 0)  # Very mild sharpening
    result = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Final color balance check
    # Ensure colors don't go out of gamut
    result_float = result.astype(np.float32) / 255.0
    for c in range(3):
        channel = result_float[:, :, c]
        # Clip extreme values gently
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        lower_bound = max(0, mean_val - 2.5 * std_val)
        upper_bound = min(1, mean_val + 2.5 * std_val)
        channel = np.clip(channel, lower_bound, upper_bound)
        result_float[:, :, c] = channel
    
    result = np.clip(result_float * 255, 0, 255).astype(np.uint8)
    
    final_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    final_avg = np.mean(final_gray) / 255.0
    print(f"  Final average brightness: {final_avg:.3f}")

    # 🔥 Convert final RGB output → BGR (so OpenCV can save/display correctly)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

def gentle_retinex(image):
    """
    Gentle Retinex algorithm with artifact prevention
    """
    img_float = image.astype(np.float32) / 255.0
    img_float = np.maximum(img_float, 0.001)

    scales = [30, 80, 200]  
    retinex_sum = np.zeros_like(img_float)
    
    for scale in scales:
        # Use larger sigma for smoother illumination estimation
        blurred = cv2.GaussianBlur(img_float, (0, 0), scale * 1.5)
        blurred = np.maximum(blurred, 0.001)
        
        # Gentle weight
        weight = 1.0  # Equal weights
        
        retinex_sum += weight * (np.log10(img_float) - np.log10(blurred))
    
    # Weighted average
    retinex = retinex_sum / len(scales)
    
    # Gentle color restoration
    intensity = np.mean(img_float, axis=2, keepdims=True)
    intensity = np.maximum(intensity, 0.001)
    
    for c in range(3):
        # More natural color restoration
        alpha = 80.0 + intensity[:, :, 0] * 20  # Less aggressive alpha
        retinex[:, :, c] = 30 * (np.log10(alpha * img_float[:, :, c]) - np.log10(intensity[:, :, 0]))  # Reduced multiplier
    
    # Normalize gently to avoid extreme values
    p_low, p_high = np.percentile(retinex, [5, 95])  # Use 5-95 percentile to ignore extremes
    retinex = (retinex - p_low) / (p_high - p_low + 1e-7)
    retinex = np.clip(retinex, 0.1, 0.9)  # Keep away from extremes
    
    return (retinex * 255).astype(np.uint8)


def compare_all_methods(image_path, output_dir="comparison_results"):
    """
    Apply all methods and save side-by-side comparison
    """
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    
    methods = {
        "1_Original": img_np,
        "3_Adaptive_Gamma": method2_adaptive_gamma(img_np),
        "5_Tone_Mapping": method4_tone_mapping(img_np),
        "6_Hybrid": method5_hybrid(img_np),
    }
    

    basename = os.path.splitext(os.path.basename(image_path))[0]
    for name, result in methods.items():
        output_path = os.path.join(output_dir, f"{basename}_{name}.jpg")
        Image.fromarray(result).save(output_path)
    
    print(f"✓ Saved {len(methods)} comparison images to {output_dir}/")
    
    return methods


def process_directory_with_method(input_dir, output_dir, method_func, method_name):
    """
    Process all images in a directory with a specific method
    """
    os.makedirs(output_dir, exist_ok=True)
    

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob(os.path.join(input_dir, ext.upper())))
    
    print(f"\nProcessing {len(image_paths)} images with {method_name}...")
    
    for img_path in tqdm(image_paths, desc=f"{method_name}"):
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            

            result = method_func(img_np)
            
            # Save
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_corrected{ext}")
            Image.fromarray(result).save(output_path)
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
    
    print(f"✓ Completed! Results saved to {output_dir}/")


if __name__ == "__main__":
    print("="*70)
    print("TRADITIONAL OVEREXPOSURE CORRECTION ALGORITHMS")
    print("="*70)
    print("\nAvailable methods:")
    print("  1. CLAHE - Fast, good for moderate overexposure")
    print("  2. Adaptive Gamma - Simple, good for global overexposure")
    print("  3. Retinex - Best quality, slower")
    print("  4. Tone Mapping - HDR-style, good for extreme overexposure")
    print("  5. Hybrid - Combines multiple methods (RECOMMENDED)")
    print("  6. Unsharp + Exposure - Good detail recovery")

    print("\n" + "="*70)
    print("EXAMPLE 1: Compare all methods on one image")
    print("="*70)
    
    test_image = "C:/Users/allan/Downloads/glare_dataset/processed/overexposed/sample_overexposed.jpg"

    print("\n" + "="*70)
    print("EXAMPLE 2: Process directory with Hybrid method")
    print("="*70)
    
    input_directory = "C:/Users/allan/Downloads/glare_dataset/chosen_processed/overexposed"
    output_directory = "C:/Users/allan/Downloads/glare_dataset/corrected"

    process_directory_with_method(
        input_directory, 
        output_directory, 
        method5_hybrid,  
        "Hybrid Method"
    )
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)