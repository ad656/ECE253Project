import cv2
import numpy as np
from PIL import Image
from glob import glob
import os
from tqdm import tqdm


# ========================================
# METHOD 1: ADAPTIVE HISTOGRAM EQUALIZATION (CLAHE)
# ========================================
def method1_clahe(image):
    """
    Contrast Limited Adaptive Histogram Equalization
    - Fast and simple
    - Good for moderate overexposure
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge and convert back
    lab_clahe = cv2.merge([l_clahe, a, b])
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return result


# ========================================
# METHOD 2: GAMMA CORRECTION WITH AUTO-ADJUSTMENT
# ========================================
def method2_adaptive_gamma(image):
    """
    Adaptive gamma correction based on image statistics
    - Adjusts gamma based on average brightness
    - Good for global overexposure
    """
    # Calculate average luminance
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    avg_luminance = np.mean(gray) / 255.0
    
    # More aggressive gamma calculation
    # If image is bright (overexposed), use higher gamma to darken more
    if avg_luminance > 0.5:
        gamma = 1.5 + (avg_luminance - 0.5) * 3.0  # Range: 1.5 to 3.0 (much more aggressive)
    else:
        gamma = 1.0
    
    # Apply gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    result = cv2.LUT(image, table)
    
    return result


# ========================================
# METHOD 3: RETINEX ALGORITHM (Multi-Scale)
# ========================================
def method3_retinex(image):
    """
    Multi-Scale Retinex with Color Restoration (MSRCR)
    - Separates illumination from reflectance
    - Best quality for overexposure
    - Slower but most effective
    """
    img_float = image.astype(np.float32) / 255.0
    img_float = np.maximum(img_float, 0.001)  # Avoid log(0)
    
    # Multi-scale Gaussian blur (simulates different surround sizes)
    scales = [15, 80, 250]
    retinex = np.zeros_like(img_float)
    
    for scale in scales:
        # Gaussian blur
        blurred = cv2.GaussianBlur(img_float, (0, 0), scale)
        blurred = np.maximum(blurred, 0.001)
        
        # Log domain division (removes illumination)
        retinex += np.log10(img_float) - np.log10(blurred)
    
    retinex = retinex / len(scales)

    alpha = 125.0
    beta = 46.0
    

    intensity = np.mean(img_float, axis=2, keepdims=True)
    intensity = np.maximum(intensity, 0.001)
    log_intensity = np.log10(intensity)
    
    # Color restoration factor
    for c in range(3):
        retinex[:, :, c] = beta * (np.log10(alpha * img_float[:, :, c]) - log_intensity[:, :, 0])
    
    # Normalize to 0-255
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))
    result = (retinex * 255).astype(np.uint8)
    
    return result



def method4_tone_mapping(image):
    """
    Reinhard tone mapping operator
    - Compresses bright values while preserving darker areas
    - Inspired by HDR photography
    """
    img_float = image.astype(np.float32) / 255.0
    
    # Calculate luminance
    luminance = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
    
    # Calculate log-average luminance
    epsilon = 1e-5
    log_avg_lum = np.exp(np.mean(np.log(luminance + epsilon)))
    

    key = 0.10  
    scaled_lum = (key / log_avg_lum) * luminance
    

    white_point = 1.0  
    tone_mapped_lum = (scaled_lum * (1 + scaled_lum / (white_point ** 2))) / (1 + scaled_lum)
    
    # Apply to each channel
    result = np.zeros_like(img_float)
    for c in range(3):
        result[:, :, c] = img_float[:, :, c] * (tone_mapped_lum / (luminance + epsilon))
    
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)
    
    return result



def method5_hybrid(image):
    """
    Combines multiple techniques for best results:
    1. Aggressive global darkening first
    2. Adaptive gamma for tonal correction
    3. Retinex for local details
    4. CLAHE for final enhancement
    """
 
    img_float = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    avg_brightness = np.mean(gray) / 255.0
    
    if avg_brightness > 0.6:

        darken_factor = 0.5 + (1.0 - avg_brightness) * 0.5 
        result = (img_float * darken_factor * 255).astype(np.uint8)
    else:
        result = image
    

    result = method2_adaptive_gamma(result)
    

    retinex_result = method3_retinex(result)
    result = cv2.addWeighted(result, 0.5, retinex_result, 0.5, 0)
    

    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.merge([l, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    
    return result



def method6_unsharp_exposure(image):
    """
    Combines exposure correction with detail enhancement
    - Good for recovering detail in bright areas
    """

    img_float = image.astype(np.float32) / 255.0
    

    luminance = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
    overexposed_mask = np.clip((luminance - 0.5) / 0.5, 0, 1)  
    
  
    darkening_factor = 0.5  
    corrected = img_float * (1 - overexposed_mask[:, :, np.newaxis] * (1 - darkening_factor))
    

    avg_luminance = np.mean(luminance)
    if avg_luminance > 0.6:
        global_darken = 0.7 + (1.0 - avg_luminance) * 0.5
        corrected = corrected * global_darken
    

    gaussian = cv2.GaussianBlur(corrected, (0, 0), 2.0)
    unsharp = cv2.addWeighted(corrected, 1.5, gaussian, -0.5, 0)
    
    # Combine
    result = np.clip(unsharp, 0, 1)
    result = (result * 255).astype(np.uint8)
    
    return result



def compare_all_methods(image_path, output_dir="comparison_results"):
    """
    Apply all methods and save side-by-side comparison
    """
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    
    methods = {
        "1_Original": img_np,
        "2_CLAHE": method1_clahe(img_np),
        "3_Adaptive_Gamma": method2_adaptive_gamma(img_np),
        "4_Retinex": method3_retinex(img_np),
        "5_Tone_Mapping": method4_tone_mapping(img_np),
        "6_Hybrid": method5_hybrid(img_np),
        "7_Unsharp_Exposure": method6_unsharp_exposure(img_np)
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
    
    # Uncomment to process:
    process_directory_with_method(
        input_directory, 
        output_directory, 
        method5_hybrid,  
        "Hybrid Method"
    )
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)