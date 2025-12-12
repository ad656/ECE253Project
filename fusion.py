import numpy as np
import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from DCP.dehaze import DCP
from non_local_dehazing.haze_line import haze_line
import Simplified_trad_method as stm


def exposure_fix(img_bgr, mode='hybrid',
                 base_reduction=0.12, extra_reduction=0.20,
                 severity_start=220.0, severity_range=35.0,
                 blend_factor=0.85, gamma_exponent=1.2, reduction_cap=0.75,
                 smooth_sigma=2.0, feather_sigma=5.0, min_glare_size=0.005):

    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr * 255.0, 0, 255).astype('uint8')
    arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype('float32')

    Y, S = stm.compute_Y_S(arr)
    S_TH = np.exp(2.4 * (Y - 1.0))
    over_mask = (Y > 0.95) & (S < S_TH)
    max_chan = np.maximum.reduce([arr[..., 0], arr[..., 1], arr[..., 2]])
    over_mask = over_mask | (max_chan >= 250) | (Y > 0.99)

    total_pixels = arr.shape[0] * arr.shape[1]
    bright_mask = max_chan >= 245
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bright_opened = cv2.morphologyEx((bright_mask.astype('uint8') * 255),
                                     cv2.MORPH_OPEN, kernel, iterations=1)
    bright_labeled, ncomp = stm.ndimage.label((bright_opened > 0))

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

    glare_mask, white_mask = stm.classify_glare_vs_white(arr, over_mask, min_glare_size=min_glare_size)

    if np.any(glare_mask):
        glare_mask_uint8 = (glare_mask.astype('uint8') * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask_dilated = cv2.dilate(glare_mask_uint8, kernel, iterations=2)
        glare_mask_soft = cv2.GaussianBlur(glare_mask_dilated, (15, 15), sigmaX=5.0, sigmaY=5.0)
        glare_mask_soft_norm = glare_mask_soft.astype('float32') / 255.0

        if mode == 'darken':
            arr_uint8 = np.clip(arr, 0, 255).astype('uint8')
            final = arr_uint8.copy().astype('float32')
            clahe_img = stm.apply_clahe(arr_uint8)
            final = stm.blend_enhancement(final, clahe_img, glare_mask_soft_norm, blend_factor=blend_factor)

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
            S_chan = hsv[..., 1]
            V = hsv[..., 2]

            V_new = V * (1.0 - reduction)
            gamma_mask = V_new > 235
            if np.any(gamma_mask):
                V_new[gamma_mask] = 255.0 * np.power((V_new[gamma_mask] / 255.0), float(gamma_exponent))

            S_enhanced = np.clip(S_chan * (1.0 + reduction * 0.3), 0, 255)
            hsv_darkened = np.stack([H, S_enhanced, V_new], axis=-1)
            final_rgb = cv2.cvtColor(np.clip(hsv_darkened, 0, 255).astype('uint8'),
                                     cv2.COLOR_HSV2RGB).astype('float32')

            min_allowed = 0.85 * orig_rgb
            final_rgb = np.maximum(final_rgb, min_allowed)
            final = final_rgb
        else:
            white_clip, colored_glare = stm.classify_glare_type(arr, glare_mask_soft_norm > 0.5)
            mode_map = np.zeros(arr.shape[:2], dtype='uint8')
            mode_map[colored_glare] = 1
            mode_map[white_clip] = 2
            final = stm.process_overexposed_hybrid(arr, glare_mask_soft_norm > 0.5, mode_map)
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
        final = arr.copy()

    final_rgb_0_1 = np.clip(final, 0, 255).astype('float32') / 255.0
    return final_rgb_0_1


def _pad_to_pow2(img, levels):
    h, w = img.shape[:2]
    m = 2**levels
    nh = (h + m - 1) // m * m
    nw = (w + m - 1) // m * m
    if nh == h and nw == w:
        return img
    return cv2.copyMakeBorder(img, 0, nh-h, 0, nw-w, cv2.BORDER_REFLECT_101)

def haar_dwt2_single(gray, levels=1):
    coeffs = []
    cur = gray.astype(np.float32)
    for _ in range(levels):
        L = (cur[:, 0::2] + cur[:, 1::2]) * 0.5
        H = (cur[:, 0::2] - cur[:, 1::2]) * 0.5
        A = (L[0::2, :] + L[1::2, :]) * 0.5
        V = (L[0::2, :] - L[1::2, :]) * 0.5
        H2 = (H[0::2, :] + H[1::2, :]) * 0.5
        D = (H[0::2, :] - H[1::2, :]) * 0.5
        coeffs.append({'A': A, 'H': H2, 'V': V, 'D': D})
        cur = A
    return coeffs

def haar_idwt2_single(coeffs):
    curA = coeffs[-1]['A']
    for l in range(len(coeffs)-1, -1, -1):
        A, H, V, D = coeffs[l]['A'], coeffs[l]['H'], coeffs[l]['V'], coeffs[l]['D']
        L0 = (A + V)
        L1 = (A - V)
        H0 = (H + D)
        H1 = (H - D)
        upL = np.empty((L0.shape[0]*2, L0.shape[1]), np.float32)
        upH = np.empty_like(upL)
        upL[0::2, :], upL[1::2, :] = L0, L1
        upH[0::2, :], upH[1::2, :] = H0, H1
        out = np.empty((upL.shape[0], upL.shape[1]*2), np.float32)
        out[:, 0::2] = (upL + upH)
        out[:, 1::2] = (upL - upH)
        curA = out
        coeffs[l]['A'] = out
    return curA

def fuse_luminance_wavelet_dcp_exp(img_dcp, img_exp, levels=5, base_chroma='exp'):
    def to_uint8(bgr):
        if bgr.dtype == np.uint8:
            return bgr
        return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)

    dcp_u8 = to_uint8(img_dcp)
    exp_u8 = to_uint8(img_exp)

    lab_dcp = cv2.cvtColor(dcp_u8, cv2.COLOR_RGB2LAB)
    lab_exp = cv2.cvtColor(exp_u8, cv2.COLOR_RGB2LAB)
    L_dcp, a_dcp, b_dcp = cv2.split(lab_dcp)
    L_exp, a_exp, b_exp = cv2.split(lab_exp)

    L_dcp_f = _pad_to_pow2(L_dcp.astype(np.float32), levels)
    L_exp_f = _pad_to_pow2(L_exp.astype(np.float32), levels)

    coeff_dcp = haar_dwt2_single(L_dcp_f, levels)
    coeff_exp = haar_dwt2_single(L_exp_f, levels)

    coeff_fused = []
    for l in range(levels):
        A1, H1, V1, D1 = (coeff_dcp[l]['A'], coeff_dcp[l]['H'],
                          coeff_dcp[l]['V'], coeff_dcp[l]['D'])
        A2, H2, V2, D2 = (coeff_exp[l]['A'], coeff_exp[l]['H'],
                          coeff_exp[l]['V'], coeff_exp[l]['D'])
        Af = 0.5 * (A1 + A2)
        Hf = np.where(np.abs(H1) >= np.abs(H2), H1, H2)
        Vf = np.where(np.abs(V1) >= np.abs(V2), V1, V2)
        Df = np.where(np.abs(D1) >= np.abs(D2), D1, D2)
        coeff_fused.append({'A': Af, 'H': Hf, 'V': Vf, 'D': Df})

    L_fused = haar_idwt2_single(coeff_fused)
    H, W = L_dcp.shape
    L_fused = L_fused[:H, :W]

    L_fused_u8 = np.clip(L_fused, 0, 255).astype(np.uint8)
    if base_chroma == 'exp':
        a_base, b_base = a_exp, b_exp
    else:
        a_base, b_base = a_dcp, b_dcp

    lab_fused = cv2.merge((L_fused_u8, a_base, b_base))
    fused_rgb = cv2.cvtColor(lab_fused, cv2.COLOR_LAB2RGB)
    return fused_rgb.astype(np.float32) / 255.0


def resize(image, pixel_num = 1024*512):
    h, w = image.shape[:2]
    scale = (pixel_num / float(h * w)) ** 0.5
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def save_rgb_float01_as_bgr_uint8(path, rgb_float01):
    rgb_clipped = np.clip(rgb_float01 * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb_clipped, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory with input images')
    parser.add_argument('--output_dir', required=True, help='Directory to save augmented images')
    parser.add_argument('--fusion', type=str, choices=['yes', 'no'], required=True, help='Save fusion or resize only')
    parser.add_argument('--dehaze', type=str, choices=['dcp', 'hazeline'], default='dcp', help='Dehazing method')
    parser.add_argument('--method', type=str, choices=['hybrid', 'retinex', 'gamma', 'tone'], default='hybrid', 
                       help='Overexposure correction method to use')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp'}
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    if not files:
        print('No images found in', input_dir)
        return

    for p in files:
        out_path = output_dir / (p.stem + ".webp")
        src = cv2.imread(str(p))
        resize_src = resize(src)

        if args.fusion == 'no':
            cv2.imwrite(str(out_path), resize_src)
            continue

        try:
            # Dehaze
            if args.dehaze == 'dcp':
                dehaze = DCP(resize_src)
                dehaze = cv2.cvtColor(dehaze, cv2.COLOR_BGR2RGB)
            elif args.dehaze == 'hazeline':
                dehaze, _ = haze_line(resize_src)

            # Fix overexposure using the selected method
            if args.method == 'hybrid':
                # Use method5_hybrid directly for better results
                exposure_fixed = stm.method5_hybrid(resize_src)
                exposure_fixed = exposure_fixed.astype(np.float32) / 255.0
            else:
                # Use the original exposure_fix function
                exposure_fixed = exposure_fix(resize_src, mode='hybrid')

            # Wavelet fusion
            fused = fuse_luminance_wavelet_dcp_exp(dehaze, exposure_fixed, levels=5, base_chroma='exp')

            save_rgb_float01_as_bgr_uint8(out_path, fused)

            # Optional visualization
            plt.subplot(2, 2, 1)
            plt.title("Original")
            plt.imshow(cv2.cvtColor(resize_src, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.title("Dehazed")
            plt.imshow(dehaze)
            plt.axis("off")
            plt.subplot(2, 2, 3)
            plt.title("De-overexposure")
            plt.imshow(exposure_fixed)
            plt.axis("off")
            plt.subplot(2, 2, 4)
            plt.title("Fused result")
            plt.imshow(fused)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

            print('Processed', p.name)
        except Exception as e:
            print('Failed', p.name, 'error:', e)

if __name__ == '__main__':
    main()