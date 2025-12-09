import numpy as np
import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from DCP.dehaze import DCP
from non_local_dehazing.haze_line import haze_line
from Simplified_trad_method import exposure_fix

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
    """
    input dehaze and de-overexposure images (RGB) and output RGB fused image
    """
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

    # clamp and cast to uint8
    L_fused_u8 = np.clip(L_fused, 0, 255).astype(np.uint8)

    # --- choose base chroma (a,b) channels ---
    if base_chroma == 'exp':
        a_base, b_base = a_exp, b_exp
    else:
        a_base, b_base = a_dcp, b_dcp

    lab_fused = cv2.merge((L_fused_u8, a_base, b_base))
    fused_rgb = cv2.cvtColor(lab_fused, cv2.COLOR_LAB2RGB)

    fused_rgb = fused_rgb.astype(np.float32) / 255.0
    return fused_rgb

def resize(image, pixel_num = 1024*512):
    h, w = image.shape[:2]
    scale = (pixel_num / float(h * w)) ** 0.5
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory with input images')
    parser.add_argument('--output_dir', required=True, help='Directory to save augmented images')
    parser.add_argument('--dehaze', type=str, choices=['dcp', 'hazeline'], help='method to dehaze', default='dcp')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.JPG', '.jpeg', '.png', '.tif', '.bmp', '.webp'}

    files = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    if not files:
        print('No images found in', input_dir)

    for p in files:
        src = cv2.imread(str(p))
        resize_src = resize(src)
        try:
            if args.dehaze == 'dcp':
                dehaze = DCP(resize_src)
                dehaze = cv2.cvtColor(dehaze, cv2.COLOR_BGR2RGB)
            elif args.dehaze == 'hazeline':
                dehaze, _ = haze_line(resize_src)

            exposure_fixed = exposure_fix(resize_src)
            fused = fuse_luminance_wavelet_dcp_exp(
                dehaze,
                exposure_fixed,
                levels=5,
                base_chroma='exp'
            )

            # result display
            plt.subplot(2, 2, 1)   
            plt.title("Original")
            plt.imshow(exposure_fixed) 
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.title("Dehazed")
            plt.imshow(dehaze)
            plt.axis("off")
            plt.tight_layout()
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