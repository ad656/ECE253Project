import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from .wls import wls_optimization

HERE = Path(__file__).resolve().parent
def im2double(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    else:
        # assume already float in [0,1]
        return img.astype(np.float64)

def im2uint8(img):
    img = np.asarray(img, dtype=np.float64)
    img = np.clip(img * 255.0, 0, 255)
    return img.astype(np.uint8)

def adjust_global_contrast(img, adj_percent=(0.005, 0.995)):
    """
    Rough equivalent of global linear contrast stretch:
    clip low/high percentiles and map to [0,1].
    """
    img = np.asarray(img, dtype=np.float64)
    low_p, high_p = adj_percent

    # compute percentiles per channel
    if img.ndim == 3 and img.shape[2] == 3:
        out = np.zeros_like(img)
        for c in range(3):
            ch = img[..., c]
            lo = np.quantile(ch, low_p)
            hi = np.quantile(ch, high_p)
            if hi <= lo:
                out[..., c] = ch
            else:
                out[..., c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        return out
    else:
        lo = np.quantile(img, low_p)
        hi = np.quantile(img, high_p)
        if hi <= lo:
            return img
        return np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    
def non_local_dehazing(img_hazy, air_light, gamma=1.0,
                       n_points=1000,
                       tessellation_prefix='TR',
                       lambda_reg=0.1,
                       trans_min=0.1):

    img_hazy = np.asarray(img_hazy)
    h, w, n_colors = img_hazy.shape

    # --- Validate input ---
    if n_colors != 3:
        raise ValueError(
            f"Non-Local Dehazing requires an RGB image, got {n_colors} channels"
        )

    air_light = np.asarray(air_light, dtype=np.float64).reshape(3,)
    if air_light.size != 3:
        raise ValueError("Dehazing on sphere requires an RGB airlight (3 values)")

    if gamma is None:
        gamma = 1.0

    # --- Radiometric correction ---
    img_hazy = im2double(img_hazy)
    img_hazy_corrected = img_hazy ** gamma

    # --- Find Haze-lines ---
    # Translate coordinates to be air_light-centric
    # dist_from_airlight(y,x,c) = img_hazy_corrected(y,x,c) - air_light(c)
    dist_from_airlight = img_hazy_corrected - air_light.reshape(1, 1, 3)

    # radius (Eq. (5))
    radius = np.sqrt(np.sum(dist_from_airlight ** 2, axis=2))  # (h, w)

    # Cluster pixels to haze-lines on the unit sphere
    dist_unit_radius = dist_from_airlight.reshape(-1, n_colors)  # (h*w, 3)
    dist_norm = np.sqrt(np.sum(dist_unit_radius ** 2, axis=1, keepdims=True))
    dist_norm[dist_norm == 0] = 1e-12  # avoid division by zero
    dist_unit_radius = dist_unit_radius / dist_norm  # normalize to unit sphere

    # Load pre-calculated uniform tessellation of the unit-sphere
    HERE = Path(__file__).resolve().parent
    tess_name = f"{tessellation_prefix}{n_points}.txt"
    tess_path = HERE / tess_name 
    points = np.loadtxt(tess_path)

    # KD-tree search for nearest tessellation point for each pixel
    tree = cKDTree(points)
    _, ind = tree.query(dist_unit_radius)

    # --- Estimating Initial Transmission ---
    radius_flat = radius.ravel()
    K = np.zeros(n_points, dtype=np.float64)
    np.maximum.at(K, ind, radius_flat)

    radius_new = K[ind].reshape(h, w)
    eps = 1e-12
    transmission_estimation = radius / (radius_new + eps)

    # Limit transmission to [trans_min, 1]
    transmission_estimation = np.clip(transmission_estimation, trans_min, 1.0)

    # --- Regularization ---
    air_light_reshaped = air_light.reshape(1, 1, 3)
    ratio = img_hazy_corrected / (air_light_reshaped + eps)
    trans_lower_bound = 1.0 - np.min(ratio, axis=2)
    transmission_estimation = np.maximum(transmission_estimation, trans_lower_bound)
    bin_count = np.zeros(n_points, dtype=np.int32)
    np.add.at(bin_count, ind, 1)
    bin_count_map = bin_count[ind].reshape(h, w)

    bin_eval_fun = lambda x: np.minimum(1.0, x / 50.0)
    sum_r = np.zeros(n_points, dtype=np.float64)
    sum_r2 = np.zeros(n_points, dtype=np.float64)
    np.add.at(sum_r, ind, radius_flat)
    np.add.at(sum_r2, ind, radius_flat ** 2)

    # avoid division by zero
    cnt = bin_count.astype(np.float64)
    mask_nonzero = cnt > 0
    mean_r = np.zeros_like(sum_r)
    var_r = np.zeros_like(sum_r)

    mean_r[mask_nonzero] = sum_r[mask_nonzero] / cnt[mask_nonzero]
    var_r[mask_nonzero] = (sum_r2[mask_nonzero] / cnt[mask_nonzero]) - mean_r[mask_nonzero] ** 2
    var_r[var_r < 0] = 0 

    K_std = np.sqrt(var_r)
    radius_std = K_std[ind].reshape(h, w)

    if radius_std.max() > 0:
        radius_std_norm = radius_std / (radius_std.max() + eps)
    else:
        radius_std_norm = radius_std

    radius_eval_fun = lambda r: np.minimum(1.0, 3.0 * np.maximum(0.001, r - 0.1))
    radius_reliability = radius_eval_fun(radius_std_norm)

    data_term_weight = bin_eval_fun(bin_count_map) * radius_reliability

    # Solve optimization problem (Eq. (15)) via weighted least-squares
    transmission = wls_optimization(
        transmission_estimation,
        data_term_weight,
        img_hazy, 
        lambda_reg
    )

    img_dehazed = np.zeros_like(img_hazy_corrected)
    leave_haze = 1.06  

    # broadcast transmission & air_light correctly
    transmission_clamped = np.maximum(transmission, trans_min)

    for c in range(3):
        img_dehazed[..., c] = (
            img_hazy_corrected[..., c]
            - (1.0 - leave_haze * transmission_clamped) * air_light[c]
        ) / transmission_clamped

    # Clamp to [0,1]
    img_dehazed = np.clip(img_dehazed, 0.0, 1.0)

    # inverse gamma correction
    img_dehazed = img_dehazed ** (1.0 / gamma)

    # Global linear contrast stretch
    img_dehazed = adjust_global_contrast(img_dehazed, adj_percent=(0.005, 0.995))

    # Back to uint8
    img_dehazed_uint8 = im2uint8(img_dehazed)

    return img_dehazed_uint8, transmission