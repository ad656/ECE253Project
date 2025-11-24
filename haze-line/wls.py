import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

def wls_optimization(in_img, data_weight, guidance, lam=0.05):
    """
    Python equivalent of the MATLAB wls_optimization.m

    in_img      : (h, w) input image (double)
    data_weight : (h, w) data term weights
    guidance    : (h, w, 3) or (h, w) guidance image
    lam         : regularization parameter (lambda in MATLAB)
    """
    small_num = 1e-5

    in_img = np.asarray(in_img, dtype=np.float64)
    data_weight = np.asarray(data_weight, dtype=np.float64)
    guidance = np.asarray(guidance, dtype=np.float64)

    # --- sizes ---
    h, w = guidance.shape[:2]
    k = h * w

    # --- convert guidance to grayscale (rgb2gray) ---
    if guidance.ndim == 3 and guidance.shape[2] == 3:
        # standard luminance weights
        R = guidance[..., 0]
        G = guidance[..., 1]
        B = guidance[..., 2]
        guidance_gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    else:
        guidance_gray = guidance

    # --- Compute affinities between adjacent pixels ---

    # vertical gradients (dy), size: (h-1, w)
    dy = np.diff(guidance_gray, n=1, axis=0)
    dy = -lam / (dy**2 + small_num)
    dy = np.pad(dy, ((0, 1), (0, 0)), mode='constant')  # pad bottom
    dy_flat = dy.ravel(order='F')  # column-major like MATLAB dy(:)

    # horizontal gradients (dx), size: (h, w-1)
    dx = np.diff(guidance_gray, n=1, axis=1)
    dx = -lam / (dx**2 + small_num)
    dx = np.pad(dx, ((0, 0), (0, 1)), mode='constant')  # pad right
    dx_flat = dx.ravel(order='F')  # column-major like MATLAB dx(:)

    # --- Construct 5-point Laplacian ---

    # B is k-by-2 in MATLAB; SciPy spdiags expects (ndiags, k)
    B = np.vstack((dx_flat, dy_flat))     # shape (2, k)
    diags = np.array([-h, -1], dtype=int)

    # tmp = spdiags(B, d, k, k);
    tmp = spdiags(B, diags, k, k)

    # east / west / south / north contributions
    ea = dx_flat
    we = np.pad(dx_flat, (h, 0), mode='constant')[:k]
    so = dy_flat
    no = np.pad(dy_flat, (1, 0), mode='constant')[:-1]

    D = -(ea + we + so + no)

    Asmoothness = tmp + tmp.T + spdiags(D, 0, k, k)

    # --- Normalize data_weight ---

    dw = data_weight.copy()
    dw -= dw.min()
    dw /= (dw.max() + small_num)

    # --- Boundary condition on top row ---
    # data_weight(1,:) < 0.6 in MATLAB (1-based) -> row 0 in Python
    reliability_mask = dw[0, :] < 0.6

    # in_row1 = min(in,[],1); -> min over rows
    in_row1 = in_img.min(axis=0)

    dw[0, reliability_mask] = 0.8
    in_img[0, reliability_mask] = in_row1[reliability_mask]

    # --- Build data term matrix and RHS ---

    dw_flat = dw.ravel(order='F')        # k-vector
    in_flat = in_img.ravel(order='F')    # k-vector

    Adata = spdiags(dw_flat, 0, k, k)

    A = (Adata + Asmoothness).tocsr()
    # b = Adata * in(:) -> elementwise, since Adata is diagonal
    b = dw_flat * in_flat

    # --- Solve linear system A * out = b ---
    x = spsolve(A, b)
    out = x.reshape((h, w), order='F')

    return out
