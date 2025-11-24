import numpy as np
import cv2
import matplotlib.pyplot as plt
from non_local_dehaze import non_local_dehazing
from scipy.io import loadmat

import numpy as np

def generate_Avals(Avals1, Avals2):
    Avals1 = np.asarray(Avals1).reshape(-1)
    Avals2 = np.asarray(Avals2).reshape(-1)

    # meshgrid with 'ij' gives same ordering as the kron construction
    A1, A2 = np.meshgrid(Avals1, Avals2, indexing='ij')  # each is (len1, len2)
    Aall = np.column_stack([A1.ravel(), A2.ravel()])     # (len1*len2, 2)
    return Aall

def vote_2D(points, points_weight, directions_all, Aall, thres):
    points = np.asarray(points, dtype=float)           # (P, 2)
    points_weight = np.asarray(points_weight, float)   # (P,)
    directions_all = np.asarray(directions_all, float) # (D, 2)
    Aall = np.asarray(Aall, float)                     # (NA, 2)

    n_directions = directions_all.shape[0]
    n_A = Aall.shape[0]
    n_points = points.shape[0]

    # accumulator_votes_idx: (n_A, n_points, n_directions)
    accumulator_votes_idx = np.zeros((n_A, n_points, n_directions), dtype=bool)

    # --- First loop: mark which (A, point, direction) combinations are valid ---
    for i_point in range(n_points):
        px, py = points[i_point]

        # idx_to_use: A candidates larger than this point in both channels
        mask_to_use = (Aall[:, 0] > px) & (Aall[:, 1] > py)
        idx_to_use = np.where(mask_to_use)[0]
        if idx_to_use.size == 0:
            continue

        Ax = Aall[idx_to_use, 0]
        Ay = Aall[idx_to_use, 1]

        # distance between this point and the candidate A
        dx = Ax - px
        dy = Ay - py
        dist1 = np.sqrt(dx * dx + dy * dy)
        dist1 = dist1 / np.sqrt(2.0) + 1.0

        for i_direction in range(n_directions):
            vx, vy = directions_all[i_direction]  # (sinθ, cosθ)

            # distance to line defined by point & direction
            dist = (
                -px * vy +
                py * vx +
                Ax * vy -
                Ay * vx
            )

            idx = np.abs(dist) < 2.0 * thres * dist1
            if not np.any(idx):
                continue

            idx_full = idx_to_use[idx]
            accumulator_votes_idx[idx_full, i_point, i_direction] = True

    # --- keep only haze-lines supported by >= 2 points ---
    # sum over points axis (axis=1): shape (n_A, n_directions)
    accumulator_votes_idx2 = (accumulator_votes_idx.sum(axis=1) >= 2)
    # broadcast to (n_A, n_points, n_directions)
    accumulator_votes_idx &= accumulator_votes_idx2[:, np.newaxis, :]

    # --- Accumulate unique votes per A candidate ---
    accumulator_unique = np.zeros(n_A, dtype=float)

    for iA in range(n_A):
        Ax, Ay = Aall[iA]

        # points that lie "below" this A in both channels
        mask_points = (Ax > points[:, 0]) & (Ay > points[:, 1])
        idx_to_use = np.where(mask_points)[0]
        if idx_to_use.size == 0:
            continue

        px = points[idx_to_use, 0]
        py = points[idx_to_use, 1]

        points_dist = np.sqrt((Ax - px) ** 2 + (Ay - py) ** 2)

        # distance-weighted cluster weights
        # points_weight_dist = points_weight(idx_to_use).*(5.*exp(-points_dist)+1)
        weights = points_weight[idx_to_use] * (5.0 * np.exp(-points_dist) + 1.0)

        # only count points that actually voted for this A (any direction)
        has_support = accumulator_votes_idx[iA, idx_to_use, :].any(axis=1)

        accumulator_unique[iA] = weights[has_support].sum()

    Aestimate_idx = np.argmax(accumulator_unique)
    Aout = Aall[Aestimate_idx, :]
    Avote2 = accumulator_unique

    return Aout, Avote2

def estimate_airlight(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    height, width, _ = image.shape
    thres = 0.01
    spacing = 0.02
    n_color = 1000
    n_angle = 40

    # search range for air light
    Amin = np.array([0, 0.05, 0.1])
    Amax = np.array([1, 1, 1])

    """ Quantify image into n_color """
    data = image.reshape(-1, 3).astype(np.float32)

    # k-means parameters (you can tweak these)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        data,
        n_color,
        None,
        criteria,
        1,
        cv2.KMEANS_PP_CENTERS
    )

    # img_ind: 0..N-1, shape (h, w)
    img_ind = labels.reshape(height, width)
    points = centers  # shape (N, 3)

    # --- Remove empty clusters ---
    idx_in_use = np.unique(img_ind)                      # used cluster indices (0..N-1)
    num_points = points.shape[0]
    all_idx = np.arange(num_points)
    idx_to_remove = np.setdiff1d(all_idx, idx_in_use)    # clusters not used

    if idx_to_remove.size > 0:
        points = np.delete(points, idx_to_remove, axis=0)

    # --- Build sequential index image (1..M, like MATLAB) ---
    img_ind_sequential = np.zeros((height, width), dtype=np.int32)
    idx_in_use_sorted = np.sort(idx_in_use)

    # kk goes 1..M; old_idx is original 0-based cluster index
    for kk, old_idx in enumerate(idx_in_use_sorted, start=1):
        img_ind_sequential[img_ind == old_idx] = kk

    # Now min(img_ind_sequential) = 1 and max = M
    # and index k corresponds to row k-1 in `points` (Python is 0-based).

    # --- Count occurrences of each index = cluster weights ---
    M = points.shape[0]
    flat_seq = img_ind_sequential.ravel()
    points_weight = np.bincount(flat_seq - 1, minlength=M).astype(np.float64)
    points_weight /= (height * width)

    # d = loadmat('debug_airlight.mat')
    # points = d['points']          # (M,3)
    # points_weight = d['points_weight'].astype(np.float64).ravel()

    """ Show quantify image """
    # img_quant = points[img_ind]
    # if img_quant.max() > 1.5:
    #     img_quant_disp = np.clip(img_quant, 0, 255).astype(np.uint8)
    # else:
    #     img_quant_disp = img_quant 
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Original")
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
    # plt.axis("off")
    # plt.subplot(1, 2, 2)
    # plt.title("Quantized")
    # plt.imshow(cv2.cvtColor(img_quant_disp, cv2.COLOR_BGR2RGB))
    # plt.axis("off")

    # plt.tight_layout()
    # plt.show()


    """ spheriacal coordinate """
    angle_list = np.linspace(0.0, np.pi, n_angle).reshape(-1, 1)   # shape (K, 1)

    # Use angle_list[0:-1] since angle_list[-1] == pi, which is the same line
    # in 2D as angle_list[0] == 0
    directions_all = np.column_stack([
        np.sin(angle_list[:-1, 0]),
        np.cos(angle_list[:-1, 0])
    ])   # shape (K-1, 2)

    # --- Air-light candidates in each color channel ---
    # Amin, Amax are length-3 arrays: [Ar_min, Ag_min, Ab_min], etc.
    # Use np.arange with a tiny epsilon to mimic MATLAB's inclusive colon behavior
    eps = 1e-9

    ArangeR = np.arange(Amin[0], Amax[0] + eps, spacing)
    ArangeG = np.arange(Amin[1], Amax[1] + eps, spacing)
    ArangeB = np.arange(Amin[2], Amax[2] + eps, spacing)

    Aall = generate_Avals(ArangeR, ArangeG)
    _, AvoteRG = vote_2D(points[:, 0:2], points_weight, directions_all, Aall, thres)

    # GB
    Aall = generate_Avals(ArangeG, ArangeB)
    _, AvoteGB = vote_2D(points[:, 1:3], points_weight, directions_all, Aall, thres)

    # RB
    Aall = generate_Avals(ArangeR, ArangeB)
    _, AvoteRB = vote_2D(points[:, [0, 2]], points_weight, directions_all, Aall, thres)

    # --- Find most probable airlight from marginal probabilities (2D arrays) ---
    max_val = max(AvoteRB.max(), AvoteRG.max(), AvoteGB.max())
    if not np.isfinite(max_val) or max_val <= 0:
        # Fallback: no meaningful votes, skip normalization
        AvoteRG2, AvoteGB2, AvoteRB2 = AvoteRG, AvoteGB, AvoteRB
    else:
        AvoteRG2 = AvoteRG / max_val
        AvoteGB2 = AvoteGB / max_val
        AvoteRB2 = AvoteRB / max_val

    lenR = len(ArangeR)
    lenG = len(ArangeG)
    lenB = len(ArangeB)

    # AvoteRG2 corresponds to (R,G) pairs in order:
    #   m = r * lenG + g
    # so reshape to (lenR, lenG)
    A11_2d = AvoteRG2.reshape((lenR, lenG))          # (lenR, lenG)
    A11 = np.repeat(A11_2d[:, :, np.newaxis], lenB, axis=2)   # (lenR, lenG, lenB)

    # AvoteRB2 corresponds to (R,B) pairs: m = r * lenB + b
    tmp = AvoteRB2.reshape((lenR, lenB))             # (lenR, lenB)
    A22 = np.repeat(tmp[:, np.newaxis, :], lenG, axis=1)      # (lenR, lenG, lenB)

    # AvoteGB2 corresponds to (G,B) pairs: m = g * lenB + b
    tmp2 = AvoteGB2.reshape((lenG, lenB))            # (lenG, lenB)
    A33 = np.repeat(tmp2[np.newaxis, :, :], lenR, axis=0)     # (lenR, lenG, lenB)

    # Combine
    AvoteAll = A11 * A22 * A33

    # Find global maximum
    idx_flat = np.argmax(AvoteAll)
    idx_r, idx_g, idx_b = np.unravel_index(idx_flat, AvoteAll.shape)

    Aout = np.array([
        ArangeR[idx_r],
        ArangeG[idx_g],
        ArangeB[idx_b]
    ])
    return Aout

if __name__ == "__main__":
    image = cv2.imread("./images/train_input.png")
    # image = cv2.imread("./images/cityscape_input.png")
    # image = cv2.imread("./images/IMG_4091.JPG")
    # scale_factor_x = 0.25
    # scale_factor_y = 0.25
    # image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    A = estimate_airlight(image)
    print(A)
    [img_dehazed, trans_refined] = non_local_dehazing(image, A, gamma=1)

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(image) 
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Dehazed")
    plt.imshow(img_dehazed)
    plt.axis("off")

    plt.tight_layout()
    plt.show()