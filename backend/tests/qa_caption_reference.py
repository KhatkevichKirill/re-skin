"""The caption score exactly as committed in be302d7 (before the per-project
config): a frozen reference so tests can prove the default config is
byte-identical to the AICP-332 detector."""
import cv2
import numpy as np

_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def legacy_caption_score(frame, band_y=(0.50, 0.72), band_x=(0.12, 0.88)):
    H, W = frame.shape[:2]
    y0, y1 = int(H * band_y[0]), int(H * band_y[1])
    x0, x1 = int(W * band_x[0]), int(W * band_x[1])
    band = frame[y0:y1, x0:x1]
    mn = band.min(axis=2).astype(np.int16)
    mx = band.max(axis=2).astype(np.int16)
    white = (mn > 225) & (mx - mn < 25)
    dark = (mx < 60).astype(np.uint8)
    darkn = cv2.dilate(dark, _CROSS, iterations=2).astype(bool)
    edge = white & darkn
    if not edge.any():
        return 0, None
    grown = cv2.dilate(edge.astype(np.uint8), _CROSS, iterations=3)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=4)
    best, best_box = 0, None
    for i in range(1, n):
        bx, by, bw, bh = (int(v) for v in stats[i, :4])
        if not (12 <= bh <= 60 and bw >= bh * 0.8):
            continue
        count = int(edge[by:by + bh, bx:bx + bw].sum())
        if count > best:
            best = count
            best_box = ((bx + x0) / W, (by + y0) / H,
                        (bx + bw + x0) / W, (by + bh + y0) / H)
    return best, best_box
