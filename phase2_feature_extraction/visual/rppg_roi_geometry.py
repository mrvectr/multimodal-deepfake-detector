import cv2
import numpy as np


MIN_ROI_PIXELS = 50
YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
YCRCB_UPPER = np.array([255, 180, 135], dtype=np.uint8)


def validate_frame(frame):
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise ValueError("frame must be a valid HxWx3 image")
    if frame.shape[2] != 3:
        raise ValueError("frame must have 3 color channels")


def validate_landmarks(landmarks):
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (68, 2):
        raise ValueError(f"Expected landmarks with shape (68, 2), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Landmarks contain NaN or infinite values")
    return points


def _polygon_mask(shape, points):
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.round(points).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    if len(polygon) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(polygon), 255)
    return mask


def _shrink_polygon(points, factor):
    center = np.mean(points, axis=0)
    return center + (points - center) * factor


def _forehead_polygon(points, frame_shape):
    # OpenFace's 68-point model has no hairline points. Estimate forehead
    # height from eyebrow width; inspect overlays for each dataset/camera.
    brow = points[17:27]
    left = np.mean(points[17:22], axis=0)
    right = np.mean(points[22:27], axis=0)
    brow_width = np.linalg.norm(right - left)
    if brow_width < 5:
        raise ValueError("Eyebrow landmarks are too close or unreliable")

    top_y = np.min(brow[:, 1]) - 0.22 * brow_width
    inner_left = points[21]
    inner_right = points[22]
    outer_left = points[17]
    outer_right = points[26]

    poly = np.array([
        [outer_left[0], top_y],
        [inner_left[0], top_y],
        inner_left,
        inner_right,
        [outer_right[0], top_y],
    ], dtype=np.float32)

    h, w = frame_shape[:2]
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    return poly


def _cheek_polygons(points):
    left_idx = [2, 3, 4, 31, 48, 41, 40]
    right_idx = [14, 13, 12, 35, 54, 46, 47]
    left = _shrink_polygon(points[left_idx], 0.82)
    right = _shrink_polygon(points[right_idx], 0.82)
    return left, right


def _exclusion_mask(points, frame_shape):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    # Eye, nose and mouth contours are expanded slightly to avoid boundary pixels.
    regions = [
        points[36:42], points[42:48], points[27:36], points[48:68]
    ]
    for region in regions:
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(region).astype(np.int32)), 255)
    kernel = np.ones((5, 5), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def _skin_color_mask(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, YCRCB_LOWER, YCRCB_UPPER)


def create_roi_masks(landmarks, frame, use_skin_filter=False):
    validate_frame(frame)
    points = validate_landmarks(landmarks)

    polygons = {
        "forehead": _forehead_polygon(points, frame.shape),
    }
    left, right = _cheek_polygons(points)
    polygons["left_cheek"] = left
    polygons["right_cheek"] = right

    exclusion = _exclusion_mask(points, frame.shape)
    skin = _skin_color_mask(frame) if use_skin_filter else np.full(
        frame.shape[:2], 255, dtype=np.uint8
    )

    masks = {}
    for name, polygon in polygons.items():
        candidate = _polygon_mask(frame.shape, polygon)
        candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(exclusion))
        masks[name] = cv2.bitwise_and(candidate, skin)

    return {
        "polygons": polygons,
        "masks": masks,
        "exclusion_mask": exclusion,
        "skin_mask": skin,
    }


def validate_roi_masks(roi_data, min_pixels=MIN_ROI_PIXELS):
    results = {}
    for name, mask in roi_data["masks"].items():
        count = int(cv2.countNonZero(mask))
        results[name] = {
            "valid": count >= min_pixels,
            "valid_pixels": count,
        }
    return results


def extract_masked_rgb(frame, mask):
    validate_frame(frame)
    pixels = frame[mask > 0]
    if pixels.size == 0:
        return {"r": np.nan, "g": np.nan, "b": np.nan, "pixel_count": 0}
    # OpenCV frames are BGR; return conventional RGB channel means.
    mean_bgr = pixels.mean(axis=0)
    return {
        "r": float(mean_bgr[2]),
        "g": float(mean_bgr[1]),
        "b": float(mean_bgr[0]),
        "pixel_count": int(len(pixels)),
    }


def visualize_roi_masks(frame, roi_data, validations=None):
    validate_frame(frame)
    preview = frame.copy()
    colors = {
        "forehead": (0, 255, 255),
        "left_cheek": (0, 255, 0),
        "right_cheek": (255, 0, 255),
    }
    overlay = preview.copy()
    for name, mask in roi_data["masks"].items():
        overlay[mask > 0] = colors[name]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, colors[name], 2)
        ys, xs = np.where(mask > 0)
        if len(xs):
            label = name
            if validations and name in validations:
                label += " OK" if validations[name]["valid"] else " INVALID"
            cv2.putText(
                preview, label, (int(xs.min()), max(20, int(ys.min()) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1, cv2.LINE_AA
            )
    return cv2.addWeighted(overlay, 0.18, preview, 0.82, 0)
