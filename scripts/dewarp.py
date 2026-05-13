"""
dewarp.py — Crop and dewarp an EC8A result sheet photo.

Usage:
    python scripts/dewarp.py <input_image> <output_image>
    python scripts/dewarp.py compare <input_image> <output_dir>

Examples:
    python scripts/dewarp.py test_images/19-41-08-011.jpg dewarped.jpg
    python scripts/dewarp.py compare test_images/19-41-08-011.jpg /tmp/ab/

Pipeline:
    1. YOLO OBB layout detection — detects logo/header to correct rotation,
       then crops to the document ROI.
    2. UVDocNet — predicts a dense 2D deformation grid and applies it via
       bilinear grid_sample, outputting a flat 1200×1700 image.

Public API (importable):
    load_models(yolo_path, rect_path) -> (det_model, rect_model)
    dewarp_to_bytes(image, det_model, rect_model, out_size) -> bytes  # raises DewarpError
    DewarpError

Model paths (relative to repo root):
    models/rotation_22may.pt   — YOLO OBB layout model
    models/rect_model.pkl      — UVDocNet rectification weights
"""

import argparse
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from shapely.geometry import Polygon
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths — resolved relative to repo root (one level up from this script)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = os.path.join(REPO_ROOT, "models", "rotation_22may.pt")
RECT_MODEL_PATH = os.path.join(REPO_ROOT, "models", "rect_model.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class DewarpError(Exception):
    """Raised when document orientation cannot be determined."""


# ---------------------------------------------------------------------------
# UVDocNet architecture (verbatim from Yuxi's notebook)
# ---------------------------------------------------------------------------

def conv3x3(in_channels, out_channels, kernel_size, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=kernel_size // 2)


def dilated_conv_bn_act(in_channels, out_channels, act_fn, BatchNorm, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=3,
                  stride=1, padding=dilation, dilation=dilation),
        BatchNorm(out_channels),
        act_fn,
    )


def dilated_conv(in_channels, out_channels, kernel_size, dilation, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=dilation * (kernel_size // 2),
                  dilation=dilation)
    )


class ResidualBlockWithDilation(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm, kernel_size,
                 stride=1, downsample=None, is_activation=True, is_top=False):
        super().__init__()
        self.downsample = downsample
        if stride != 1 or is_top:
            self.conv1 = conv3x3(in_channels, out_channels, kernel_size, stride)
            self.conv2 = conv3x3(out_channels, out_channels, kernel_size)
        else:
            self.conv1 = dilated_conv(in_channels, out_channels, kernel_size, dilation=3)
            self.conv2 = dilated_conv(out_channels, out_channels, kernel_size, dilation=3)
        self.bn1 = BatchNorm(out_channels)
        self.bn2 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResnetStraight(nn.Module):
    def __init__(self, num_filter, map_num, BatchNorm,
                 block_nums=(3, 4, 6), kernel_size=5, stride=(1, 2, 2)):
        super().__init__()
        self.in_channels = num_filter * map_num[0]
        self.kernel_size = kernel_size
        self.layer1 = self._make_layer(ResidualBlockWithDilation, num_filter * map_num[0],
                                       block_nums[0], BatchNorm, kernel_size, stride[0])
        self.layer2 = self._make_layer(ResidualBlockWithDilation, num_filter * map_num[1],
                                       block_nums[1], BatchNorm, kernel_size, stride[1])
        self.layer3 = self._make_layer(ResidualBlockWithDilation, num_filter * map_num[2],
                                       block_nums[2], BatchNorm, kernel_size, stride[2])

    def _make_layer(self, block, out_channels, n_blocks, BatchNorm, kernel_size, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, kernel_size, stride),
                BatchNorm(out_channels),
            )
        layers = [block(self.in_channels, out_channels, BatchNorm, kernel_size,
                        stride, downsample, is_top=True)]
        self.in_channels = out_channels
        for _ in range(1, n_blocks):
            layers.append(block(out_channels, out_channels, BatchNorm, kernel_size, is_top=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


class UVDocnet(nn.Module):
    def __init__(self, num_filter=32, kernel_size=5):
        super().__init__()
        BN = nn.BatchNorm2d
        act = nn.ReLU(inplace=True)
        mp = [1, 2, 4, 8, 16]
        nf = num_filter

        self.resnet_head = nn.Sequential(
            nn.Conv2d(3, nf*mp[0], bias=False, kernel_size=kernel_size,
                      stride=2, padding=kernel_size//2),
            BN(nf*mp[0]), act,
            nn.Conv2d(nf*mp[0], nf*mp[0], bias=False, kernel_size=kernel_size,
                      stride=2, padding=kernel_size//2),
            BN(nf*mp[0]), act,
        )
        self.resnet_down = ResnetStraight(nf, mp, BN, block_nums=(3, 4, 6),
                                          kernel_size=kernel_size, stride=(1, 2, 2))
        ch = nf * mp[2]  # 128
        self.bridge_1 = nn.Sequential(dilated_conv_bn_act(ch, ch, act, BN, 1))
        self.bridge_2 = nn.Sequential(dilated_conv_bn_act(ch, ch, act, BN, 2))
        self.bridge_3 = nn.Sequential(dilated_conv_bn_act(ch, ch, act, BN, 5))
        self.bridge_4 = nn.Sequential(*[dilated_conv_bn_act(ch, ch, act, BN, d) for d in (8, 3, 2)])
        self.bridge_5 = nn.Sequential(*[dilated_conv_bn_act(ch, ch, act, BN, d) for d in (12, 7, 4)])
        self.bridge_6 = nn.Sequential(*[dilated_conv_bn_act(ch, ch, act, BN, d) for d in (18, 12, 6)])
        self.bridge_concat = nn.Sequential(
            nn.Conv2d(ch * 6, ch, bias=False, kernel_size=1),
            BN(ch), act,
        )

        def out_head(out_ch):
            return nn.Sequential(
                nn.Conv2d(ch, nf*mp[0], bias=False, kernel_size=kernel_size,
                          padding=kernel_size//2, padding_mode="reflect"),
                BN(nf*mp[0]), nn.PReLU(),
                nn.Conv2d(nf*mp[0], out_ch, kernel_size=kernel_size,
                          padding=kernel_size//2, padding_mode="reflect"),
            )

        self.out_point_positions2D = out_head(2)
        self.out_point_positions3D = out_head(3)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight, gain=0.2)

    def forward(self, x):
        feat = self.resnet_down(self.resnet_head(x))
        bridges = torch.cat([
            self.bridge_1(feat), self.bridge_2(feat), self.bridge_3(feat),
            self.bridge_4(feat), self.bridge_5(feat), self.bridge_6(feat),
        ], dim=1)
        bridge = self.bridge_concat(bridges)
        return self.out_point_positions2D(bridge), self.out_point_positions3D(bridge)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_rect_model(path):
    model = UVDocnet(num_filter=32, kernel_size=5)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model


def load_models(yolo_path=None, rect_path=None):
    """Load and return (det_model, rect_model). Uses default paths if not specified."""
    yolo_path = yolo_path or YOLO_MODEL_PATH
    rect_path = rect_path or RECT_MODEL_PATH
    det_model = YOLO(yolo_path)
    rect_model = load_rect_model(rect_path)
    return det_model, rect_model


# ---------------------------------------------------------------------------
# YOLO helpers (verbatim logic from Yuxi's notebook)
# ---------------------------------------------------------------------------

def _iou_polygon(a, b):
    poly_a = Polygon([(a[i], a[i+1]) for i in range(0, len(a), 2)])
    poly_b = Polygon([(b[i], b[i+1]) for i in range(0, len(b), 2)])
    return poly_a.intersection(poly_b).area / poly_a.union(poly_b).area


def _nms_classwise(df):
    import pandas as pd

    def _nms(group):
        rows = group.sort_values("confidence", ascending=False).to_dict("records")
        kept = []
        for cur in rows:
            cur_poly = [v for i in range(4) for v in (cur[f"x{i+1}"], cur[f"y{i+1}"])]
            if not any(_iou_polygon(cur_poly,
                                    [v for i in range(4) for v in (s[f"x{i+1}"], s[f"y{i+1}"])]) > 0.3
                       for s in kept):
                kept.append(cur)
        return pd.DataFrame(kept)

    import pandas as pd
    return pd.concat([_nms(g) for _, g in df.groupby("label")], ignore_index=True)


def _angle_to_rotation(df):
    """Return median adjusted angle if distribution is clean, else None."""
    pivot = 90
    adj = df["angle"].apply(lambda x: x if x <= pivot else 180 - x)
    counts, _ = np.histogram(adj, bins=[0, 15, 30, 45, 60, 75, 90])
    top2 = counts.argsort()[-2:]
    if counts[top2].sum() / len(df) > 0.8:
        return float(np.median(adj))
    return None


def _infer_orientation(image_size, detections, median_rot):
    """Infer document rotation from logo/header position.

    Args:
        image_size: (width, height) of the image being analysed.
        detections: DataFrame of YOLO detections.
        median_rot: median adjusted angle from _angle_to_rotation.

    Returns:
        Rotation angle in degrees, or -1 if orientation cannot be determined.
    """
    w, h = image_size

    def from_elements(elems):
        if elems.empty:
            return -1
        best = elems.sort_values(["confidence", "centroid_y"], ascending=[False, True]).iloc[0]
        if median_rot < 25:
            if best["centroid_y"] < h / 3:
                return median_rot
            if best["centroid_y"] > 2 * h / 3:
                return 180 - median_rot
        elif median_rot > 75:
            if best["centroid_x"] < w / 3:
                return 90
            if best["centroid_x"] > 2 * w / 3:
                return -90
        return -1

    result = from_elements(detections[detections["label"] == "logo"])
    if result == -1:
        result = from_elements(detections[detections["label"] == "header"])
    return result


def _get_yolo_boxes(det_model, img_pil):
    """Run YOLO on a PIL image. Returns (detections_df, rotation_degrees)."""
    import pandas as pd

    im = np.array(img_pil)[:, :, ::-1].copy()  # PIL RGB → BGR
    preds = det_model(im, imgsz=800, conf=0.05, max_det=10000, verbose=False)
    names = ["box", "table", "column", "header", "signature",
             "figure", "paragraph", "logo", "kv", "stamp"]

    rows = []
    for xywhr, xyxyxyxy, conf, cls in zip(
            preds[0].obb.xywhr.tolist(),
            preds[0].obb.xyxyxyxy.tolist(),
            preds[0].obb.conf.tolist(),
            preds[0].obb.cls.tolist()):
        cx, cy, w, h, rad = xywhr
        pts = [[int(xyxyxyxy[j][0]), int(xyxyxyxy[j][1])] for j in range(4)]
        if conf > 0.1:
            rows.append(dict(x1=pts[0][0], y1=pts[0][1], x2=pts[1][0], y2=pts[1][1],
                             x3=pts[2][0], y3=pts[2][1], x4=pts[3][0], y4=pts[3][1],
                             centroid_x=cx, centroid_y=cy,
                             angle=rad * 180 / 3.14159,
                             label=names[int(cls)], confidence=conf))

    if not rows:
        return pd.DataFrame(), 0

    df = pd.DataFrame(rows)
    df_nms = _nms_classwise(df)
    median_rot = _angle_to_rotation(df)
    if median_rot is None:
        return df_nms, 0
    rotation = _infer_orientation(img_pil.size, df_nms, median_rot)
    return df_nms, rotation


def _rotate_image(img_pil, angle):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rad = np.deg2rad(angle)
    new_w = int(h * abs(np.sin(rad)) + w * abs(np.cos(rad)))
    new_h = int(h * abs(np.cos(rad)) + w * abs(np.sin(rad)))
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2
    rotated = cv2.warpAffine(img, M, (new_w, new_h))
    return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))


def _get_roi(detections, img_h, img_w):
    x1, x2, y1, y2 = 0, img_w, 0, img_h
    tables = detections[detections["label"] == "table"]
    if not tables.empty:
        x1 = tables[["x1", "x2", "x3", "x4"]].min().min()
        x2 = tables[["x1", "x2", "x3", "x4"]].max().max()
    logos = detections[detections["label"].isin(["logo", "header"])]
    if not logos.empty:
        y1 = logos[["y1", "y2", "y3", "y4"]].min().min()
    stamps = detections[detections["label"] == "stamp"]
    if not stamps.empty:
        y2 = stamps[["y1", "y2", "y3", "y4"]].max().max()

    m = 0.15
    x1 = max(int(x1 - m * (x2 - x1)), 0)
    x2 = min(int(x2 + m * (x2 - x1)), img_w)
    y1 = max(int(y1 - m * (y2 - y1)), 0)
    y2 = min(int(y2 + m * (y2 - y1)), img_h)

    if x1 >= img_w // 2: x1 = 0
    if y1 > img_h // 4:  y1 = 0
    if x2 < img_w // 2:  x2 = img_w
    if y2 < 3 * img_h // 4: y2 = img_h
    return x1, y1, x2, y2


def crop_document(det_model, image):
    """YOLO-based rotation correction + crop.

    Args:
        det_model: loaded YOLO model.
        image: file path (str/Path) OR BGR numpy array.

    Returns:
        BGR ndarray of the cropped, rotation-corrected document.

    Raises:
        DewarpError: if orientation cannot be determined.
    """
    if isinstance(image, (str, Path)):
        img_pil = Image.open(image).convert("RGB")
    else:
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    df, rotation = _get_yolo_boxes(det_model, img_pil)

    if rotation == -1:
        raise DewarpError("Could not determine document orientation (no logo/header detected)")

    if rotation in (90, -90):
        img_pil = _rotate_image(img_pil, -rotation)
        df, _ = _get_yolo_boxes(det_model, img_pil)
    elif rotation > 150:
        img_pil = _rotate_image(img_pil, rotation)
        df, _ = _get_yolo_boxes(det_model, img_pil)

    w, h = img_pil.size
    relevant = df[df["label"].isin(["stamp", "table", "logo", "header"])]
    if not relevant.empty:
        x1, y1, x2, y2 = _get_roi(relevant, h, w)
    else:
        x1, y1, x2, y2 = 0, 0, w, h

    cropped = img_pil.crop((x1, y1, x2, y2)).convert("RGB")
    return np.array(cropped)[:, :, ::-1]  # → BGR


# ---------------------------------------------------------------------------
# UVDocNet inference
# ---------------------------------------------------------------------------

def dewarp(rect_model, bgr_image, out_size=(1200, 1700)):
    """Apply UVDocNet dewarping. bgr_image is a BGR ndarray; returns BGR ndarray."""
    IMG_SIZE = [488, 712]  # model's fixed input (width, height)
    img_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = torch.from_numpy(
        cv2.resize(img_rgb, IMG_SIZE).transpose(2, 0, 1)
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        grid2d, _ = rect_model(inp)

    grid = F.interpolate(grid2d, size=(out_size[1], out_size[0]),
                         mode="bilinear", align_corners=True)
    src = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    unwarped = F.grid_sample(src, grid.transpose(1, 2).transpose(2, 3), align_corners=True)
    result = (unwarped[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dewarp_to_bytes(image, det_model, rect_model, out_size=(1200, 1700)):
    """Crop, dewarp, and return JPEG-encoded bytes.

    Args:
        image: file path (str/Path) OR BGR numpy array.
        det_model: loaded YOLO model (from load_models).
        rect_model: loaded UVDocNet model (from load_models).
        out_size: (width, height) of the output image.

    Returns:
        JPEG bytes at quality=85.

    Raises:
        DewarpError: if orientation cannot be determined.
    """
    cropped_bgr = crop_document(det_model, image)
    result_bgr = dewarp(rect_model, cropped_bgr, out_size=out_size)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(result_rgb).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_compare(argv):
    """compare subcommand: produce side-by-side raw vs dewarped images."""
    parser = argparse.ArgumentParser(
        prog="dewarp.py compare",
        description="A/B comparison: raw vs dewarped. Writes original.jpg, dewarped.jpg, comparison.jpg.",
    )
    parser.add_argument("input", help="Path to input photo")
    parser.add_argument("output_dir", help="Directory to write comparison images")
    parser.add_argument("--out-size", default="1200x1700", help="Dewarp output WxH (default: 1200x1700)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        sys.exit(f"Input not found: {args.input}")
    for p, label in [(YOLO_MODEL_PATH, "YOLO model"), (RECT_MODEL_PATH, "rect model")]:
        if not os.path.exists(p):
            sys.exit(f"{label} not found: {p}")

    out_w, out_h = (int(x) for x in args.out_size.split("x"))
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading models ...")
    det_model, rect_model = load_models()

    print(f"Dewarping ...")
    try:
        dewarped_bgr = cv2.cvtColor(
            np.frombuffer(dewarp_to_bytes(args.input, det_model, rect_model, out_size=(out_w, out_h)), np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        # Re-decode from bytes to ensure we're comparing the same JPEG artefacts
        dewarped_bgr = cv2.imdecode(
            np.frombuffer(dewarp_to_bytes(args.input, det_model, rect_model, out_size=(out_w, out_h)), np.uint8),
            cv2.IMREAD_COLOR,
        )
    except DewarpError as e:
        sys.exit(f"Dewarp failed: {e}")

    original_bgr = cv2.imread(args.input)

    # Resize original to same height as dewarped for side-by-side
    orig_h, orig_w = original_bgr.shape[:2]
    scale = out_h / orig_h
    orig_resized = cv2.resize(original_bgr, (int(orig_w * scale), out_h))

    # Side-by-side composite
    comparison = cv2.hconcat([orig_resized, dewarped_bgr])

    orig_out = os.path.join(args.output_dir, "original.jpg")
    dew_out  = os.path.join(args.output_dir, "dewarped.jpg")
    cmp_out  = os.path.join(args.output_dir, "comparison.jpg")

    cv2.imwrite(orig_out,  orig_resized)
    cv2.imwrite(dew_out,   dewarped_bgr)
    cv2.imwrite(cmp_out,   comparison)

    print(f"  original.jpg  → {orig_out}")
    print(f"  dewarped.jpg  → {dew_out}")
    print(f"  comparison.jpg→ {cmp_out}")

    # Pixel-level stats (resize both to same dims for fair comparison)
    orig_f = orig_resized.astype(np.float32)
    dew_f  = dewarped_bgr.astype(np.float32)
    if orig_f.shape != dew_f.shape:
        dew_f = cv2.resize(dew_f, (orig_f.shape[1], orig_f.shape[0])).astype(np.float32)
    mad = float(np.mean(np.abs(orig_f - dew_f)))
    print(f"\nPixel stats (original vs dewarped):")
    print(f"  Mean absolute difference: {mad:.1f} / 255")

    try:
        from skimage.metrics import structural_similarity as ssim
        score = ssim(orig_f, dew_f, channel_axis=2, data_range=255)
        print(f"  SSIM: {score:.4f}  (1.0 = identical, lower = more changed)")
    except ImportError:
        print("  (install scikit-image for SSIM: pip install scikit-image)")


def main():
    # Route 'compare' subcommand before argparse sees it
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        _run_compare(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="Crop and dewarp an EC8A result sheet.")
    parser.add_argument("input", help="Path to input photo")
    parser.add_argument("output", help="Path to write dewarped image")
    parser.add_argument("--no-dewarp", action="store_true",
                        help="Crop and rotate only, skip UVDocNet dewarping")
    parser.add_argument("--out-size", default="1200x1700",
                        help="Output dimensions WxH (default: 1200x1700)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Input not found: {args.input}")
    for p, label in [(YOLO_MODEL_PATH, "YOLO model"), (RECT_MODEL_PATH, "rect model")]:
        if not os.path.exists(p):
            sys.exit(f"{label} not found: {p}")

    out_w, out_h = (int(x) for x in args.out_size.split("x"))

    print(f"Loading YOLO layout model ...")
    det_model = YOLO(YOLO_MODEL_PATH)

    print(f"Cropping and correcting rotation ...")
    try:
        cropped_bgr = crop_document(det_model, args.input)
    except DewarpError as e:
        sys.exit(f"Failed: {e}")

    if args.no_dewarp:
        cv2.imwrite(args.output, cropped_bgr)
        print(f"Saved cropped image (no dewarp) → {args.output}")
        return

    print(f"Loading UVDocNet ...")
    rect_model = load_rect_model(RECT_MODEL_PATH)

    print(f"Dewarping to {out_w}×{out_h} ...")
    result = dewarp(rect_model, cropped_bgr, out_size=(out_w, out_h))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cv2.imwrite(args.output, result)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
