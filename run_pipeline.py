#!/usr/bin/env python3
"""
RoomFormer End-to-End Floorplan Reconstruction Pipeline

Features:
- Ingests single files or nested folders of .ply, .npz (world_points_from_depth / depth maps), or density images.
- Preserves full folder hierarchy under the output path.
- Automatically detects the vertical (up) axis (Y vs Z) and generates isotropic 256x256 density maps.
- Automatically downloads and caches official RoomFormer checkpoints if not present on disk.
- Runs RoomFormer model inference on GPU (CUDA) or CPU.
- Exports publication-quality visual floorplans, density overlays, and structured vector JSON.
"""

import os
import sys
import glob
import json
import time
import shutil
import urllib.request
import zipfile
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Try importing shapely Polygon for polygon filtering
try:
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except ImportError:
    Polygon = None
    HAS_SHAPELY = False

# Try importing plyfile
try:
    from plyfile import PlyData
    HAS_PLYFILE = True
except ImportError:
    PlyData = None
    HAS_PLYFILE = False


# Checkpoint definitions
CHECKPOINT_URL = "https://polybox.ethz.ch/index.php/s/vlBo66X0NTrcsTC/download"
MODEL_CONFIGS = {
    "tight": {
        "filename": "roomformer_stru3d_tight.pth",
        "num_queries": 800,
        "num_polys": 20,
        "semantic_classes": -1,
        "description": "Structured3D with tight, watertight polygon boundaries (Recommended)"
    },
    "semantic": {
        "filename": "roomformer_stru3d_semantic_rich.pth",
        "num_queries": 2800,
        "num_polys": 70,
        "semantic_classes": 19,
        "description": "Structured3D with 16 semantic room labels, doors, and windows"
    },
    "scenecad": {
        "filename": "roomformer_scenecad.pth",
        "num_queries": 800,
        "num_polys": 20,
        "semantic_classes": -1,
        "description": "Trained on real-world ScanNet sensor scans"
    },
    "standard": {
        "filename": "roomformer_stru3d.pth",
        "num_queries": 800,
        "num_polys": 20,
        "semantic_classes": -1,
        "description": "Standard benchmark Structured3D model"
    }
}

SEMANTIC_LABELS = {
    0: "Living Room", 1: "Kitchen", 2: "Bedroom", 3: "Bathroom",
    4: "Balcony", 5: "Corridor", 6: "Dining Room", 7: "Study",
    8: "Studio", 9: "Store Room", 10: "Garden", 11: "Laundry Room",
    12: "Office", 13: "Basement", 14: "Garage", 15: "Misc",
    16: "Door", 17: "Window"
}

PALETTE_COLORS = [
    (75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0),
    (48, 130, 245), (180, 30, 145), (240, 240, 70), (230, 50, 240),
    (60, 245, 210), (190, 190, 250), (128, 128, 0), (255, 190, 230),
    (40, 110, 170), (200, 250, 255), (0, 0, 128), (195, 255, 170),
    (0, 128, 128), (180, 215, 255), (128, 0, 128), (0, 165, 255)
]


def download_checkpoints_if_needed(target_ckpt_path, model_type="tight"):
    """
    Checks if target checkpoint exists. If not, downloads checkpoints.zip
    from official ETH Polybox and extracts it into the checkpoints folder.
    """
    if os.path.isfile(target_ckpt_path):
        return target_ckpt_path

    ckpt_dir = os.path.dirname(os.path.abspath(target_ckpt_path)) or "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    zip_dest = os.path.join(ckpt_dir, "checkpoints.zip")

    print(f"\n[Checkpoint Manager] Model checkpoint '{target_ckpt_path}' not found locally.")
    print(f"[Checkpoint Manager] Downloading checkpoints (~1.62 GB) from ETH Polybox: {CHECKPOINT_URL}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            mb_down = downloaded / (1024 * 1024)
            mb_tot = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Downloading: {mb_down:.1f} MB / {mb_tot:.1f} MB ({percent:.1f}%)")
        else:
            mb_down = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  Downloading: {mb_down:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, zip_dest, reporthook=progress_hook)
        print("\n[Checkpoint Manager] Download complete. Extracting archive...")
        with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
            # Check top-level folder in zip
            names = zip_ref.namelist()
            for member in names:
                filename = os.path.basename(member)
                if not filename:
                    continue
                # Extract to ckpt_dir
                source = zip_ref.open(member)
                target = open(os.path.join(ckpt_dir, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)
        if os.path.exists(zip_dest):
            os.remove(zip_dest)
        print(f"[Checkpoint Manager] Successfully extracted models to '{ckpt_dir}'.\n")
    except Exception as e:
        print(f"\n[Checkpoint Manager Error] Failed to auto-download: {e}")
        print("Please manually download checkpoints from: https://polybox.ethz.ch/index.php/s/vlBo66X0NTrcsTC")
        raise

    if not os.path.isfile(target_ckpt_path):
        # Fallback search inside ckpt_dir
        expected_name = os.path.basename(target_ckpt_path)
        candidates = glob.glob(os.path.join(ckpt_dir, "**", expected_name), recursive=True)
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"Expected checkpoint '{expected_name}' not found after extraction in '{ckpt_dir}'")

    return target_ckpt_path


def read_ply_points(ply_path):
    """Reads 3D vertex points (N, 3) from a .ply file."""
    if HAS_PLYFILE and PlyData is not None:
        ply = PlyData.read(ply_path)
        v = ply['vertex']
        xyz = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], axis=1)
        return xyz.astype(np.float32)

    # Basic fallback parser for binary/ascii PLY
    with open(ply_path, 'rb') as f:
        header = ""
        while True:
            line = f.readline().decode('latin1')
            header += line
            if line.strip() == "end_header":
                break
        
        # Read format and count
        is_ascii = "format ascii" in header
        v_count = 0
        for line in header.splitlines():
            if line.startswith("element vertex"):
                v_count = int(line.split()[-1])
        
        if is_ascii:
            data = np.loadtxt(f, max_rows=v_count)
            return data[:, :3].astype(np.float32)
        else:
            # binary_little_endian float32 x, y, z
            raw = f.read(v_count * 12)
            pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3)
            return pts


def read_npz_points(npz_path, depth_conf_thresh=0.1):
    """
    Extracts 3D point coordinates from a .npz prediction or depth file.
    Supports keys: 'world_points_from_depth', 'xyz', 'points', 'depth'.
    """
    data = np.load(npz_path)
    keys = list(data.keys())

    if 'world_points_from_depth' in data:
        pts = data['world_points_from_depth'] # shape (N, H, W, 3) or (H, W, 3)
        pts = pts.reshape(-1, 3)
        if 'depth_conf' in data:
            conf = data['depth_conf'].reshape(-1)
            valid = (conf > depth_conf_thresh) & (~np.isnan(pts).any(axis=1)) & (~np.isinf(pts).any(axis=1))
            pts = pts[valid]
        else:
            valid = (~np.isnan(pts).any(axis=1)) & (~np.isinf(pts).any(axis=1))
            pts = pts[valid]
        return pts.astype(np.float32)

    for key in ['xyz', 'points', 'point_cloud']:
        if key in data:
            pts = data[key].reshape(-1, 3)
            valid = (~np.isnan(pts).any(axis=1)) & (~np.isinf(pts).any(axis=1))
            return pts[valid].astype(np.float32)

    raise ValueError(f"Could not find 3D point keys in '{npz_path}'. Available keys: {keys}")


def project_points_to_density(xyz, up_axis="auto", width=256, height=256, padding=0.05):
    """
    Projects 3D points along the vertical up-axis to create an isotropic
    256x256 density map matching RoomFormer training standards.
    
    Returns:
        density_uint8: (256, 256) uint8 image [0, 255]
        norm_dict: metadata required to project 2D pixel coordinates back to real world meters
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # Auto-detect up axis if requested
    if up_axis == "auto":
        span_x = float(x.max() - x.min())
        span_y = float(y.max() - y.min())
        span_z = float(z.max() - z.min())
        # In indoor building scans, vertical ceiling-to-floor height is the shortest span (~2.4-3.0m)
        spans = {'x': span_x, 'y': span_y, 'z': span_z}
        detected_up = min(spans, key=spans.get)
        up_axis = detected_up

    up_axis = up_axis.lower()
    if up_axis == 'y':
        coords_2d = np.stack([x, z], axis=1) # Ground plane: X and Z
    elif up_axis == 'z':
        coords_2d = np.stack([x, y], axis=1) # Ground plane: X and Y
    elif up_axis == 'x':
        coords_2d = np.stack([y, z], axis=1) # Ground plane: Y and Z
    else:
        raise ValueError(f"Unsupported up_axis '{up_axis}'. Choose 'auto', 'x', 'y', or 'z'.")

    # Isotropic square bounding box with padding
    mins = coords_2d.min(axis=0)
    maxs = coords_2d.max(axis=0)
    max_range = float(np.max(maxs - mins))
    pad = max_range * padding

    center = (maxs + mins) / 2.0
    box_min = center - (max_range / 2.0) - pad
    full_range = max_range + 2.0 * pad

    # Normalize to [0, 1]
    norm_coords = (coords_2d - box_min) / full_range

    # Map to grid
    grid_x = np.clip(np.round(norm_coords[:, 0] * (width - 1)), 0, width - 1)
    grid_y = np.clip(np.round(norm_coords[:, 1] * (height - 1)), 0, height - 1)
    coordinates = np.stack([grid_x, grid_y], axis=1).astype(np.int32)

    unique_coords, counts = np.unique(coordinates, return_counts=True, axis=0)

    density = np.zeros((height, width), dtype=np.float32)
    density[unique_coords[:, 1], unique_coords[:, 0]] = counts.astype(np.float32)

    if density.max() > 0:
        density = density / density.max()

    density_uint8 = (density * 255.0).astype(np.uint8)

    norm_dict = {
        "up_axis": up_axis,
        "box_min": box_min.tolist(),
        "full_range": full_range,
        "image_size": [width, height]
    }

    return density_uint8, norm_dict


def pixel_to_world_coords(corners_px, norm_dict):
    """Converts 2D pixel coordinates [0, 256] back to real-world metric plane coordinates."""
    box_min = np.array(norm_dict["box_min"])
    full_range = norm_dict["full_range"]
    w, h = norm_dict["image_size"]
    
    corners = np.array(corners_px, dtype=np.float32)
    norm_x = corners[:, 0] / (w - 1)
    norm_y = corners[:, 1] / (h - 1)
    
    world_x = box_min[0] + norm_x * full_range
    world_y = box_min[1] + norm_y * full_range
    
    return np.stack([world_x, world_y], axis=1).tolist()


def polygon_area_and_validity(corners):
    """Computes area and checks if corners form a valid non-degenerate polygon."""
    if len(corners) < 3:
        return 0.0, False
    if HAS_SHAPELY and Polygon is not None:
        try:
            poly = Polygon(corners)
            return float(poly.area), poly.is_valid
        except Exception:
            pass
    # Shoelace formula fallback
    x = corners[:, 0]
    y = corners[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return float(area), (area >= 10.0)


def draw_floorplan_visualizations(density_img, rooms, window_doors=None):
    """
    Generates:
    1. Clean colored floorplan map (room regions)
    2. Overlay map (room boundaries plotted directly over density image)
    """
    h, w = density_img.shape[:2]

    # Clean floorplan
    clean_floorplan = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Overlay map
    overlay_floorplan = cv2.cvtColor(density_img, cv2.COLOR_GRAY2BGR)

    # 1. Fill room regions on clean floorplan
    for i, room in enumerate(rooms):
        poly_pts = np.array(room["polygon_256"], dtype=np.int32).reshape((-1, 1, 2))
        color = PALETTE_COLORS[i % len(PALETTE_COLORS)]
        cv2.fillPoly(clean_floorplan, [poly_pts], color)

    # 2. Draw wall outlines on both
    for i, room in enumerate(rooms):
        poly_pts = np.array(room["polygon_256"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(clean_floorplan, [poly_pts], isClosed=True, color=(30, 30, 30), thickness=2)
        cv2.polylines(overlay_floorplan, [poly_pts], isClosed=True, color=(0, 255, 100), thickness=2)
        # Draw corners
        for pt in room["polygon_256"]:
            cv2.circle(overlay_floorplan, (int(pt[0]), int(pt[1])), radius=3, color=(0, 0, 255), thickness=-1)

    # 3. Draw doors and windows if present
    if window_doors:
        for item in window_doors:
            line_pts = np.array(item["coords_256"], dtype=np.int32)
            pt1, pt2 = (int(line_pts[0][0]), int(line_pts[0][1])), (int(line_pts[1][0]), int(line_pts[1][1]))
            if item.get("type_id") == 16:  # Door
                color = (0, 165, 255) # Orange
            else:  # Window
                color = (255, 255, 0) # Cyan
            cv2.line(clean_floorplan, pt1, pt2, color=color, thickness=3)
            cv2.line(overlay_floorplan, pt1, pt2, color=color, thickness=3)

    return clean_floorplan, overlay_floorplan


def find_candidate_scenes(input_path):
    """
    Recursively scans input_path for .ply, .npz, and density image files.
    Returns list of dicts: {"type": ..., "path": ..., "rel_dir": ..., "scene_name": ...}
    """
    input_path = os.path.abspath(input_path)
    candidates = []

    if os.path.isfile(input_path):
        base_dir = os.path.dirname(input_path)
        rel_dir = ""
        scene_name = Path(input_path).stem
        ext = Path(input_path).suffix.lower()
        if ext in ['.ply', '.npz', '.png', '.jpg', '.jpeg']:
            candidates.append({
                "path": input_path,
                "ext": ext,
                "rel_dir": rel_dir,
                "scene_name": scene_name
            })
        return candidates, base_dir

    # input_path is a directory
    base_dir = input_path
    for root, dirs, files in os.walk(base_dir):
        # Prioritize .ply and .npz in each folder
        has_ply = any(f.lower().endswith('.ply') for f in files)
        has_npz = any(f.lower().endswith('.npz') for f in files)
        has_density = any(f.lower() in ['density_map.png', 'density.png'] for f in files)

        for f in files:
            ext = os.path.splitext(f)[-1].lower()
            file_path = os.path.join(root, f)
            rel_dir = os.path.relpath(root, base_dir)
            if rel_dir == ".":
                rel_dir = ""

            if ext == '.ply':
                candidates.append({
                    "path": file_path, "ext": ext, "rel_dir": rel_dir, "scene_name": Path(f).stem
                })
            elif ext == '.npz' and not has_ply:
                candidates.append({
                    "path": file_path, "ext": ext, "rel_dir": rel_dir, "scene_name": Path(f).stem
                })
            elif ext == '.png' and has_density and not (has_ply or has_npz):
                candidates.append({
                    "path": file_path, "ext": ext, "rel_dir": rel_dir, "scene_name": Path(f).stem
                })

    return candidates, base_dir


def load_model(args, model_type, checkpoint_path, device):
    """Initializes and returns RoomFormer model loaded with trained checkpoint."""
    from models import build_model
    from eval import get_args_parser

    parser = argparse.ArgumentParser('RoomFormer Eval', parents=[get_args_parser()])
    model_args = parser.parse_args([])

    # Apply model specific settings
    cfg = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["tight"])
    model_args.num_queries = cfg["num_queries"]
    model_args.num_polys = cfg["num_polys"]
    model_args.semantic_classes = cfg["semantic_classes"]
    model_args.device = str(device)
    model_args.checkpoint = checkpoint_path

    print(f"[Model Loader] Building RoomFormer ({model_type.upper()}: {model_args.num_polys} polys, {model_args.num_queries} queries)...")
    model = build_model(model_args, train=False)
    model.to(device)
    model.eval()

    print(f"[Model Loader] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)

    return model, model_args


def run_model_inference(model, density_img, device, conf_thresh=0.5, semantic_rich=False):
    """
    Feeds density map into RoomFormer and decodes valid polygons.
    """
    # Prepare single-channel tensor (1, 1, 256, 256)
    img_tensor = (1.0 / 255.0) * torch.as_tensor(density_img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        outputs = model([img_tensor[0]])
        pred_logits = outputs['pred_logits'][0]      # (num_polys, max_corners)
        pred_coords = outputs['pred_coords'][0]      # (num_polys, max_corners, 2)
        fg_mask = torch.sigmoid(pred_logits) > conf_thresh

        pred_room_label = None
        if semantic_rich and 'pred_room_logits' in outputs:
            prob = F.softmax(outputs['pred_room_logits'][0], dim=-1)
            _, pred_room_label = prob[..., :-1].max(dim=-1)
            pred_room_label = pred_room_label.cpu().numpy()

    rooms = []
    window_doors = []

    for j in range(fg_mask.shape[0]):
        valid_corners = pred_coords[j][fg_mask[j]]
        if len(valid_corners) == 0:
            continue

        corners_px = (valid_corners * 255.0).cpu().numpy().round().astype(np.int32)

        if not semantic_rich:
            area, is_valid = polygon_area_and_validity(corners_px)
            if len(corners_px) >= 4 and area >= 100:
                rooms.append({
                    "room_id": len(rooms),
                    "polygon_256": corners_px.tolist(),
                    "area_px": round(area, 2),
                    "num_corners": len(corners_px)
                })
        else:
            cls_id = int(pred_room_label[j]) if pred_room_label is not None else 0
            cls_name = SEMANTIC_LABELS.get(cls_id, f"Class_{cls_id}")

            # Regular rooms
            if cls_id not in [16, 17]:
                area, is_valid = polygon_area_and_validity(corners_px)
                if len(corners_px) >= 4 and area >= 100:
                    rooms.append({
                        "room_id": len(rooms),
                        "type": cls_name,
                        "type_id": cls_id,
                        "polygon_256": corners_px.tolist(),
                        "area_px": round(area, 2),
                        "num_corners": len(corners_px)
                    })
            elif len(corners_px) == 2:
                # Door / Window segment
                window_doors.append({
                    "id": len(window_doors),
                    "type": cls_name,
                    "type_id": cls_id,
                    "coords_256": corners_px.tolist()
                })

    return rooms, window_doors


def main():
    parser = argparse.ArgumentParser(description="RoomFormer End-to-End Floorplan Reconstruction Pipeline")
    parser.add_argument("--input_path", required=True, type=str,
                        help="Path to input .ply, .npz, or folder of scenes")
    parser.add_argument("--output_path", default="output_floorplans", type=str,
                        help="Target output directory preserving input folder structure")
    parser.add_argument("--model_type", default="tight", choices=["tight", "semantic", "scenecad", "standard", "custom"],
                        help="Pretrained RoomFormer variant to use")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="Explicit path to checkpoint file (required if --model_type custom)")
    parser.add_argument("--device", default="auto", type=str,
                        help="Device: 'cuda', 'cpu', or 'auto' (detects cuda availability)")
    parser.add_argument("--up_axis", default="auto", choices=["auto", "x", "y", "z"],
                        help="Vertical axis for 2D density projection")
    parser.add_argument("--conf_thresh", default=0.5, type=float,
                        help="Corner detection confidence threshold (default: 0.5)")
    parser.add_argument("--depth_conf_thresh", default=0.1, type=float,
                        help="Confidence threshold for filtering NPZ depth points")
    parser.add_argument("--density_only", action="store_true",
                        help="Only generate density maps, skipping model inference")
    parser.add_argument("--save_overlay", default=True, type=lambda v: str(v).lower() in ['true', '1', 'yes'],
                        help="Export density map with polygon overlay")
    parser.add_argument("--save_json", default=True, type=lambda v: str(v).lower() in ['true', '1', 'yes'],
                        help="Export vector polygon coordinates in JSON")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print(" RoomFormer End-to-End Floorplan Reconstruction Pipeline")
    print("=" * 70)
    print(f" Input Path    : {args.input_path}")
    print(f" Output Path   : {args.output_path}")
    print(f" Model Type    : {args.model_type}")
    print(f" Device        : {device}")
    print(f" Up Axis       : {args.up_axis}")
    print(f" Density Only  : {args.density_only}")
    print("=" * 70)

    # 1. Discover scenes
    candidates, base_dir = find_candidate_scenes(args.input_path)
    if not candidates:
        print(f"[Warning] No valid .ply, .npz, or density images found in '{args.input_path}'. Exiting.")
        return

    print(f"[Discovery] Found {len(candidates)} scene(s) to process.")

    # 2. Checkpoint handling (if not density_only)
    model = None
    if not args.density_only:
        if args.model_type == "custom":
            if not args.checkpoint or not os.path.isfile(args.checkpoint):
                raise ValueError("Must specify an existing checkpoint via --checkpoint when using --model_type custom.")
            ckpt_path = args.checkpoint
        else:
            default_filename = MODEL_CONFIGS[args.model_type]["filename"]
            ckpt_path = os.path.join("checkpoints", default_filename)
            ckpt_path = download_checkpoints_if_needed(ckpt_path, args.model_type)

        semantic_rich = (args.model_type == "semantic")
        model, model_args = load_model(args, args.model_type, ckpt_path, device)

    # 3. Process each candidate scene
    success_count = 0
    t0 = time.time()

    for idx, item in enumerate(candidates, 1):
        scene_file = item["path"]
        rel_dir = item["rel_dir"]
        scene_name = item["scene_name"]
        ext = item["ext"]

        # Preserve directory hierarchy
        target_scene_dir = os.path.join(args.output_path, rel_dir, scene_name)
        os.makedirs(target_scene_dir, exist_ok=True)

        print(f"\n[{idx}/{len(candidates)}] Processing '{scene_name}' from {rel_dir or '.'}...")

        norm_dict = {}
        # Load or generate density map
        if ext in ['.png', '.jpg', '.jpeg']:
            density_img = cv2.imread(scene_file, cv2.IMREAD_GRAYSCALE)
            if density_img.shape != (256, 256):
                density_img = cv2.resize(density_img, (256, 256), interpolation=cv2.INTER_AREA)
        elif ext == '.ply':
            pts = read_ply_points(scene_file)
            print(f"  Loaded {len(pts)} points from PLY.")
            density_img, norm_dict = project_points_to_density(pts, up_axis=args.up_axis)
        elif ext == '.npz':
            pts = read_npz_points(scene_file, depth_conf_thresh=args.depth_conf_thresh)
            print(f"  Extracted {len(pts)} 3D points from NPZ.")
            density_img, norm_dict = project_points_to_density(pts, up_axis=args.up_axis)
        else:
            continue

        # Save density map
        density_out_path = os.path.join(target_scene_dir, "density_map.png")
        cv2.imwrite(density_out_path, density_img)
        print(f"  Saved: {density_out_path}")

        if args.density_only:
            success_count += 1
            continue

        # Run RoomFormer model inference
        rooms, window_doors = run_model_inference(
            model, density_img, device,
            conf_thresh=args.conf_thresh,
            semantic_rich=(args.model_type == "semantic")
        )
        print(f"  Predicted {len(rooms)} room(s)" + (f", {len(window_doors)} door/window(s)" if window_doors else ""))

        # Convert pixel coordinates to world coordinates if projection info is available
        if norm_dict:
            for r in rooms:
                r["polygon_world"] = pixel_to_world_coords(r["polygon_256"], norm_dict)
            for wd in window_doors:
                wd["coords_world"] = pixel_to_world_coords(wd["coords_256"], norm_dict)

        # Generate & save visualizations
        clean_fp, overlay_fp = draw_floorplan_visualizations(density_img, rooms, window_doors)
        fp_path = os.path.join(target_scene_dir, "floorplan.png")
        cv2.imwrite(fp_path, clean_fp)
        print(f"  Saved: {fp_path}")

        if args.save_overlay:
            overlay_path = os.path.join(target_scene_dir, "floorplan_overlay.png")
            cv2.imwrite(overlay_path, overlay_fp)
            print(f"  Saved: {overlay_path}")

        # Save JSON output
        if args.save_json:
            result_json = {
                "scene_name": scene_name,
                "input_file": scene_file,
                "model_type": args.model_type,
                "projection_info": norm_dict,
                "num_rooms": len(rooms),
                "rooms": rooms,
                "openings": window_doors
            }
            json_path = os.path.join(target_scene_dir, "floorplan.json")
            with open(json_path, 'w') as jf:
                json.dump(result_json, jf, indent=2)
            print(f"  Saved: {json_path}")

        success_count += 1

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Pipeline Completed! Successfully processed {success_count}/{len(candidates)} scenes in {elapsed:.1f}s.")
    print(f"All outputs preserved in: {os.path.abspath(args.output_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
