#!/usr/bin/env python3
"""
Comparator: original vs improved Baccarat dot detectors
Place this file next to your baccarat_analyzer.py (the script you provided).
Usage:
  - With args:
      python compare.py --input_dir ./images --output_dir ./debug_reports
  - Without args (double-click or run without parameters): a folder picker will open.
"""
import os
import sys
import argparse
import json
import csv
import math
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Try to import the user's existing analyzer
try:
    from baccarat_analyzer import BaccaratSheetAnalyzer
except Exception as e:
    print("ERROR: Could not import BaccaratSheetAnalyzer from baccarat_analyzer.py")
    print("Make sure your original script is named 'baccarat_analyzer.py' and is in the same folder.")
    print("Import error:", e)
    sys.exit(1)


class ImprovedBaccaratSheetAnalyzer(BaccaratSheetAnalyzer):
    """Extends the original analyzer with more robust detection + debug output."""

    def _rectify_board(self, img):
        """Try to find board contour and warp to top-down. Return (warped, M) or (img, None)."""
        try:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 51, 10)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w//40), max(15, h//40)))
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return img, None
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            large = None
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 0.03 * w * h:
                    large = cnt
                    break
            if large is None:
                return img, None
            peri = cv2.arcLength(large, True)
            approx = cv2.approxPolyDP(large, 0.02 * peri, True)
            if len(approx) < 4:
                x, y, ww, hh = cv2.boundingRect(large)
                src = np.array([[x, y], [x + ww, y], [x + ww, y + hh], [x, y + hh]], dtype=np.float32)
            else:
                pts = approx.reshape(-1, 2)
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)[:, 0]
                tl = pts[np.argmin(s)]
                br = pts[np.argmax(s)]
                tr = pts[np.argmin(diff)]
                bl = pts[np.argmax(diff)]
                src = np.array([tl, tr, br, bl], dtype=np.float32)
            (tl, tr, br, bl) = src
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = int(max(widthA, widthB)) or w
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = int(max(heightA, heightB)) or h
            dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
                           dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
            return warped, M
        except Exception:
            return img, None

    def _preprocess_clahe(self, img):
        """Improve contrast via CLAHE on V channel."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint8)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _detect_dots_robust(self, img, debug_dir=None):
        """
        Robust multi-method dot detection. Returns list of dicts.
        """
        debug_images = {}
        try:
            rectified, M = self._rectify_board(img)
            proc = self._preprocess_clahe(rectified)
            height, width = proc.shape[:2]
            hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)

            detected = []
            k_small = max(3, int(min(width, height) / 200))
            k_large = max(7, int(min(width, height) / 100))
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_large, k_large))

            for color_name, mapping in [('red', self.color_ranges['red']),
                                        ('blue', self.color_ranges['blue']),
                                        ('green', self.color_ranges['green'])]:
                if color_name == 'red':
                    m1 = cv2.inRange(hsv, mapping['lower1'], mapping['upper1'])
                    m2 = cv2.inRange(hsv, mapping['lower2'], mapping['upper2'])
                    mask = cv2.bitwise_or(m1, m2)
                else:
                    mask = cv2.inRange(hsv, mapping['lower'], mapping['upper'])
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                debug_images[f'{color_name}_mask'] = mask.copy()

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for i in range(1, num_labels):
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if area < 12 or area > 20000:
                        continue
                    cx, cy = centroids[i]
                    cx_i, cy_i = int(cx), int(cy)
                    r = int(max(3, math.sqrt(area / math.pi) * 1.5))
                    x1, x2 = max(0, cx_i - r), min(width - 1, cx_i + r)
                    y1, y2 = max(0, cy_i - r), min(height - 1, cy_i + r)
                    roi = mask[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    cnt = max(contours, key=cv2.contourArea)
                    cnt_area = cv2.contourArea(cnt)
                    perim = cv2.arcLength(cnt, True)
                    circularity = 0.0
                    if perim > 0:
                        circularity = 4 * math.pi * cnt_area / (perim * perim)
                    if circularity < 0.25 and cnt_area < 200:
                        continue
                    symbol = self.color_mapping.get(color_name, '?')
                    detected.append({'x': cx_i, 'y': cy_i, 'radius': r, 'color': color_name, 'symbol': symbol})

            gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            minR = max(3, int(min(width, height) / 200))
            maxR = max(8, int(min(width, height) / 40))
            try:
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=minR * 2,
                                           param1=100, param2=12, minRadius=minR, maxRadius=maxR)
            except Exception:
                circles = None
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (cx, cy, r) in circles:
                    too_close = False
                    for d in detected:
                        if math.hypot(d['x'] - cx, d['y'] - cy) < max(6, r * 0.6):
                            too_close = True
                            break
                    if too_close:
                        continue
                    sample_r = max(2, int(r / 2))
                    x1, x2 = max(0, cx - sample_r), min(width - 1, cx + sample_r)
                    y1, y2 = max(0, cy - sample_r), min(height - 1, cy + sample_r)
                    region = hsv[y1:y2, x1:x2]
                    if region.size == 0:
                        continue
                    avg_h = int(np.mean(region[:, :, 0]))
                    avg_s = int(np.mean(region[:, :, 1]))
                    color_guess = None
                    if avg_s > 40:
                        if (0 <= avg_h <= 15) or (165 <= avg_h <= 180):
                            color_guess = 'red'
                        elif 85 <= avg_h <= 135:
                            color_guess = 'blue'
                        elif 35 <= avg_h <= 85:
                            color_guess = 'green'
                    if not color_guess:
                        color_guess = 'red'
                    detected.append({'x': int(cx), 'y': int(cy), 'radius': int(r), 'color': color_guess,
                                     'symbol': self.color_mapping.get(color_guess, '?')})

            if M is not None and len(detected) > 0:
                try:
                    Minv = np.linalg.inv(M)
                    for d in detected:
                        pt = np.array([[d['x'], d['y'], 1.0]]).T
                        src_pt = Minv.dot(pt)
                        src_pt = src_pt / src_pt[2]
                        d['x'] = int(round(src_pt[0, 0]))
                        d['y'] = int(round(src_pt[1, 0]))
                except Exception:
                    pass

            if debug_dir:
                try:
                    os.makedirs(debug_dir, exist_ok=True)
                    for k, v in debug_images.items():
                        cv2.imwrite(os.path.join(debug_dir, f"{k}.png"), v)
                except Exception:
                    pass

            return detected
        except Exception:
            return []

    def analyze_image_local_improved(self, image_path, debug_dir=None):
        """Analyze a single image path using improved detector (for comparator)."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "error": "Could not load image"}
            detected = self._detect_dots_robust(img, debug_dir=debug_dir)
            if not detected:
                return {"success": False, "error": "No dots detected (improved)"}
            grid = self._smart_grid_detection(detected, img.shape[1], img.shape[0])
            analysis = self._analyze_grid(grid)
            return {"success": True, "mode": "local_improved", "total_dots": len(detected),
                    "detected_dots": detected, "grid": grid, "analysis": analysis}
        except Exception as e:
            return {"success": False, "error": f"Improved processing failed: {e}"}


def sample_patch_stats(img, x, y, radius=8):
    """Return dict with average HSV, sat, val and simple counts for three color masks in local patch."""
    h, w = img.shape[:2]
    x1, x2 = max(0, int(x - radius)), min(w - 1, int(x + radius))
    y1, y2 = max(0, int(y - radius)), min(h - 1, int(y + radius))
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return {}
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    avg_h = float(np.mean(hsv[:, :, 0]))
    avg_s = float(np.mean(hsv[:, :, 1]))
    avg_v = float(np.mean(hsv[:, :, 2]))
    return {'avg_h': avg_h, 'avg_s': avg_s, 'avg_v': avg_v, 'w': patch.shape[1], 'h': patch.shape[0]}


def match_detections(orig_list, imp_list, image_size, tol_px=None):
    """
    Greedy matching: for each improved dot find nearest original dot within tol.
    Returns matches list of tuples (orig_idx, imp_idx, dist), lists of unmatched indices.
    """
    if tol_px is None:
        tol_px = max(12, int(min(image_size) * 0.02))
    orig_pts = np.array([[d['x'], d['y']] for d in orig_list]) if orig_list else np.empty((0, 2))
    imp_pts = np.array([[d['x'], d['y']] for d in imp_list]) if imp_list else np.empty((0, 2))

    matched_orig = set()
    matched_imp = set()
    matches = []

    for imp_i, imp_p in enumerate(imp_pts):
        if orig_pts.size == 0:
            break
        dists = np.linalg.norm(orig_pts - imp_p, axis=1)
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        if best_dist <= tol_px and best_idx not in matched_orig:
            matches.append((best_idx, imp_i, best_dist))
            matched_orig.add(best_idx)
            matched_imp.add(imp_i)

    unmatched_orig = [i for i in range(len(orig_list)) if i not in matched_orig]
    unmatched_imp = [i for i in range(len(imp_list)) if i not in matched_imp]
    return matches, unmatched_orig, unmatched_imp, tol_px


def draw_overlay_save(image_path, orig_list, imp_list, matches, unmatched_orig, unmatched_imp, out_path):
    """Draw overlay: original = yellow, improved = cyan, matches = green, unmatched = red. Save PNG."""
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, d in enumerate(orig_list):
        x, y = d['x'], d['y']
        r = max(4, d.get('radius', 6))
        bbox = [x - r, y - r, x + r, y + r]
        color = (255, 215, 0, 200)
        draw.ellipse(bbox, outline=color, width=2)
        if font:
            draw.text((x + r + 2, y - r), f"O{i}:{d.get('symbol','?')}", fill=(255,215,0,200), font=font)

    for i, d in enumerate(imp_list):
        x, y = d['x'], d['y']
        r = max(4, d.get('radius', 6))
        bbox = [x - r, y - r, x + r, y + r]
        color = (0, 255, 255, 180)
        draw.ellipse(bbox, outline=color, width=2)
        if font:
            draw.text((x + r + 2, y + r), f"I{i}:{d.get('symbol','?')}", fill=(0,255,255,180), font=font)

    for orig_i, imp_i, dist in matches:
        o = orig_list[orig_i]
        p = imp_list[imp_i]
        mx = (o['x'] + p['x']) // 2
        my = (o['y'] + p['y']) // 2
        draw.line([(o['x'], o['y']), (p['x'], p['y'])], fill=(0, 255, 0, 200), width=2)
        draw.ellipse([mx - 3, my - 3, mx + 3, my + 3], fill=(0, 255, 0, 220))

    for i in unmatched_orig:
        d = orig_list[i]
        x, y = d['x'], d['y']
        r = max(4, d.get('radius', 6))
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0, 200), width=3)
        if font:
            draw.text((x + r + 2, y - r), f"O{i}:MISS", fill=(255,0,0,200), font=font)

    for i in unmatched_imp:
        d = imp_list[i]
        x, y = d['x'], d['y']
        r = max(4, d.get('radius', 6))
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 255, 200), width=3)
        if font:
            draw.text((x + r + 2, y + r), f"I{i}:NEW", fill=(255,0,255,200), font=font)

    img.save(out_path)


def run_comparison_on_folder(input_dir, output_dir, orig_module_name='baccarat_analyzer'):
    os.makedirs(output_dir, exist_ok=True)
    orig_analyzer = BaccaratSheetAnalyzer()
    imp_analyzer = ImprovedBaccaratSheetAnalyzer()
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    images.sort()
    summary_rows = []
    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        base_out = os.path.join(output_dir, os.path.splitext(img_name)[0])
        os.makedirs(base_out, exist_ok=True)
        print(f"[{datetime.now().isoformat()}] Processing {img_name} ...")
        orig_res = orig_analyzer.analyze_image_local(img_path)
        orig_list = orig_res.get('detected_dots', []) if orig_res.get('success') else []
        imp_res = imp_analyzer.analyze_image_local_improved(img_path, debug_dir=base_out)
        imp_list = imp_res.get('detected_dots', []) if imp_res.get('success') else []

        img_cv = cv2.imread(img_path)
        h, w = img_cv.shape[:2]
        matches, unmatched_orig, unmatched_imp, tol = match_detections(orig_list, imp_list, (h, w))
        color_mismatches = 0
        matched_details = []
        for orig_i, imp_i, dist in matches:
            o = orig_list[orig_i]
            p = imp_list[imp_i]
            mismatch = o.get('symbol') != p.get('symbol')
            if mismatch:
                color_mismatches += 1
            matched_details.append({
                'orig_idx': orig_i, 'imp_idx': imp_i, 'dist': dist,
                'orig_symbol': o.get('symbol'), 'imp_symbol': p.get('symbol'),
                'orig_color': o.get('color'), 'imp_color': p.get('color')
            })
        per_dot_debug = {'original': [], 'improved': []}
        for i, d in enumerate(orig_list):
            stats = sample_patch_stats(img_cv, d['x'], d['y'], radius=max(6, d.get('radius', 6)))
            per_dot_debug['original'].append({'idx': i, 'x': d['x'], 'y': d['y'], 'symbol': d.get('symbol'), 'stats': stats})
        for i, d in enumerate(imp_list):
            stats = sample_patch_stats(img_cv, d['x'], d['y'], radius=max(6, d.get('radius', 6)))
            per_dot_debug['improved'].append({'idx': i, 'x': d['x'], 'y': d['y'], 'symbol': d.get('symbol'), 'stats': stats})

        overlay_path = os.path.join(base_out, f"{os.path.splitext(img_name)[0]}_overlay.png")
        draw_overlay_save(img_path, orig_list, imp_list, matches, unmatched_orig, unmatched_imp, overlay_path)

        debug_images = imp_res.get('debug_images', {}) if imp_res.get('success') else {}
        for k, v in debug_images.items():
            try:
                save_p = os.path.join(base_out, f"debug_{k}.png")
                cv2.imwrite(save_p, v)
            except Exception:
                pass

        report = {
            'image': img_name,
            'orig_success': orig_res.get('success', False),
            'imp_success': imp_res.get('success', False),
            'orig_count': len(orig_list),
            'imp_count': len(imp_list),
            'matches_count': len(matches),
            'only_in_original': len(unmatched_orig),
            'only_in_improved': len(unmatched_imp),
            'color_mismatches': color_mismatches,
            'match_tolerance_px': tol,
            'matched_details': matched_details,
            'per_dot_debug': per_dot_debug
        }
        with open(os.path.join(base_out, "report.json"), "w") as fh:
            json.dump(report, fh, indent=2)

        summary_rows.append({
            'image': img_name,
            'orig_count': len(orig_list),
            'imp_count': len(imp_list),
            'matches': len(matches),
            'only_orig': len(unmatched_orig),
            'only_imp': len(unmatched_imp),
            'color_mismatches': color_mismatches
        })
    csv_path = os.path.join(output_dir, "aggregate_summary.csv")
    with open(csv_path, "w", newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['image', 'orig_count', 'imp_count', 'matches', 'only_orig', 'only_imp', 'color_mismatches'])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print("Done. Reports saved to:", output_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Compare original vs improved Baccarat detectors")
    p.add_argument("--input_dir", "-i", help="Folder with input images")
    p.add_argument("--output_dir", "-o", default="./debug_reports", help="Folder to write reports and overlays")
    args = p.parse_args()
    if not args.input_dir:
        # Prompt user to pick a folder using a simple Tk dialog
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            folder = filedialog.askdirectory(title="Select input images folder")
            root.destroy()
            if not folder:
                print("No input folder selected. Exiting.")
                sys.exit(1)
            args.input_dir = folder
        except Exception as e:
            print("Could not open folder picker. Please provide --input_dir. Error:", e)
            sys.exit(1)
    return args


if __name__ == "__main__":
    args = parse_args()
    run_comparison_on_folder(args.input_dir, args.output_dir)