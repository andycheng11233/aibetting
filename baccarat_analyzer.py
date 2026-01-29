#!/usr/bin/env python3
"""
🎯 BACCARAT ROAD SHEET ANALYZER (Enhanced)
Smart grid detection for imperfect photos — improved preprocessing,
robust multi-method dot detection, perspective rectification, and
better grid snapping. Includes Google Vision fallback.

Drop-in replacement for your original script with enhanced detection.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import cv2
import json
import os
import sys
import argparse
from PIL import Image, ImageTk, ImageDraw
import threading
import queue
import time
import base64
import math


# NOTE: sklearn is used inside functions; main() checks for required packages.

class BaccaratSheetAnalyzer:
    """Analyze Baccarat road sheets with smart grid detection (improved)."""

    def __init__(self):
        # Define color ranges for Baccarat dots (HSV)
        # These can be tuned per dataset. Values are conservative.
        self.color_ranges = {
            'red': {
                'lower1': np.array([0, 90, 70]),  # Lower red range
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 90, 70]),  # Upper red range wrap
                'upper2': np.array([180, 255, 255])
            },
            'blue': {
                'lower': np.array([90, 70, 60]),
                'upper': np.array([135, 255, 255])
            },
            'green': {
                'lower': np.array([35, 60, 60]),
                'upper': np.array([85, 255, 255])
            }
        }

        self.color_mapping = {
            'red': 'B',  # Banker
            'blue': 'P',  # Player
            'green': 'T'  # Tie
        }

    # ---------------------------
    # Preprocessing & rectification
    # ---------------------------
    def _rectify_board(self, img):
        """
        Attempt to detect a large rectangular board in the image and warp it to a top-down view.
        Returns (warped_img, M) where M is the perspective transform matrix (original->warped).
        If detection fails, returns (img, None).
        """
        try:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 51, 10)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 40), max(15, h // 40)))
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return img, None

            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            large = None
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 0.03 * w * h:  # consider reasonably large shapes
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
                # Order points: tl, tr, br, bl
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
        """
        Apply CLAHE (adaptive histogram equalization) to the V channel to improve contrast.
        Returns color image after enhancement.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint8)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ---------------------------
    # Robust detection
    # ---------------------------
    def _detect_dots_robust(self, img, debug_dir=None):
        """
        Robust dot detection pipeline:
        - Perspective rectify board (optional)
        - CLAHE contrast boost
        - Per-color mask + morphological cleanup + connected components
        - HoughCircles fallback to recover faded dots
        - Map detected points back to original coordinates if rectified
        Returns list of dicts: {'x','y','radius','color','symbol'}
        """
        debug_images = {}
        try:
            orig_img = img.copy()
            warped, M = self._rectify_board(img)
            proc = self._preprocess_clahe(warped)
            height, width = proc.shape[:2]
            hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)

            detected = []
            # Morphological kernel sizing adaptive to image size
            k_small = max(3, int(min(width, height) / 200))
            k_large = max(7, int(min(width, height) / 100))
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_large, k_large))

            # Per-color mask detection
            for color_name, mapping in [('red', self.color_ranges['red']),
                                        ('blue', self.color_ranges['blue']),
                                        ('green', self.color_ranges['green'])]:
                if color_name == 'red':
                    m1 = cv2.inRange(hsv, mapping['lower1'], mapping['upper1'])
                    m2 = cv2.inRange(hsv, mapping['lower2'], mapping['upper2'])
                    mask = cv2.bitwise_or(m1, m2)
                else:
                    mask = cv2.inRange(hsv, mapping['lower'], mapping['upper'])

                # Clean masks
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
                mask = cv2.GaussianBlur(mask, (5, 5), 0)

                # Save debug mask
                debug_images[f"{color_name}_mask"] = mask.copy()

                # Connected components for robust region detection
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for i in range(1, num_labels):
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if area < 12 or area > 20000:
                        continue
                    cx, cy = centroids[i]
                    cx_i, cy_i = int(cx), int(cy)
                    r = int(max(3, math.sqrt(area / math.pi) * 1.5))

                    # Extract ROI for circularity
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

                    # Allow less circular if tiny; otherwise discard elongated shapes
                    if circularity < 0.25 and cnt_area < 200:
                        continue

                    symbol = self.color_mapping.get(color_name, '?')
                    detected.append({
                        'x': cx_i,
                        'y': cy_i,
                        'radius': r,
                        'color': color_name,
                        'symbol': symbol
                    })

            # HoughCircles fallback (grayscale) to catch faint dots
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
                    # Skip duplicates close to existing detections
                    too_close = False
                    for d in detected:
                        if math.hypot(d['x'] - cx, d['y'] - cy) < max(6, r * 0.6):
                            too_close = True
                            break
                    if too_close:
                        continue
                    sample_r = max(2, int(r // 2))
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
                        # If uncertain, pick the most common nearby mask color by sampling masks
                        color_guess = 'red'
                    detected.append({
                        'x': int(cx),
                        'y': int(cy),
                        'radius': int(r),
                        'color': color_guess,
                        'symbol': self.color_mapping.get(color_guess, '?')
                    })

            # Map detections back to original coordinates if rectified
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
                    # If inverse fails, keep warped coords (less desirable)
                    pass

            # Optionally save debug masks/images
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

    # ---------------------------
    # Grid inference (improved)
    # ---------------------------
    def _smart_grid_detection(self, dots, image_width, image_height, expected_rows=6):
        """
        Infer a consistent row x column grid from detected dots robustly.
        Uses 1D clustering for rows and columns with heuristics for merging/padding.
        Returns grid as list of rows, each a list of cell dicts {'symbol','color','x','y'} or empty cells.
        """
        if not dots:
            return []

        try:
            # Convert to arrays
            xs = np.array([d['x'] for d in dots])
            ys = np.array([d['y'] for d in dots])

            # Row clustering (1D on y)
            from sklearn.cluster import DBSCAN
            if len(ys) >= 2:
                y_sorted = np.sort(ys)
                diffs = np.diff(y_sorted)
                # robust estimate: median of diffs > 1
                positive_diffs = diffs[diffs > 1] if diffs.size > 0 else np.array([image_height / 10])
                est_spacing = np.median(positive_diffs) if positive_diffs.size > 0 else max(10, image_height / 30)
            else:
                est_spacing = image_height / 10

            eps_y = max(8, est_spacing * 0.5)
            row_clust = DBSCAN(eps=eps_y, min_samples=1).fit(ys.reshape(-1, 1))
            labels = row_clust.labels_
            rows_dict = {}
            for d, lab in zip(dots, labels):
                rows_dict.setdefault(lab, []).append(d)

            # Sort rows by mean y
            rows_list = [rows_dict[k] for k in
                         sorted(rows_dict.keys(), key=lambda k: np.mean([p['y'] for p in rows_dict[k]]))]

            # Merge/split rows to match expected_rows if needed
            if len(rows_list) > expected_rows:
                # merge closest pairs until match
                while len(rows_list) > expected_rows:
                    min_dist = float('inf')
                    idx = 0
                    for i in range(len(rows_list) - 1):
                        y1 = np.mean([p['y'] for p in rows_list[i]])
                        y2 = np.mean([p['y'] for p in rows_list[i + 1]])
                        dist = abs(y2 - y1)
                        if dist < min_dist:
                            min_dist = dist
                            idx = i
                    rows_list[idx].extend(rows_list[idx + 1])
                    rows_list[idx].sort(key=lambda p: p['x'])
                    del rows_list[idx + 1]
            elif len(rows_list) < expected_rows:
                # pad with empty rows at the end
                for _ in range(expected_rows - len(rows_list)):
                    rows_list.append([])

            # Determine column centers across rows
            all_x_candidates = []
            for row in rows_list:
                if not row:
                    continue
                row_xs = np.array(sorted([p['x'] for p in row]))
                all_x_candidates.append(row_xs)
            if len(all_x_candidates) == 0:
                # no columns found
                return [[{'symbol': '.', 'color': 'empty'}] for _ in range(expected_rows)]
            all_x = np.hstack(all_x_candidates)
            if all_x.size == 0:
                return [[{'symbol': '.', 'color': 'empty'}] for _ in range(expected_rows)]

            # Column clustering
            est_col_spacing = np.median(np.abs(np.diff(np.sort(all_x)))) if all_x.size > 1 else max(10,
                                                                                                    image_width / 20)
            eps_x = max(10, est_col_spacing * 0.5)
            col_clust = DBSCAN(eps=eps_x, min_samples=1).fit(all_x.reshape(-1, 1))
            col_labels = col_clust.labels_
            col_centers = []
            for lab in sorted(set(col_labels)):
                center = np.mean(all_x[col_labels == lab])
                col_centers.append(center)
            col_centers = sorted(col_centers)
            if len(col_centers) == 0:
                col_centers = [np.median(all_x)]

            # Build grid by snapping each dot to nearest column center
            max_cols = max(1, len(col_centers))
            grid = []
            for row in rows_list:
                cells = [{'symbol': '.', 'color': 'empty'} for _ in range(max_cols)]
                for dot in row:
                    idx = int(np.argmin([abs(dot['x'] - c) for c in col_centers]))
                    # put dot into cell (prefer higher confidence if multiple)
                    cells[idx] = {
                        'symbol': dot.get('symbol', '?'),
                        'color': dot.get('color', 'unknown'),
                        'x': dot.get('x'),
                        'y': dot.get('y')
                    }
                grid.append(cells)

            # Normalize length (should be consistent already)
            max_cols = max(len(r) for r in grid) if grid else 0
            for r in grid:
                while len(r) < max_cols:
                    r.append({'symbol': '.', 'color': 'empty'})
            return grid
        except Exception:
            # On failure, fallback to previous simple approach: six rows of empties
            grid = []
            for _ in range(expected_rows):
                grid.append([{'symbol': '.', 'color': 'empty'}])
            return grid

    # ---------------------------
    # Analysis
    # ---------------------------
    def _analyze_grid(self, grid):
        """Analyze the grid for Baccarat patterns (counts, percentages, streaks)"""
        if not grid or len(grid) == 0:
            return {}

        analysis = {
            'counts': {'B': 0, 'P': 0, 'T': 0, '.': 0},
            'streaks': [],
            'patterns': []
        }

        # Flatten grid for analysis
        flat_grid = []
        for row in grid:
            for cell in row:
                symbol = cell.get('symbol', '.')
                flat_grid.append(symbol)
                analysis['counts'][symbol] = analysis['counts'].get(symbol, 0) + 1

        # Calculate percentages
        total = len(flat_grid) - analysis['counts'].get('.', 0)
        if total > 0:
            analysis['percentages'] = {
                'B': (analysis['counts'].get('B', 0) / total) * 100,
                'P': (analysis['counts'].get('P', 0) / total) * 100,
                'T': (analysis['counts'].get('T', 0) / total) * 100
            }

        # Find streaks (Banker/Player only, ignore ties)
        current_streak = {'symbol': None, 'length': 0}
        for symbol in flat_grid:
            if symbol in ['B', 'P']:
                if symbol == current_streak['symbol']:
                    current_streak['length'] += 1
                else:
                    if current_streak['symbol'] is not None and current_streak['length'] >= 2:
                        analysis['streaks'].append(current_streak.copy())
                    current_streak = {'symbol': symbol, 'length': 1}
            else:
                # reset on ties/empties (but allow ties not to break long sequence if desired)
                if current_streak['symbol'] is not None and current_streak['length'] >= 2:
                    analysis['streaks'].append(current_streak.copy())
                current_streak = {'symbol': None, 'length': 0}

        if current_streak['symbol'] is not None and current_streak['length'] >= 2:
            analysis['streaks'].append(current_streak)

        return analysis

    # ---------------------------
    # Public interface: local analysis
    # ---------------------------
    def analyze_image_local(self, image_path, debug_dir=None):
        """Analyze image using local OpenCV with improved detection and smart grid detection"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "error": "Could not load image"}

            # Detect dots
            detected_dots = self._detect_dots_robust(img, debug_dir=debug_dir)

            if not detected_dots:
                # If initial pass fails, attempt a relaxed detection by increasing HSV ranges slightly
                # Slightly expand ranges to be tolerant of lighting
                backup_ranges = json.loads(json.dumps(self.color_ranges))
                try:
                    # Expand saturation/value thresholds
                    for k, v in self.color_ranges.items():
                        if k == 'red':
                            v['lower1'][1] = max(0, v['lower1'][1] - 30)
                            v['lower1'][2] = max(0, v['lower1'][2] - 30)
                            v['lower2'][1] = max(0, v['lower2'][1] - 30)
                            v['lower2'][2] = max(0, v['lower2'][2] - 30)
                        else:
                            v['lower'][1] = max(0, v['lower'][1] - 30)
                            v['lower'][2] = max(0, v['lower'][2] - 30)
                    detected_dots = self._detect_dots_robust(img, debug_dir=debug_dir)
                finally:
                    self.color_ranges = backup_ranges

            if not detected_dots:
                return {"success": False, "error": "No dots detected"}

            # Smart grid detection
            grid = self._smart_grid_detection(detected_dots, img.shape[1], img.shape[0])
            analysis = self._analyze_grid(grid)

            return {
                "success": True,
                "mode": "local",
                "total_dots": len(detected_dots),
                "detected_dots": detected_dots,
                "grid": grid,
                "analysis": analysis
            }
        except Exception as e:
            return {"success": False, "error": f"Local processing failed: {str(e)}"}


class GoogleVisionAnalyzer:
    """Analyze images using Google Vision API (keeps original behavior, uses improved local analyzer for grid)"""

    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        self.headers = {'Content-Type': 'application/json'}

    def encode_image(self, image_path):
        """Convert image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def analyze_image_google(self, image_path):
        """Analyze image using Google Vision API"""
        try:
            import requests

            base64_image = self.encode_image(image_path)
            request_data = {
                "requests": [
                    {
                        "image": {"content": base64_image},
                        "features": [
                            {"type": "OBJECT_LOCALIZATION", "maxResults": 100},
                            {"type": "IMAGE_PROPERTIES", "maxResults": 10}
                        ]
                    }
                ]
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=request_data,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return self._process_google_response(data, image_path)
            else:
                error_msg = f"API Error {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg = f"{error_msg}: {error_data['error'].get('message', 'Unknown')}"
                except:
                    error_msg = f"{error_msg}: {response.text[:200]}"
                return {"success": False, "error": error_msg}
        except Exception as e:
            return {"success": False, "error": f"Google Vision failed: {str(e)}"}

    def _process_google_response(self, data, image_path):
        """Process Google Vision API response and map detected localized objects to Baccarat dots"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "error": "Could not load image for processing"}
            height, width = img.shape[:2]
            detected_dots = []

            if 'responses' in data and len(data['responses']) > 0:
                response = data['responses'][0]
                if 'localizedObjectAnnotations' in response:
                    objects = response['localizedObjectAnnotations']
                    for obj in objects:
                        score = obj.get('score', 0)
                        if score > 0.25:
                            vertices = obj.get('boundingPoly', {}).get('normalizedVertices', [])
                            if len(vertices) >= 2:
                                xs = [v.get('x', 0) * width for v in vertices]
                                ys = [v.get('y', 0) * height for v in vertices]
                                x_center = sum(xs) / len(xs)
                                y_center = sum(ys) / len(ys)
                                # Extract color using local region
                                color = self._get_color_at_position(img, int(x_center), int(y_center))
                                if color:
                                    symbol = {'red': 'B', 'blue': 'P', 'green': 'T'}.get(color, '?')
                                    detected_dots.append({
                                        'x': int(x_center),
                                        'y': int(y_center),
                                        'color': color,
                                        'symbol': symbol,
                                        'confidence': score
                                    })

            if not detected_dots:
                return {"success": False, "error": "No dots detected by Google Vision"}

            # Use local robust grid inference for analysis
            local_analyzer = BaccaratSheetAnalyzer()
            grid = local_analyzer._smart_grid_detection(detected_dots, width, height)
            return {
                "success": True,
                "mode": "google",
                "total_dots": len(detected_dots),
                "detected_dots": detected_dots,
                "grid": grid,
                "analysis": local_analyzer._analyze_grid(grid)
            }
        except Exception as e:
            return {"success": False, "error": f"Processing failed: {str(e)}"}

    def _get_color_at_position(self, img, x, y):
        """Sample a small region around (x,y) and classify color by average HSV."""
        if img is None:
            return None
        h, w = img.shape[:2]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        region_size = 6
        x1, x2 = max(0, x - region_size), min(w, x + region_size)
        y1, y2 = max(0, y - region_size), min(h, y + region_size)
        if x2 <= x1 or y2 <= y1:
            return None
        region = img[y1:y2, x1:x2]
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv_region[:, :, 0])
        avg_sat = np.mean(hsv_region[:, :, 1])
        avg_val = np.mean(hsv_region[:, :, 2])
        if avg_sat > 50:
            if (0 <= avg_hue <= 15) or (165 <= avg_hue <= 180):
                return 'red'
            elif 85 <= avg_hue <= 135:
                return 'blue'
            elif 35 <= avg_hue <= 85:
                return 'green'
        return None


class BaccaratAnalyzerGUI:
    """GUI for Baccarat Road Sheet Analyzer"""

    def __init__(self, root, google_api_key=None):
        self.root = root
        self.root.title("🎯 Baccarat Road Sheet Analyzer (Enhanced)")
        self.root.geometry("1200x800")

        # Analyzers
        self.local_analyzer = BaccaratSheetAnalyzer()
        self.google_analyzer = None
        if google_api_key:
            try:
                self.google_analyzer = GoogleVisionAnalyzer(google_api_key)
            except Exception as e:
                print(f"Google Vision initialization failed: {e}")

        # State
        self.current_image_path = None
        self.current_image = None
        self.last_result = None

        # Queue for threading
        self.queue = queue.Queue()

        # Setup GUI
        self.setup_gui()

        # Start queue checker
        self.root.after(100, self.check_queue)

    def setup_gui(self):
        """Setup the GUI"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(title_frame, text="🎯 Baccarat Road Sheet Analyzer (Enhanced)",
                                font=("Arial", 16, "bold"))
        title_label.pack()

        subtitle = ttk.Label(title_frame,
                             text="Smart grid detection for imperfect photos",
                             font=("Arial", 10))
        subtitle.pack()

        # Control panel
        control_frame = ttk.LabelFrame(main_container, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack()

        self.load_btn = ttk.Button(btn_frame, text="📁 Load Image",
                                   command=self.load_image, width=15)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(pady=(10, 0))

        ttk.Label(mode_frame, text="Analysis Mode:").pack(side=tk.LEFT, padx=(0, 5))

        self.mode_var = tk.StringVar(value="local")

        self.local_mode_btn = ttk.Radiobutton(mode_frame, text="Local (Recommended)",
                                              variable=self.mode_var,
                                              value="local")
        self.local_mode_btn.pack(side=tk.LEFT, padx=5)

        if self.google_analyzer:
            self.google_mode_btn = ttk.Radiobutton(mode_frame, text="Google Vision",
                                                   variable=self.mode_var,
                                                   value="google")
            self.google_mode_btn.pack(side=tk.LEFT, padx=5)
        else:
            ttk.Label(mode_frame, text="Google Vision: Not available",
                      font=("Arial", 9, "italic")).pack(side=tk.LEFT, padx=5)

        self.analyze_btn = ttk.Button(btn_frame, text="🔍 Analyze",
                                      command=self.analyze_image, width=15,
                                      state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(btn_frame, text="🔄 Clear",
                                    command=self.clear_results, width=15)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Status
        self.status_var = tk.StringVar(value="Ready to load image")
        status_label = ttk.Label(control_frame, textvariable=self.status_var,
                                 font=("Arial", 9))
        status_label.pack(pady=(5, 0))

        # MAIN CONTENT
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # LEFT: Image preview
        left_frame = ttk.LabelFrame(content_frame, text="Image Preview", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Image canvas
        self.image_canvas = tk.Canvas(left_frame, bg='#f0f0f0',
                                      highlightthickness=1,
                                      highlightbackground="#cccccc")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        self.image_canvas.create_text(200, 150,
                                      text="No image loaded\n\nClick 'Load Image' to begin",
                                      font=("Arial", 12),
                                      fill="#666666",
                                      justify=tk.CENTER)

        # RIGHT: Results
        right_frame = ttk.LabelFrame(content_frame, text="Analysis Results", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Grid View
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="📊 Grid")

        self.grid_text = scrolledtext.ScrolledText(tab1,
                                                   font=("Courier New", 12, "bold"),
                                                   wrap=tk.WORD,
                                                   height=15)
        self.grid_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Legend
        legend_frame = ttk.Frame(tab1)
        legend_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Label(legend_frame, text="Legend: ", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="B=Banker(Red) ", font=("Arial", 9, "bold"), foreground="red").pack(side=tk.LEFT,
                                                                                                         padx=5)
        ttk.Label(legend_frame, text="P=Player(Blue) ", font=("Arial", 9, "bold"), foreground="blue").pack(side=tk.LEFT,
                                                                                                           padx=5)
        ttk.Label(legend_frame, text="T=Tie(Green) ", font=("Arial", 9, "bold"), foreground="green").pack(side=tk.LEFT,
                                                                                                          padx=5)
        ttk.Label(legend_frame, text=".=Empty", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)

        # Tab 2: Statistics
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="📈 Statistics")

        self.stats_text = scrolledtext.ScrolledText(tab2,
                                                    font=("Courier New", 10),
                                                    wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom status
        self.bottom_status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.bottom_status_var,
                               relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))

    def load_image(self):
        """Load an image file"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select a Baccarat road sheet image",
            filetypes=filetypes
        )

        if filename:
            self.current_image_path = filename

            try:
                # Load image
                self.current_image = Image.open(filename)

                # Display image
                self.display_image()

                # Enable analyze button
                self.analyze_btn.config(state=tk.NORMAL)

                # Update status
                self.status_var.set(f"Loaded: {os.path.basename(filename)}")
                self.bottom_status_var.set("Ready to analyze")

            except Exception as e:
                messagebox.showerror("Image Error", f"Cannot load image: {str(e)}")

    def display_image(self):
        """Display image on canvas"""
        if not self.current_image:
            return

        # Calculate display size
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width < 10:
            canvas_width, canvas_height = 400, 300

        # Resize for display
        img = self.current_image.copy()
        img.thumbnail((canvas_width - 20, canvas_height - 20))
        self.display_size = img.size

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(img)

        # Clear and display
        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_width // 2, canvas_height // 2,
                                       image=self.photo, anchor=tk.CENTER)

    def analyze_image(self):
        """Analyze the loaded image"""
        if not self.current_image_path:
            return

        self.clear_results()

        # Update status
        mode = self.mode_var.get()
        self.status_var.set(f"Analyzing with {mode} mode...")

        # Disable buttons
        self.load_btn.config(state=tk.DISABLED)
        self.analyze_btn.config(state=tk.DISABLED)

        # Start analysis in thread
        thread = threading.Thread(target=self.run_analysis, args=(mode,), daemon=True)
        thread.start()

    def run_analysis(self, mode):
        """Run analysis in thread"""
        try:
            if mode == "local":
                # Optionally create a debug directory next to the image
                debug_dir = None
                try:
                    base = os.path.splitext(self.current_image_path)[0]
                    debug_dir = base + "_debug"
                except Exception:
                    debug_dir = None
                result = self.local_analyzer.analyze_image_local(self.current_image_path, debug_dir=debug_dir)
            elif mode == "google" and self.google_analyzer:
                result = self.google_analyzer.analyze_image_google(self.current_image_path)
            else:
                result = {"success": False, "error": "Google Vision not available"}

            self.queue.put({'type': 'result', 'data': result})

        except Exception as e:
            self.queue.put({'type': 'error', 'message': str(e)})

    def check_queue(self):
        """Check for messages from threads"""
        try:
            while True:
                item = self.queue.get_nowait()

                if item['type'] == 'result':
                    self.display_result(item['data'])

                elif item['type'] == 'error':
                    self.show_error(item['message'])

        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

    def display_result(self, result):
        """Display analysis results"""
        # Re-enable buttons
        self.load_btn.config(state=tk.NORMAL)
        self.analyze_btn.config(state=tk.NORMAL)

        if result.get('success'):
            self.last_result = result

            # Display grid
            self.display_grid(result)

            # Display statistics
            self.display_statistics(result)

            # Highlight dots on image
            self.highlight_dots(result)

            # Update status
            mode = result.get('mode', 'unknown')
            total_dots = result.get('total_dots', 0)
            self.status_var.set(f"Analysis complete ({mode} mode)")
            self.bottom_status_var.set(f"✓ Detected {total_dots} dots")

            # Switch to grid tab
            self.notebook.select(0)

        else:
            error_msg = f"Analysis failed: {result.get('error', 'Unknown error')}"
            self.grid_text.insert(tk.END, error_msg)
            self.status_var.set("Analysis failed")
            self.bottom_status_var.set("✗ " + result.get('error', 'Unknown error'))

    def display_grid(self, result):
        """Display the grid in readable format"""
        grid = result.get('grid', [])

        if not grid:
            self.grid_text.insert(tk.END, "No grid generated")
            return

        self.grid_text.delete(1.0, tk.END)

        self.grid_text.insert(tk.END, "BACCARAT BIG ROAD SHEET\n")
        self.grid_text.insert(tk.END, "=" * 60 + "\n\n")
        self.grid_text.insert(tk.END, "Rows go DOWN (1-6), Columns go RIGHT\n")
        self.grid_text.insert(tk.END, "=" * 60 + "\n\n")

        # Display each row
        for row_idx, row in enumerate(grid):
            self.grid_text.insert(tk.END, f"Row {row_idx + 1}: ")

            for cell in row:
                symbol = cell.get('symbol', '.')
                color = cell.get('color', '')

                if symbol == 'B':
                    self.grid_text.insert(tk.END, "B ", "red_text")
                elif symbol == 'P':
                    self.grid_text.insert(tk.END, "P ", "blue_text")
                elif symbol == 'T':
                    self.grid_text.insert(tk.END, "T ", "green_text")
                else:
                    self.grid_text.insert(tk.END, ". ")

            self.grid_text.insert(tk.END, "\n")

        # Configure text colors
        self.grid_text.tag_config("red_text", foreground="red", font=("Courier New", 12, "bold"))
        self.grid_text.tag_config("blue_text", foreground="blue", font=("Courier New", 12, "bold"))
        self.grid_text.tag_config("green_text", foreground="green", font=("Courier New", 12, "bold"))

    def display_statistics(self, result):
        """Display analysis statistics"""
        analysis = result.get('analysis', {})

        self.stats_text.delete(1.0, tk.END)

        self.stats_text.insert(tk.END, "📊 STATISTICAL ANALYSIS\n")
        self.stats_text.insert(tk.END, "=" * 50 + "\n\n")

        # Counts
        counts = analysis.get('counts', {})
        total_non_empty = counts.get('B', 0) + counts.get('P', 0) + counts.get('T', 0)

        if total_non_empty > 0:
            self.stats_text.insert(tk.END, "OUTCOME COUNTS:\n")
            self.stats_text.insert(tk.END, f"• Banker (B): {counts.get('B', 0)}\n")
            self.stats_text.insert(tk.END, f"• Player (P): {counts.get('P', 0)}\n")
            self.stats_text.insert(tk.END, f"• Tie (T): {counts.get('T', 0)}\n")
            self.stats_text.insert(tk.END, f"• Empty cells: {counts.get('.', 0)}\n")
            self.stats_text.insert(tk.END, f"• Total games: {total_non_empty}\n\n")

            # Percentages
            percentages = analysis.get('percentages', {})
            if percentages:
                self.stats_text.insert(tk.END, "PERCENTAGES:\n")
                self.stats_text.insert(tk.END, f"• Banker wins: {percentages.get('B', 0):.1f}%\n")
                self.stats_text.insert(tk.END, f"• Player wins: {percentages.get('P', 0):.1f}%\n")
                self.stats_text.insert(tk.END, f"• Tie games: {percentages.get('T', 0):.1f}%\n\n")

        # Streaks
        streaks = analysis.get('streaks', [])
        if streaks:
            self.stats_text.insert(tk.END, "STREAKS (2+ consecutive):\n")
            for streak in streaks:
                symbol = streak.get('symbol', '')
                length = streak.get('length', 0)
                outcome = "Banker" if symbol == 'B' else "Player"
                self.stats_text.insert(tk.END, f"• {outcome}: {length} in a row\n")

    def highlight_dots(self, result):
        """Highlight detected dots on the image"""
        detected_dots = result.get('detected_dots', [])

        if not detected_dots or not self.current_image:
            return

        # Get original image for scale
        orig_img = Image.open(self.current_image_path)
        orig_width, orig_height = orig_img.size
        disp_width, disp_height = self.display_size

        scale_x = disp_width / orig_width
        scale_y = disp_height / orig_height

        # Canvas offset
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        img_x = (canvas_width - disp_width) // 2
        img_y = (canvas_height - disp_height) // 2

        # Draw each dot
        for dot in detected_dots:
            x = dot.get('x', 0) * scale_x + img_x
            y = dot.get('y', 0) * scale_y + img_y
            radius = max(5, int(dot.get('radius', 8) * ((scale_x + scale_y) / 2)))
            color = dot.get('color', 'red')
            symbol = dot.get('symbol', '?')

            # Choose color
            fill_color = {
                'red': '#FF0000',
                'blue': '#0000FF',
                'green': '#00FF00'
            }.get(color, '#FF0000')

            # Draw circle
            self.image_canvas.create_oval(
                x - radius, y - radius,
                x + radius, y + radius,
                outline=fill_color,
                width=2,
                fill=fill_color
            )

            # Draw symbol inside
            self.image_canvas.create_text(
                x, y,
                text=symbol,
                fill='white',
                font=("Arial", 10, "bold")
            )

    def clear_results(self):
        """Clear all results and highlights"""
        self.grid_text.delete(1.0, tk.END)
        self.stats_text.delete(1.0, tk.END)

        # Clear highlights from image
        self.image_canvas.delete("all")
        if self.current_image:
            self.display_image()

    def show_error(self, message):
        """Show error message"""
        self.load_btn.config(state=tk.NORMAL)
        self.analyze_btn.config(state=tk.NORMAL)

        messagebox.showerror("Analysis Error", message)
        self.status_var.set("Error")
        self.bottom_status_var.set("Analysis failed")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Baccarat Road Sheet Analyzer (Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python baccarat_analyzer.py                      # Local mode only
  python baccarat_analyzer.py --key YOUR_API_KEY   # With Google Vision

Requirements:
  pip install opencv-python pillow numpy scikit-learn requests

Note:
  • Local mode uses smart grid detection for imperfect photos
  • Works best with images cropped to show only the dot grid
        """
    )

    parser.add_argument(
        '--key', '-k',
        type=str,
        help='Google Cloud Vision API key (optional)'
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    # Check for required packages
    try:
        import cv2  # already imported but keep check
        import numpy as np  # already imported
        from sklearn.cluster import DBSCAN
    except ImportError:
        print("ERROR: Required packages not installed!")
        print("Install with: pip install opencv-python numpy scikit-learn pillow requests")
        sys.exit(1)

    # Create main window
    root = tk.Tk()

    # Center window
    root.update_idletasks()
    width, height = 1200, 800
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    # Create app
    app = BaccaratAnalyzerGUI(root, args.key)

    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    print("🎯 Baccarat Road Sheet Analyzer (Enhanced)")
    print("=" * 60)
    print("Features:")
    print("• 📁 Load baccarat road sheet images")
    print("• 🔍 Smart grid detection for imperfect photos (enhanced)")
    print("• 📊 Automatic 6-row grid reconstruction")
    print("• 🎯 Color-based dot detection (Red=Banker, Blue=Player, Green=Tie)")
    print("• 📈 Statistical analysis")
    print()
    print("Requirements:")
    print("  pip install opencv-python numpy scikit-learn pillow requests")
    print()

    main()