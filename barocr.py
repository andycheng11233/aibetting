#!/usr/bin/env python3
"""
🎯 BACCARRAT ROAD SHEET ANALYZER - IMPROVED VERSION
With color detection fixes and area threshold adjustment
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import cv2
import json
import os
import sys
import argparse
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import queue
import time
import base64
from collections import defaultdict


class BaccaratSheetAnalyzer:
    """Analyze Baccarat road sheets with IMPROVED color detection"""

    def __init__(self):
        # IMPROVED COLOR RANGES - Fixed blue range to catch violet-blue
        self.color_ranges = {
            'red': {
                'lower1': np.array([0, 100, 100]),  # Lower range for red
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 100, 100]),  # Red wraps around
                'upper2': np.array([180, 255, 255])
            },
            'blue': {
                'lower': np.array([85, 80, 80]),  # EXPANDED: More permissive blue
                'upper': np.array([140, 255, 255])  # Now includes violet-blue
            },
            'green': {
                'lower': np.array([40, 100, 100]),
                'upper': np.array([80, 255, 255])
            }
        }

        self.color_mapping = {
            'red': 'B',  # Banker
            'blue': 'P',  # Player
            'green': 'T'  # Tie
        }

        # Debug storage
        self.debug_info = {
            'steps': [],
            'contours_found': 0,
            'contours_accepted': 0,
            'rejection_reasons': defaultdict(int),
            'color_decisions': [],
            'grid_steps': [],
            'color_stats': defaultdict(int)
        }

    def analyze_image_local(self, image_path, debug_mode=False):
        """Analyze image using local OpenCV with IMPROVED color detection"""
        try:
            # Clear debug
            self.debug_info = {
                'steps': [],
                'contours_found': 0,
                'contours_accepted': 0,
                'rejection_reasons': defaultdict(int),
                'color_decisions': [],
                'grid_steps': [],
                'color_stats': defaultdict(int)
            }

            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "error": "Could not load image"}

            height, width = img.shape[:2]
            self.debug_info['steps'].append(f"Image loaded: {width}x{height}")

            # Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Create masks for each color
            masks = {}

            # Red mask
            red_lower1 = self.color_ranges['red']['lower1']
            red_upper1 = self.color_ranges['red']['upper1']
            red_lower2 = self.color_ranges['red']['lower2']
            red_upper2 = self.color_ranges['red']['upper2']

            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            masks['red'] = cv2.bitwise_or(mask_red1, mask_red2)

            # Blue mask (with improved range)
            masks['blue'] = cv2.inRange(hsv,
                                        self.color_ranges['blue']['lower'],
                                        self.color_ranges['blue']['upper'])

            # Green mask
            masks['green'] = cv2.inRange(hsv,
                                         self.color_ranges['green']['lower'],
                                         self.color_ranges['green']['upper'])

            # Combine all masks to find dots
            combined_mask = masks['red'] | masks['blue'] | masks['green']

            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.debug_info['contours_found'] = len(contours)
            self.debug_info['steps'].append(f"Found {len(contours)} contours")

            detected_dots = []

            for contour_idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                # FIXED: Lower area threshold from 50 to 30 to catch slightly smaller dots
                if 30 < area < 5000:  # CHANGED from 50 to 30
                    # Get bounding circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    radius = int(radius)

                    # FIXED: Lower circularity threshold from 0.5 to 0.4
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)

                        if circularity > 0.4:  # CHANGED from 0.5 to 0.4
                            # IMPROVED: Check a patch around the dot for color voting
                            patch_size = int(radius * 1.2)  # Check area slightly larger than dot
                            y1 = max(0, int(y) - patch_size)
                            y2 = min(height, int(y) + patch_size)
                            x1 = max(0, int(x) - patch_size)
                            x2 = min(width, int(x) + patch_size)

                            if y2 > y1 and x2 > x1:
                                # Count pixels of each color in the patch
                                red_count = np.sum(masks['red'][y1:y2, x1:x2] > 0)
                                blue_count = np.sum(masks['blue'][y1:y2, x1:x2] > 0)
                                green_count = np.sum(masks['green'][y1:y2, x1:x2] > 0)

                                # Debug color counts
                                if debug_mode:
                                    self.debug_info['color_stats'][f'contour_{contour_idx}_red'] = red_count
                                    self.debug_info['color_stats'][f'contour_{contour_idx}_blue'] = blue_count
                                    self.debug_info['color_stats'][f'contour_{contour_idx}_green'] = green_count

                                # Find dominant color with some minimum threshold
                                min_pixels = max(1, int(area * 0.3))  # At least 30% of dot area

                                best_color = None
                                max_count = 0

                                if red_count >= min_pixels and red_count > max_count:
                                    best_color = 'red'
                                    max_count = red_count
                                if blue_count >= min_pixels and blue_count > max_count:
                                    best_color = 'blue'
                                    max_count = blue_count
                                if green_count >= min_pixels and green_count > max_count:
                                    best_color = 'green'
                                    max_count = green_count

                                if best_color:
                                    detected_dots.append({
                                        'x': int(x),
                                        'y': int(y),
                                        'radius': radius,
                                        'color': best_color,
                                        'symbol': self.color_mapping[best_color],
                                        'contour_idx': contour_idx,
                                        'area': area,
                                        'circularity': circularity,
                                        'color_counts': {
                                            'red': red_count,
                                            'blue': blue_count,
                                            'green': green_count
                                        }
                                    })

                                    self.debug_info['contours_accepted'] += 1
                                    if debug_mode:
                                        self.debug_info['color_decisions'].append(
                                            f"Contour {contour_idx}: Accepted as {best_color} ({self.color_mapping[best_color]}) "
                                            f"at ({int(x)}, {int(y)}), area={area:.1f}, circularity={circularity:.2f}\n"
                                            f"       Color counts: R={red_count}, B={blue_count}, G={green_count}"
                                        )
                                else:
                                    self.debug_info['rejection_reasons']['no dominant color'] += 1
                                    if debug_mode:
                                        self.debug_info['color_decisions'].append(
                                            f"Contour {contour_idx}: Rejected - No dominant color found\n"
                                            f"       Color counts: R={red_count}, B={blue_count}, G={green_count}, min_required={min_pixels}"
                                        )
                            else:
                                self.debug_info['rejection_reasons']['patch out of bounds'] += 1
                        else:
                            self.debug_info['rejection_reasons'][f'circularity {circularity:.2f}'] += 1
                else:
                    self.debug_info['rejection_reasons'][f'area {area:.1f}'] += 1

            self.debug_info['steps'].append(f"Accepted {len(detected_dots)} out of {len(contours)} contours")

            if not detected_dots:
                return {"success": False, "error": "No dots detected", "debug": self.debug_info if debug_mode else None}

            # SMART GRID DETECTION (unchanged from working version)
            self.debug_info['grid_steps'].append(f"Starting grid detection with {len(detected_dots)} dots")
            grid = self._smart_grid_detection(detected_dots, width, height, debug_mode)

            # Analyze the grid
            analysis = self._analyze_grid(grid)

            # Calculate confidence based on original detection stats
            confidence = self._calculate_confidence(detected_dots, grid)

            return {
                "success": True,
                "mode": "local",
                "total_dots": len(detected_dots),
                "detected_dots": detected_dots,
                "grid": grid,
                "analysis": analysis,
                "debug": self.debug_info if debug_mode else None,
                "confidence": confidence
            }

        except Exception as e:
            self.debug_info['steps'].append(f"ERROR: {str(e)}")
            return {"success": False, "error": f"Local processing failed: {str(e)}",
                    "debug": self.debug_info if debug_mode else None}

    def _smart_grid_detection(self, dots, image_width, image_height, debug_mode=False):
        """Smart grid detection with noise filtering"""
        if not dots:
            self.debug_info['grid_steps'].append("No dots for grid detection")
            return []

        # STEP 1: Filter out obvious noise dots
        # Remove dots at the very top or bottom (likely not part of grid)
        filtered_dots = []

        # First pass: find the main cluster y-range
        y_values = [d['y'] for d in dots]
        if y_values:
            y_mean = np.mean(y_values)
            y_std = np.std(y_values)

            # Keep dots within 2 standard deviations of mean (95% of data if normal)
            for dot in dots:
                if abs(dot['y'] - y_mean) < 2 * y_std:
                    filtered_dots.append(dot)
                elif debug_mode:
                    self.debug_info['grid_steps'].append(
                        f"Filtered outlier dot at y={dot['y']} (mean={y_mean:.1f}, std={y_std:.1f})")

        if len(filtered_dots) < len(dots):
            self.debug_info['grid_steps'].append(
                f"After noise filtering: {len(filtered_dots)} of {len(dots)} dots kept")

        # Use filtered dots for grid detection
        if not filtered_dots:
            return []

        dots = filtered_dots

        # ========== ORIGINAL CODE FROM YOUR WORKING VERSION ==========
        # Sort dots by y-coordinate first
        dots.sort(key=lambda d: d['y'])

        # Find natural row breaks using clustering on y-coordinates
        y_coords = np.array([d['y'] for d in dots]).reshape(-1, 1)

        # Use DBSCAN for row clustering (handles varying distances)
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=10, min_samples=1).fit(y_coords)
            row_labels = clustering.labels_
            self.debug_info['grid_steps'].append(f"DBSCAN found {len(set(row_labels))} row clusters")
        except Exception as e:
            self.debug_info['grid_steps'].append(f"DBSCAN failed: {e}, using simple clustering")
            # Simple fallback
            row_labels = [0] * len(dots)
            for i in range(1, len(dots)):
                if dots[i]['y'] - dots[i - 1]['y'] > 15:
                    row_labels[i] = row_labels[i - 1] + 1
                else:
                    row_labels[i] = row_labels[i - 1]

        # Group dots by row
        rows_dict = {}
        for dot, label in zip(dots, row_labels):
            if label not in rows_dict:
                rows_dict[label] = []
            rows_dict[label].append(dot)

        # Sort rows by average y
        rows = []
        for label in sorted(rows_dict.keys(), key=lambda k: np.mean([d['y'] for d in rows_dict[k]])):
            rows.append(rows_dict[label])

        self.debug_info['grid_steps'].append(f"Initial row count: {len(rows)}")

        # Debug: Show each row's stats
        for i, row in enumerate(rows):
            if row:
                y_vals = [d['y'] for d in row]
                self.debug_info['grid_steps'].append(
                    f"Row {i + 1}: {len(row)} dots, y-range: {min(y_vals)}-{max(y_vals)}"
                )

        # Ensure we have exactly 6 rows (pad if necessary)
        if len(rows) < 6:
            self.debug_info['grid_steps'].append(f"Padding from {len(rows)} to 6 rows")
            # Pad with empty rows
            for i in range(len(rows), 6):
                rows.append([])
        elif len(rows) > 6:
            self.debug_info['grid_steps'].append(f"Merging from {len(rows)} to 6 rows")
            # Merge closest rows to get exactly 6
            while len(rows) > 6:
                # Find two closest rows by average y
                closest_pair = None
                min_distance = float('inf')
                for i in range(len(rows) - 1):
                    if rows[i] and rows[i + 1]:
                        y1 = np.mean([d['y'] for d in rows[i]])
                        y2 = np.mean([d['y'] for d in rows[i + 1]])
                        distance = abs(y1 - y2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_pair = (i, i + 1)

                if closest_pair:
                    # Merge the two closest rows
                    self.debug_info['grid_steps'].append(
                        f"Merging rows {closest_pair[0] + 1} and {closest_pair[1] + 1} (distance: {min_distance:.1f})"
                    )
                    rows[closest_pair[0]].extend(rows[closest_pair[1]])
                    rows[closest_pair[0]].sort(key=lambda d: d['x'])
                    del rows[closest_pair[1]]
                else:
                    # Just remove the last row if no valid pair found
                    rows.pop()

        # Now process each row
        grid = []
        max_cols = 0

        for row_idx, row_dots in enumerate(rows):
            self.debug_info['grid_steps'].append(f"Processing row {row_idx + 1}: {len(row_dots)} dots")

            if not row_dots:
                # Empty row
                grid.append([{'symbol': '.', 'color': 'empty'}])
                continue

            # Sort dots in this row by x-coordinate
            row_dots.sort(key=lambda d: d['x'])

            # Find natural column breaks in this row
            x_coords = np.array([d['x'] for d in row_dots]).reshape(-1, 1)
            try:
                from sklearn.cluster import DBSCAN
                clustering_x = DBSCAN(eps=15, min_samples=1).fit(x_coords)
                col_labels = clustering_x.labels_
                self.debug_info['grid_steps'].append(
                    f"Row {row_idx + 1}: DBSCAN found {len(set(col_labels))} column clusters")
            except Exception as e:
                self.debug_info['grid_steps'].append(f"Row {row_idx + 1}: DBSCAN failed: {e}, using simple clustering")
                # Simple fallback
                col_labels = [0] * len(row_dots)
                for i in range(1, len(row_dots)):
                    if row_dots[i]['x'] - row_dots[i - 1]['x'] > 15:
                        col_labels[i] = col_labels[i - 1] + 1
                    else:
                        col_labels[i] = col_labels[i - 1]

            # Group dots by column in this row
            cols_dict = {}
            for dot, label in zip(row_dots, col_labels):
                if label not in cols_dict:
                    cols_dict[label] = []
                cols_dict[label].append(dot)

            # Sort columns by average x
            sorted_cols = []
            for label in sorted(cols_dict.keys(), key=lambda k: np.mean([d['x'] for d in cols_dict[k]])):
                sorted_cols.append(cols_dict[label])

            self.debug_info['grid_steps'].append(f"Row {row_idx + 1}: Organized into {len(sorted_cols)} columns")

            # Process this row
            row_cells = []
            for col_idx, col_dots in enumerate(sorted_cols):
                if col_dots:
                    # Take the first dot in this column (should be only one)
                    dot = col_dots[0]
                    row_cells.append({
                        'symbol': dot['symbol'],
                        'color': dot['color'],
                        'x': dot['x'],
                        'y': dot['y']
                    })
                    self.debug_info['grid_steps'].append(
                        f"Row {row_idx + 1}, Col {col_idx + 1}: {dot['symbol']} at ({dot['x']}, {dot['y']})"
                    )
                else:
                    row_cells.append({'symbol': '.', 'color': 'empty'})

            max_cols = max(max_cols, len(row_cells))
            grid.append(row_cells)

        # Make all rows the same length
        for row_idx, row in enumerate(grid):
            while len(row) < max_cols:
                row.append({'symbol': '.', 'color': 'empty'})
            self.debug_info['grid_steps'].append(f"Row {row_idx + 1}: Final length = {len(row)} cells")

        self.debug_info['grid_steps'].append(f"Final grid: {len(grid)} rows x {max_cols} columns")

        return grid

    def _analyze_grid(self, grid):
        """Analyze the grid for Baccarat patterns"""
        analysis = {
            'counts': {'B': 0, 'P': 0, 'T': 0, '.': 0},
            'streaks': [],
            'patterns': []
        }

        if not grid or len(grid) == 0:
            return analysis

        # Flatten grid for analysis
        flat_grid = []
        for row in grid:
            for cell in row:
                symbol = cell.get('symbol', '.')
                flat_grid.append(symbol)
                analysis['counts'][symbol] = analysis['counts'].get(symbol, 0) + 1

        # Calculate percentages
        total = len(flat_grid) - analysis['counts']['.']
        if total > 0:
            analysis['percentages'] = {
                'B': (analysis['counts']['B'] / total) * 100,
                'P': (analysis['counts']['P'] / total) * 100,
                'T': (analysis['counts']['T'] / total) * 100
            }

        # Find streaks (Banker/Player only, ignore ties for streaks)
        current_streak = {'symbol': None, 'length': 0}
        for symbol in flat_grid:
            if symbol in ['B', 'P']:
                if symbol == current_streak['symbol']:
                    current_streak['length'] += 1
                else:
                    if current_streak['symbol'] is not None and current_streak['length'] >= 2:
                        analysis['streaks'].append(current_streak.copy())
                    current_streak = {'symbol': symbol, 'length': 1}

        # Add last streak
        if current_streak['symbol'] is not None and current_streak['length'] >= 2:
            analysis['streaks'].append(current_streak)

        return analysis

    def _calculate_confidence(self, detected_dots, grid):
        """Calculate confidence score based on detection quality"""
        if not detected_dots:
            return 0.0

        # Base confidence on detection rate
        detection_rate = len(detected_dots) / 30  # Assuming ~30 dots in a full sheet

        # Grid completeness
        total_cells = sum(len(row) for row in grid)
        filled_cells = sum(1 for row in grid for cell in row if cell.get('symbol', '.') != '.')
        completeness = filled_cells / total_cells if total_cells > 0 else 0

        # Row consistency
        if grid:
            row_lengths = [len(row) for row in grid]
            avg_length = np.mean(row_lengths)
            row_consistency = 1.0 - (np.std(row_lengths) / avg_length if avg_length > 0 else 1.0)
        else:
            row_consistency = 0.0

        # Overall confidence
        confidence = (
                detection_rate * 0.4 +
                completeness * 0.3 +
                row_consistency * 0.3
        )

        return min(max(confidence, 0), 1)

    def generate_debug_report(self, result):
        """Generate debug report for analysis"""
        if not result.get('debug'):
            return "No debug information available."

        debug = result['debug']
        report = []
        report.append("=" * 80)
        report.append("DEBUG REPORT - IMPROVED VERSION")
        report.append("=" * 80)

        # Basic info
        report.append(f"\n📊 BASIC INFORMATION:")
        report.append(f"  Success: {result.get('success', False)}")
        report.append(f"  Mode: {result.get('mode', 'unknown')}")
        report.append(f"  Total dots detected: {result.get('total_dots', 0)}")

        if 'confidence' in result:
            report.append(f"  Overall confidence: {result['confidence']:.1%}")

        # Steps
        if 'steps' in debug:
            report.append(f"\n🔍 PROCESSING STEPS:")
            for step in debug['steps']:
                report.append(f"  • {step}")

        # Contour statistics
        # Contour statistics
        if 'contours_found' in debug:
            accepted = debug.get('contours_accepted', 0)
            rejected = debug['contours_found'] - accepted

            report.append(f"\n📈 CONTOUR STATISTICS:")
            report.append(f"  Total contours: {debug['contours_found']}")  # FIXED HERE
            report.append(f"  Accepted: {accepted} ({accepted / debug['contours_found'] * 100:.1f}%)")
            report.append(f"  Rejected: {rejected} ({rejected / debug['contours_found'] * 100:.1f}%)")

            # Rejection reasons
            if debug['rejection_reasons']:
                report.append(f"\n  REASONS FOR REJECTION (top 10):")
                reasons = sorted(debug['rejection_reasons'].items(), key=lambda x: x[1], reverse=True)[:10]
                for reason, count in reasons:
                    report.append(f"    • {reason}: {count}")

        # Color decisions
        if 'color_decisions' in debug and debug['color_decisions']:
            report.append(f"\n🎨 COLOR DECISIONS (first 15):")
            for i, decision in enumerate(debug['color_decisions'][:15]):
                report.append(f"  {i + 1}. {decision}")
            if len(debug['color_decisions']) > 15:
                report.append(f"  ... and {len(debug['color_decisions']) - 15} more")

        # Grid steps
        if 'grid_steps' in debug and debug['grid_steps']:
            report.append(f"\n🏗️ GRID CONSTRUCTION (key steps):")
            key_steps = [s for s in debug['grid_steps'] if
                         'Row' in s or 'Final' in s or 'DBSCAN' in s or 'Merging' in s]
            for step in key_steps[:20]:
                report.append(f"  • {step}")

        # Analysis statistics
        if result.get('analysis'):
            analysis = result['analysis']
            report.append(f"\n📊 ANALYSIS STATISTICS:")
            report.append(f"  Banker (B): {analysis['counts'].get('B', 0)}")
            report.append(f"  Player (P): {analysis['counts'].get('P', 0)}")
            report.append(f"  Tie (T): {analysis['counts'].get('T', 0)}")
            report.append(f"  Empty (.): {analysis['counts'].get('.', 0)}")

            if 'percentages' in analysis:
                report.append(f"\n  PERCENTAGES:")
                report.append(f"    Banker: {analysis['percentages'].get('B', 0):.1f}%")
                report.append(f"    Player: {analysis['percentages'].get('P', 0):.1f}%")
                report.append(f"    Tie: {analysis['percentages'].get('T', 0):.1f}%")

            if 'streaks' in analysis and analysis['streaks']:
                report.append(f"\n  STREAKS:")
                for streak in analysis['streaks']:
                    outcome = "Banker" if streak.get('symbol') == 'B' else "Player"
                    report.append(f"    {outcome}: {streak.get('length', 0)} in a row")

        report.append("\n" + "=" * 80)
        report.append("IMPROVEMENTS APPLIED:")
        report.append("1. Blue HSV range expanded: [85,80,80] to [140,255,255]")
        report.append("2. Area threshold lowered: 30 instead of 50")
        report.append("3. Circularity threshold lowered: 0.4 instead of 0.5")
        report.append("4. Improved color voting: Patch-based majority vote")
        report.append("=" * 80)

        return "\n".join(report)


class ImprovedBaccaratGUI:
    """GUI for the improved analyzer"""

    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Baccarat Road Sheet Analyzer - IMPROVED VERSION")
        self.root.geometry("1200x800")

        # Analyzer
        self.analyzer = BaccaratSheetAnalyzer()

        # State
        self.current_image_path = None
        self.current_image = None
        self.last_result = None
        self.debug_mode = tk.BooleanVar(value=True)

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(title_frame, text="🎯 Baccarat Road Sheet Analyzer - IMPROVED VERSION",
                                font=("Arial", 16, "bold"))
        title_label.pack()

        subtitle = ttk.Label(title_frame,
                             text="With color detection fixes: Expanded blue range, better color voting",
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

        self.analyze_btn = ttk.Button(btn_frame, text="🔍 Analyze",
                                      command=self.analyze_image, width=15,
                                      state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        self.debug_btn = ttk.Button(btn_frame, text="🐛 Debug",
                                    command=self.show_debug, width=15,
                                    state=tk.DISABLED)
        self.debug_btn.pack(side=tk.LEFT, padx=5)

        self.export_btn = ttk.Button(btn_frame, text="💾 Export Results",
                                     command=self.export_results, width=15,
                                     state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5)

        # Debug toggle
        debug_frame = ttk.Frame(control_frame)
        debug_frame.pack(pady=(10, 0))

        self.debug_cb = ttk.Checkbutton(debug_frame, text="Enable Debug Mode",
                                        variable=self.debug_mode)
        self.debug_cb.pack(side=tk.LEFT)

        # Status
        self.status_var = tk.StringVar(value="Ready to load image")
        status_label = ttk.Label(control_frame, textvariable=self.status_var,
                                 font=("Arial", 9))
        status_label.pack(pady=(5, 0))

        # Confidence
        self.confidence_var = tk.StringVar(value="Confidence: --")
        confidence_label = ttk.Label(control_frame, textvariable=self.confidence_var,
                                     font=("Arial", 9, "bold"))
        confidence_label.pack()

        # Main content
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Image preview
        left_frame = ttk.LabelFrame(content_frame, text="Image Preview", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_canvas = tk.Canvas(left_frame, bg='#f0f0f0',
                                      highlightthickness=1,
                                      highlightbackground="#cccccc")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        self.image_canvas.create_text(200, 150,
                                      text="No image loaded\n\nClick 'Load Image' to begin",
                                      font=("Arial", 12),
                                      fill="#666666",
                                      justify=tk.CENTER)

        # Right: Results
        right_frame = ttk.LabelFrame(content_frame, text="Results", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Grid
        grid_tab = ttk.Frame(self.notebook)
        self.notebook.add(grid_tab, text="📊 Grid")

        self.grid_text = scrolledtext.ScrolledText(grid_tab,
                                                   font=("Courier New", 12, "bold"),
                                                   wrap=tk.WORD,
                                                   height=15)
        self.grid_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Legend
        legend_frame = ttk.Frame(grid_tab)
        legend_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Label(legend_frame, text="Legend: ", font=("Arial", 9)).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="B=Banker(Red) ",
                  font=("Arial", 9), foreground="red").pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="P=Player(Blue) ",
                  font=("Arial", 9), foreground="blue").pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="T=Tie(Green) ",
                  font=("Arial", 9), foreground="green").pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text=".=Empty",
                  font=("Arial", 9)).pack(side=tk.LEFT, padx=5)

        # Tab 2: Statistics
        stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(stats_tab, text="📈 Statistics")

        self.stats_text = scrolledtext.ScrolledText(stats_tab,
                                                    font=("Courier New", 10),
                                                    wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 3: Debug
        debug_tab = ttk.Frame(self.notebook)
        self.notebook.add(debug_tab, text="🐛 Debug")

        self.debug_text = scrolledtext.ScrolledText(debug_tab,
                                                    font=("Courier New", 9),
                                                    wrap=tk.WORD)
        self.debug_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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
                self.debug_btn.config(state=tk.DISABLED)
                self.export_btn.config(state=tk.DISABLED)

                # Update status
                self.status_var.set(f"Loaded: {os.path.basename(filename)}")
                self.bottom_status_var.set("Ready to analyze")
                self.confidence_var.set("Confidence: --")

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
        self.status_var.set("Analyzing image...")
        self.bottom_status_var.set("Processing...")

        # Disable buttons during analysis
        self.load_btn.config(state=tk.DISABLED)
        self.analyze_btn.config(state=tk.DISABLED)

        # Run analysis in thread
        thread = threading.Thread(
            target=self.run_analysis,
            daemon=True
        )
        thread.start()

    def run_analysis(self):
        """Run analysis in background thread"""
        try:
            result = self.analyzer.analyze_image_local(
                self.current_image_path,
                self.debug_mode.get()
            )

            # Schedule result display in main thread
            self.root.after(0, lambda: self.display_result(result))

        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Analysis failed: {str(e)}"))

    def display_result(self, result):
        """Display analysis results"""
        # Re-enable buttons
        self.load_btn.config(state=tk.NORMAL)
        self.analyze_btn.config(state=tk.NORMAL)
        self.debug_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)

        if result.get('success'):
            self.last_result = result

            # Display grid
            self.display_grid(result)

            # Display statistics
            self.display_statistics(result)

            # Display debug info if available
            if self.debug_mode.get():
                debug_report = self.analyzer.generate_debug_report(result)
                self.debug_text.delete(1.0, tk.END)
                self.debug_text.insert(tk.END, debug_report)

            # Highlight dots on image
            self.highlight_dots(result)

            # Update confidence display
            confidence = result.get('confidence', 0)
            self.confidence_var.set(f"Confidence: {confidence:.1%}")

            # Update status
            total_dots = result.get('total_dots', 0)
            self.status_var.set("Analysis complete")
            self.bottom_status_var.set(
                f"✓ Detected {total_dots} dots with {confidence:.1%} confidence"
            )

            # Switch to grid tab
            self.notebook.select(0)

        else:
            error_msg = result.get('error', 'Unknown error')
            self.grid_text.insert(tk.END, f"ERROR: {error_msg}")
            self.status_var.set("Analysis failed")
            self.bottom_status_var.set(f"✗ {error_msg}")
            self.confidence_var.set("Confidence: 0%")

    def display_grid(self, result):
        """Display the grid in readable format"""
        grid = result.get('grid', [])

        if not grid:
            self.grid_text.insert(tk.END, "No grid generated")
            return

        self.grid_text.delete(1.0, tk.END)

        # Header with confidence
        confidence = result.get('confidence', 0)
        self.grid_text.insert(tk.END, f"BACCARAT BIG ROAD SHEET (IMPROVED VERSION)\n")
        self.grid_text.insert(tk.END, f"Confidence: {confidence:.1%}\n")
        self.grid_text.insert(tk.END, "=" * 60 + "\n\n")
        self.grid_text.insert(tk.END, "Rows go DOWN (1-6), Columns go RIGHT\n")
        self.grid_text.insert(tk.END, "=" * 60 + "\n\n")

        # Display each row
        for row_idx, row in enumerate(grid):
            self.grid_text.insert(tk.END, f"Row {row_idx + 1}: ")

            for cell in row:
                symbol = cell.get('symbol', '.')

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
        self.grid_text.tag_config("red_text", foreground="red",
                                  font=("Courier New", 12, "bold"))
        self.grid_text.tag_config("blue_text", foreground="blue",
                                  font=("Courier New", 12, "bold"))
        self.grid_text.tag_config("green_text", foreground="green",
                                  font=("Courier New", 12, "bold"))

    def display_statistics(self, result):
        """Display analysis statistics"""
        analysis = result['analysis']

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
                self.stats_text.insert(tk.END,
                                       f"• Banker wins: {percentages.get('B', 0):.1f}%\n")
                self.stats_text.insert(tk.END,
                                       f"• Player wins: {percentages.get('P', 0):.1f}%\n")
                self.stats_text.insert(tk.END,
                                       f"• Tie games: {percentages.get('T', 0):.1f}%\n\n")

            # Streaks
            streaks = analysis.get('streaks', [])
            if streaks:
                self.stats_text.insert(tk.END, "STREAKS (2+ consecutive):\n")
                for streak in streaks:
                    outcome = "Banker" if streak.get('symbol') == 'B' else "Player"
                    self.stats_text.insert(tk.END,
                                           f"• {outcome}: {streak.get('length', 0)} in a row\n")

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
            radius = max(5, dot.get('radius', 10) * scale_x)
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

    def show_debug(self):
        """Show debug information"""
        if not self.last_result:
            return

        # Switch to debug tab
        self.notebook.select(2)  # Debug tab

    def export_results(self):
        """Export results to file"""
        if not self.last_result:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="baccarat_results.txt"
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    # Write grid
                    f.write("BACCARAT BIG ROAD SHEET (IMPROVED VERSION)\n")
                    f.write("=" * 60 + "\n\n")

                    if self.grid_text.get(1.0, tk.END).strip():
                        f.write(self.grid_text.get(1.0, tk.END))

                    # Write statistics
                    f.write("\n" + "=" * 60 + "\n")
                    f.write("STATISTICS\n")
                    f.write("=" * 60 + "\n\n")

                    if self.stats_text.get(1.0, tk.END).strip():
                        f.write(self.stats_text.get(1.0, tk.END))

                    # Write debug if available
                    if self.debug_mode.get() and self.debug_text.get(1.0, tk.END).strip():
                        f.write("\n" + "=" * 60 + "\n")
                        f.write("DEBUG INFORMATION\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(self.debug_text.get(1.0, tk.END))

                self.bottom_status_var.set(f"Results saved to {filename}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Cannot save file: {str(e)}")

    def clear_results(self):
        """Clear all results and highlights"""
        self.grid_text.delete(1.0, tk.END)
        self.stats_text.delete(1.0, tk.END)
        self.debug_text.delete(1.0, tk.END)

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
        description="Baccarat Road Sheet Analyzer - Improved Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python baccarat_improved.py

Improvements Applied:
  1. Blue HSV range expanded: [85,80,80] to [140,255,255]
  2. Area threshold lowered: 30 instead of 50
  3. Circularity threshold lowered: 0.4 instead of 0.5
  4. Improved color voting: Patch-based majority vote

Requirements:
  pip install opencv-python pillow numpy scikit-learn
        """
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    # Check for required packages
    try:
        import cv2
        import numpy as np
        from sklearn.cluster import DBSCAN
    except ImportError:
        print("ERROR: Required packages not installed!")
        print("Install with: pip install opencv-python numpy scikit-learn pillow")
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
    app = ImprovedBaccaratGUI(root)

    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    print("🎯 Baccarat Road Sheet Analyzer - IMPROVED VERSION")
    print("=" * 60)
    print("IMPROVEMENTS APPLIED:")
    print("1. Blue HSV range expanded: [85,80,80] to [140,255,255]")
    print("2. Area threshold lowered: 30 instead of 50")
    print("3. Circularity threshold lowered: 0.4 instead of 0.5")
    print("4. Improved color voting: Patch-based majority vote")
    print()
    print("Expected fixes:")
    print("  • Fix color misclassification (blue vs red dots)")
    print("  • Capture slightly smaller dots")
    print("  • More robust color detection")
    print()
    print("Requirements:")
    print("  pip install opencv-python numpy scikit-learn pillow")
    print()

    main()