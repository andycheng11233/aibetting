#!/usr/bin/env python3
"""
🔍 BACCARAT PHOTO DIAGNOSTIC ANALYZER WITH MENU
Analyzes photo properties to understand why some work and others don't
"""
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import sys
from collections import defaultdict


class BaccaratPhotoDiagnostic:
    """Diagnose why photos work or don't work"""

    def __init__(self):
        # Store stats from successful photos
        self.success_stats = {
            'brightness_range': (100, 180),
            'contrast_range': (30, 100),
            'sharpness_range': (150, 500),
            'color_stats': defaultdict(list),
            'dot_stats': defaultdict(list),
            'grid_stats': defaultdict(list)
        }

    def analyze_photo_properties(self, image_path):
        """Comprehensive analysis of photo properties"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not load image"}

            height, width = img.shape[:2]

            # Convert color spaces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Basic properties
            properties = {
                'dimensions': {'width': width, 'height': height},
                'basic_stats': self._get_basic_stats(gray, hsv),
                'color_analysis': self._analyze_colors(hsv),
                'dot_detection': self._detect_dots(img),
                'grid_analysis': self._analyze_grid_structure(img),
                'focus_quality': self._analyze_focus(gray),
                'lighting_analysis': self._analyze_lighting(gray, hsv),
                'potential_issues': []
            }

            # Identify potential issues
            properties['potential_issues'] = self._identify_issues(properties)

            return {"success": True, "properties": properties}

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _get_basic_stats(self, gray, hsv):
        """Get basic image statistics"""
        stats = {
            'brightness': float(np.mean(gray)),
            'brightness_std': float(np.std(gray)),
            'contrast': float(np.std(gray)),
            'hue_mean': float(np.mean(hsv[:, :, 0])),
            'hue_std': float(np.std(hsv[:, :, 0])),
            'saturation_mean': float(np.mean(hsv[:, :, 1])),
            'saturation_std': float(np.std(hsv[:, :, 1])),
            'value_mean': float(np.mean(hsv[:, :, 2])),
            'value_std': float(np.std(hsv[:, :, 2]))
        }
        return stats

    def _analyze_colors(self, hsv):
        """Analyze color distribution for Baccarat dots"""
        h, s, v = cv2.split(hsv)

        # Define color ranges (using our successful parameters)
        color_ranges = {
            'red_low': (0, 10),
            'red_high': (160, 180),
            'blue': (85, 140),
            'green': (40, 80)
        }

        color_stats = {}
        for color_name, (hue_low, hue_high) in color_ranges.items():
            # Create mask for this color range
            if 'red' in color_name:
                mask1 = cv2.inRange(h, color_ranges['red_low'][0], color_ranges['red_low'][1])
                mask2 = cv2.inRange(h, color_ranges['red_high'][0], color_ranges['red_high'][1])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(h, hue_low, hue_high)

            # Apply saturation and value thresholds
            sat_mask = cv2.inRange(s, 80, 255)
            val_mask = cv2.inRange(v, 80, 255)
            mask = cv2.bitwise_and(mask, sat_mask)
            mask = cv2.bitwise_and(mask, val_mask)

            # Calculate coverage
            coverage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
            color_stats[color_name] = {
                'coverage': float(coverage),
                'pixel_count': int(np.sum(mask > 0))
            }

        return color_stats

    def _detect_dots(self, img):
        """Detect and analyze dots"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)

        # Try different detection methods
        methods = []

        # Method 1: Hough Circles
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=30
        )

        circle_count = 0
        if circles is not None:
            circle_count = len(circles[0])
            methods.append({
                'method': 'hough_circles',
                'count': circle_count,
                'params': {'dp': 1, 'minDist': 20, 'param1': 50, 'param2': 30}
            })

        # Method 2: Simple thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size and circularity
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 5000:  # Our successful range
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.4:  # Our successful threshold
                        valid_contours.append({
                            'area': float(area),
                            'circularity': float(circularity)
                        })

        methods.append({
            'method': 'threshold_contours',
            'count': len(valid_contours),
            'avg_area': np.mean([c['area'] for c in valid_contours]) if valid_contours else 0,
            'avg_circularity': np.mean([c['circularity'] for c in valid_contours]) if valid_contours else 0
        })

        # Method 3: Our actual color-based detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red mask
        mask_red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Blue mask (our successful expanded range)
        mask_blue = cv2.inRange(hsv, np.array([85, 80, 80]), np.array([140, 255, 255]))

        # Green mask
        mask_green = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([80, 255, 255]))

        # Combine masks
        combined_mask = mask_red | mask_blue | mask_green

        # Find contours in combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color_dots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 5000:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.4:
                        color_dots.append({
                            'area': float(area),
                            'circularity': float(circularity)
                        })

        methods.append({
            'method': 'color_based',
            'count': len(color_dots),
            'avg_area': np.mean([c['area'] for c in color_dots]) if color_dots else 0,
            'avg_circularity': np.mean([c['circularity'] for c in color_dots]) if color_dots else 0
        })

        return {
            'detection_methods': methods,
            'best_estimate': max(m['count'] for m in methods),
            'expected_range': (25, 31)  # 28±3 dots
        }

    def _analyze_grid_structure(self, img):
        """Analyze grid-like structure"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=10)

        horizontal_angles = []
        vertical_angles = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                if abs(angle) < 20:  # Horizontal-ish
                    horizontal_angles.append(angle)
                elif abs(angle - 90) < 20:  # Vertical-ish
                    vertical_angles.append(angle)

        # Calculate alignment metrics
        h_alignment = 1.0 - (np.std(horizontal_angles) / 10 if horizontal_angles else 1.0)
        v_alignment = 1.0 - (np.std(vertical_angles) / 10 if vertical_angles else 1.0)

        return {
            'horizontal_lines': len(horizontal_angles),
            'vertical_lines': len(vertical_angles),
            'horizontal_alignment': float(h_alignment),
            'vertical_alignment': float(v_alignment),
            'overall_alignment': float((h_alignment + v_alignment) / 2)
        }

    def _analyze_focus(self, gray):
        """Analyze image focus/sharpness"""
        # Laplacian variance method
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Edge density method
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        return {
            'laplacian_variance': float(laplacian_var),
            'edge_density': float(edge_density),
            'sharpness_category': self._categorize_sharpness(laplacian_var)
        }

    def _categorize_sharpness(self, laplacian_var):
        """Categorize sharpness level"""
        if laplacian_var > 200:
            return "Very Sharp"
        elif laplacian_var > 100:
            return "Sharp"
        elif laplacian_var > 50:
            return "Acceptable"
        else:
            return "Blurry"

    def _analyze_lighting(self, gray, hsv):
        """Analyze lighting conditions"""
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # Check for shadows (variance across regions)
        h, w = gray.shape
        regions = []
        for i in range(3):
            for j in range(3):
                y1, y2 = i * h // 3, (i + 1) * h // 3
                x1, x2 = j * w // 3, (j + 1) * w // 3
                region_mean = np.mean(gray[y1:y2, x1:x2])
                regions.append(region_mean)

        region_variance = np.std(regions) / np.mean(regions) if np.mean(regions) > 0 else 0

        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'uniformity': float(1.0 - min(1.0, region_variance)),
            'lighting_category': self._categorize_lighting(brightness, contrast)
        }

    def _categorize_lighting(self, brightness, contrast):
        """Categorize lighting conditions"""
        if brightness < 50:
            return "Too Dark"
        elif brightness > 200:
            return "Too Bright"
        elif contrast < 20:
            return "Low Contrast"
        elif 50 <= brightness <= 180 and contrast >= 30:
            return "Optimal"
        else:
            return "Acceptable"

    def _identify_issues(self, properties):
        """Identify potential issues that could affect detection"""
        issues = []
        stats = properties['basic_stats']

        # Brightness issues
        if stats['brightness'] < 50:
            issues.append(f"⚠️ Too dark (brightness: {stats['brightness']:.1f})")
        elif stats['brightness'] > 200:
            issues.append(f"⚠️ Too bright (brightness: {stats['brightness']:.1f})")

        # Contrast issues
        if stats['contrast'] < 20:
            issues.append(f"⚠️ Low contrast (std: {stats['contrast']:.1f})")

        # Focus issues
        focus = properties['focus_quality']
        if focus['sharpness_category'] in ["Blurry", "Acceptable"]:
            issues.append(f"⚠️ {focus['sharpness_category'].lower()} (variance: {focus['laplacian_variance']:.1f})")

        # Color saturation issues
        if stats['saturation_mean'] < 50:
            issues.append(f"⚠️ Low color saturation (mean: {stats['saturation_mean']:.1f})")

        # Dot count issues
        dot_analysis = properties['dot_detection']
        expected_min, expected_max = dot_analysis['expected_range']
        best_count = dot_analysis['best_estimate']

        if best_count < expected_min:
            issues.append(f"⚠️ Too few dots detected ({best_count} < {expected_min})")
        elif best_count > expected_max:
            issues.append(f"⚠️ Too many dots detected ({best_count} > {expected_max}) - possible noise")

        # Grid alignment issues
        grid = properties['grid_analysis']
        if grid['overall_alignment'] < 0.6:
            issues.append(f"⚠️ Poor grid alignment (score: {grid['overall_alignment']:.2f})")

        return issues

    def compare_photos(self, working_path, non_working_path):
        """Compare a working photo with a non-working one"""
        working = self.analyze_photo_properties(working_path)
        non_working = self.analyze_photo_properties(non_working_path)

        if not working.get('success') or not non_working.get('success'):
            return {"error": "Could not analyze one or both photos"}

        comparison = {
            'differences': [],
            'working_issues': working['properties']['potential_issues'],
            'non_working_issues': non_working['properties']['potential_issues'],
            'key_metrics': {}
        }

        # Compare key metrics
        w_props = working['properties']
        n_props = non_working['properties']

        metrics = [
            ('brightness', w_props['basic_stats']['brightness'], n_props['basic_stats']['brightness']),
            ('contrast', w_props['basic_stats']['contrast'], n_props['basic_stats']['contrast']),
            ('focus', w_props['focus_quality']['laplacian_variance'], n_props['focus_quality']['laplacian_variance']),
            ('saturation', w_props['basic_stats']['saturation_mean'], n_props['basic_stats']['saturation_mean']),
            ('dots_detected', w_props['dot_detection']['best_estimate'], n_props['dot_detection']['best_estimate']),
            ('grid_alignment', w_props['grid_analysis']['overall_alignment'],
             n_props['grid_analysis']['overall_alignment'])
        ]

        for metric_name, w_val, n_val in metrics:
            diff_pct = abs(w_val - n_val) / max(w_val, 1) * 100
            comparison['key_metrics'][metric_name] = {
                'working': w_val,
                'non_working': n_val,
                'difference_pct': diff_pct,
                'significant': diff_pct > 30
            }

            if diff_pct > 30:
                comparison['differences'].append(
                    f"📊 {metric_name}: Working={w_val:.1f}, Non-working={n_val:.1f} ({diff_pct:.1f}% diff)"
                )

        # Identify the most likely cause
        if comparison['differences']:
            # Find the biggest difference
            max_diff = max(comparison['key_metrics'].values(),
                           key=lambda x: x['difference_pct'])
            comparison[
                'most_likely_issue'] = f"Biggest difference in {list(comparison['key_metrics'].keys())[list(comparison['key_metrics'].values()).index(max_diff)]}"

        return {"success": True, "comparison": comparison}


class DiagnosticGUI:
    """GUI for the diagnostic analyzer"""

    def __init__(self, root):
        self.root = root
        self.root.title("🔍 Baccarat Photo Diagnostic Analyzer")
        self.root.geometry("1200x800")

        # Diagnostic analyzer
        self.diagnostic = BaccaratPhotoDiagnostic()

        # State
        self.working_photo_path = None
        self.non_working_photo_path = None
        self.current_photo_path = None
        self.current_photo = None

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

        title_label = ttk.Label(title_frame,
                                text="🔍 BACCARAT PHOTO DIAGNOSTIC ANALYZER",
                                font=("Arial", 16, "bold"))
        title_label.pack()

        subtitle = ttk.Label(title_frame,
                             text="Analyze why some photos work and others don't",
                             font=("Arial", 10))
        subtitle.pack()

        # Control panel
        control_frame = ttk.LabelFrame(main_container, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Photo selection buttons
        selection_frame = ttk.Frame(control_frame)
        selection_frame.pack()

        self.working_btn = ttk.Button(selection_frame, text="📷 Select Working Photo",
                                      command=lambda: self.select_photo('working'), width=20)
        self.working_btn.pack(side=tk.LEFT, padx=5)

        self.non_working_btn = ttk.Button(selection_frame, text="📷 Select Non-Working Photo",
                                          command=lambda: self.select_photo('non_working'), width=20)
        self.non_working_btn.pack(side=tk.LEFT, padx=5)

        self.single_btn = ttk.Button(selection_frame, text="📷 Analyze Single Photo",
                                     command=lambda: self.select_photo('single'), width=20)
        self.single_btn.pack(side=tk.LEFT, padx=5)

        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(pady=(10, 0))

        self.analyze_single_btn = ttk.Button(action_frame, text="🔍 Analyze Single",
                                             command=self.analyze_single, width=15,
                                             state=tk.DISABLED)
        self.analyze_single_btn.pack(side=tk.LEFT, padx=5)

        self.compare_btn = ttk.Button(action_frame, text="🔄 Compare Photos",
                                      command=self.compare_photos, width=15,
                                      state=tk.DISABLED)
        self.compare_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(action_frame, text="🗑️ Clear All",
                                    command=self.clear_all, width=15)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Status indicators
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(pady=(10, 0))

        self.working_status = tk.StringVar(value="No working photo selected")
        ttk.Label(status_frame, textvariable=self.working_status,
                  font=("Arial", 9)).pack(side=tk.LEFT, padx=10)

        self.non_working_status = tk.StringVar(value="No non-working photo selected")
        ttk.Label(status_frame, textvariable=self.non_working_status,
                  font=("Arial", 9)).pack(side=tk.LEFT, padx=10)

        # Main content
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Photo preview
        left_frame = ttk.LabelFrame(content_frame, text="Photo Preview", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_canvas = tk.Canvas(left_frame, bg='#f0f0f0',
                                      highlightthickness=1,
                                      highlightbackground="#cccccc")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        self.image_canvas.create_text(200, 150,
                                      text="No photo loaded\n\nSelect a photo to begin",
                                      font=("Arial", 12),
                                      fill="#666666",
                                      justify=tk.CENTER)

        # Photo info
        self.photo_info_var = tk.StringVar(value="")
        photo_info_label = ttk.Label(left_frame, textvariable=self.photo_info_var,
                                     font=("Arial", 9))
        photo_info_label.pack(pady=(5, 0))

        # Right: Results
        right_frame = ttk.LabelFrame(content_frame, text="Diagnostic Results", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Analysis Results
        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(analysis_tab, text="📊 Analysis")

        self.analysis_text = scrolledtext.ScrolledText(analysis_tab,
                                                       font=("Courier New", 9),
                                                       wrap=tk.WORD,
                                                       height=15)
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 2: Comparison Results
        compare_tab = ttk.Frame(self.notebook)
        self.notebook.add(compare_tab, text="🔄 Comparison")

        self.compare_text = scrolledtext.ScrolledText(compare_tab,
                                                      font=("Courier New", 9),
                                                      wrap=tk.WORD)
        self.compare_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 3: Recommendations
        rec_tab = ttk.Frame(self.notebook)
        self.notebook.add(rec_tab, text="💡 Recommendations")

        self.rec_text = scrolledtext.ScrolledText(rec_tab,
                                                  font=("Arial", 10),
                                                  wrap=tk.WORD)
        self.rec_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom status
        self.bottom_status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.bottom_status_var,
                               relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))

    def select_photo(self, photo_type):
        """Select a photo file"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title=f"Select a {'Working' if photo_type == 'working' else 'Non-working' if photo_type == 'non_working' else 'Baccarat'} photo",
            filetypes=filetypes
        )

        if filename:
            try:
                # Load and display image
                self.current_photo = Image.open(filename)
                self.current_photo_path = filename

                # Display image
                self.display_image()

                # Update status based on photo type
                if photo_type == 'working':
                    self.working_photo_path = filename
                    self.working_status.set(f"✓ {os.path.basename(filename)}")
                elif photo_type == 'non_working':
                    self.non_working_photo_path = filename
                    self.non_working_status.set(f"✓ {os.path.basename(filename)}")
                else:  # single
                    self.photo_info_var.set(f"Loaded: {os.path.basename(filename)}")

                # Enable appropriate buttons
                if photo_type == 'single':
                    self.analyze_single_btn.config(state=tk.NORMAL)
                elif self.working_photo_path and self.non_working_photo_path:
                    self.compare_btn.config(state=tk.NORMAL)

                # Update bottom status
                self.bottom_status_var.set(f"Loaded {os.path.basename(filename)}")

            except Exception as e:
                messagebox.showerror("Image Error", f"Cannot load image: {str(e)}")

    def display_image(self):
        """Display image on canvas"""
        if not self.current_photo:
            return

        # Calculate display size
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width < 10:
            canvas_width, canvas_height = 400, 300

        # Resize for display
        img = self.current_photo.copy()
        img.thumbnail((canvas_width - 20, canvas_height - 20))
        self.display_size = img.size

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(img)

        # Clear and display
        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_width // 2, canvas_height // 2,
                                       image=self.photo, anchor=tk.CENTER)

    def analyze_single(self):
        """Analyze a single photo"""
        if not self.current_photo_path:
            return

        self.analysis_text.delete(1.0, tk.END)
        self.rec_text.delete(1.0, tk.END)

        self.bottom_status_var.set("Analyzing photo...")

        # Run analysis
        result = self.diagnostic.analyze_photo_properties(self.current_photo_path)

        if result.get('success'):
            self.display_analysis_results(result['properties'])
            self.display_recommendations(result['properties'])
            self.bottom_status_var.set("Analysis complete")
            self.notebook.select(0)  # Switch to analysis tab
        else:
            messagebox.showerror("Analysis Error", result.get('error', 'Unknown error'))
            self.bottom_status_var.set("Analysis failed")

    def display_analysis_results(self, properties):
        """Display analysis results"""
        self.analysis_text.delete(1.0, tk.END)

        # Format the report
        report = []
        report.append("=" * 80)
        report.append("🔍 BACCARAT PHOTO DIAGNOSTIC REPORT")
        report.append("=" * 80)

        # Basic info
        dims = properties['dimensions']
        report.append(f"\n📏 DIMENSIONS: {dims['width']} x {dims['height']}")

        # Basic stats
        stats = properties['basic_stats']
        report.append(f"\n📊 BASIC STATISTICS:")
        report.append(f"  • Brightness: {stats['brightness']:.1f} (std: {stats['brightness_std']:.1f})")
        report.append(f"  • Contrast: {stats['contrast']:.1f}")
        report.append(f"  • Hue: mean={stats['hue_mean']:.1f}, std={stats['hue_std']:.1f}")
        report.append(f"  • Saturation: mean={stats['saturation_mean']:.1f}, std={stats['saturation_std']:.1f}")
        report.append(f"  • Value: mean={stats['value_mean']:.1f}, std={stats['value_std']:.1f}")

        # Dot detection
        dots = properties['dot_detection']
        report.append(f"\n🎯 DOT DETECTION:")
        report.append(f"  • Best estimate: {dots['best_estimate']} dots")
        report.append(f"  • Expected range: {dots['expected_range'][0]}-{dots['expected_range'][1]}")

        for method in dots['detection_methods']:
            if method['method'] == 'color_based':
                report.append(f"  • Color-based detection: {method['count']} dots")
                if method['count'] > 0:
                    report.append(f"    - Avg area: {method['avg_area']:.1f}")
                    report.append(f"    - Avg circularity: {method['avg_circularity']:.3f}")

        # Focus quality
        focus = properties['focus_quality']
        report.append(f"\n📸 FOCUS QUALITY:")
        report.append(f"  • Sharpness: {focus['sharpness_category']}")
        report.append(f"  • Laplacian variance: {focus['laplacian_variance']:.1f}")
        report.append(f"  • Edge density: {focus['edge_density']:.3f}")

        # Lighting
        lighting = properties['lighting_analysis']
        report.append(f"\n💡 LIGHTING ANALYSIS:")
        report.append(f"  • Category: {lighting['lighting_category']}")
        report.append(f"  • Brightness: {lighting['brightness']:.1f}")
        report.append(f"  • Contrast: {lighting['contrast']:.1f}")
        report.append(f"  • Uniformity: {lighting['uniformity']:.3f}")

        # Grid analysis
        grid = properties['grid_analysis']
        report.append(f"\n🏗️ GRID STRUCTURE:")
        report.append(f"  • Horizontal lines: {grid['horizontal_lines']}")
        report.append(f"  • Vertical lines: {grid['vertical_lines']}")
        report.append(f"  • Alignment score: {grid['overall_alignment']:.3f}")

        # Color analysis
        colors = properties['color_analysis']
        report.append(f"\n🎨 COLOR ANALYSIS:")
        for color_name, color_stats in colors.items():
            coverage_pct = color_stats['coverage'] * 100
            if coverage_pct > 0.1:  # Only show significant colors
                report.append(f"  • {color_name}: {coverage_pct:.2f}% coverage")

        # Potential issues
        issues = properties['potential_issues']
        if issues:
            report.append(f"\n⚠️ POTENTIAL ISSUES:")
            for issue in issues:
                report.append(f"  • {issue}")
        else:
            report.append(f"\n✅ No major issues detected")

        report.append("\n" + "=" * 80)

        # Insert into text widget
        self.analysis_text.insert(tk.END, "\n".join(report))

    def display_recommendations(self, properties):
        """Display recommendations based on analysis"""
        self.rec_text.delete(1.0, tk.END)

        recommendations = []
        issues = properties['potential_issues']

        if issues:
            recommendations.append("💡 RECOMMENDATIONS FOR IMPROVEMENT:")
            recommendations.append("=" * 60 + "\n")

            for issue in issues:
                if "Too dark" in issue:
                    recommendations.append("• 💡 Increase lighting or use flash")
                elif "Too bright" in issue:
                    recommendations.append("• 🔆 Reduce glare or move away from direct light")
                elif "Low contrast" in issue:
                    recommendations.append("• 🎨 Improve lighting to increase contrast")
                elif "Blurry" in issue or "Acceptable" in issue:
                    recommendations.append("• 📸 Hold phone steady, wait for auto-focus")
                elif "Low color saturation" in issue:
                    recommendations.append("• 🎨 Ensure good lighting for vibrant colors")
                elif "Too few dots detected" in issue:
                    recommendations.append("• 🔍 Get closer or ensure all dots are visible")
                elif "Too many dots detected" in issue:
                    recommendations.append("• 🧹 Clean camera lens, reduce background noise")
                elif "Poor grid alignment" in issue:
                    recommendations.append("• 📐 Hold phone parallel to the sheet")

        else:
            recommendations.append("✅ PHOTO IS OPTIMAL!")
            recommendations.append("=" * 60 + "\n")
            recommendations.append("This photo meets all requirements for accurate analysis.")
            recommendations.append("\n📸 Photo guidelines followed:")
            recommendations.append("  • Good lighting and contrast")
            recommendations.append("  • Sharp focus")
            recommendations.append("  • Proper grid alignment")
            recommendations.append("  • Adequate color saturation")

        self.rec_text.insert(tk.END, "\n".join(recommendations))

    def compare_photos(self):
        """Compare working and non-working photos"""
        if not self.working_photo_path or not self.non_working_photo_path:
            messagebox.showwarning("Missing Photos",
                                   "Please select both working and non-working photos")
            return

        self.compare_text.delete(1.0, tk.END)
        self.bottom_status_var.set("Comparing photos...")

        # Run comparison
        result = self.diagnostic.compare_photos(
            self.working_photo_path,
            self.non_working_photo_path
        )

        if result.get('success'):
            self.display_comparison_results(result['comparison'])
            self.bottom_status_var.set("Comparison complete")
            self.notebook.select(1)  # Switch to comparison tab
        else:
            messagebox.showerror("Comparison Error", result.get('error', 'Unknown error'))
            self.bottom_status_var.set("Comparison failed")

    def display_comparison_results(self, comparison):
        """Display comparison results"""
        self.compare_text.delete(1.0, tk.END)

        report = []
        report.append("=" * 80)
        report.append("🔄 WORKING vs NON-WORKING PHOTO COMPARISON")
        report.append("=" * 80)

        report.append(f"\n✅ WORKING PHOTO ISSUES: {len(comparison['working_issues'])}")
        for issue in comparison['working_issues']:
            report.append(f"  • {issue}")

        report.append(f"\n❌ NON-WORKING PHOTO ISSUES: {len(comparison['non_working_issues'])}")
        for issue in comparison['non_working_issues']:
            report.append(f"  • {issue}")

        if comparison['differences']:
            report.append(f"\n📊 KEY DIFFERENCES:")
            for diff in comparison['differences']:
                report.append(f"  • {diff}")

            if 'most_likely_issue' in comparison:
                report.append(f"\n🔍 MOST LIKELY CAUSE:")
                report.append(f"  • {comparison['most_likely_issue']}")

            report.append(f"\n💡 SUGGESTED FIXES:")
            for diff in comparison['differences']:
                if "brightness" in diff.lower():
                    if "Working=" in diff:
                        working_val = float(diff.split("Working=")[1].split(",")[0])
                        non_working_val = float(diff.split("Non-working=")[1].split(" ")[0])
                        if non_working_val < working_val:
                            report.append("  • 💡 Increase lighting for non-working photo")
                        else:
                            report.append("  • 🔆 Reduce brightness for non-working photo")
                elif "contrast" in diff.lower():
                    report.append("  • 🎨 Improve lighting to increase contrast")
                elif "focus" in diff.lower() or "sharpness" in diff.lower():
                    report.append("  • 📸 Hold phone steady, ensure focus is sharp")
                elif "saturation" in diff.lower():
                    report.append("  • 🎨 Improve lighting for better color saturation")
                elif "dots" in diff.lower():
                    report.append("  • 🔍 Check if dots are visible and not obstructed")
                elif "alignment" in diff.lower():
                    report.append("  • 📐 Hold phone parallel to sheet")
        else:
            report.append(f"\n📊 No significant differences found")
            report.append("  Photos are similar - issue may be with the analysis algorithm")

        report.append("\n" + "=" * 80)

        self.compare_text.insert(tk.END, "\n".join(report))

    def clear_all(self):
        """Clear all selections and results"""
        self.working_photo_path = None
        self.non_working_photo_path = None
        self.current_photo_path = None
        self.current_photo = None

        self.working_status.set("No working photo selected")
        self.non_working_status.set("No non-working photo selected")
        self.photo_info_var.set("")

        self.analyze_single_btn.config(state=tk.DISABLED)
        self.compare_btn.config(state=tk.DISABLED)

        self.image_canvas.delete("all")
        self.image_canvas.create_text(200, 150,
                                      text="No photo loaded\n\nSelect a photo to begin",
                                      font=("Arial", 12),
                                      fill="#666666",
                                      justify=tk.CENTER)

        self.analysis_text.delete(1.0, tk.END)
        self.compare_text.delete(1.0, tk.END)
        self.rec_text.delete(1.0, tk.END)

        self.bottom_status_var.set("Cleared all selections")


def main():
    """Main function"""
    # Create main window
    root = tk.Tk()
    root.title("Baccarat Photo Diagnostic Analyzer")

    # Center window
    root.update_idletasks()
    width, height = 1200, 800
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    # Create app
    app = DiagnosticGUI(root)

    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    print("🔍 BACCARAT PHOTO DIAGNOSTIC ANALYZER")
    print("=" * 60)
    print("Analyze why some photos work and others don't")
    print()
    print("Features:")
    print("  • 📊 Analyze single photo properties")
    print("  • 🔄 Compare working vs non-working photos")
    print("  • 💡 Get improvement recommendations")
    print()
    print("Requirements:")
    print("  pip install opencv-python numpy pillow")
    print()

    main()