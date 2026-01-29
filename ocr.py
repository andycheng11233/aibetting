#!/usr/bin/env python3
"""
🎰 GOOGLE VISION API OCR EXTRACTOR
Reliable API-based OCR with free tier
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import requests
import base64
import json
import os
import sys
import argparse
from PIL import Image, ImageTk
import threading
import queue
import time
import csv
from datetime import datetime


class GoogleVisionOCR:
    """Google Cloud Vision API OCR client"""

    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        self.headers = {'Content-Type': 'application/json'}

    def encode_image(self, image_path):
        """Convert image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def extract_text(self, image_path):
        """Extract text using Google Vision API"""
        try:
            # Encode image
            base64_image = self.encode_image(image_path)

            # Prepare request
            request_data = {
                "requests": [
                    {
                        "image": {
                            "content": base64_image
                        },
                        "features": [
                            {
                                "type": "TEXT_DETECTION",
                                "maxResults": 1
                            },
                            {
                                "type": "DOCUMENT_TEXT_DETECTION",
                                "maxResults": 1
                            }
                        ]
                    }
                ]
            }

            # Make API call
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=request_data,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                # Try DOCUMENT_TEXT_DETECTION first (better for documents)
                if 'responses' in data and len(data['responses']) > 0:
                    response_data = data['responses'][0]

                    # Get full text
                    full_text = ""
                    if 'fullTextAnnotation' in response_data:
                        full_text = response_data['fullTextAnnotation'].get('text', '')

                    # Get text with positions
                    text_blocks = []
                    if 'textAnnotations' in response_data and len(response_data['textAnnotations']) > 1:
                        # First item is the entire text, skip it
                        for annotation in response_data['textAnnotations'][1:]:
                            text = annotation.get('description', '').strip()
                            if text:
                                vertices = annotation.get('boundingPoly', {}).get('vertices', [])
                                if len(vertices) >= 2:
                                    x1 = vertices[0].get('x', 0)
                                    y1 = vertices[0].get('y', 0)
                                    x2 = vertices[2].get('x', x1) if len(vertices) > 2 else x1
                                    y2 = vertices[2].get('y', y1) if len(vertices) > 2 else y1

                                    text_blocks.append({
                                        'text': text,
                                        'x': x1,
                                        'y': y1,
                                        'width': x2 - x1,
                                        'height': y2 - y1
                                    })

                    return {
                        "success": True,
                        "full_text": full_text,
                        "text_blocks": text_blocks,
                        "raw_response": data
                    }
                else:
                    return {
                        "success": False,
                        "error": "No text found in response"
                    }

            else:
                error_msg = f"API Error {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg = f"{error_msg}: {error_data['error'].get('message', 'Unknown')}"
                except:
                    error_msg = f"{error_msg}: {response.text[:100]}"

                return {
                    "success": False,
                    "error": error_msg,
                    "raw_response": response.text
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"API call failed: {str(e)}"
            }


class APITextExtractorGUI:
    """GUI for API-based text extraction"""

    def __init__(self, root, api_key):
        self.root = root
        self.root.title("🎰 Google Vision API OCR Extractor")
        self.root.geometry("1300x800")

        # API
        self.api_key = api_key
        self.ocr_client = GoogleVisionOCR(api_key)

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

        # TOP: Title and controls
        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # Title
        title_label = ttk.Label(top_frame, text="🎰 Google Vision API OCR Extractor",
                                font=("Arial", 18, "bold"))
        title_label.pack()

        subtitle = ttk.Label(top_frame,
                             text="Reliable API-based OCR with free tier | No local processing",
                             font=("Arial", 10))
        subtitle.pack()

        # Control panel
        control_frame = ttk.LabelFrame(main_container, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack()

        self.load_btn = ttk.Button(btn_frame, text="📁 Load Image",
                                   command=self.load_image, width=20)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.extract_btn = ttk.Button(btn_frame, text="🔍 Extract Text with API",
                                      command=self.extract_text, width=25,
                                      state=tk.DISABLED)
        self.extract_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(btn_frame, text="💾 Save Results",
                                   command=self.save_results, width=20,
                                   state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Progress
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, pady=(10, 0))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame,
                                            variable=self.progress_var,
                                            maximum=100, length=300)
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 10))

        self.status_var = tk.StringVar(value="Ready to load image")
        ttk.Label(progress_frame, textvariable=self.status_var,
                  font=("Arial", 9)).pack(side=tk.LEFT)

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

        # Image info
        self.image_info_var = tk.StringVar(value="")
        ttk.Label(left_frame, textvariable=self.image_info_var,
                  font=("Arial", 9)).pack(pady=(5, 0))

        # RIGHT: Results area
        right_frame = ttk.LabelFrame(content_frame, text="Extracted Text", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Full Text
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Full Text")

        self.full_text_area = scrolledtext.ScrolledText(tab1,
                                                        font=("Courier New", 10),
                                                        wrap=tk.WORD)
        self.full_text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 2: Text Blocks (with positions)
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Text Blocks")

        self.blocks_text_area = scrolledtext.ScrolledText(tab2,
                                                          font=("Courier New", 9),
                                                          wrap=tk.WORD)
        self.blocks_text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 3: Raw Response
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="Raw API Response")

        self.raw_text_area = scrolledtext.ScrolledText(tab3,
                                                       font=("Consolas", 8),
                                                       wrap=tk.WORD)
        self.raw_text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom: Status bar
        self.bottom_status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.bottom_status_var,
                               relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))

    def load_image(self):
        """Load an image file"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=filetypes
        )

        if filename:
            self.current_image_path = filename

            try:
                # Load image
                self.current_image = Image.open(filename)

                # Display image on canvas
                self.display_image()

                # Update info
                width, height = self.current_image.size
                file_size = os.path.getsize(filename) / 1024  # KB
                self.image_info_var.set(
                    f"File: {os.path.basename(filename)}\n"
                    f"Size: {width}x{height} pixels | {file_size:.1f} KB"
                )

                # Enable extract button
                self.extract_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.DISABLED)

                # Clear previous results
                self.full_text_area.delete(1.0, tk.END)
                self.blocks_text_area.delete(1.0, tk.END)
                self.raw_text_area.delete(1.0, tk.END)

                self.status_var.set("Ready to extract")
                self.bottom_status_var.set(f"Loaded: {os.path.basename(filename)}")

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

        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(img)

        # Clear and display
        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_width // 2, canvas_height // 2,
                                       image=self.photo, anchor=tk.CENTER)

    def extract_text(self):
        """Extract text using Google Vision API"""
        if not self.current_image_path:
            return

        # Clear previous results
        self.full_text_area.delete(1.0, tk.END)
        self.full_text_area.insert(tk.END, "🔄 Calling Google Vision API...\nPlease wait...\n")

        # Disable buttons
        self.load_btn.config(state=tk.DISABLED)
        self.extract_btn.config(state=tk.DISABLED)

        # Update status
        self.status_var.set("Calling API...")
        self.progress_var.set(20)

        # Start extraction in thread
        thread = threading.Thread(target=self.run_extraction, daemon=True)
        thread.start()

        # Start progress animation
        self.animate_progress()

    def animate_progress(self):
        """Animate progress bar"""
        current = self.progress_var.get()
        if current < 70:
            self.progress_var.set(current + 5)
            self.root.after(200, self.animate_progress)

    def run_extraction(self):
        """Run extraction in thread"""
        try:
            self.queue.put({'type': 'progress', 'value': 40})

            result = self.ocr_client.extract_text(self.current_image_path)

            self.queue.put({'type': 'progress', 'value': 90})
            self.queue.put({'type': 'result', 'data': result})

        except Exception as e:
            self.queue.put({'type': 'error', 'message': str(e)})

    def check_queue(self):
        """Check for messages from threads"""
        try:
            while True:
                item = self.queue.get_nowait()

                if item['type'] == 'progress':
                    self.progress_var.set(item['value'])

                elif item['type'] == 'result':
                    self.display_result(item['data'])

                elif item['type'] == 'error':
                    self.show_error(item['message'])

        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

    def display_result(self, result):
        """Display extraction results"""
        self.progress_var.set(100)

        # Re-enable buttons
        self.load_btn.config(state=tk.NORMAL)
        self.extract_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)

        # Store result
        self.last_result = result

        # Clear text areas
        self.full_text_area.delete(1.0, tk.END)
        self.blocks_text_area.delete(1.0, tk.END)
        self.raw_text_area.delete(1.0, tk.END)

        if result.get('success'):
            # Display full text
            full_text = result.get('full_text', '')
            self.full_text_area.insert(tk.END, full_text)

            # Display text blocks with positions
            text_blocks = result.get('text_blocks', [])
            if text_blocks:
                self.blocks_text_area.insert(tk.END, f"Found {len(text_blocks)} text blocks:\n")
                self.blocks_text_area.insert(tk.END, "=" * 60 + "\n\n")

                # Group by approximate rows
                rows = {}
                for block in text_blocks:
                    y = block['y']
                    row_key = y // 20  # Group by 20-pixel rows
                    if row_key not in rows:
                        rows[row_key] = []
                    rows[row_key].append(block)

                # Sort rows and blocks within rows
                for row_key in sorted(rows.keys()):
                    row_blocks = rows[row_key]
                    row_blocks.sort(key=lambda b: b['x'])

                    self.blocks_text_area.insert(tk.END, f"Row (approx y={row_key * 20}):\n")
                    for block in row_blocks:
                        self.blocks_text_area.insert(tk.END,
                                                     f"  '{block['text']}' at ({block['x']}, {block['y']}) "
                                                     f"size: {block['width']}x{block['height']}\n"
                                                     )
                    self.blocks_text_area.insert(tk.END, "\n")

            # Display raw response (truncated)
            raw_response = result.get('raw_response', {})
            self.raw_text_area.insert(tk.END, json.dumps(raw_response, indent=2)[:5000] + "...")

            # Update status
            self.status_var.set("Extraction complete")
            self.bottom_status_var.set(
                f"✓ Extracted {len(full_text.split())} words, {len(text_blocks)} text blocks"
            )

            # Switch to full text tab
            self.notebook.select(0)

        else:
            # Display error
            error_msg = f"❌ ERROR: {result.get('error', 'Unknown error')}"
            self.full_text_area.insert(tk.END, error_msg)

            # Show raw response if available
            raw_response = result.get('raw_response', '')
            if raw_response:
                self.raw_text_area.insert(tk.END, f"Raw response: {raw_response}")

            self.status_var.set("Extraction failed")
            self.bottom_status_var.set("API error occurred")

    def show_error(self, message):
        """Show error message"""
        self.progress_var.set(0)
        self.load_btn.config(state=tk.NORMAL)
        self.extract_btn.config(state=tk.NORMAL)

        messagebox.showerror("Extraction Error", message)
        self.status_var.set("Error")
        self.bottom_status_var.set("Extraction failed")

    def save_results(self):
        """Save extracted text to file"""
        if not self.last_result or not self.current_image_path:
            messagebox.showwarning("No Results", "No results to save")
            return

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]

        filename = filedialog.asksaveasfilename(
            title="Save OCR Results",
            defaultextension=".json",
            initialfile=f"{base_name}_google_ocr_{timestamp}.json",
            filetypes=[
                ("JSON files", "*.json"),
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if filename:
            try:
                if filename.endswith('.csv'):
                    # Save as CSV
                    with open(filename, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Text', 'X', 'Y', 'Width', 'Height'])

                        text_blocks = self.last_result.get('text_blocks', [])
                        for block in text_blocks:
                            writer.writerow([
                                block['text'],
                                block['x'],
                                block['y'],
                                block['width'],
                                block['height']
                            ])

                elif filename.endswith('.txt'):
                    # Save as text
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"OCR Results for: {self.current_image_path}\n")
                        f.write(f"Extraction time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=" * 60 + "\n\n")

                        full_text = self.last_result.get('full_text', '')
                        f.write(full_text)

                else:
                    # Save as JSON (default)
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump({
                            'image_file': self.current_image_path,
                            'extraction_time': datetime.now().isoformat(),
                            'result': self.last_result
                        }, f, indent=2, ensure_ascii=False)

                self.bottom_status_var.set(f"✓ Saved to: {os.path.basename(filename)}")

            except Exception as e:
                messagebox.showerror("Save Error", f"Cannot save file: {str(e)}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Google Vision API OCR Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python google_ocr.py --key YOUR_GOOGLE_API_KEY
  python google_ocr.py -k YOUR_GOOGLE_API_KEY
  python google_ocr.py --env (uses GOOGLE_API_KEY environment variable)

Get a Google API Key:
  1. Go to https://console.cloud.google.com/
  2. Create a project (or use existing)
  3. Enable Cloud Vision API
  4. Create API credentials
  5. Get API key
        """
    )

    parser.add_argument(
        '--key', '-k',
        type=str,
        help='Google Cloud Vision API key'
    )

    parser.add_argument(
        '--env', '-e',
        action='store_true',
        help='Use GOOGLE_API_KEY environment variable'
    )

    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()

    # Get API key
    api_key = args.key

    if args.env and not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            print("Using API key from GOOGLE_API_KEY environment variable")

    if not api_key:
        # Ask user for API key
        print("Google Cloud Vision API Key required!")
        print()
        print("To get an API key:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a project")
        print("3. Enable 'Cloud Vision API'")
        print("4. Create API credentials")
        print("5. Copy the API key")
        print()
        api_key = input("Enter your Google Cloud Vision API key: ").strip()

        if not api_key:
            print("No API key provided. Exiting.")
            sys.exit(1)

    # Check API key format (Google keys are typically long strings)
    if len(api_key) < 20:
        print("Warning: API key seems too short. Make sure it's a valid Google API key.")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            sys.exit(0)

    # Check requirements
    try:
        import requests
        from PIL import Image
    except ImportError:
        print("Missing required packages!")
        print("Install with: pip install requests pillow")
        sys.exit(1)

    # Test the API key
    print("Testing Google Vision API key...")
    test_client = GoogleVisionOCR(api_key)

    # Create a simple test image for testing
    try:
        from PIL import Image, ImageDraw, ImageFont
        # Create a test image with text
        test_img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(test_img)

        # Try to use a font, but if not available, just draw text
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        draw.text((10, 40), "API Test", fill='black', font=font)

        # Save test image
        test_img.save("test_api_image.png")

        # Test API call
        test_result = test_client.extract_text("test_api_image.png")

        if test_result.get('success'):
            print("✅ API key is valid!")
        else:
            print(f"❌ API key test failed: {test_result.get('error', 'Unknown error')}")
            print("Please check your API key and ensure Cloud Vision API is enabled.")

            response = input("Continue anyway? (y/n): ").lower()
            if response != 'y':
                sys.exit(1)

        # Clean up test file
        try:
            os.remove("test_api_image.png")
        except:
            pass

    except Exception as e:
        print(f"⚠️ Could not complete API test: {e}")
        print("Continuing anyway...")

    # Create main window
    root = tk.Tk()

    # Center window
    root.update_idletasks()
    width = 1300
    height = 800
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    # Create app
    app = APITextExtractorGUI(root, api_key)

    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    print("🎰 Google Vision API OCR Extractor")
    print("=" * 60)
    print("Features:")
    print("• 📁 Load any image (JPEG, PNG, etc.)")
    print("• 🔍 Extract text using Google Vision API (very accurate)")
    print("• 📊 See text with positions and grouping")
    print("• 💾 Save results in JSON, CSV, or text format")
    print("• 🆓 Free tier: 1000 units/month (~1000 pages)")
    print()
    print("Note: You need a Google Cloud account with Cloud Vision API enabled.")
    print()

    main()