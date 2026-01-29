#!/usr/bin/env python3
"""
🛠️ BACCARAT IMAGE FIXER - STANDALONE
Fixes blurry, dark, or low-contrast photos before analysis
"""
import cv2
import numpy as np
import sys
import os


class BaccaratImagePreprocessor:
    """Preprocess images to fix common issues before analysis"""

    def __init__(self):
        # Target specifications
        self.target_brightness = 140  # Optimal for dot detection
        self.target_contrast = 40  # Optimal contrast
        self.min_sharpness = 100  # Minimum Laplacian variance

    def preprocess_image(self, image_path, debug_mode=False):
        """Fix common image issues before processing"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "error": "Could not load image", "processed": None}

            original = img.copy()
            steps = []

            # Step 1: Measure original quality
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            orig_brightness = np.mean(gray)
            orig_contrast = np.std(gray)
            orig_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            steps.append(f"Original: Brightness={orig_brightness:.1f}, "
                         f"Contrast={orig_contrast:.1f}, Sharpness={orig_sharpness:.1f}")

            # Step 2: Fix blurriness (if needed)
            if orig_sharpness < self.min_sharpness:
                img = self._sharpen_image(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                new_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                steps.append(f"Sharpened: {orig_sharpness:.1f} → {new_sharpness:.1f}")

            # Step 3: Fix brightness (if needed)
            if abs(orig_brightness - self.target_brightness) > 30:
                img = self._adjust_brightness(img, self.target_brightness)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                new_brightness = np.mean(gray)
                steps.append(f"Brightness adjusted: {orig_brightness:.1f} → {new_brightness:.1f}")

            # Step 4: Fix contrast (if needed)
            if orig_contrast < 25:  # Too low contrast
                img = self._enhance_contrast(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                new_contrast = np.std(gray)
                steps.append(f"Contrast enhanced: {orig_contrast:.1f} → {new_contrast:.1f}")

            # Step 5: Remove noise
            img = self._remove_noise(img)
            steps.append("Noise reduction applied")

            # Step 6: Enhance colors for better detection
            img = self._enhance_colors(img)
            steps.append("Colors enhanced for detection")

            return {
                "success": True,
                "processed": img,
                "original": original,
                "steps": steps,
                "improvements": {
                    "brightness_change": abs(orig_brightness - np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))),
                    "contrast_change": abs(orig_contrast - np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))),
                    "sharpness_change": abs(
                        orig_sharpness - cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Preprocessing failed: {str(e)}", "processed": None}

    def _sharpen_image(self, img):
        """Sharpen blurry image"""
        # Method 1: Unsharp masking (best for photos)
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

        # Method 2: Kernel sharpening (backup)
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(sharpened, -1, kernel)

        return sharpened

    def _adjust_brightness(self, img, target_brightness):
        """Adjust image brightness to target level"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray)

        # Calculate adjustment factor
        if current_brightness > 0:
            factor = target_brightness / current_brightness
            # Limit adjustment to reasonable range
            factor = max(0.5, min(2.0, factor))

            # Apply brightness adjustment
            adjusted = cv2.convertScaleAbs(img, alpha=factor, beta=0)
            return adjusted

        return img

    def _enhance_contrast(self, img):
        """Enhance image contrast"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge back
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return enhanced

    def _remove_noise(self, img):
        """Remove noise while preserving edges"""
        # Non-local means denoising (preserves edges better than blur)
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return denoised

    def _enhance_colors(self, img):
        """Enhance colors for better dot detection"""
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Enhance saturation (make colors more vibrant)
        s = cv2.multiply(s, 1.2)
        s = np.clip(s, 0, 255).astype(np.uint8)

        # Enhance value (brightness of colors)
        v = cv2.multiply(v, 1.1)
        v = np.clip(v, 0, 255).astype(np.uint8)

        # Merge back
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

        return enhanced

    def show_comparison(self, original, processed, title="Preprocessing Results"):
        """Show side-by-side comparison"""
        # Resize for display
        height = max(original.shape[0], processed.shape[0])
        width = original.shape[1] + processed.shape[1] + 10

        # Create comparison image
        comparison = np.zeros((height, width, 3), dtype=np.uint8)

        # Place original on left
        comparison[0:original.shape[0], 0:original.shape[1]] = original

        # Place processed on right
        start_x = original.shape[1] + 10
        comparison[0:processed.shape[0], start_x:start_x + processed.shape[1]] = processed

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Processed", (start_x + 10, 30), font, 1, (255, 255, 255), 2)

        # Show
        cv2.imshow(title, comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) != 2:
        print("🛠️ BACCARAT IMAGE FIXER")
        print("=" * 60)
        print("Fixes blurry, dark, or low-contrast photos for analysis")
        print()
        print("Usage: python fixphoto.py <image_path>")
        print("Example: python fixphoto.py bad_photo.jpg")
        print()
        print("Features:")
        print("  • 🔪 Sharpens blurry images")
        print("  • 💡 Adjusts brightness")
        print("  • 🎨 Enhances contrast")
        print("  • 🧹 Reduces noise")
        print("  • 🌈 Improves colors for detection")
        print()
        print("Requirements:")
        print("  pip install opencv-python numpy")
        return

    image_path = sys.argv[1]

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return

    # Create preprocessor
    preprocessor = BaccaratImagePreprocessor()

    # Process image
    print(f"🛠️ Processing: {os.path.basename(image_path)}")
    result = preprocessor.preprocess_image(image_path, debug_mode=True)

    if result['success']:
        print("✅ Preprocessing successful!")
        print("\n📋 Processing steps:")
        for step in result['steps']:
            print(f"  • {step}")

        # Save processed image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_fixed.jpg"
        cv2.imwrite(output_path, result['processed'])
        print(f"\n💾 Saved processed image to: {output_path}")

        # Show comparison
        try:
            preprocessor.show_comparison(result['original'], result['processed'])
        except Exception as e:
            print(f"⚠️ Could not display comparison: {e}")
            print("   (Image saved successfully)")

        # Show what was improved
        improvements = result['improvements']
        print("\n📈 Improvements made:")
        print(f"  • Brightness change: {improvements['brightness_change']:.1f}")
        print(f"  • Contrast change: {improvements['contrast_change']:.1f}")
        print(f"  • Sharpness change: {improvements['sharpness_change']:.1f}")

        # Check if image is now good for analysis
        gray = cv2.cvtColor(result['processed'], cv2.COLOR_BGR2GRAY)
        final_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        final_brightness = np.mean(gray)
        final_contrast = np.std(gray)

        print(f"\n🎯 Final image quality:")
        print(f"  • Sharpness: {final_sharpness:.1f} {'✅' if final_sharpness >= 100 else '⚠️'}")
        print(f"  • Brightness: {final_brightness:.1f} {'✅' if 100 <= final_brightness <= 180 else '⚠️'}")
        print(f"  • Contrast: {final_contrast:.1f} {'✅' if final_contrast >= 30 else '⚠️'}")

        if final_sharpness >= 100 and 100 <= final_brightness <= 180 and final_contrast >= 30:
            print("\n🎉 Image is now READY for analysis!")
        else:
            print("\n⚠️ Image may still have issues. Consider retaking the photo.")

    else:
        print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    main()