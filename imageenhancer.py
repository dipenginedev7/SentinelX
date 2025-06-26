import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import torch
import torchvision.transforms as transforms
import requests
import os
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class AIPortraitDesigner:
    def __init__(self):
        """Initialize the AI Portrait Designer with necessary models and settings."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Download and load ESRGAN model for super-resolution (if available)
        self.setup_super_resolution()
        
    def setup_super_resolution(self):
        """Setup super-resolution model for image enhancement."""
        try:
            # You can download ESRGAN or Real-ESRGAN models
            # For this example, we'll use basic upscaling
            self.sr_model = None
            print("Super-resolution: Using basic upscaling")
        except Exception as e:
            print(f"Super-resolution model not available: {e}")
            self.sr_model = None
    
    def detect_face_and_eyes(self, image: np.ndarray) -> Tuple[Optional[tuple], list]:
        """Detect face and eyes in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        face_roi = faces[0] if len(faces) > 0 else None
        
        # Detect eyes
        eyes = []
        if face_roi is not None:
            x, y, w, h = face_roi
            roi_gray = gray[y:y+h, x:x+w]
            detected_eyes = self.eye_cascade.detectMultiScale(roi_gray)
            eyes = [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in detected_eyes]
        
        return face_roi, eyes
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality using AI techniques."""
        # Convert to PIL for better processing
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply AI-like enhancements
        # 1. Sharpening
        pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # 2. Contrast enhancement
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.3)
        
        # 3. Color enhancement
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # 4. Brightness adjustment
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.05)
        
        # 5. Detail enhancement using edge enhancement
        pil_image = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def apply_neural_style_effect(self, image: np.ndarray) -> np.ndarray:
        """Apply neural style-like effects for artistic enhancement."""
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply bilateral filter for smooth artistic effect
        enhanced = cv2.bilateralFilter(enhanced, 15, 35, 35)
        
        return enhanced
    
    def create_high_contrast_bw(self, image: np.ndarray) -> np.ndarray:
        """Create high-contrast black and white version."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold for high contrast
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Smooth and enhance
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to 3 channel for color overlays
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def add_geometric_overlays(self, image: np.ndarray, eyes: list, name: str) -> np.ndarray:
        """Add geometric overlays and design elements."""
        height, width = image.shape[:2]
        overlay = image.copy()
        
        # Create red rectangular overlay on one eye
        if eyes:
            eye_x, eye_y, eye_w, eye_h = eyes[0]  # First eye
            # Create stylish rectangular overlay
            cv2.rectangle(overlay, 
                         (eye_x - 10, eye_y - 5), 
                         (eye_x + eye_w + 10, eye_y + eye_h + 5), 
                         (0, 0, 255), -1)
            
            # Add geometric frame around the eye area
            frame_thickness = 3
            cv2.rectangle(overlay, 
                         (eye_x - 20, eye_y - 15), 
                         (eye_x + eye_w + 20, eye_y + eye_h + 15), 
                         (0, 0, 255), frame_thickness)
        
        # Add vertical red line (design element)
        line_x = width // 6
        cv2.line(overlay, (line_x, 0), (line_x, height), (0, 0, 255), 4)
        
        # Add name text vertically (repeated effect)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        text_color = (50, 50, 50)  # Dark gray
        
        # Vertical text effect
        for i in range(3):
            y_pos = 100 + i * 150
            cv2.putText(overlay, name.upper(), (30, y_pos), font, font_scale, text_color, 2)
        
        return overlay
    
    def add_professional_text(self, image: np.ndarray, name: str) -> np.ndarray:
        """Add professional text elements using PIL for better typography."""
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        width, height = pil_image.size
        
        try:
            # Try to load a nice font
            font_large = ImageFont.truetype("arial.ttf", 36)
            font_medium = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            # Fallback to default font
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Add main slogan
        slogan = "WORK SMART NOT HARD"
        draw.text((width - 400, height - 100), slogan, fill=(255, 0, 0), font=font_medium)
        
        # Add signature
        draw.text((width - 200, height - 60), "GRAPHICS", fill=(255, 0, 0), font=font_small)
        
        # Add hashtag
        hashtag = f"#{name.upper()}"
        draw.text((50, height - 50), hashtag, fill=(255, 0, 0), font=font_small)
        
        # Add Nike-style logo placeholder (you'd replace with actual logo)
        draw.ellipse([80, 50, 120, 90], fill=(255, 0, 0))
        draw.text((95, 65), "✓", fill=(255, 255, 255), font=font_medium)
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def create_textured_background(self, width: int, height: int) -> np.ndarray:
        """Create a textured grey background."""
        # Create base grey background
        background = np.full((height, width, 3), 128, dtype=np.uint8)
        
        # Add noise for texture
        noise = np.random.normal(0, 15, (height, width, 3)).astype(np.int16)
        background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Apply subtle gradient
        for i in range(height):
            factor = 0.8 + 0.4 * (i / height)
            background[i] = np.clip(background[i] * factor, 0, 255)
        
        return background
    
    def process_portrait(self, input_path: str, name: str, output_path: str = "enhanced_portrait.jpg"):
        """Main processing pipeline for creating the enhanced portrait."""
        print("Loading and processing image...")
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image from {input_path}")
        
        original_height, original_width = image.shape[:2]
        
        # Resize for processing if too large
        if original_width > 1200:
            scale = 1200 / original_width
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        print("Enhancing image quality...")
        # AI Enhancement pipeline
        enhanced = self.enhance_image_quality(image)
        enhanced = self.apply_neural_style_effect(enhanced)
        
        print("Detecting facial features...")
        # Detect face and eyes
        face_roi, eyes = self.detect_face_and_eyes(enhanced)
        
        if face_roi is None:
            print("Warning: No face detected, proceeding without face-specific enhancements")
        else:
            print(f"Detected {len(eyes)} eyes")
        
        print("Creating artistic effects...")
        # Create high-contrast version for artistic effect
        bw_version = self.create_high_contrast_bw(enhanced)
        
        # Combine original enhanced with BW for mixed effect
        # Use the enhanced color version as base
        result = enhanced.copy()
        
        # Apply BW effect to parts of the image for artistic contrast
        mask = np.zeros(enhanced.shape[:2], dtype=np.uint8)
        if face_roi is not None:
            x, y, w, h = face_roi
            # Apply BW effect around face area
            cv2.rectangle(mask, (x-50, y-50), (x+w+50, y+h+50), 255, -1)
        
        # Blend BW and color versions
        mask_3d = cv2.merge([mask, mask, mask]) / 255.0
        result = (result * (1 - mask_3d * 0.7) + bw_version * mask_3d * 0.7).astype(np.uint8)
        
        print("Adding design elements...")
        # Add geometric overlays
        result = self.add_geometric_overlays(result, eyes, name)
        
        # Add professional text
        result = self.add_professional_text(result, name)
        
        print("Finalizing design...")
        # Final enhancement pass
        result = self.enhance_image_quality(result)
        
        # Save result
        cv2.imwrite(output_path, result)
        print(f"Enhanced portrait saved to: {output_path}")
        
        return result
    
    def create_instagram_version(self, image: np.ndarray, output_path: str = "instagram_version.jpg"):
        """Create an Instagram-optimized version (1080x1080)."""
        # Resize to Instagram square format
        instagram_size = 1080
        resized = cv2.resize(image, (instagram_size, instagram_size))
        
        # Apply Instagram-style filter
        # Increase saturation slightly
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255)  # Increase saturation
        resized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        cv2.imwrite(output_path, resized)
        print(f"Instagram version saved to: {output_path}")
        return resized

# Example usage
def main():
    """Example usage of the AI Portrait Designer."""
    designer = AIPortraitDesigner()
    
    # Process the portrait
    try:
        input_image = "arghya.jpg"  # Replace with your image path
        name = "ARGHYADIP"
        
        print("Starting AI-enhanced portrait creation...")
        result = designer.process_portrait(input_image, name, "ai_enhanced_portrait.jpg")
        
        # Create Instagram version
        designer.create_instagram_version(result, "instagram_portrait.jpg")
        
        print("✅ Portrait creation completed successfully!")
        print("Files created:")
        print("- ai_enhanced_portrait.jpg (Main version)")
        print("- instagram_portrait.jpg (Square format)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have an image file named 'input_photo.jpg' in the same directory")

if __name__ == "__main__":
    main()