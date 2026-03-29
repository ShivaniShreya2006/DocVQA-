import cv2
import numpy as np
from PIL import Image

class ImageProcessor:
    """
    Handles image preprocessing for document understanding.
    Includes techniques to improve image quality for OCR.
    """

    def __init__(self):
        pass

    def load_image(self, image_path):
        """Loads an image using OpenCV."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        return image

    def convert_to_grayscale(self, image):
        """Converts the image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image):
        """Applies median blur to remove salt-and-pepper noise."""
        return cv2.medianBlur(image, 3)

    def rescale(self, image, scale_factor=2):
        """Rescales the image by a scale factor to improve OCR on small text."""
        h, w = image.shape[:2]
        new_size = (int(w * scale_factor), int(h * scale_factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

    def apply_thresholding(self, image):
        """Applies Gaussian Adaptive Thresholding for better document clarity."""
        # Using adaptive thresholding instead of simple OTSU for documents with varying lighting/scanning quality
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def deskew(self, image):
        """
        Attempts to deskew the image by finding the orientation of the text.
        (Simplified version for initial implementation).
        """
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def preprocess_for_ocr(self, image_path, output_path=None):
        """
        Full preprocessing pipeline.
        """
        image = self.load_image(image_path)
        gray = self.convert_to_grayscale(image)
        rescaled = self.rescale(gray, scale_factor=2)
        denoised = self.remove_noise(rescaled)
        thresholded = self.apply_thresholding(denoised)
        
        if output_path:
            cv2.imwrite(output_path, thresholded)
            
        return thresholded

if __name__ == "__main__":
    # Example usage (test)
    import os
    processor = ImageProcessor()
    # Assuming there's a sample image in data/
    # processor.preprocess_for_ocr("data/sample.png", "output/preprocessed.png")
    print("ImageProcessor initialized.")
