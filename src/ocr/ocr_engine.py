import pytesseract
from PIL import Image
import json
import os

try:
    import easyocr
    import numpy as np
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class OCREngine:
    """
    Handles text and spatial information extraction using Tesseract or EasyOCR.
    """

    def __init__(self, tesseract_cmd=None, use_easyocr=None):
        """
        Initializes the OCR engine.
        :param tesseract_cmd: Path to the tesseract executable.
        :param use_easyocr: Force use of EasyOCR (True/False). If None, tries Tesseract first.
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.use_easyocr = use_easyocr
        if self.use_easyocr is None:
            # Check if tesseract is available
            import shutil
            if not shutil.which("tesseract") and not tesseract_cmd:
                print("Tesseract not found. Falling back to EasyOCR.")
                self.use_easyocr = True
            else:
                self.use_easyocr = False
        
        if self.use_easyocr and EASYOCR_AVAILABLE:
            self.reader = easyocr.Reader(['en']) # Initialize for English

    def extract_text_with_layout(self, image_path):
        """
        Extracts text along with bounding box information.
        Returns a list of dictionaries, each containing text and its coordinates.
        """
        if self.use_easyocr and EASYOCR_AVAILABLE:
            return self._extract_with_easyocr(image_path)
        else:
            return self._extract_with_tesseract(image_path)

    def _extract_with_tesseract(self, image_path):
        try:
            data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
            structured_data = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                if int(data['conf'][i]) > 0:
                    box = {
                        "text": data['text'][i],
                        "left": data['left'][i],
                        "top": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i],
                        "conf": data['conf'][i]
                    }
                    structured_data.append(box)
            return structured_data
        except Exception as e:
            print(f"Error during Tesseract extraction: {e}")
            if EASYOCR_AVAILABLE:
                print("Attempting fallback to EasyOCR...")
                self.use_easyocr = True
                self.reader = easyocr.Reader(['en'])
                return self._extract_with_easyocr(image_path)
            return []

    def _extract_with_easyocr(self, image_path):
        try:
            # EasyOCR returns [[bbox], text, conf]
            # bbox is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            results = self.reader.readtext(image_path)
            structured_data = []
            for (bbox, text, conf) in results:
                left = int(bbox[0][0])
                top = int(bbox[0][1])
                width = int(bbox[2][0] - left)
                height = int(bbox[2][1] - top)
                
                box = {
                    "text": text,
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "conf": float(conf) * 100 # Scale to match Tesseract
                }
                structured_data.append(box)
            return structured_data
        except Exception as e:
            print(f"Error during EasyOCR extraction: {e}")
            return []

    def save_output(self, data, output_path):
        """Saves the extracted data to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Example usage
    engine = OCREngine()
    # Assume data is available
    # results = engine.extract_text_with_layout("output/preprocessed.png")
    # engine.save_output(results, "output/ocr_results.json")
    print("OCREngine initialized.")
