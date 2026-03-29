import os
import argparse
from src.preprocessing.image_processor import ImageProcessor
from src.ocr.ocr_engine import OCREngine
from src.layout.layout_encoder import LayoutEncoder
from src.visualize import visualize_results

def process_single_image(image_path, output_dir, tesseract_cmd=None):
    """
    Orchestrates the multimodal document understanding pipeline for a single image.
    """
    print(f"\n--- Processing: {os.path.basename(image_path)} ---")
    
    # 1. Image Preprocessing
    processor = ImageProcessor()
    preprocessed_path = os.path.join(output_dir, "preprocessed_" + os.path.basename(image_path))
    print("Step 1: Preprocessing image...")
    processor.preprocess_for_ocr(image_path, preprocessed_path)
    
    # 2. OCR & Spatial Extraction
    engine = OCREngine(tesseract_cmd=tesseract_cmd)
    print("Step 2: Extracting text and spatial data...")
    ocr_results = engine.extract_text_with_layout(preprocessed_path)
    
    if not ocr_results:
        print(f"Warning: No text extracted for {image_path}. Skipping layout.")
        return None
    
    # 3. Layout Encoding
    encoder = LayoutEncoder()
    print("Step 3: Encoding layout and grouping...")
    multimodal_representation = encoder.generate_multimodal_representation(ocr_results)
    
    # 4. Save Output
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_json_path = os.path.join(output_dir, base_name + "_multimodal.json")
    engine.save_output(multimodal_representation, output_json_path)
    
    # 5. Visualize
    print("Step 4: Generating visualization...")
    vis_output_path = os.path.join(output_dir, "visualization_" + os.path.basename(image_path))
    visualize_results(preprocessed_path, output_json_path, vis_output_path)
    
    print(f"--- Completed {os.path.basename(image_path)}! Output saved to: {output_json_path} ---")
    return multimodal_representation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Document Understanding Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input document image or directory containing images")
    parser.add_argument("--output", type=str, default="output", help="Directory for output files")
    parser.add_argument("--tesseract", type=str, help="Path to tesseract executable")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    if os.path.isdir(args.input):
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(valid_extensions)]
        print(f"Found {len(files)} images in directory '{args.input}'. Starting batch processing...")
        for f in files:
            base_name = os.path.splitext(os.path.basename(f))[0]
            expected_output = os.path.join(args.output, base_name + "_multimodal.json")
            if os.path.exists(expected_output):
                print(f"Skipping {os.path.basename(f)} - already processed.")
                continue
            process_single_image(f, args.output, args.tesseract)
        print("\nBatch processing complete!")
    elif os.path.isfile(args.input):
        process_single_image(args.input, args.output, args.tesseract)
    else:
        print(f"Error: Invalid input path '{args.input}'. Must be an image file or a directory.")
