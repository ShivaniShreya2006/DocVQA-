import cv2
import json

def visualize_results(image_path, json_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    encoding = data.get('visual_layout_encoding', {})
    text_blocks = {b['id']: b for b in encoding.get('text_blocks', [])}
    
    # Draw all text blocks (Blue)
    for block_id, block in text_blocks.items():
        x, y, w, h = block['left'], block['top'], block['width'], block['height']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

    entities = encoding.get('entities', {})
    
    # Draw Key-Value pairs (Green connections)
    for kv in entities.get('key_value_pairs', []):
        k_id = kv.get('key_id')
        v_id = kv.get('value_id')
        if k_id in text_blocks and v_id in text_blocks:
            k_b = text_blocks[k_id]
            v_b = text_blocks[v_id]
            # Center of key
            cx1 = k_b['left'] + k_b['width'] // 2
            cy1 = k_b['top'] + k_b['height'] // 2
            # Center of value
            cx2 = v_b['left'] + v_b['width'] // 2
            cy2 = v_b['top'] + v_b['height'] // 2
            cv2.line(image, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            cv2.rectangle(image, (k_b['left'], k_b['top']), (k_b['left']+k_b['width'], k_b['top']+k_b['height']), (0, 255, 0), 2)
            cv2.rectangle(image, (v_b['left'], v_b['top']), (v_b['left']+v_b['width'], v_b['top']+v_b['height']), (0, 200, 0), 2)
            
    # Draw Tables (Orange)
    for table in entities.get('tables', []):
        bbox = table.get('bbox', [])
        if bbox:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 3)

    cv2.imwrite(output_path, image)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    visualize_results(
        "output/preprocessed_sample_invoice.png",
        "output/sample_invoice_multimodal.json",
        "output/visualization_sample_invoice.png"
    )
