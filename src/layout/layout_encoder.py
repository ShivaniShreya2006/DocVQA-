import json
import re

class LayoutEncoder:
    """
    Handles advanced layout encoding: OCR post-processing, key-value extraction,
    table detection, and high-level data organization.
    """

    def __init__(self):
        # Default financial keywords for key-value extraction
        self.target_keys = [
            "total", "amount due", "invoice no", "invoice number", "date", 
            "balance due", "tax", "subtotal", "account no", "amount"
        ]

    def sort_by_reading_order(self, ocr_data, margin=10):
        """Sorts text blocks into a logical reading order."""
        return sorted(ocr_data, key=lambda x: (x['top'] // margin, x['left']))

    def merge_fragmented_boxes(self, blocks, x_tolerance=20, y_tolerance=10):
        """Merges boxes that are part of the same text line and are horizontally close."""
        if not blocks:
            return []
        
        # Sort by reading order first
        sorted_blocks = self.sort_by_reading_order(blocks, margin=y_tolerance)
        merged = [sorted_blocks[0].copy()]
        merged[0]['id'] = "blk_000"
        
        for i, curr_original in enumerate(sorted_blocks[1:]):
            curr = curr_original.copy()
            curr['id'] = f"blk_{i+1:03d}"
            
            prev = merged[-1]
            vertical_diff = abs(curr['top'] - prev['top'])
            horizontal_gap = curr['left'] - (prev['left'] + prev['width'])
            
            # If on the same line and very close horizontally, merge them
            if vertical_diff <= y_tolerance and -5 <= horizontal_gap <= x_tolerance:
                prev['text'] += " " + curr['text']
                # new width is the right edge of curr minus left edge of prev
                new_right = max(prev['left'] + prev['width'], curr['left'] + curr['width'])
                prev['width'] = new_right - prev['left']
                prev['top'] = min(prev['top'], curr['top'])
                prev['height'] = max(prev['height'], curr['height'])
                prev['conf'] = (prev['conf'] + curr['conf']) / 2.0
            else:
                merged.append(curr)
                
        # Reassign IDs after merging to be sequentially clean
        for i, m in enumerate(merged):
            m['id'] = f"blk_{i:03d}"
            
        return merged

    def find_nearest_right(self, block, blocks, y_tol=10, max_x_dist=200):
        candidates = []
        for b in blocks:
            if b['id'] == block['id']: continue
            dy = abs(b['top'] - block['top'])
            dx = b['left'] - (block['left'] + block['width'])
            if dy <= y_tol and 0 <= dx <= max_x_dist:
                candidates.append(b)
        if candidates:
            return min(candidates, key=lambda x: x['left'] - (block['left'] + block['width']))
        return None

    def find_nearest_below(self, block, blocks, x_tol=20, max_y_dist=100):
        candidates = []
        for b in blocks:
            if b['id'] == block['id']: continue
            dx = abs(b['left'] - block['left'])
            dy = b['top'] - (block['top'] + block['height'])
            if dx <= x_tol and 0 <= dy <= max_y_dist:
                candidates.append(b)
        if candidates:
            return min(candidates, key=lambda x: b['top'] - (block['top'] + block['height']))
        return None

    def extract_key_values(self, blocks):
        """Extracts key-value pairs using spatial heuristics."""
        kv_pairs = []
        used_values = set()
        
        for block in blocks:
            text_lower = block['text'].lower()
            for key in self.target_keys:
                # Match if key is in the text
                if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
                    # Find a block to the right
                    value_block = self.find_nearest_right(block, blocks)
                    if not value_block:
                        # Fallback: look directly below
                        value_block = self.find_nearest_below(block, blocks)
                    
                    if value_block and value_block['id'] not in used_values:
                        kv_pairs.append({
                            "key": block['text'],
                            "key_id": block['id'],
                            "value": value_block['text'],
                            "value_id": value_block['id'],
                            "confidence": round((block['conf'] + value_block['conf']) / 200.0, 4) # Normalized 0-1
                        })
                        used_values.add(value_block['id'])
                    break # Stop looking for other keys for this block
        return kv_pairs

    def detect_tables(self, blocks, y_tolerance=10, min_cols=3):
        """Basic heuristic table detection."""
        if not blocks:
            return []
            
        # 1. Cluster into rows
        rows = []
        sorted_blocks = sorted(blocks, key=lambda b: b['top'])
        current_row = [sorted_blocks[0]]
        
        for block in sorted_blocks[1:]:
            if abs(block['top'] - current_row[0]['top']) <= y_tolerance:
                current_row.append(block)
            else:
                rows.append(sorted(current_row, key=lambda b: b['left']))
                current_row = [block]
        rows.append(sorted(current_row, key=lambda b: b['left']))
        
        # 2. Filter rows that look like tables
        tables = []
        current_table_rows = []
        
        for row in rows:
            if len(row) >= min_cols:
                current_table_rows.append(row)
            else:
                if len(current_table_rows) >= 2:
                    tables.append(self._format_table(current_table_rows, f"tbl_{len(tables):03d}"))
                current_table_rows = []
                
        if len(current_table_rows) >= 2:
            tables.append(self._format_table(current_table_rows, f"tbl_{len(tables):03d}"))
            
        return tables
        
    def _format_table(self, table_rows, table_id):
        rows_data = []
        all_blocks = []
        for row in table_rows:
            row_data = []
            for item in row:
                row_data.append({
                    "text": item['text'],
                    "block_id": item['id']
                })
                all_blocks.append(item)
            rows_data.append(row_data)
            
        # Compute bounding box for the whole table
        left = min(b['left'] for b in all_blocks)
        top = min(b['top'] for b in all_blocks)
        right = max(b['left'] + b['width'] for b in all_blocks)
        bottom = max(b['top'] + b['height'] for b in all_blocks)
        
        return {
            "table_id": table_id,
            "bbox": [left, top, right - left, bottom - top],
            "rows": rows_data
        }

    def generate_multimodal_representation(self, ocr_data):
        """
        Generates the final JSON representation with advanced layout encoding.
        """
        if not ocr_data:
            return {"metadata": {"version": "2.0"}, "visual_layout_encoding": {}}
            
        # 1. Post-process: merge fragmented blocks
        merged_blocks = self.merge_fragmented_boxes(ocr_data)
        
        # 2. Extract Key-Values
        kv_pairs = self.extract_key_values(merged_blocks)
        
        # 3. Detect Tables
        tables = self.detect_tables(merged_blocks)
        
        # 4. Construct the output schema
        representation = {
            "metadata": {
                "version": "2.0",
                "format": "multimodal_document_encoding"
            },
            "visual_layout_encoding": {
                "text_blocks": merged_blocks,
                "entities": {
                    "key_value_pairs": kv_pairs,
                    "tables": tables
                }
            }
        }
        
        return representation

if __name__ == "__main__":
    encoder = LayoutEncoder()
    print("Advanced LayoutEncoder initialized.")
