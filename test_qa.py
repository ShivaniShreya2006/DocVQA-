import json
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.qa.qa_engine import MultimodalQAEngine

def main():
    # Simulated output from LayoutEncoder
    multimodal_data = {
        "visual_layout_encoding": {
            "text_blocks": [
                {"id": "blk_01", "text": "Invoice Number:", "left": 50, "top": 50, "width": 180, "height": 20},
                {"id": "blk_02", "text": "INV-5524A", "left": 300, "top": 50, "width": 100, "height": 20},
                {"id": "blk_03", "text": "Subtotal:", "left": 50, "top": 350, "width": 100, "height": 20},
                {"id": "blk_04", "text": "$4,500.00", "left": 350, "top": 350, "width": 90, "height": 20},
                {"id": "blk_05", "text": "Total Amount:", "left": 50, "top": 400, "width": 160, "height": 20},
                {"id": "blk_06", "text": "$5,000.00", "left": 350, "top": 400, "width": 90, "height": 20}
            ],
            "entities": {
                "key_value_pairs": [],
                "tables": []
            }
        }
    }

    engine = MultimodalQAEngine()
    
    questions = [
        "What is the total amount?",       # Tests total intent + amount value
        "How much is the subtotal?",       # Tests finding the subtotal explicitly
        "Tell me the invoice number",      # Tests non-monetary value fetching
        "What is the grand total due?",    # Tests synonym intent matching
        "What is the tax amount?"          # Tests intentional failure case (missing label)
    ]
    
    print("\n" + "="*70)
    print("   MULTIMODAL FIN-DOC QA ENGINE SIMULATION   ")
    print("="*70 + "\n")
    
    for q in questions:
        print(f"Q: \"{q}\"")
        result = engine.answer_question(q, multimodal_data)
        print(f"A: {result['answer']}")
        print(f"Trace:\n  {result['reasoning']}")
        print("\n" + "-"*70 + "\n")

if __name__ == "__main__":
    main()
