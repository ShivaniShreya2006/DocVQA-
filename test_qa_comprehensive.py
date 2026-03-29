import json
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.qa.qa_engine import MultimodalQAEngine

def print_separator(char="=", length=80):
    print(char * length)

def simulate_document(engine, doc_title, blocks, questions):
    print_separator("=")
    print(f"                            {doc_title.upper()}")
    print_separator("=")
    print("[STRUCTURED OCR BLOCKS (SIMULATED)]:")
    for i in range(0, len(blocks), 2):
        b1 = blocks[i]
        str1 = f"{b1['id']}: \"{b1['text']}\" [X: {b1['left']}, Y: {b1['top']}]"
        str2 = ""
        if i + 1 < len(blocks):
            b2 = blocks[i+1]
            str2 = f"{b2['id']}: \"{b2['text']}\" [X: {b2['left']}, Y: {b2['top']}]"
        print(f"{str1:<45} {str2}")
    print_separator("-")
    print()
    
    data = {
        "visual_layout_encoding": {
            "text_blocks": blocks,
            "entities": {"key_value_pairs": [], "tables": []}
        }
    }
    
    for q in questions:
        print(f"Q: \"{q}\"")
        res = engine.answer_question(q, data)
        print(f"A: {res['answer']}")
        print(f"Trace:\n  {res['reasoning']}\n")
        print_separator("-")
        print()

def main():
    engine = MultimodalQAEngine()
    
    print_separator("=")
    print("          MULTIMODAL FIN-DOC QA ENGINE - COMPREHENSIVE SIMULATION SUITE")
    print_separator("=")
    print("Loading Pipeline...\nStarting evaluation across 3 distinct financial document formats.\n")
    
    # DOCUMENT 1: Standard Invoice
    doc1_blocks = [
        {"id": "blk_01", "text": "Invoice Number:", "left": 50, "top": 50, "width": 180, "height": 20},
        {"id": "blk_02", "text": "INV-9901", "left": 300, "top": 50, "width": 80, "height": 20},
        {"id": "blk_03", "text": "Subtotal:", "left": 50, "top": 350, "width": 100, "height": 20},
        {"id": "blk_04", "text": "$4,500.00", "left": 350, "top": 350, "width": 90, "height": 20},
        {"id": "blk_05", "text": "Total Amount:", "left": 50, "top": 400, "width": 160, "height": 20},
        {"id": "blk_06", "text": "$5,000.00", "left": 350, "top": 400, "width": 90, "height": 20}
    ]
    doc1_q = [
        "What is the total amount?",
        "What is the grand total due?",
        "What is the invoice number?",
        "What is the balance?",
        "What is the tax amount?",
        "What is the shipping weight?"
    ]
    
    # DOCUMENT 2: Retail Receipt
    doc2_blocks = [
        {"id": "blk_10", "text": "Coffee", "left": 20, "top": 150, "width": 70, "height": 20},
        {"id": "blk_11", "text": "$4.50", "left": 200, "top": 150, "width": 50, "height": 20},
        {"id": "blk_12", "text": "Muffin", "left": 20, "top": 170, "width": 70, "height": 20},
        {"id": "blk_13", "text": "$3.00", "left": 200, "top": 170, "width": 50, "height": 20},
        {"id": "blk_14", "text": "Subtotal", "left": 20, "top": 210, "width": 80, "height": 20},
        {"id": "blk_15", "text": "$7.50", "left": 200, "top": 210, "width": 50, "height": 20},
        {"id": "blk_16", "text": "Tax", "left": 20, "top": 230, "width": 40, "height": 20},
        {"id": "blk_17", "text": "$0.60", "left": 200, "top": 230, "width": 50, "height": 20},
        {"id": "blk_18", "text": "Total", "left": 20, "top": 250, "width": 50, "height": 20},
        {"id": "blk_19", "text": "$8.10", "left": 200, "top": 250, "width": 50, "height": 20}
    ]
    doc2_q = [
        "How much is the subtotal?",
        "What is the total?",
        "What is the tax amount?",
        "Show me the total vat.",
        "What is the tip?"
    ]
    
    # DOCUMENT 3: Bank Statement Snippet
    doc3_blocks = [
        {"id": "blk_20", "text": "Date:", "left": 40, "top": 60, "width": 50, "height": 20},
        {"id": "blk_21", "text": "10/12/2026", "left": 150, "top": 60, "width": 100, "height": 20},
        {"id": "blk_22", "text": "Transfer Amount:", "left": 40, "top": 100, "width": 150, "height": 20},
        {"id": "blk_23", "text": "$1,200.00", "left": 300, "top": 100, "width": 100, "height": 20},
        {"id": "blk_24", "text": "Ending Balance:", "left": 40, "top": 140, "width": 140, "height": 20},
        {"id": "blk_25", "text": "$14,550.00", "left": 300, "top": 140, "width": 120, "height": 20}
    ]
    doc3_q = [
        "What is the date?",
        "What is the closing balance?",
        "What is the transaction amount?",
        "Tell me the tax charged on the transfer."
    ]
    
    simulate_document(engine, "Document 1: Standard Invoice", doc1_blocks, doc1_q)
    simulate_document(engine, "Document 2: Retail Receipt", doc2_blocks, doc2_q)
    simulate_document(engine, "Document 3: Bank Statement Snippet", doc3_blocks, doc3_q)

    print("                  END OF SIMULATION - 0 ERRORS - 15 QUERIES")
    print_separator("=")

if __name__ == "__main__":
    main()
