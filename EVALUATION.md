# Multimodal QA Engine Evaluation

This document details the performance metrics and simulated evaluation of the Multimodal QA Engine across three distinct financial document formats: Standard Invoices, Retail Receipts, and Bank Statements.

## Executive Summary
The system currently operates at a **93.3% accuracy rate** on the 15-question evaluation suite. It leverages lightweight heuristic logic, specifically:
- **Intent Normalization:** Dynamically recognizing synonyms without strict programmatic matches.
- **Strict Data-Type Filtering:** Using regex strings to safely guarantee the algorithm ignores alphabetically misaligned noise when hunting for numeric currency answers.
- **Geometrical Tracing:** Replacing blind "closest neighbor" mapping with weighted horizontal Right/Same-Line prioritization.

---

## Evaluation Results

| # | Question / Query | Predicted Answer | Expected Answer (Ground Truth) | Result |
|---|---|---|---|---|
| **1.** | What is the total amount? | $5,000.00 | $5,000.00 | ✔ Correct Answer |
| **2.** | What is the grand total due? | $5,000.00 | $5,000.00 | ✔ Correct Answer |
| **3.** | What is the invoice number? | INV-9901 | INV-9901 | ✔ Correct Answer |
| **4.** | What is the balance? | $5,000.00 | $5,000.00 | ✔ Correct Answer |
| **5.** | What is the tax amount? *(No tax listed)* | Relevant information not found. | Relevant information not found. | ✔ Correct Failure |
| **6.** | What is the shipping weight? | Relevant information not found. | Relevant information not found. | ✔ Correct Failure |
| **7.** | How much is the subtotal? | $7.50 | $7.50 | ✔ Correct Answer |
| **8.** | What is the total? | $8.10 | $8.10 | ✔ Correct Answer |
| **9.** | What is the tax amount? | $0.60 | $0.60 | ✔ Correct Answer |
| **10.**| Show me the total vat. | $0.60 | $0.60 | ✔ Correct Answer |
| **11.**| What is the tip? | Relevant information not found. | Relevant information not found. | ✔ Correct Failure |
| **12.**| What is the date? | 10/12/2026 | 10/12/2026 | ✔ Correct Answer |
| **13.**| What is the closing balance? | $1,200.00 | $14,550.00 | ❌ Incorrect Answer |
| **14.**| What is the transaction amount? | $1,200.00 | $1,200.00 | ✔ Correct Answer |
| **15.**| Tell me the tax charged on the transfer. | Relevant information not found. | Relevant information not found. | ✔ Correct Failure |

### Metric Breakdown
- **Correct Responses:** 11  
- **Graceful / Correct Failures:** 3  
- **Incorrect Guesses:** 1  
- **Overall Accuracy:** 93.3%  

---

## Architectural Analysis

### Where the Architecture Shines
1. **Safety over Hallucination:** The largest structural benefit is its failure handling. As seen in queries 5, 6, 11, and 15, if the engine cannot strictly identify an anchor keyword to build its spatial geometry around, it entirely refuses to answer rather than guessing on the biggest number on the page. In financial technology, avoiding false positives is critical.
2. **Dynamic Semantic Routing:** The system successfully evaluates unique human phrasing. E.g., interpreting "What is the grand total due?" natively into the target label *"Total Amount:"* by passing tokens through the unified dictionary mapping.

### Known Vulnerabilities
**Target Collision Drop-off (Query #13 Failure):**
The one evaluated failure occurred during Q13 ("*What is the closing balance?*"). Because our intent dictionary lumps "balance" and "amount" together under `total_amount`, the engine simply searched the document block sequentially. It struck `"Transfer Amount:"` before it ever hit `"Ending Balance:"`, mistakenly halting execution and returning $1,200.00 instead of $14,550.00. 

**Future Resolution (V2 Model):** To permanently fix target collision, the architecture must transition from heuristic dictionary loops to contextual vector-embeddings (e.g., LayoutLMv3), allowing it to evaluate document context holistically rather than stopping upon discovering the first partial token overlap.
