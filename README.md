# Multimodal Fin-Doc QA Engine
A Vision-Language System for Intelligent Question Answering and Information Extraction from Financial Documents.

## The Goal
This system bridges the gap between raw document text (OCR) and high-level reasoning. 
Our QA module isn't just reading text—it evaluates the **physical layout geometry** (where text sits logically on the page) to accurately map user questions to exact financial values.

### Why not just use a massive Language Model?
Financial environments require **zero hallucinations**. Instead of deploying a black-box LLM that might blindly guess a total, our engineering leverages strict **Heuristics & Spatial Mathematics**. It mathematically verifies an answer's location before ever returning it.

---

## How It Works (Step-by-Step Logic)
If a user asks: *"What is the grand total due?"*

1. **Intent Normalization:** The engine groups human synonyms ("grand", "due", "balance") into strict technical intents (e.g., `total_amount`). Strong intents like "Tax" are programmed to securely override weak intents.
2. **Anchor Hunting (Label Match):** It queries the structured OCR groupings to mathematically lock onto the textual anchor (e.g., finding the exact bounding box for "Total Amount:").
3. **Candidate Filtering (Type Safety):** It establishes a search radius and applies extreme regex formatting. If the user asks for a monetary value, the engine aggressively ignores any adjacent text lacking numerical digits/currency formats.
4. **Spatial Geometric Selection:** It calculates the distance between `[X, Y, Width, Height]` bounding arrays. A target value existing perfectly on the exact identical horizontal Y-axis and strictly to the right receives immense prioritization.
5. **Explainable Output:** It publishes a step-by-step trace `[Intent] -> [Label Match] -> [Filter] -> [Spatial]` so human auditors can permanently trace *why* an algorithm made a decision.

---

## 93.3% Accuracy Evaluation

We ran a Comprehensive Simulation Suite (`test_qa_comprehensive.py`) feeding 15 unique semantic queries into 3 structured mock documents (Standard Invoices, Retail Receipts, and Bank Statements). 

**See `EVALUATION.md` for the massive, full dataset breakdown.**

### Where The Architecture Shines
**Safety over Hallucination:** The single greatest benefit to our heuristic algorithm is its failure handling. If asked *"What is the tax amount?"* on a document that physically does not contain tax labels, the engine's geometry math fails to map an anchor. It then entirely refuses to respond rather than guessing the biggest number nearby. In financial technology, avoiding false positives is critical.

### Known Vulnerabilities
**Target Collision Drop-off:**
The only evaluated failure occurred during query #13 (*"What is the closing balance?"*). Because our intent dictionary currently groups "balance" and "amount" together, the engine searched the document and accidentally struck `"Transfer Amount:"` before it reached `"Ending Balance:"`, resulting in an incorrect extraction. 

**Future Roadmap Resolution:** 
To permanently fix sequential target collision in V2, the architecture must eventually transition from heuristic dictionary loops to contextual vector embeddings (e.g., `LayoutLMv3`). This will allow the machine to evaluate surrounding sentence context globally rather than violently halting upon discovering the first overlapping semantic token!
