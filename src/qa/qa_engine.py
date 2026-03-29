import re
import math
import string

"""
MULTIMODAL FIN-DOC QA ENGINE (v1.0 - Heuristic Baseline)

This module represents the algorithmic staging ground for transforming visual coordinates 
and text layouts into explainable, structured answers. It avoids LLM/hallucinations by using:
    1. Intent Normalization: Grouping user queries into manageable intent sets.
    2. Candidate Type Safety: Applying rigorous currency/numeric format checking.
    3. Mathematical Geometric Matching: Prioritizing spatial distance formulas over nearest-neighbor reading orders.
    4. Explanatory Tracing: Generating the 'Why' behind every programmatic block selection.
"""

class MultimodalQAEngine:
    def __init__(self):
        # A simple list of common English stopwords to ignore in questions
        self.stopwords = {
            "what", "is", "the", "of", "in", "a", "an", "find", "show", "tell", "me", "are", "do", "does", "for", "on", "at", "to", "how", "much", "many"
        }
        
        # Normalized intent mapping to handle synonyms and variations
        # Ordered by specificity so specific categories can override broad ones
        self.intent_map = {
            "tax": {"tax", "vat", "gst"},
            "date": {"date", "when", "day", "year"},
            "invoice_number": {"invoice", "number", "id", "no"},
            "total_amount": {"total", "amount", "balance", "due", "net", "sum", "grand"}
        }
        
        # Regex for financial amounts to filter irrelevant answers
        self.amount_regex = re.compile(r'[$€£¥]?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?')

    def tokenize_and_normalize(self, text):
        """Standardizes text by lowercasing and removing punctuation."""
        if not text:
            return []
        text = text.lower()
        # Replace punctuation with space to separate words properly
        for p in string.punctuation:
            text = text.replace(p, ' ')
        return [word for word in text.split() if word.strip()]

    def detect_intent(self, question):
        """Maps user question to a predefined intent category and returns expanded keywords."""
        tokens = set(self.tokenize_and_normalize(question)) - self.stopwords
        best_intent = "unknown"
        
        # Priority resolution: strict ordered evaluation over generic overlap counting
        # This prevents generic words like "amount" from overriding specific queries like "tax"
        for intent, synonyms in self.intent_map.items():
            if tokens.intersection(synonyms):
                best_intent = intent
                break
                
        # If an intent is cleanly formulated, strictly search using its synonyms.
        # This prevents generic user words (like "amount") from causing cross-talk with "total" labels when asking for "tax".
        if best_intent != "unknown":
            keywords = self.intent_map[best_intent]
        else:
            keywords = tokens
            
        return best_intent, keywords

    def is_valid_candidate(self, text, intent):
        """Filters answers based on expected conceptual type."""
        # Both total amounts and tax values should strictly be numeric currencies
        if intent in ("total_amount", "tax"):
            return bool(self.amount_regex.search(text))
        return len(text.strip()) > 0

    def answer_question(self, question, multimodal_data):
        """
        Interprets a user question, maps it to the multimodal representation,
        and returns an answer with explainability context (reasoning and evidence).
        """
        intent, keywords = self.detect_intent(question)
        if not keywords:
            return {
                "answer": "Please provide a more specific question.",
                "evidence": [],
                "reasoning": "[Intent: unknown] -> [Label Match: None] -> [Candidate Filtering: Failed] -> [Spatial Selection: N/A] -> [Final Answer: Please specify clearly]"
            }
            
        visual_encoding = multimodal_data.get("visual_layout_encoding", {})
        text_blocks = visual_encoding.get("text_blocks", [])
        entities = visual_encoding.get("entities", {})
        kv_pairs = entities.get("key_value_pairs", [])
        tables = entities.get("tables", [])

        if not text_blocks:
            return {
                "answer": "No valid text detected in the document.",
                "evidence": [],
                "reasoning": "[Intent: Evaluated] -> [Label Match: Failed] -> [Candidate Filtering: N/A] -> [Spatial Selection: N/A] -> [Final Answer: OCR returned empty]"
            }

        # Step 1: Check existing key-value pairs from the LayoutEncoder.
        # This takes advantage of the system's structural understanding.
        best_kv = None
        best_kv_score = 0
        for kv in kv_pairs:
            key_tokens = set(self.tokenize_and_normalize(kv["key"]))
            match_score = len(keywords.intersection(key_tokens))
            if match_score > best_kv_score:
                best_kv_score = match_score
                best_kv = kv
                
        if best_kv and best_kv_score > 0:
            evidence_blocks = [
                b for b in text_blocks 
                if b["id"] in (best_kv["key_id"], best_kv["value_id"])
            ]
            
            reasoning = (
                f"[Intent: {intent}] -> "
                f"[Label Match: Matched pre-extracted Key '{best_kv['key']}'] -> "
                f"[Candidate Filtering: Accepted pre-extracted Value '{best_kv['value']}'] -> "
                f"[Spatial Selection: Validated by layout engine grouping] -> "
                f"[Final Answer: {best_kv['value']}]"
            )
            return {
                "answer": best_kv["value"],
                "evidence": evidence_blocks,
                "reasoning": reasoning
            }

        # Step 2: Spatial Reasoning fallback. Find the 'label' text block and find nearest 
        # value in reading order (usually to the right or below).
        best_block = None
        best_block_score = 0
        for block in text_blocks:
            block_tokens = set(self.tokenize_and_normalize(block["text"]))
            match_score = len(keywords.intersection(block_tokens))
            if match_score > best_block_score:
                best_block_score = match_score
                best_block = block

        if best_block and best_block_score > 0:
            candidates = []
            for other_block in text_blocks:
                if other_block["id"] == best_block["id"]:
                    continue
                
                # Answer filtering: only keep candidates matching expected format
                if not self.is_valid_candidate(other_block["text"], intent):
                    continue
                    
                # Advanced spatial relation
                dx = other_block["left"] - (best_block["left"] + best_block["width"])
                dy = other_block["top"] - best_block["top"]
                
                # Discard candidates clearly above or strictly left
                if dy < -10 or (other_block["left"] + other_block["width"] < best_block["left"]):
                    continue
                
                is_same_line = abs(dy) <= 15
                is_right = dx >= -10
                
                score = 0
                if is_same_line and is_right:
                    # Priority 1: Right adjacent on the identically aligned layout line
                    score = 1000 - max(0, dx)
                elif not is_same_line and dy > 0:
                    # Priority 2: Directly below the label
                    score = 500 - max(0, dy) - abs(other_block["left"] - best_block["left"])
                
                if score > 0:
                    candidates.append((score, other_block))
                
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                nearest_block = candidates[0][1]
                
                spatial_reason = "Selected nearest item below label."
                if abs(nearest_block["top"] - best_block["top"]) <= 15:
                    spatial_reason = "Selected adjacent item on same horizontal line to the right."
                    
                reasoning = (
                    f"[Intent: {intent}] -> "
                    f"[Label Match: Found text anchor '{best_block['text']}'] -> "
                    f"[Candidate Filtering: Verified '{nearest_block['text']}' fits numeric/text format] -> "
                    f"[Spatial Selection: {spatial_reason}] -> "
                    f"[Final Answer: {nearest_block['text']}]"
                )
                
                return {
                    "answer": nearest_block["text"],
                    "evidence": [best_block, nearest_block],
                    "reasoning": reasoning
                }
            else:
                return {
                    "answer": best_block["text"],
                    "evidence": [best_block],
                    "reasoning": f"[Intent: {intent}] -> [Label Match: Found anchor '{best_block['text']}'] -> [Candidate Filtering: All nearby candidates failed type check] -> [Spatial Selection: Failed] -> [Final Answer: {best_block['text']}]"
                }
                
        return {
            "answer": "Relevant information not found.",
            "evidence": [],
            "reasoning": f"[Intent: {intent}] -> [Label Match: No labels matched keywords '{', '.join(keywords)}'] -> [Candidate Filtering: N/A] -> [Spatial Selection: N/A] -> [Final Answer: Not found]"
        }
