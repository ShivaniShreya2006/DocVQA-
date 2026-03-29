import re
import math
import string

class MultimodalQAEngine:
    def __init__(self):
        # A simple list of common English stopwords to ignore in questions
        self.stopwords = {
            "what", "is", "the", "of", "in", "a", "an", "find", "show", "tell", "me", "are", "do", "does", "for", "on", "at", "to"
        }

    def tokenize_and_normalize(self, text):
        """Standardizes text by lowercasing and removing punctuation."""
        if not text:
            return []
        text = text.lower()
        # Replace punctuation with space to separate words properly
        for p in string.punctuation:
            text = text.replace(p, ' ')
        return [word for word in text.split() if word.strip()]

    def extract_keywords(self, question):
        """Extracts significant terms from the user query."""
        tokens = self.tokenize_and_normalize(question)
        keywords = set(tokens) - self.stopwords
        return keywords

    def answer_question(self, question, multimodal_data):
        """
        Interprets a user question, maps it to the multimodal representation,
        and returns an answer with explainability context (reasoning and evidence).
        """
        keywords = self.extract_keywords(question)
        if not keywords:
            return {
                "answer": "Please provide a more specific question.",
                "evidence": [],
                "reasoning": "The question only contained stop words."
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
                "reasoning": "The OCR did not find text blocks."
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
            return {
                "answer": best_kv["value"],
                "evidence": evidence_blocks,
                "reasoning": f"Found direct matching key-value pair extracted from layout: Key='{best_kv['key']}', Value='{best_kv['value']}'"
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
                # Calculate positional distance
                # Only consider blocks roughly to the right or below, avoiding blocks fully above or left
                dx = other_block["left"] - (best_block["left"] + best_block["width"])
                dy = other_block["top"] - (best_block["top"] + best_block["height"])
                
                if dy < -10 and other_block["left"] + other_block["width"] < best_block["left"]:
                    continue  # The block is above and to the left
                
                # Approximate distance metric prioritizing horizontal over vertical
                dist = math.sqrt(max(0, dx)**2 + max(0, dy)**2 + (other_block["top"] - best_block["top"])**2)
                candidates.append((dist, other_block))
                
            if candidates:
                candidates.sort(key=lambda x: x[0])
                nearest_block = candidates[0][1]
                return {
                    "answer": nearest_block["text"],
                    "evidence": [best_block, nearest_block],
                    "reasoning": f"Found keyword match '{best_block['text']}'. Associated it with nearest spatial text block '{nearest_block['text']}' based on bounding box proximity."
                }
            else:
                return {
                    "answer": best_block["text"],
                    "evidence": [best_block],
                    "reasoning": f"Found text matching keywords: '{best_block['text']}', but could not find a distinct adjacent value block."
                }
                
        return {
            "answer": "I couldn't find relevant information in the document to answer your question.",
            "evidence": [],
            "reasoning": f"No text in the document matched the keywords: {', '.join(keywords)}."
        }
