#!/usr/bin/env python3
"""
Answer extractor for multi-choice questions using LLM.
"""

import re
import json
import torch
from typing import List, Optional
from unsloth import FastLanguageModel

class AnswerExtractor:
    """LLM-based answer extractor for multi-choice questions."""
    
    def __init__(self, model_name: str, max_seq_length: int = 2048, 
                 dtype=None, load_in_4bit: bool = True, device: str = "cuda"):
        """
        Initialize the answer extractor.
        
        Args:
            model_name: Name of the model to use for extraction
            max_seq_length: Maximum sequence length
            dtype: Data type (None for auto detection)
            load_in_4bit: Use 4bit quantization
            device: Device to run inference on
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.device = device
        
        # Load model and tokenizer
        print(f"Loading extraction model: {model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        # Enable inference mode
        FastLanguageModel.for_inference(self.model)
        print("Extraction model loaded successfully!")
    
    def extract_answers(self, response: str, question: str = None) -> List[str]:
        """
        Extract answer choices from model response.
        
        Args:
            response: Model response string
            question: Original question for context (optional)
            
        Returns:
            List of extracted answer choices (e.g., ['A', 'B'])
        """
        try:
            # Create extraction prompt
            extraction_prompt = f"""You are an expert at extracting answer choices from model responses for multiple choice questions.

Question: {question or 'Multiple choice question'}

Model Response: {response}

Please extract ONLY the answer choices (A, B, C, D, E) that the model selected. 
- Return them as a JSON array
- Include ALL selected answers
- Use only uppercase letters
- If no clear answer is given, return an empty array []

Example format: ["A", "C"] or ["B"] or []

Extracted answers:"""

            # Prepare input
            input_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": extraction_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length
            ).to(self.device)
            
            # Generate extraction response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Short response for extraction
                    use_cache=True,
                    repetition_penalty=1.1,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            extraction_response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Parse the extraction response
            return self._parse_extraction_response(extraction_response)
            
        except Exception as e:
            print(f"Error in LLM extraction: {e}")
            # Fallback to regex extraction
            return self._regex_extraction(response)
    
    def _parse_extraction_response(self, response: str) -> List[str]:
        """
        Parse the extraction response using multiple methods.
        
        Args:
            response: Extraction response from LLM
            
        Returns:
            List of extracted answer choices
        """
        try:
            # Method 1: Try to parse as JSON directly
            try:
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    # Filter only valid answer choices
                    valid_answers = [str(item).upper() for item in parsed if str(item).upper() in ['A', 'B', 'C', 'D', 'E']]
                    return list(set(valid_answers))  # Remove duplicates
            except json.JSONDecodeError:
                pass
            
            # Method 2: Look for JSON array pattern
            json_match = re.search(r'\[["\']?([A-E])["\']?(?:,\s*["\']?([A-E])["\']?)*\]', response)
            if json_match:
                # Extract letters from the match
                letters = re.findall(r'[A-E]', json_match.group())
                return list(set(letters))  # Remove duplicates
            
            # Method 3: Look for individual letters in order
            letters = re.findall(r'\b([A-E])\b', response.upper())
            if letters:
                return list(set(letters))  # Remove duplicates
            
            # Method 4: Look for "Answer: A, B" pattern
            answer_match = re.search(r'(?:answer|answers?)[:\s]+([A-E](?:,\s*[A-E])*)', response.lower())
            if answer_match:
                letters = re.findall(r'[A-E]', answer_match.group(1).upper())
                return list(set(letters))  # Remove duplicates
            
            # If no valid answers found, return empty list
            return []
            
        except Exception as e:
            print(f"Error parsing extraction response: {e}")
            return []
    
    def _regex_extraction(self, response: str) -> List[str]:
        """
        Fallback regex extraction method.
        
        Args:
            response: Model response string
            
        Returns:
            List of extracted answer choices
        """
        # Look for patterns like "A", "B", "C", etc. in the response
        answer_pattern = r'\b([A-E])\b'
        matches = re.findall(answer_pattern, response.upper())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_answers = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                unique_answers.append(match)
        
        return unique_answers

# Example usage
if __name__ == "__main__":
    # Test the extractor
    extractor = AnswerExtractor(
        model_name="Qwen/Qwen2.5-1.5B",  # Use a smaller model for extraction
        max_seq_length=2048,
        load_in_4bit=True
    )
    
    # Test cases
    test_cases = [
        {
            "response": "根据沸腾钢的特点，正确答案是A和C。",
            "question": "1. 沸腾钢的特点是什么?\nA: 金属表面外层较纯.\nB: 夹杂物分布均匀.\nC: 有偏析区.",
            "expected": ["A", "C"]
        },
        {
            "response": "I think the answer is A, C",
            "question": "Test question",
            "expected": ["A", "C"]
        },
        {
            "response": '["A", "C"]',
            "question": "Test question",
            "expected": ["A", "C"]
        }
    ]
    
    print("Testing Answer Extractor:")
    print("=" * 40)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Response: {test['response']}")
        
        extracted = extractor.extract_answers(test['response'], test['question'])
        print(f"Extracted: {extracted}")
        print(f"Expected: {test['expected']}")
        print(f"Correct: {set(extracted) == set(test['expected'])}")
