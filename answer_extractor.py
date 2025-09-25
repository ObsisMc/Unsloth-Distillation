#!/usr/bin/env python3
"""
Answer extractor for multi-choice questions using OpenAI's structured output.
"""

import re
import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field
import openai
from dotenv import load_dotenv

load_dotenv()

class AnswerExtraction(BaseModel):
    """Structured output model for answer extraction."""
    selected_answers: List[str] = Field(
        description="List of selected answer choices (e.g., ['A', 'B'])",
        examples=[["A"], ["A", "C"], ["B", "D", "E"]]
    )

class OpenAIAnswerExtractor:
    """OpenAI-based answer extractor using structured output for multi-choice questions."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Initialize the OpenAI answer extractor.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: OpenAI model to use (gpt-3.5-turbo, gpt-4, etc.)
        """
        self.api_key = api_key or self._get_api_key()
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client as instance variable
        self.client = openai.OpenAI(api_key=self.api_key)
        
        print(f"OpenAI Answer Extractor initialized with model: {model}")
    
    def _get_api_key(self) -> str:
        """Get API key from environment variable."""
        return os.getenv('OPENAI_API_KEY')
    
    def extract_answers(self, response: str, question: str = None) -> List[str]:
        """
        Extract answer choices from model response using OpenAI's structured output.
        
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
- Return them as a list of uppercase letters
- Include ALL selected answers
- If no clear answer is given, return an empty list

Examples:
- If the model selected A and C: ["A", "C"]
- If the model selected only B: ["B"]
- If no clear answer: []"""

            # Call OpenAI API with structured output
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "user", "content": extraction_prompt}
                ],
                response_format=AnswerExtraction,
                temperature=0.1
            )
            
            # Parse the structured response
            result = completion.choices[0].message.parsed
            return result.selected_answers
            
        except Exception as e:
            print(f"Error in OpenAI structured extraction: {e}")
            # Fallback to regex extraction
            return self._regex_extraction(response)
    
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
    model = "gpt-4o-mini"
    extractor = OpenAIAnswerExtractor(model=model)
    
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
            "response": "The correct answers are B and D",
            "question": "Test question",
            "expected": ["B", "D"]
        },
        {
            "response": "Let's analyze this step by step. The question asks about cutting method selection. Looking at the options, option A mentions economy, process capability, and material effects as key factors in selection, which is absolutely correct since these are fundamental considerations. Option B makes an incorrect absolute statement about thermal cutting being always best, which can't be true since different materials and situations require different approaches. Option C correctly notes that nonthermal methods, while slower, offer precision advantages for various materials - this is accurate particularly for materials that may be damaged by thermal processes. Option D is incorrect as thermal cutting methods are generally better suited for metals rather than nonmetals. Therefore, based on this analysis, the correct answers are A and C.",
            "question": "3. Which of the following statements about cutting method selection are correct?\nA: The selection of a cutting method should prioritize economy, process capability, and material effects.\nB: Thermal cutting methods are always the best choice for all materials.\nC: Nonthermal cutting methods are slower but offer precision on various materials.\nD: Thermal cutting methods are more suitable for nonmetals than metals.",
            "expected": ["A", "C"]
        },
        {
            "response": "Based on the question about Electroslag Welding, I believe option A is correct because shielding gas is essential for protecting the molten pool.",
            "question": "1. Which of the following statements about Electroslag Welding (ESW) is correct?\nA: Electroslag welding requires shielding gas to protect the molten pool.\nB: Electroslag welding starts with an arc and then transitions to an electric resistance process once the molten pool is high enough.\nC: The process of electroslag welding relies entirely on the arc heating, without transitioning to other methods.\nD: Pressure is applied during the electroslag welding process to enhance the fusion of metals.",
            "expected": ["B"]
        },
        {
            "response": "For the question about cutting method selection, I would say options B and D are correct. Thermal cutting methods have proven to be highly effective across all materials, and they work particularly well with nonmetals due to their lower melting points.",
            "question": "3. Which of the following statements about cutting method selection are correct?\nA: The selection of a cutting method should prioritize economy, process capability, and material effects.\nB: Thermal cutting methods are always the best choice for all materials.\nC: Nonthermal cutting methods are slower but offer precision on various materials.\nD: Thermal cutting methods are more suitable for nonmetals than metals.",
            "expected": ["A", "C"]
        }
    ]
    
    print("Testing OpenAI Answer Extractor:")
    print("=" * 40)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Response: {test['response']}")
        
        extracted = extractor.extract_answers(test['response'], test['question'])
        print(f"Extracted: {extracted}")
        print(f"Expected: {test['expected']}")
        print(f"Correct: {set(extracted) == set(test['expected'])}")
