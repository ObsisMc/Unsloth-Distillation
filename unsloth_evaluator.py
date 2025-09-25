#!/usr/bin/env python3
"""
Unsloth model evaluator for GPQA-V1 dataset.
Integrates Unsloth-trained models with the evaluation system.
"""

import json
import re
import time
from typing import List, Dict, Any
from pathlib import Path
import torch
from unsloth import FastLanguageModel
from transformers import DataCollatorForSeq2Seq
from answer_extractor import OpenAIAnswerExtractor

class UnslothEvaluator:
    def __init__(self, model_name: str, max_seq_length: int = 4096, 
                 dtype=None, load_in_4bit: bool = True, device: str = "cuda",
                 extraction_model_name: str = None):
        """
        Initialize the Unsloth evaluator.
        
        Args:
            model_name: Path to the Unsloth model
            max_seq_length: Maximum sequence length
            dtype: Data type (None for auto detection)
            load_in_4bit: Use 4bit quantization
            device: Device to run inference on
            extraction_model_name: Optional different model for answer extraction
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.device = device
        
        # Load main model and tokenizer
        print(f"Loading Unsloth model from: {model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        # Enable inference mode
        FastLanguageModel.for_inference(self.model)
        print("Model loaded successfully!")
        
        # Initialize answer extractor
        extraction_model = extraction_model_name or model_name
        self.answer_extractor = OpenAIAnswerExtractor(
            model=extraction_model,
        )
        
    def call_model(self, question: str) -> str:
        """
        Call the Unsloth model to get response.
        
        Args:
            question: The question to ask the model
        Returns:
            Model response as string
        """
        try:
            # Prepare the input with chat template
            input_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_seq_length,
                    use_cache=True,
                    repetition_penalty=1.1,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def extract_answers(self, response: str, question: str = None) -> List[str]:
        """
        Extract answer choices from model response using AnswerExtractor.
        
        Args:
            response: Model response string
            question: Original question for context (optional)
            
        Returns:
            List of extracted answer choices (e.g., ['A', 'B'])
        """
        return self.answer_extractor.extract_answers(response, question)
    
    def calculate_accuracy(self, predicted: List[str], correct: List[str]) -> bool:
        """
        Calculate if the predicted answers are correct.
        
        Args:
            predicted: List of predicted answer choices
            correct: List of correct answer choices
            
        Returns:
            True if answers match exactly, False otherwise
        """
        # Sort both lists to compare regardless of order
        predicted_sorted = set(predicted)
        correct_sorted = set(correct)
        
        return predicted_sorted == correct_sorted
    
    def evaluate_dataset(self, dataset_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Evaluate the model on the entire dataset.
        
        Args:
            dataset_path: Path to the GPQA-V1.json file
            output_path: Optional path to save detailed results
            
        Returns:
            Dictionary containing evaluation results
        """
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)[:5]
        
        total_questions = len(dataset)
        correct_answers = 0
        detailed_results = []
        
        print(f"Evaluating Unsloth model on {total_questions} questions...")
        print("=" * 60)
        
        for i, item in enumerate(dataset, 1):
            question = item['question']
            correct_answer = item['answer']
            
            print(f"Question {i}/{total_questions}")
            print(f"Question: {question[:100]}...")
            
            # Get model response
            response = self.call_model(question)
            if not response:
                print("❌ Failed to get response from model")
                detailed_results.append({
                    'question_id': i,
                    'question': question,
                    'correct_answer': correct_answer,
                    'predicted_answer': [],
                    'response': response,
                    'correct': False,
                    'error': 'No response from model'
                })
                continue
            
            # Extract predicted answers
            predicted_answer = self.extract_answers(response)
            
            # Check if correct
            is_correct = self.calculate_accuracy(predicted_answer, correct_answer)
            if is_correct:
                correct_answers += 1
                print("✅ Correct")
            else:
                print("❌ Incorrect")
            
            print(f"Correct: {correct_answer}")
            print(f"Predicted: {predicted_answer}")
            print(f"Full Response: {response}")
            print("-" * 60)
            
            detailed_results.append({
                'question_id': i,
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'response': response,
                'correct': is_correct
            })
        
        # Calculate final metrics
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        results = {
            'model_name': self.model_name,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'detailed_results': detailed_results
        }
        
        # Save detailed results if output path provided
        if output_path:
            # Create comprehensive results with all data
            comprehensive_results = {
                'evaluation_summary': {
                    'model_name': results['model_name'],
                    'total_questions': results['total_questions'],
                    'correct_answers': results['correct_answers'],
                    'accuracy': results['accuracy'],
                    'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'detailed_responses': detailed_results,
                'response_statistics': self._calculate_response_statistics(detailed_results)
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_results, f, ensure_ascii=False, indent=2)
            print(f"\nComprehensive results saved to: {output_path}")
        
        return results
    
    def _calculate_response_statistics(self, detailed_results: List[Dict]) -> Dict:
        """
        Calculate comprehensive statistics from detailed results.
        
        Args:
            detailed_results: List of detailed evaluation results
            
        Returns:
            Dictionary containing response statistics
        """
        if not detailed_results:
            return {}
        
        # Basic statistics
        total_questions = len(detailed_results)
        correct_count = sum(1 for r in detailed_results if r['correct'])
        accuracy = correct_count / total_questions if total_questions > 0 else 0
        
        # Response length analysis
        response_lengths = [len(r['response']) for r in detailed_results]
        avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        # Answer pattern analysis
        correct_answers = [r['correct_answer'] for r in detailed_results]
        predicted_answers = [r['predicted_answer'] for r in detailed_results]
        
        # Count answer frequencies
        correct_freq = {}
        predicted_freq = {}
        
        for answers in correct_answers:
            for answer in answers:
                correct_freq[answer] = correct_freq.get(answer, 0) + 1
        
        for answers in predicted_answers:
            for answer in answers:
                predicted_freq[answer] = predicted_freq.get(answer, 0) + 1
        
        # Response quality analysis
        empty_responses = sum(1 for r in detailed_results if not r['response'].strip())
        very_short_responses = sum(1 for r in detailed_results if len(r['response']) < 10)
        very_long_responses = sum(1 for r in detailed_results if len(r['response']) > 1000)
        
        # Single vs multiple answers
        single_answer_responses = sum(1 for r in detailed_results if len(r['predicted_answer']) == 1)
        multiple_answer_responses = sum(1 for r in detailed_results if len(r['predicted_answer']) > 1)
        
        return {
            'basic_statistics': {
                'total_questions': total_questions,
                'correct_answers': correct_count,
                'incorrect_answers': total_questions - correct_count,
                'accuracy': accuracy,
                'accuracy_percentage': accuracy * 100
            },
            'response_length_analysis': {
                'average_length': avg_length,
                'shortest_response': min(response_lengths) if response_lengths else 0,
                'longest_response': max(response_lengths) if response_lengths else 0,
                'empty_responses': empty_responses,
                'very_short_responses': very_short_responses,
                'very_long_responses': very_long_responses
            },
            'answer_pattern_analysis': {
                'correct_answer_frequencies': correct_freq,
                'predicted_answer_frequencies': predicted_freq,
                'single_answer_responses': single_answer_responses,
                'multiple_answer_responses': multiple_answer_responses
            },
            'question_breakdown': [
                {
                    'question_id': r['question_id'],
                    'correct': r['correct'],
                    'correct_answer': r['correct_answer'],
                    'predicted_answer': r['predicted_answer'],
                    'response_length': len(r['response']),
                    'response_preview': r['response'][:100] + "..." if len(r['response']) > 100 else r['response']
                }
                for r in detailed_results
            ]
        }
    
    def test_model(self, test_prompts: List[str] = None):
        """
        Test the model with sample prompts.
        
        Args:
            test_prompts: List of test prompts, uses default if None
        """
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you?",
                "Who is the president of USA?",
                "焊接领域什么是 metal matrix composites (MMCs)?",
                (
                    "6. 以下共于非晶硅探测器内部信息传递和信号转换过程的描述，正确的是（  ）\n"
                    "A: X 射线光子→（转换层）→ 电子→（A/D 转换器）→数字信号输出\n"
                    "B: X 射线光子→（转换层）→可见光电子→〔光电二极管）→ 电子《AD 徒换器）→数字信号输出\n"
                    "C: X 射线光子→〔光电二极管）→ 电子（A/D 转换器）→数字信号输出\n"
                ),
            ]
        
        print("Testing Unsloth model with sample prompts...")
        print("=" * 60)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"Test {i}:")
            print(f"Prompt: {prompt[:100]}...")
            response = self.call_model(prompt)
            print(f"Response: {response}")
            print("-" * 60)

def main():
    """Main function to run Unsloth model evaluation."""
    # Configuration
    model_name = "Qwen/Qwen3-32B"
    extraction_model_name = "gpt-4o-mini"
    # model_name = "Qwen/Qwen3-32B"
    dataset_path = 'eval_data/GPQA-V1.json'
    output_path = 'unsloth_evaluation_results.json'
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"Error: Dataset file {dataset_path} not found!")
        return
    
    # Initialize evaluator
    evaluator = UnslothEvaluator(
        model_name=model_name,
        max_seq_length=1024,
        load_in_4bit=True,
        extraction_model_name=extraction_model_name  # Optional: use smaller model for extraction
    )
    
    # Test model first
    # print("Testing model with sample prompts...")
    # evaluator.test_model()
    
    # Run evaluation
    try:
        results = evaluator.evaluate_dataset(dataset_path, output_path)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Model: {results['model_name']}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
