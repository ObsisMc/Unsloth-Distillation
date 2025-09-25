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
        Calculate if the predicted answers are correct (exact match).
        
        Args:
            predicted: List of predicted answer choices
            correct: List of correct answer choices
            
        Returns:
            True if answers match exactly, False otherwise
        """
        metrics = self.calculate_metrics(predicted, correct)
        return metrics['exact_match']

        
    def calculate_metrics(self, predicted: List[str], correct: List[str]) -> Dict[str, float]:
        """
        Calculate multiple metrics for predicted vs correct answers.
        
        Args:
            predicted: List of predicted answer choices
            correct: List of correct answer choices
            
        Returns:
            Dictionary containing various metrics
        """
        # Convert to sets for set operations
        pred_set = set(predicted)
        correct_set = set(correct)
        
        # 1. Exact Match Accuracy (original metric)
        exact_match = sorted(predicted) == sorted(correct)
        
        # 2. Jaccard Similarity (intersection over union)
        intersection = len(pred_set & correct_set)
        union = len(pred_set | correct_set)
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # 3. Precision (how many predicted answers are correct)
        precision = intersection / len(pred_set) if len(pred_set) > 0 else 0.0
        
        # 4. Recall (how many correct answers were predicted)
        recall = intersection / len(correct_set) if len(correct_set) > 0 else 0.0
        
        # 5. F1 Score (harmonic mean of precision and recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 6. Partial Credit Score (proportion of correct answers found)
        partial_credit = intersection / len(correct_set) if len(correct_set) > 0 else 0.0
        
        # 7. Over-prediction Penalty (penalty for extra wrong answers)
        over_prediction_penalty = max(0, len(pred_set) - len(correct_set)) / max(len(correct_set), 1)
        
        # 8. Under-prediction Penalty (penalty for missing correct answers)
        under_prediction_penalty = max(0, len(correct_set) - len(pred_set)) / max(len(correct_set), 1)
        
        return {
            'exact_match': exact_match,
            'jaccard_similarity': jaccard_similarity,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'partial_credit': partial_credit,
            'over_prediction_penalty': over_prediction_penalty,
            'under_prediction_penalty': under_prediction_penalty,
            'intersection_size': intersection,
            'union_size': union,
            'predicted_size': len(pred_set),
            'correct_size': len(correct_set)
        }
    
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
            dataset = json.load(f)
        
        total_questions = len(dataset)
        correct_answers = 0
        detailed_results = []
        
        # Initialize metrics accumulators
        total_jaccard = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_partial_credit = 0.0
        
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
                # Use empty metrics for failed responses
                metrics = {
                    'exact_match': False,
                    'jaccard_similarity': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'partial_credit': 0.0,
                    'over_prediction_penalty': 0.0,
                    'under_prediction_penalty': 0.0,
                    'intersection_size': 0,
                    'union_size': len(correct_answer),
                    'predicted_size': 0,
                    'correct_size': len(correct_answer)
                }
                detailed_results.append({
                    'question_id': i,
                    'question': question,
                    'correct_answer': correct_answer,
                    'predicted_answer': [],
                    'response': response,
                    'correct': False,
                    'error': 'No response from model',
                    'metrics': metrics
                })
                continue
            
            # Extract predicted answers
            predicted_answer = self.extract_answers(response)
            
            # Calculate all metrics
            metrics = self.calculate_metrics(predicted_answer, correct_answer)
            
            # Accumulate metrics
            total_jaccard += metrics['jaccard_similarity']
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1_score']
            total_partial_credit += metrics['partial_credit']
            
            if metrics['exact_match']:
                correct_answers += 1
                print("✅ Correct")
            else:
                print("❌ Incorrect")
            
            print(f"Correct: {correct_answer}")
            print(f"Predicted: {predicted_answer}")
            print(f"Jaccard: {metrics['jaccard_similarity']:.3f}, F1: {metrics['f1_score']:.3f}")
            print(f"Full Response: {response}")
            print("-" * 60)
            
            detailed_results.append({
                'question_id': i,
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'response': response,
                'correct': metrics['exact_match'],
                'metrics': metrics
            })
        
        # Calculate final metrics
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        avg_jaccard = total_jaccard / total_questions if total_questions > 0 else 0
        avg_precision = total_precision / total_questions if total_questions > 0 else 0
        avg_recall = total_recall / total_questions if total_questions > 0 else 0
        avg_f1 = total_f1 / total_questions if total_questions > 0 else 0
        avg_partial_credit = total_partial_credit / total_questions if total_questions > 0 else 0
        
        results = {
            'model_name': self.model_name,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'metrics': {
                'exact_match_accuracy': accuracy,
                'jaccard_similarity': avg_jaccard,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1,
                'partial_credit': avg_partial_credit
            },
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
                    'metrics': results['metrics'],
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
        
        # Calculate average metrics
        if detailed_results and 'metrics' in detailed_results[0]:
            avg_jaccard = sum(r['metrics']['jaccard_similarity'] for r in detailed_results) / total_questions
            avg_precision = sum(r['metrics']['precision'] for r in detailed_results) / total_questions
            avg_recall = sum(r['metrics']['recall'] for r in detailed_results) / total_questions
            avg_f1 = sum(r['metrics']['f1_score'] for r in detailed_results) / total_questions
            avg_partial_credit = sum(r['metrics']['partial_credit'] for r in detailed_results) / total_questions
        else:
            avg_jaccard = avg_precision = avg_recall = avg_f1 = avg_partial_credit = 0.0
        
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
            'metrics_summary': {
                'average_jaccard_similarity': avg_jaccard,
                'average_precision': avg_precision,
                'average_recall': avg_recall,
                'average_f1_score': avg_f1,
                'average_partial_credit': avg_partial_credit
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
    model_name = "results/model/sft/Qwen3-32B/checkpoint-31"
    # model_name = "results/model/sft/Qwen3-32B_fft/checkpoint-62"
    extraction_model_name = "gpt-4o-mini"
    # model_name = "Qwen/Qwen3-32B"
    dataset_path = 'eval_data/GPQA-V1.json'
    output_root = Path("results/eval")
    output_file = "qwen3_32b_lora_results_more_metric.json"
    # output_file = "qwen3_32b_lora_results.json"
    # output_file = "qwen3_32b_fft_2epoch_results_more_metric.json"
    output_path = output_root / output_file

    output_root.mkdir(parents=True, exist_ok=True)

    max_seq_length = 1024
    load_in_4bit = False
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"Error: Dataset file {dataset_path} not found!")
        return
    
    # Initialize evaluator
    evaluator = UnslothEvaluator(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
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
        print("\n📊 METRICS:")
        metrics = results['metrics']
        print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
        print(f"  Jaccard Similarity:   {metrics['jaccard_similarity']:.4f} ({metrics['jaccard_similarity']*100:.2f}%)")
        print(f"  Precision:             {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:                {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1 Score:              {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"  Partial Credit:        {metrics['partial_credit']:.4f} ({metrics['partial_credit']*100:.2f}%)")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
