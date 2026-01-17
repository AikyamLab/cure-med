"""
Reward computation script for multilingual medical reasoning fine-tuning.
I
"""

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class RewardConfig:
    """Configuration for reward computation"""
    gpt_model: str = "gpt-4.1"  
    # Weights for composite reward 
    weight_accuracy: float = 0.65 
    weight_language: float = 0.30  
    weight_format: float = 0.05  
    # GPT prompt templates (removed question_type dependency; treat all as open for simplicity)
    accuracy_prompt_template: str = """
    You are an expert multilingual medical evaluator. Score the generated response for correctness and helpfulness on a continuous scale from 0.0 to 1.0.
    Compare to ground truth: Give 1.0 if reasoning is sound and answer conveys the correct information semantically, even if phrased differently. Partial scores for partially correct answers.
    Focus on factual accuracy and medical correctness as compared to ground truth solution over wording.
    Question: {question}
    Ground truth reasoning and answer: {ground_truth}
    Generated response: {generated}
    Output only the score as a float between 0 and 1.
    """
    language_prompt_template: str = """
    You are an expert multilingual medical evaluator. Determine if the answer of the model is entirely in the language of the question. Score based on the language of the question and the answer. 
    Question language: {language}
    Generated response: {generated}
    If the response is in the same language, output 1.0.
    If it is in a different language, output 0.0.
    Output only the score as a float (0.0 or 1.0).
    """

# ============================================================================
# REWARD MODEL CLASS
# ============================================================================
class CompositeRewardModel:
    """Composite reward model implementation (cosine and repetition removed)"""
    def __init__(self, config: RewardConfig):
        self.config = config
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def compute_quality_score(self, sample: Dict[str, Any], generated: str) -> float:
        """Compute quality/accuracy score using GPT-4.1"""
        ground_truth = f"{sample['reasoning']} {sample['answer']}" if 'reasoning' in sample and 'answer' in sample else sample.get('ground_truth', '')
        prompt = self.config.accuracy_prompt_template.format(
            question=sample['question'],
            ground_truth=ground_truth,
            generated=generated
        )
        response = self.client.chat.completions.create(
            model=self.config.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            print("Warning: Invalid GPT response for quality score")
            return 0.0

    def compute_language_score(self, sample: Dict[str, Any], generated: str) -> float:
        """Compute language consistency score using GPT-4.1-mini"""
        prompt = self.config.language_prompt_template.format(
            language=sample['language'],
            generated=generated
        )
        response = self.client.chat.completions.create(
            model=self.config.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            print("Warning: Invalid GPT response for language score")
            return 0.0

    def compute_format_reward(self, generated: str) -> float:
        """Check for proper <thinking> and <answer> tags"""
        think_pattern = r"<thinking>.*?</thinking>"
        answer_pattern = r"<answer>.*?</answer>"
        think_matches = re.findall(think_pattern, generated, re.DOTALL)
        answer_matches = re.findall(answer_pattern, generated, re.DOTALL)
        # Exactly one pair, properly formatted
        if len(think_matches) == 1 and len(answer_matches) == 1:
            return 1.0
        return 0.0

    def compute_single_reward(self, sample: Dict[str, Any], generated: str) -> Dict[str, Any]:
        """Compute reward for a single sample"""
        quality_score = self.compute_quality_score(sample, generated)
        language_score = self.compute_language_score(sample, generated)
        format_reward = self.compute_format_reward(generated)
        # Weighted components (only accuracy, language, format)
        components = {
            "accuracy": self.config.weight_accuracy * quality_score,
            "language": self.config.weight_language * language_score,
            "format": self.config.weight_format * format_reward,
        }
        total_reward = sum(components.values())
        return {
            "total_reward": total_reward,
            "accuracy_score": quality_score,
            "language_score": language_score,
            "format_reward": format_reward,
            "components": components
        }

    def compute_rewards_sync(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute rewards for multiple samples synchronously"""
        return [self.compute_single_reward(sample, sample['generated_answer']) for sample in samples]
