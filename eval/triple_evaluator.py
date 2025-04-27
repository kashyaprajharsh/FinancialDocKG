import json
import os
import csv
import random
import logging
from typing import Dict, List, Any, Tuple, Optional
from pydantic import BaseModel, Field
import pandas as pd
import sys
sys.path.append('.')  

from Models.llm import generate_response

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TripleEvaluation(BaseModel):
    """Structured output for triple evaluation results from LLM"""
    is_valid: bool = Field(description="Whether the triple is valid based on the chunk content")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)
    reasoning: str = Field(description="Brief explanation of why the triple is valid or invalid")
    classification: str = Field(description="One of: CORRECT, PARTIAL, INCORRECT")

class TripleEvaluator:
    """Evaluates triples extracted from document chunks using LLM"""
    
    def __init__(self, canonical_triples_path: str, source_chunks_path: str):
        """Initialize with paths to triples and source chunks"""
        self.canonical_triples_path = canonical_triples_path
        self.source_chunks_path = source_chunks_path
        self.triples = []
        self.chunks = {}
        self.load_data()
    
    def load_data(self) -> None:
        """Load triples and chunks from files"""
        try:
            with open(self.canonical_triples_path, 'r', encoding='utf-8') as f:
                self.triples = json.load(f)
            logger.info(f"Loaded {len(self.triples)} triples from {self.canonical_triples_path}")
            
            with open(self.source_chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            logger.info(f"Loaded {len(self.chunks)} source chunks from {self.source_chunks_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_triples_by_chunk(self) -> Dict[int, List[Dict]]:
        """Group triples by their source chunk ID"""
        triples_by_chunk = {}
        for triple in self.triples:
            chunk_id = triple.get("source_chunk")
            if chunk_id is not None:
                if chunk_id not in triples_by_chunk:
                    triples_by_chunk[chunk_id] = []
                triples_by_chunk[chunk_id].append(triple)
        return triples_by_chunk
    
    def sample_triples(self, n_per_chunk: int = 5) -> Dict[int, List[Dict]]:
        """Sample n triples from each chunk"""
        triples_by_chunk = self.get_triples_by_chunk()
        sampled_triples = {}
        
        for chunk_id, chunk_triples in triples_by_chunk.items():
            sample_size = min(n_per_chunk, len(chunk_triples))
            sampled_triples[chunk_id] = random.sample(chunk_triples, sample_size)
            
        return sampled_triples
    
    def sample_random_triples(self, total_triples: int = 50) -> Dict[int, List[Dict]]:
        """
        Sample a fixed total number of triples randomly from all chunks
        
        Args:
            total_triples: Total number of triples to sample across all chunks
            
        Returns:
            Dictionary mapping chunk IDs to lists of sampled triples
        """
        # Get all triples grouped by chunk
        triples_by_chunk = self.get_triples_by_chunk()
        
        # Flatten the triples list while keeping track of their chunk
        all_triples_with_chunks = []
        for chunk_id, triples in triples_by_chunk.items():
            for triple in triples:
                all_triples_with_chunks.append((chunk_id, triple))
        
        # Sample the requested number of triples (or all if there are fewer)
        sample_size = min(total_triples, len(all_triples_with_chunks))
        sampled_items = random.sample(all_triples_with_chunks, sample_size)
        
        # Group the sampled triples back by chunk
        sampled_triples = {}
        for chunk_id, triple in sampled_items:
            if chunk_id not in sampled_triples:
                sampled_triples[chunk_id] = []
            sampled_triples[chunk_id].append(triple)
        
        logger.info(f"Sampled {sample_size} triples from {len(sampled_triples)} chunks")
        return sampled_triples
    
    def evaluate_triple_with_llm(self, triple: Dict, chunk_text: str) -> TripleEvaluation:
        """Evaluate a single triple using the LLM"""
        system_prompt = f"""
You are a specialized AI for evaluating knowledge triples extracted from financial documents.

SOURCE TEXT:
```
{chunk_text}
```

TRIPLE TO EVALUATE:
Subject: {triple['subject']}
Predicate: {triple['predicate']}
Object: {triple['object']}

Your task is to determine if this triple is a valid representation of information in the source text.
Evaluate ONLY based on the provided source text, not on external knowledge.

Classify the triple as:
- CORRECT: The triple accurately represents a fact from the text with the correct relationship
- PARTIAL: The triple has some correct elements but contains minor inaccuracies or lacks context
- INCORRECT: The triple misrepresents the text or cannot be inferred from the given text

Provide your reasoning process and assign a confidence score from 0 to 1.
"""
        try:
            result = generate_response(system_prompt, TripleEvaluation)
            return result
        except Exception as e:
            logger.error(f"Error evaluating triple with LLM: {e}")
            # Return fallback evaluation if LLM fails
            return TripleEvaluation(
                is_valid=False,
                confidence=0.0,
                reasoning=f"LLM evaluation failed: {str(e)}",
                classification="INCORRECT"
            )
    
    def evaluate_sample(self, n_per_chunk: int = 0, total_triples: int = 0) -> Dict[str, Any]:
        """
        Evaluate a sample of triples using LLM
        
        Args:
            n_per_chunk: Number of triples to sample from each chunk (if > 0)
            total_triples: Total number of triples to sample across all chunks (if > 0)
            
        Note: You should use either n_per_chunk OR total_triples, not both
        """
        # Determine sampling method based on parameters
        if total_triples > 0:
            logger.info(f"Sampling {total_triples} triples randomly across all chunks")
            sampled_triples = self.sample_random_triples(total_triples)
        else:
            logger.info(f"Sampling {n_per_chunk} triples from each chunk")
            sampled_triples = self.sample_triples(n_per_chunk)
        
        results = {
            "total_chunks": len(self.chunks),
            "chunks_with_triples": len(sampled_triples),
            "total_sampled_triples": sum(len(triples) for triples in sampled_triples.values()),
            "evaluated_triples": []
        }
        
        # Counters for summary
        classifications = {"CORRECT": 0, "PARTIAL": 0, "INCORRECT": 0, "ERROR": 0}
        total_confidence = 0.0
        
        # Evaluate each triple
        for chunk_id, triples in sampled_triples.items():
            chunk_text = self.chunks.get(str(chunk_id), "")
            
            for triple in triples:
                logger.info(f"Evaluating triple: {triple['subject']} | {triple['predicate']} | {triple['object']}")
                
                try:
                    evaluation = self.evaluate_triple_with_llm(triple, chunk_text)
                    
                    # Update counters
                    classifications[evaluation.classification] = classifications.get(evaluation.classification, 0) + 1
                    total_confidence += evaluation.confidence
                    
                    # Add to results
                    results["evaluated_triples"].append({
                        "chunk_id": chunk_id,
                        "triple": triple,
                        "evaluation": {
                            "is_valid": evaluation.is_valid,
                            "confidence": evaluation.confidence,
                            "reasoning": evaluation.reasoning,
                            "classification": evaluation.classification
                        }
                    })
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
                    classifications["ERROR"] += 1
                    results["evaluated_triples"].append({
                        "chunk_id": chunk_id,
                        "triple": triple,
                        "evaluation": {
                            "is_valid": False,
                            "confidence": 0.0,
                            "reasoning": f"Evaluation error: {str(e)}",
                            "classification": "ERROR"
                        }
                    })
        
        # Calculate summary statistics
        total = results["total_sampled_triples"]
        if total > 0:
            results["summary"] = {
                "correct_rate": classifications["CORRECT"] / total,
                "partial_rate": classifications["PARTIAL"] / total,
                "incorrect_rate": classifications["INCORRECT"] / total,
                "error_rate": classifications["ERROR"] / total,
                "avg_confidence": total_confidence / total,
                "classification_counts": classifications
            }
        else:
            results["summary"] = {
                "correct_rate": 0,
                "partial_rate": 0,
                "incorrect_rate": 0,
                "error_rate": 0,
                "avg_confidence": 0,
                "classification_counts": classifications
            }
        
        return results
    
    def save_results_to_json(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to a JSON file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            raise
    
    def save_results_to_csv(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to a CSV file"""
        try:
            # Extract the relevant data for CSV
            rows = []
            for item in results["evaluated_triples"]:
                rows.append({
                    "chunk_id": item["chunk_id"],
                    "subject": item["triple"]["subject"],
                    "predicate": item["triple"]["predicate"],
                    "object": item["triple"]["object"],
                    "is_valid": item["evaluation"]["is_valid"],
                    "confidence": item["evaluation"]["confidence"],
                    "classification": item["evaluation"]["classification"],
                    "reasoning": item["evaluation"]["reasoning"]
                })
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(rows)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Evaluation results saved to CSV at {output_path}")
            
            # Create a summary CSV with just the counts and rates
            summary_path = output_path.replace('.csv', '_summary.csv')
            summary_data = {
                "metric": ["correct_rate", "partial_rate", "incorrect_rate", "error_rate", "avg_confidence", 
                           "correct_count", "partial_count", "incorrect_count", "error_count", "total_triples"],
                "value": [
                    results["summary"]["correct_rate"],
                    results["summary"]["partial_rate"],
                    results["summary"]["incorrect_rate"],
                    results["summary"]["error_rate"],
                    results["summary"]["avg_confidence"],
                    results["summary"]["classification_counts"]["CORRECT"],
                    results["summary"]["classification_counts"]["PARTIAL"],
                    results["summary"]["classification_counts"]["INCORRECT"],
                    results["summary"]["classification_counts"]["ERROR"],
                    results["total_sampled_triples"]
                ]
            }
            pd.DataFrame(summary_data).to_csv(summary_path, index=False)
            logger.info(f"Summary results saved to CSV at {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving CSV results: {e}")
            raise


def evaluate_triples_for_company(company_name: str, n_per_chunk: int = 0, total_triples: int = 50, output_dir: str = "eval/results"):
    """
    Evaluate triples for a specific company and save results to both JSON and CSV
    
    Args:
        company_name: Name of the company (e.g., "APPLE")
        n_per_chunk: Number of triples to sample from each chunk (if > 0)
        total_triples: Total number of triples to sample across all chunks (if > 0)
        output_dir: Directory to save evaluation results
    """
    canonical_triples_path = f"companies/{company_name}/canonical_triples.json"
    source_chunks_path = f"companies/{company_name}/source_chunks.json"
    
    # Create descriptive filename based on sampling method
    if total_triples > 0:
        sample_desc = f"random{total_triples}"
    else:
        sample_desc = f"perchunk{n_per_chunk}"
    
    json_output_path = f"{output_dir}/{company_name}_evaluation_{sample_desc}.json"
    csv_output_path = f"{output_dir}/{company_name}_evaluation_{sample_desc}.csv"
    
    try:
        evaluator = TripleEvaluator(canonical_triples_path, source_chunks_path)
        results = evaluator.evaluate_sample(n_per_chunk=n_per_chunk, total_triples=total_triples)
        
        # Save in both formats
        evaluator.save_results_to_json(results, json_output_path)
        evaluator.save_results_to_csv(results, csv_output_path)
        
        logger.info(f"Evaluation completed for {company_name}")
        return results
    except Exception as e:
        logger.error(f"Error evaluating triples for {company_name}: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate extracted triples using LLM')
    parser.add_argument('--company', type=str, required=True, help='Company name (e.g., APPLE)')
    parser.add_argument('--samples', type=int, default=0, help='Number of triples to sample per chunk')
    parser.add_argument('--total', type=int, default=50, help='Total number of triples to sample randomly')
    parser.add_argument('--output-dir', type=str, default='eval/results', help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    evaluate_triples_for_company(args.company, args.samples, args.total, args.output_dir) 