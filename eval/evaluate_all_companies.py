import os
import logging
import pandas as pd
from typing import Dict, Any, List
from triple_evaluator import evaluate_triples_for_company

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_company_directories(companies_dir="companies"):
    """Get a list of all company directories with required files"""
    company_dirs = []
    for item in os.listdir(companies_dir):
        dir_path = os.path.join(companies_dir, item)
        if os.path.isdir(dir_path):
            # Check if both required files exist
            has_triples = os.path.exists(os.path.join(dir_path, "canonical_triples.json"))
            has_chunks = os.path.exists(os.path.join(dir_path, "source_chunks.json"))
            
            if has_triples and has_chunks:
                company_dirs.append(item)
    
    return company_dirs

def evaluate_all_companies(samples_per_chunk=0, total_triples=50, output_dir="eval/results"):
    """Run evaluation for all available companies"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of companies
    companies = get_company_directories()
    logger.info(f"Found {len(companies)} companies to evaluate: {', '.join(companies)}")
    
    # Determine sampling method description for filename
    if total_triples > 0:
        sample_desc = f"random{total_triples}"
    else:
        sample_desc = f"perchunk{samples_per_chunk}"
    
    # Track results
    evaluation_results = {}
    
    # Evaluate each company
    for company in companies:
        try:
            logger.info(f"Evaluating triples for {company}...")
            results = evaluate_triples_for_company(company, samples_per_chunk, total_triples, output_dir)
            
            evaluation_results[company] = {
                "total_sampled_triples": results["total_sampled_triples"],
                "correct_rate": results["summary"]["correct_rate"],
                "partial_rate": results["summary"]["partial_rate"],
                "incorrect_rate": results["summary"]["incorrect_rate"],
                "error_rate": results["summary"]["error_rate"],
                "avg_confidence": results["summary"]["avg_confidence"],
                "correct_count": results["summary"]["classification_counts"]["CORRECT"],
                "partial_count": results["summary"]["classification_counts"]["PARTIAL"],
                "incorrect_count": results["summary"]["classification_counts"]["INCORRECT"],
                "error_count": results["summary"]["classification_counts"]["ERROR"]
            }
        except Exception as e:
            logger.error(f"Error evaluating {company}: {e}")
            evaluation_results[company] = {"error": str(e)}
    
    # Create combined CSV with all company results
    create_combined_report(evaluation_results, output_dir, sample_desc)
    
    # Log summary
    logger.info("Evaluation complete. Summary:")
    for company, result in evaluation_results.items():
        if "error" in result:
            logger.info(f"  {company}: Error - {result['error']}")
        else:
            logger.info(f"  {company}: Sampled {result['total_sampled_triples']} triples, " 
                       f"Correct: {result['correct_rate']:.2%}, "
                       f"Partial: {result['partial_rate']:.2%}, "
                       f"Incorrect: {result['incorrect_rate']:.2%}")
    
    return evaluation_results

def create_combined_report(evaluation_results: Dict[str, Dict[str, Any]], output_dir: str, sample_desc: str):
    """Create a combined CSV report with results from all companies"""
    # Prepare data for CSV
    rows = []
    for company, results in evaluation_results.items():
        if "error" not in results:
            row = {"company": company}
            row.update(results)
            rows.append(row)
    
    if rows:
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        combined_csv_path = os.path.join(output_dir, f"all_companies_evaluation_{sample_desc}.csv")
        df.to_csv(combined_csv_path, index=False)
        logger.info(f"Combined evaluation results saved to {combined_csv_path}")
        
        # Create a pivot table with just the key metrics
        pivot_csv_path = os.path.join(output_dir, f"evaluation_summary_pivot_{sample_desc}.csv")
        pivot_df = df[['company', 'correct_rate', 'partial_rate', 'incorrect_rate', 'avg_confidence', 'total_sampled_triples']]
        pivot_df.to_csv(pivot_csv_path, index=False)
        logger.info(f"Summary pivot table saved to {pivot_csv_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate triples for all companies')
    parser.add_argument('--samples', type=int, default=0, 
                        help='Number of triples to sample per chunk')
    parser.add_argument('--total', type=int, default=50, 
                        help='Total number of triples to sample randomly')
    parser.add_argument('--output-dir', type=str, default='eval/results', 
                        help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    evaluate_all_companies(args.samples, args.total, args.output_dir) 