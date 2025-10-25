"""
Evaluate the agent based on a set of trajectories.

Usage: python scripts/evaluate.py -inp <input_trajectories_file> -ev <evaluator_type>
"""

import argparse
import random
import re
import json
import os

from dotenv import load_dotenv
from toolemu.dataloader import DataLoader
from toolemu.evaluators import EVALUATORS
from toolemu.executors import FuncExecutorWithRetry
from toolemu.utils import (
    llm_register_args,
    load_openai_llm_with_args,
    print_intermediate_result_and_stop,
)
from toolemu.utils.my_typing import *
from toolemu.utils.cost_tracker import CostTracker

load_dotenv()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--eval-type",
    "-ev",
    type=str,
    default="agent_safe",
    choices=EVALUATORS.keys(),
)
parser.add_argument("--output-file-prefix", "-out", type=str, default=None)
parser.add_argument("--eval-results-out-suffix", "-es", type=str, default="")
parser.add_argument("--random-seed", "-seed", type=int, default=42)
parser.add_argument("--track-costs", action="store_true", help="Enable cost tracking for all components")
parser.add_argument("-inp", "--input-path", type=str, required=True, help="Input trajectories file")
parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Batch size for evaluation")
parser.add_argument("--start-index", type=int, default=0, help="The start index")
parser.add_argument("--trunc-num", type=int, default=None, help="The number of items to use")
parser.add_argument("--selected-indexes", nargs="+", type=int, default=None, help="The indexes of the items to select")
parser.add_argument("--removed-indexes", nargs="+", type=int, default=None, help="The indexes of the items to remove")
parser.add_argument("--num-replicates", type=int, default=1, help="The number of replicates")
parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle the data")
parser.add_argument("--critique-rounds", type=int, default=0, help="Number of critique rounds (not supported yet)")
parser.add_argument("--stop-at", type=str, default=None, help="Where to stop in the evaluation pipeline")
parser.add_argument("--threading-request-timeout", type=int, default=3000, help="Timeout for a single function execution request")
parser.add_argument("--sequential-generation", action="store_true", help="Run function sequentially over the dataset instead of in parallel")
parser.add_argument("--disable-batch-progress", action="store_true", help="Disable progress bar for multi-threaded execution")
parser.add_argument("--disable-timer", action="store_true", help="Disable timer for multi-threaded execution")
parser.add_argument("--num-retries", type=int, default=0, help="The number of retries when the function execution fails")

llm_register_args(parser, prefix="evaluator")
args = parser.parse_args()
random.seed(args.random_seed)


def clean_and_parse_score(raw_score):
    """Clean and robustly parse a numeric score from model output."""
    if isinstance(raw_score, (int, float)):
        return raw_score
    if not isinstance(raw_score, str):
        raise ValueError(f"Cannot parse score from non-string: {raw_score}")
    
    # First, try to extract just the first number we find
    number_match = re.search(r'(\d+(?:\.\d+)?)', raw_score)
    if number_match:
        try:
            score_str = number_match.group(1)
            if "." in score_str:
                return float(score_str)
            else:
                return int(score_str)
        except Exception as e:
            pass
    
    # Fallback: Remove markdown, whitespace, and non-numeric chars except . and -
    cleaned = re.sub(r"[^0-9.\-]", "", raw_score)
    if cleaned == "":
        raise ValueError(f"Score string is empty after cleaning: {raw_score}")
    try:
        if "." in cleaned:
            return float(cleaned)
        else:
            return int(cleaned)
    except Exception as e:
        raise ValueError(f"Failed to parse cleaned score '{cleaned}' from '{raw_score}': {e}")


def sanitize_model_name(model_name):
    return model_name.replace('/', '_').replace(' ', '_')


def main():
    trajs = DataLoader.from_args(args, return_mode="with_idx", item_name="trajectory")
    # Extract model name and quit type from input_path or args
    base = os.path.basename(args.input_path)
    input_dir = os.path.dirname(args.input_path)
    m = re.match(r"(.+?)_(.+?)_\d{4}_\d{6}", base)
    if m:
        modelname, quit_type = m.group(1), m.group(2)
    else:
        modelname, quit_type = "model", "quit"
    # Extract the full prefix from the input path (without extension)
    input_prefix = os.path.splitext(base)[0]
    output_filename = f"{input_prefix}_eval_{args.eval_type}.jsonl"
    if input_dir:
        output_path = os.path.join(input_dir, output_filename)
    else:
        output_path = output_filename

    if args.critique_rounds > 0:
        raise ValueError("Evaluation does not support critique rounds yet.")
    evaluator_llm = load_openai_llm_with_args(args, prefix="evaluator")
    print(f"[DEBUG] Evaluator model name: {getattr(evaluator_llm, 'model_name', None)}")
    evaluator = EVALUATORS[args.eval_type].from_args(args, evaluator_llm)
    if args.track_costs:
        evaluator.track_costs = True
        evaluator.cost_tracker = CostTracker()

    if evaluator._stop_at in ["preprocess", "prompt"]:
        result = evaluator({"trajectory": trajs[0]["item"]})
        print_intermediate_result_and_stop(result, evaluator._stop_at)

    failed_indices = []
    def evaluate_trajec(traj_with_idx):
        traj_idx, traj = traj_with_idx["idx"], traj_with_idx["item"]
        try:
            results = evaluator({"trajectory": traj})
            result = results[0]  # TODO: support merge multiple rounds of critiques
            # Clean and robustly parse all numeric scores in result
            if "eval_scores" in result:
                for k, v in result["eval_scores"].items():
                    try:
                        result["eval_scores"][k] = clean_and_parse_score(v)
                    except Exception as e:
                        raise ValueError(f"Failed to parse score for key '{k}' in idx {traj_idx}: {e}")
        except Exception as e:
            result = {"error": str(e)}
            failed_indices.append(traj_idx)
            print(f"[ERROR] Failed at idx {traj_idx}: {e}")
            return traj_with_idx, result
        result["eval_id"] = traj_idx
        return None, result

    runner = FuncExecutorWithRetry.from_args(args)
    runner.run(evaluate_trajec, output_path, trajs)
    if failed_indices:
        print(f"[SUMMARY] Failed indices: {failed_indices}")
    print(
        "You may want to use scripts to convert the result jsonl file "
        f"{output_path} to json for easier reading."
    )


if __name__ == "__main__":
    main()
