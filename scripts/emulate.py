"""
Emulate the agent's tool-use trajectories.

Usage: python scripts/emulate.py -inp <input_cases_file> -atp <agent_type> -stp <simulator_type> -ctp <case_type>
"""

import argparse
import os
import random
import time
import requests
import json

import anthropic
from openai import OpenAIError, BadRequestError
import openai
from dotenv import load_dotenv
from toolemu.agent_executor_builder import build_agent_executor
from toolemu.agents import AGENT_TYPES, SIMULATORS
from toolemu.dataloader import DataLoader
from toolemu.executors import FuncExecutorWithRetry
from toolemu.tools import *
from toolemu.utils import (
    case_to_input_dict,
    filter_keys,
    get_toolkit_names,
    llm_register_args,
    load_openai_llm_with_args,
    replace_agent_action_with_list,
    get_trajectory_token_usage,
)
from toolemu.utils.my_typing import *
from toolemu.utils.io import get_output_dir_and_prefix, get_run_id
from toolemu.utils.cost_tracker import CostTracker

load_dotenv()
ROLES = ["agent", "simulator"]

parser = argparse.ArgumentParser()
for role in ROLES:
    llm_register_args(parser, prefix=role)
DataLoader.register_args(parser)
FuncExecutorWithRetry.register_args(
    parser, default_num_retries=5, default_batch_size=5, default_timeout=1800
)
parser.add_argument("--dump-dir", "-du", type=str, default="./dumps/trajectories")
parser.add_argument(
    "--output-file-suffix",
    "-outsuf",
    type=str,
    default="",
    help="Suffix to add to the output file name",
)
parser.add_argument(
    "--agent-type",
    "-atp",
    type=str,
    default="naive",
    choices=AGENT_TYPES,
)
parser.add_argument(
    "--simulator-type",
    "-stp",
    type=str,
    default="adv_thought",
    choices=list(SIMULATORS.keys()),
)
parser.add_argument("--max-iterations", "-mi", type=int, default=15)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--random-seed", "-seed", type=int, default=42)
parser.add_argument("--run-id", type=str, default=None, help="Run ID for consistent output naming")
parser.add_argument("--track-costs", action="store_true", help="Enable cost tracking for all components")

args = parser.parse_args()
random.seed(42)

skipped_indices = []  # Collect indices of skipped data points

def sanitize_model_name(model_name):
    return model_name.replace('/', '_').replace(' ', '_')

def main():
    # Do not manually add tensor_parallel_size or quantization to llm_kwargs_agent
    llms = {
        'agent': load_openai_llm_with_args(args, prefix='agent'),
        'simulator': load_openai_llm_with_args(args, prefix='simulator')
    }
    
    cases = DataLoader.from_args(args, return_mode="with_idx", item_name="case")
    runner = FuncExecutorWithRetry.from_args(args)
    NOW = args.run_id if args.run_id is not None else get_run_id()
    
    # Add debug print statements after models are loaded
    print("\n" + "="*80)
    print(f"Loaded models:")
    print(f"  Agent:      {llms['agent'].model_name if hasattr(llms['agent'], 'model_name') else getattr(llms['agent'], '_model_name', 'Unknown')}")
    if hasattr(llms['agent'], 'tensor_parallel_size'):
        print(f"             tensor_parallel_size={llms['agent'].tensor_parallel_size}")
    if hasattr(llms['agent'], 'quantization'):
        print(f"             quantization={llms['agent'].quantization}")
    
    print(f"  Simulator:  {llms['simulator'].model_name if hasattr(llms['simulator'], 'model_name') else getattr(llms['simulator'], '_model_name', 'Unknown')}")
    if hasattr(llms['simulator'], 'tensor_parallel_size'):
        print(f"             tensor_parallel_size={llms['simulator'].tensor_parallel_size}")
    if hasattr(llms['simulator'], 'quantization'):
        print(f"             quantization={llms['simulator'].quantization}")
    
    if hasattr(args, 'evaluator_model_name'):
        print(f"  Evaluator:  {args.evaluator_model_name}")
    print("="*80 + "\n")
    
    output_dir, output_prefix = get_output_dir_and_prefix(
        agent_model_name=sanitize_model_name(args.agent_model_name),
        agent_type=args.agent_type,
        run_id=NOW,
        simulator_type=args.simulator_type,
        dump_dir=args.dump_dir,
        flat_output=True
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_prefix + ".jsonl"

    cost_tracker = CostTracker() if args.track_costs else None

    def generate_trajectory(case_with_idx):
        case_idx, case = case_with_idx["idx"], case_with_idx["item"]
        agent_executer = build_agent_executor(
            get_toolkit_names(case),
            llms["agent"],
            llms["simulator"],
            agent_type=args.agent_type,
            simulator_type=args.simulator_type,
            verbose=args.verbose,
            max_iterations=args.max_iterations,
            track_costs=args.track_costs,
            cost_tracker=cost_tracker,
        )
        inputs = filter_keys(case_to_input_dict(case), agent_executer.input_keys)
        max_retries = 5
        num_tries = 0
        while num_tries < max_retries:
            if num_tries > 0:
                # Exponential backoff, capped at 32 seconds
                sleep_time = min(2 ** num_tries, 32)
                print(f"Retrying after {sleep_time} seconds...")
                time.sleep(sleep_time)
            try:
                outputs = agent_executer(inputs)
                failed_item = None
                break
            except (BadRequestError, getattr(anthropic, 'BadRequestError', Exception)) as e:
                print(f"{case_idx}: {str(e)}")
                outputs = {"error": str(e), "error_type": type(e).__name__}
                failed_item = case_with_idx
                break
            except Exception as e:
                # Handle OpenAI/Anthropic/vLLM/network errors and prompt length errors
                import sys
                import traceback
                error_str = str(e)
                tb_str = traceback.format_exc()
                # Handle vLLM prompt too long error (ValueError)
                if isinstance(e, ValueError) and ("longer than the maximum model length" in error_str or "is too long and exceeds limit" in error_str):
                    print(f"{case_idx}: Prompt too long for model, skipping. {error_str}")
                    outputs = {"error": error_str, "error_type": type(e).__name__}
                    failed_item = case_with_idx
                    skipped_indices.append(case_idx)
                    break
                # Handle OpenAI/Anthropic/network errors
                openai_errors = (getattr(openai, 'error', type('FakeError', (), {})), requests.exceptions.RequestException)
                if (
                    hasattr(openai, 'error') and isinstance(e, (getattr(openai.error, 'RateLimitError', Exception), getattr(openai.error, 'Timeout', Exception)))
                ) or isinstance(e, requests.exceptions.RequestException):
                    print(f"{case_idx}: Network or rate limit error: {error_str}")
                    outputs = {"error": error_str, "error_type": type(e).__name__}
                    failed_item = case_with_idx
                    num_tries += 1
                    if num_tries >= max_retries:
                        print(f"{case_idx}: Failed after {max_retries} retries due to network or rate limit error. Skipping.")
                        skipped_indices.append(case_idx)
                        break
                    continue
                # Handle generic errors
                print(f"{case_idx}: Unexpected error: {type(e).__name__}: {error_str}\n{tb_str}")
                outputs = {"error": error_str, "error_type": type(e).__name__}
                failed_item = case_with_idx
                num_tries += 1
                if num_tries >= max_retries:
                    print(f"{case_idx}: Failed after {max_retries} retries due to unexpected error. Skipping.")
                    skipped_indices.append(case_idx)
                    break
                continue
        outputs = replace_agent_action_with_list(outputs)  # ensure serializable
        outputs["case"] = case
        outputs["case_idx"] = case_idx
        return failed_item, outputs

    runner.run(generate_trajectory, output_path, cases)
    print(
        "You may want to use scripts to convert the result jsonl file "
        f"{output_path} to json for easier reading."
    )
    if args.track_costs and cost_tracker is not None:
        cost_file = output_path.replace(".jsonl", "_costs.json")
        cost_tracker.save_costs(cost_file)
        print(f"Cost tracking data saved to {cost_file}")
    if skipped_indices:
        print(f"\nSkipped indices due to prompt too long: {skipped_indices}")
        skipped_path = os.path.join(output_dir, "skipped_indices.txt")
        with open(skipped_path, "w") as f:
            for idx in skipped_indices:
                f.write(str(idx) + "\n")
        print(f"Skipped indices saved to {skipped_path}")

if __name__ == "__main__":
    main()



