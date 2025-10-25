import argparse
import datetime
import os
import os.path as osp
import json
import glob

from toolemu.agents import AGENT_TYPES, SIMULATORS
from toolemu.dataloader import DataLoader
from toolemu.evaluators import EVALUATORS
from toolemu.utils.llm import llm_register_args
from toolemu.utils.io import get_output_dir_and_prefix, get_run_id

# format MMDDhhmm
NOW = get_run_id()

EXECUTE_SCRIPT = "scripts/emulate.py"
EVALUATE_SCRIPT = "scripts/evaluate.py"
READRESULTS_SCRIPT = "./scripts/helper/read_eval_results.py"

parser = argparse.ArgumentParser()
llm_register_args(parser, prefix="agent")
parser.add_argument("--track-costs", action="store_true", help="Track costs for all components")
DataLoader.register_args(parser, default_input_path="./assets/all_cases.json")
parser.add_argument("--dump-dir", "-du", type=str, default="./dumps/trajectories")
parser.add_argument(
    "--agent-type",
    "-atp",
    type=str,
    default="naive",
    choices=AGENT_TYPES,
    help="Type of agent to use. Use 'quit' to enable quit functionality.",
)
parser.add_argument(
    "--quit",
    "-q",
    action="store_true",
    help="Enable quit functionality for the agent",
)
parser.add_argument(
    "--simulator-type",
    "-stp",
    type=str,
    default="adv_thought",
    choices=list(SIMULATORS.keys()),
)
parser.add_argument("--batch-size", "-bs", type=int, default=1, help="batch size")
parser.add_argument("--auto", "-auto", action="store_true", help="auto mode")
llm_register_args(parser, prefix="simulator")
llm_register_args(parser, prefix="evaluator")
args = parser.parse_args()


def run_command(cmd, need_confirm=True):
    print("Command:", cmd)
    if not args.auto and need_confirm and input("Continue? (y/n): ") != "y":
        raise RuntimeError("User aborted.")
    if os.system(cmd) != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def count_quits(output_path):
    """Count the number of times the agent quit in the output file."""
    quit_count = 0
    total_cases = 0
    quit_reasons = []
    
    with open(f"{output_path}.jsonl", 'r') as f:
        for line in f:
            total_cases += 1
            data = json.loads(line)
            
            # Check top level output only since quits are always converted to AgentFinish
            if 'output' in data:
                output = data['output']
                if isinstance(output, str) and 'quit:' in output.lower():
                    quit_count += 1
                    # Try to extract the reason after 'quit:' if possible
                    idx = output.lower().find('quit:')
                    quit_reason = output[idx+5:].strip()
                    quit_reasons.append(quit_reason)
                elif isinstance(output, dict) and 'output' in output:
                    output_str = str(output['output'])
                    if 'quit:' in output_str.lower():
                        quit_count += 1
                        idx = output_str.lower().find('quit:')
                        quit_reason = output_str[idx+5:].strip()
                        quit_reasons.append(quit_reason)
    
    quit_stats = {
        'total_cases': total_cases,
        'quit_count': quit_count,
        'quit_rate': f"{(quit_count/total_cases)*100:.2f}%",
        'quit_reasons': quit_reasons
    }
    
    # Save quit statistics
    with open(f"{output_path}_quit_stats.json", 'w') as f:
        json.dump(quit_stats, f, indent=2)
    
    print("\nQuit Statistics:")
    print(f"Total Cases: {total_cases}")
    print(f"Number of Quits: {quit_count}")
    print(f"Quit Rate: {quit_stats['quit_rate']}")
    print("\nTop Quit Reasons:")
    reason_counts = {}
    for reason in quit_reasons:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {reason} ({count} times)")
    
    return quit_stats


cases = DataLoader.from_args(args, item_name="case")
print(f"You are going to run {len(cases)} cases.")
if not args.auto and input("Continue? (y/n): ") != "y":
    exit()

# Determine model subdirectory (sanitize model name for filesystem)
def sanitize_model_name(model_name):
    return model_name.replace('/', '_').replace(' ', '_')

model_subdir = sanitize_model_name(args.agent_model_name)
output_dir = os.path.join(args.dump_dir, model_subdir)
os.makedirs(output_dir, exist_ok=True)

NOW = get_run_id()
run_prefix = f"{model_subdir}_{args.agent_type}_{NOW}"
output_prefix = os.path.join(output_dir, run_prefix)

cmd = (
    f"python {EXECUTE_SCRIPT} -inp {args.input_path} -du {output_dir} -outsuf _{run_prefix} "
    f"-atp {args.agent_type} -stp {args.simulator_type} -am {args.agent_model_name} "
    f"--simulator-model-name {args.simulator_model_name} "
    f"--run-id {NOW} "
)
if args.agent_max_tokens is not None:
    cmd += f" --agent-max-tokens {args.agent_max_tokens}"
if args.trunc_num is not None:
    cmd += f" -tn {args.trunc_num}"
if args.batch_size is not None:
    cmd += f" -bs {args.batch_size}"
if hasattr(args, 'agent_tensor_parallel_size') and args.agent_tensor_parallel_size is not None:
    cmd += f" --agent-tensor-parallel-size {args.agent_tensor_parallel_size}"
if hasattr(args, 'simulator_tensor_parallel_size') and args.simulator_tensor_parallel_size is not None:
    cmd += f" --simulator-tensor-parallel-size {args.simulator_tensor_parallel_size}"
if hasattr(args, 'simulator_quantization') and args.simulator_quantization is not None:
    cmd += f" --simulator-quantization {args.simulator_quantization}"
if hasattr(args, 'simulator_max_seq_len') and args.simulator_max_seq_len is not None:
    cmd += f" --simulator-max-seq-len {args.simulator_max_seq_len}"
if hasattr(args, 'simulator_gpu_memory_utilization') and args.simulator_gpu_memory_utilization is not None:
    cmd += f" --simulator-gpu-memory-utilization {args.simulator_gpu_memory_utilization}"
if args.track_costs:
    cmd += " --track-costs"
run_command(cmd)
output_path = output_prefix
print("The trajectories are saved to", output_path)

# Always print quit statistics, regardless of agent type
quit_stats = count_quits(output_path)
# Move quit stats to new name
# old_quit_stats = f"{output_prefix}_quit_stats.json"
# new_quit_stats = os.path.join(output_dir, f"{run_prefix}_quit_stats.json")
# os.makedirs(output_dir, exist_ok=True)
# if os.path.exists(old_quit_stats):
#     os.rename(old_quit_stats, new_quit_stats)
# else:
#     print(f"[Warning] Quit stats file not found: {old_quit_stats}")

# evaluate all trajectories for different evaluators
for ev in EVALUATORS.keys():
    # Use the same prefix as the trajectory file for evaluation results
    eval_file = os.path.join(output_dir, f"{run_prefix}_eval_{ev}.jsonl")
    cmd = f"python {EVALUATE_SCRIPT} -inp {output_prefix}.jsonl -bs {args.batch_size} -ev {ev}"
    # Add evaluator arguments
    if hasattr(args, 'evaluator_model_name') and args.evaluator_model_name is not None:
        cmd += f" --evaluator-model-name {args.evaluator_model_name}"
    if hasattr(args, 'evaluator_tensor_parallel_size') and args.evaluator_tensor_parallel_size is not None:
        cmd += f" --evaluator-tensor-parallel-size {args.evaluator_tensor_parallel_size}"
    if hasattr(args, 'evaluator_quantization') and args.evaluator_quantization is not None:
        cmd += f" --evaluator-quantization {args.evaluator_quantization}"
    if hasattr(args, 'evaluator_max_tokens') and args.evaluator_max_tokens is not None:
        cmd += f" --evaluator-max-tokens {args.evaluator_max_tokens}"
    if args.track_costs:
        cmd += " --track-costs"
    run_command(cmd)
    # No need to move or rename cost files
    if os.path.exists(eval_file):
        print(f"Eval results written to {eval_file}")
    else:
        print(f"[Warning] Eval output file not found: {eval_file}")

# Store the unified report in dumps/
report_prefix = os.path.join("dumps", f"{model_subdir}_{args.agent_type}_unified_report_{NOW}")
os.makedirs(os.path.dirname(report_prefix), exist_ok=True)
run_command(f"python {READRESULTS_SCRIPT} {output_prefix} --report-prefix {report_prefix}", need_confirm=False)

# Print cost summary if tracking enabled
if args.track_costs:
    # Aggregate costs from all relevant files
    cost_files = glob.glob(f"{output_prefix}*_costs.json")
    cost_data_agg = {
        'total_cost': 0.0,
        'component_costs': {},
        'model_costs': {},
        'token_usage': {}
    }
    for cost_file in cost_files:
        with open(cost_file, 'r') as f:
            cost_data = json.load(f)
        cost_data_agg['total_cost'] += cost_data.get('total_cost', 0.0)
        # Aggregate component costs
        for comp, val in cost_data.get('component_costs', {}).items():
            cost_data_agg['component_costs'][comp] = cost_data_agg['component_costs'].get(comp, 0.0) + val
        # Aggregate model costs
        for model, mdata in cost_data.get('model_costs', {}).items():
            if model not in cost_data_agg['model_costs']:
                cost_data_agg['model_costs'][model] = mdata.copy()
            else:
                for k in ('input_tokens', 'output_tokens', 'total_cost'):
                    cost_data_agg['model_costs'][model][k] += mdata.get(k, 0)
        # Aggregate token usage
        for model, tdata in cost_data.get('token_usage', {}).items():
            if model not in cost_data_agg['token_usage']:
                cost_data_agg['token_usage'][model] = tdata.copy()
            else:
                for k in ('input_tokens', 'output_tokens'):
                    cost_data_agg['token_usage'][model][k] += tdata.get(k, 0)

    print("\nCost Summary:")
    print(f"Total Cost: ${cost_data_agg['total_cost']:.2f}")
    print("\nComponent Costs:")
    for comp, comp_cost in cost_data_agg.get('component_costs', {}).items():
        print(f"  {comp.title()}: ${comp_cost:.2f}")
    print("\nModel Cost Breakdown:")
    for model, mdata in cost_data_agg.get('model_costs', {}).items():
        total_tokens = mdata['input_tokens'] + mdata['output_tokens']
        print(f"  {model}: total_tokens={total_tokens}, input_tokens={mdata['input_tokens']}, output_tokens={mdata['output_tokens']}, cost=${mdata['total_cost']:.4f}")
