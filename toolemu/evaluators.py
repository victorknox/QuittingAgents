import json
import random
import re
from argparse import ArgumentParser, Namespace
from abc import ABC
from typing import Optional, Dict, Any

from langchain_core.outputs import LLMResult
from langchain_core.outputs import ChatResult
from langchain_core.language_models import BaseLLM
from procoder.functional import removed_submodules
from procoder.prompt import Module as PromptModule
from toolemu.prompts.evaluator import (
    AGENT_HELP_EVALUATOR_PROMPT,
    AGENT_HELP_EVALUATOR_SYSTEM_INFO,
    AGENT_SAFE_EVALUATOR_PROMPT,
    AGENT_SAFE_EVALUATOR_SYSTEM_INFO,
)
from toolemu.prompts.globals import GENERAL_CRITIQUE_PROMPT
from toolemu.tools import get_toolkits_by_names
from toolemu.utils import (
    METRICS_SHORT2LONG,
    case_to_input_dict,
    construct_trajec,
    get_toolkit_names,
    get_used_tools_in_trajec,
    parse_llm_response,
)
from toolemu.utils.my_typing import *
from toolemu.utils.my_typing import Any, Dict, List
from toolemu.utils.misc import get_num_tokens

from .executors import BasePromptExecutorWithCritique
from .utils.cost_tracker import CostTracker


class BaseTrajEvaluator(ABC):
    """Base class for trajectory evaluators."""
    
    def __init__(self, llm: BaseLLM, cost_tracker: Optional[CostTracker] = None):
        self._llm = llm
        self.llm_chain = self._llm
        self.cost_tracker = cost_tracker or CostTracker()
        # Ensure model_name is set from the LLM
        self.model_name = getattr(self._llm, 'model_name', None)
        
    def _parse_output(self, output: str) -> Dict[str, Any]:
        """Parse the output from the LLM."""
        try:
            # Track token usage if we have a model name
            if self.model_name:
                # Get input tokens from the trajectory text
                input_tokens = get_num_tokens(self._last_trajectory_text)
                output_tokens = get_num_tokens(output)
                print(f"[DEBUG] Evaluator cost tracking: model={self.model_name}, input_tokens={input_tokens}, output_tokens={output_tokens}")
                if input_tokens > 0 or output_tokens > 0:
                    self.cost_tracker.add_token_usage('evaluator', self.model_name, input_tokens, output_tokens)
            if isinstance(output, str):
                return json.loads(output)
            return output
        except Exception as e:
            print(f"Error parsing output: {e}")
            return {}
            
    def save_costs(self, experiment_name: str) -> str:
        """Save cost information to a file."""
        return self.cost_tracker.save_costs(experiment_name)
        
    def get_total_cost(self) -> float:
        """Get total cost across all components."""
        return self.cost_tracker.get_total_cost()
        
    def get_component_cost(self, component: str) -> float:
        """Get cost for a specific component."""
        return self.cost_tracker.get_component_cost(component)

    def __call__(self, inputs: Dict[str, Any], *args, **kwargs):
        # Ensure trajectory text is set for input token counting
        if not hasattr(self, '_last_trajectory_text') or not self._last_trajectory_text:
            traj = inputs.get('trajectory')
            if traj is not None:
                traj_text = construct_trajec(
                    traj,
                    include_input=False,
                    is_dumped_dict=True,
                    include_agent_log=False,
                    include_simulator_log=False,
                    include_final_answer=getattr(self, '_final_answer_in_traj', True),
                )
                self._last_trajectory_text = traj_text
        return super().__call__(inputs, *args, **kwargs)


class BaseTrajEvaluator(BasePromptExecutorWithCritique):
    _short_name: str = None
    _input_keys: List[str] = ["trajectory"]
    _output_keys: List[str] = [
        "Evaluator Thought",
        "Overall Qualitative Label",
        "Overall Quantitative Score",
        "Evaluator Log Summary",
    ]
    _keys2metrics: Dict[str, str] = None
    _critique_prompt_module: PromptModule = GENERAL_CRITIQUE_PROMPT
    _final_answer_in_traj: bool = True
    _toolkit_desc_detail_level: str = "low"  # use simple toolkit descriptions
    _used_tool_desc_detail_level: str = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_tracker = CostTracker()
        self.llm_chain = self._llm  # Initialize llm_chain for cost tracking

    def _preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        traj = inputs["trajectory"]
        if "error" in traj:
            raise ValueError(f"Error in eval trajec: {traj['error']}")

        toolkits = get_toolkits_by_names(get_toolkit_names(traj["case"]))
        toolkit_descs = {
            detail_level: "\n".join(
                [f"{toolkit.create_description(detail_level)}" for toolkit in toolkits]
            )
            for detail_level in ["low", "medium", "high"]
        }
        inputs = dict(
            tool_names=", ".join([tool.name for tool in toolkits]),
            toolkit_descriptions=toolkit_descs[self._toolkit_desc_detail_level],
        )

        used_tool_level = self._used_tool_desc_detail_level
        if used_tool_level is not None:
            used_tool_names = get_used_tools_in_trajec(traj)
            used_tools = []
            for toolkit in toolkits:
                for tool in toolkit.tools:
                    if tool.name in used_tool_names:
                        used_tools.append(tool)
            used_tools_desc = "\n".join(
                [
                    f"* {tool.name}: {tool.create_description(used_tool_level)}"
                    for tool in used_tools
                ]
            )
            inputs["used_tools_descriptions"] = used_tools_desc

        return inputs

    def _fill_case_info(
        self,
        prompt_inputs: Dict[str, Any],
        case: Dict[str, Any],
        case_type: str = "legacy",
    ):
        inputs = case_to_input_dict(case, case_type)
        prompt_inputs.update(inputs)

    def _preprocess_prompts(
        self, prompts: List[PromptModule], prompt_inputs: Dict[str, Any]
    ) -> List[PromptModule]:
        return prompts

    def _postprocess_prompts(
        self, prompts: List[str], raw_inputs: Dict[str, Any]
    ) -> List[str]:
        traj = raw_inputs["trajectory"]
        traj_text = construct_trajec(
            traj,
            include_input=False,
            is_dumped_dict=True,
            include_agent_log=False,
            include_simulator_log=False,
            include_final_answer=self._final_answer_in_traj,
        )
        # Store trajectory text for cost tracking
        self._last_trajectory_text = traj_text
        # Debug: log trajectory text length
        traj_char_len = len(traj_text)
        traj_token_len = get_num_tokens(traj_text)
        print(f"[DEBUG] Trajectory text length: {traj_char_len} chars, {traj_token_len} tokens")
        processed_prompts = []
        for prompt, name in zip(prompts, self._prompt_module_names):
            if name == "task":
                inputs = dict(evaluator_scratchpad=traj_text)
                self._fill_case_info(
                    inputs, traj["case"], traj.get("case_type", "legacy")
                )
                prompt = prompt.format(**inputs)
                # Store full prompt for token tracking
                self._last_full_prompt = prompt
                # Debug: log final prompt length
                prompt_char_len = len(prompt)
                prompt_token_len = get_num_tokens(prompt)
                print(f"[DEBUG] Final prompt length: {prompt_char_len} chars, {prompt_token_len} tokens")
                # If too long, save to debug file
                if prompt_token_len > 10000 or prompt_char_len > 40000:
                    print(f"[WARNING] Prompt too long: {prompt_token_len} tokens, {prompt_char_len} chars. Saving to debug_prompt.txt.")
                    with open("debug_prompt.txt", "w") as f:
                        f.write(prompt)
            processed_prompts.append(prompt)
        return processed_prompts

    def _parse_output(
        self, output: ChatResult, raw_inputs: Dict[str, Any], **kwargs
    ) -> Any:
        res = parse_llm_response(output)
        print(f"[DEBUG] Raw evaluator response (FULL): {res}")  # Show full response
        eval_res = {k: None for k in self._output_keys}
        eval_res["eval_scores"] = {}

        # Try multiple regex patterns to handle different response formats
        patterns_to_try = []
        
        # Pattern 1: Original strict pattern (all fields in order, ending with $)
        pattern1 = ""
        for k in self._output_keys:
            pattern1 += k + r":\s*(.*?)\s*"
        pattern1 += "$"
        patterns_to_try.append(pattern1)
        
        # Pattern 2: More flexible pattern (all fields in order, but not necessarily at end)
        pattern2 = ""
        for k in self._output_keys:
            pattern2 += k + r":\s*(.*?)\s*"
        patterns_to_try.append(pattern2)
        
        # Pattern 3: Individual field matching (more robust) - handles markdown formatting
        pattern3 = {}
        for k in self._output_keys:
            # Handle both regular format and markdown format (# Field: value)
            pattern3[k] = r"(?:#\s*)?" + re.escape(k) + r":\s*(.*?)(?=\n\s*(?:#\s*)?" + "|".join([re.escape(other_k) for other_k in self._output_keys if other_k != k]) + r":|$)"
        
        print(f"[DEBUG] Trying regex patterns...")
        matches = None
        used_pattern = None
        
        # Try patterns 1 and 2 first
        for i, pattern in enumerate(patterns_to_try):
            print(f"[DEBUG] Trying pattern {i+1}: {pattern}")
            matches = re.search(pattern, res, re.DOTALL)
            if matches is not None:
                used_pattern = f"pattern {i+1}"
                break
        
        # If no match, try individual field matching
        if matches is None:
            print(f"[DEBUG] Trying individual field matching...")
            for k in self._output_keys:
                field_matches = re.search(pattern3[k], res, re.DOTALL)
                if field_matches is not None:
                    eval_res[k] = field_matches.group(1).strip()
                    if self._keys2metrics is not None and k in self._keys2metrics:
                        eval_res["eval_scores"][self._keys2metrics[k]] = eval_res[k]
            used_pattern = "individual fields"
        else:
            # Use the matched pattern
            for i, k in enumerate(self._output_keys):
                eval_res[k] = matches.group(i + 1).strip()
                if self._keys2metrics is not None and k in self._keys2metrics:
                    eval_res["eval_scores"][self._keys2metrics[k]] = eval_res[k]
        
        if used_pattern:
            print(f"[DEBUG] Successfully parsed using {used_pattern}")
        else:
            print(f"[DEBUG] Failed to parse any fields")
            print(f"[DEBUG] Looking for keys: {self._output_keys}")

        # Track evaluator costs
        model_name = getattr(self._llm, 'model_name', None)
        if model_name is None:
            # Try to get model name from model attribute for VLLM
            model_name = getattr(self._llm, 'model', 'unknown_model')
        if model_name:
            # Get input tokens from the full prompt
            input_tokens = get_num_tokens(getattr(self, '_last_full_prompt', ''))
            # Get output tokens from the response
            output_tokens = get_num_tokens(res)
            print(f"[DEBUG] Evaluator cost tracking: model={model_name}, input_tokens={input_tokens}, output_tokens={output_tokens}")
            if input_tokens > 0 or output_tokens > 0:
                self.cost_tracker.add_token_usage('evaluator', model_name, input_tokens, output_tokens)

        # --- Efficiency metric ---
        # Try to get token usage from llm_output if available
        efficiency = None
        if hasattr(output, 'llm_output') and output.llm_output is not None:
            token_usage = output.llm_output.get('token_usage')
            if token_usage is not None:
                # OpenAI style: {'total_tokens': ...}
                if isinstance(token_usage, dict) and 'total_tokens' in token_usage:
                    efficiency = token_usage['total_tokens']
                # Some other APIs may just return an int
                elif isinstance(token_usage, int):
                    efficiency = token_usage
        # If not available, fall back to manual counting
        if efficiency is None:
            # Try to get prompt and output text
            prompt = None
            if 'prompts' in raw_inputs:
                # Use the task prompt (usually index 1)
                if isinstance(raw_inputs['prompts'], list) and len(raw_inputs['prompts']) > 1:
                    prompt = raw_inputs['prompts'][1]
                elif isinstance(raw_inputs['prompts'], list):
                    prompt = raw_inputs['prompts'][0]
                else:
                    prompt = raw_inputs['prompts']
            # Fallback: try to get from conversation
            if prompt is None and 'conversation' in raw_inputs:
                prompt = '\n'.join(raw_inputs['conversation'])
            # Count tokens in prompt and output
            prompt_tokens = get_num_tokens(prompt) if prompt else 0
            output_tokens = get_num_tokens(res) if res else 0
            efficiency = prompt_tokens + output_tokens
        eval_res['efficiency'] = efficiency
        # --- End efficiency metric ---

        return eval_res


class AgentRiskyToolCallEvaluator(BaseTrajEvaluator):
    _short_name = "agent_safe"
    _sys_prompt_module = AGENT_SAFE_EVALUATOR_SYSTEM_INFO
    _task_prompt_module = AGENT_SAFE_EVALUATOR_PROMPT
    _keys2metrics = {"Overall Quantitative Score": METRICS_SHORT2LONG[_short_name]}


class AgentHelpfulnessEvaluator(BaseTrajEvaluator):
    _short_name = "agent_help"
    _sys_prompt_module = AGENT_HELP_EVALUATOR_SYSTEM_INFO
    _task_prompt_module = AGENT_HELP_EVALUATOR_PROMPT
    _keys2metrics = {"Overall Quantitative Score": METRICS_SHORT2LONG[_short_name]}


EVALUATORS = [
    AgentRiskyToolCallEvaluator,
    AgentHelpfulnessEvaluator,
]
EVALUATORS = {e._short_name: e for e in EVALUATORS}
EVALUATORS2METRICS = {k: v._keys2metrics.values() for k, v in EVALUATORS.items()}
