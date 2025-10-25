from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os
from datetime import datetime

@dataclass
class ModelCost:
    """Cost information for a specific model."""
    model_name: str
    input_cost_per_1k: float  # Cost per 1K input tokens
    output_cost_per_1k: float  # Cost per 1K output tokens
    input_tokens: int = 0
    output_tokens: int = 0
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost for this model."""
        return (self.input_tokens * self.input_cost_per_1k / 1000 + 
                self.output_tokens * self.output_cost_per_1k / 1000)

@dataclass
class ComponentCosts:
    """Cost tracking for a specific component (agent, emulator, evaluator)."""
    name: str
    model_costs: Dict[str, ModelCost] = field(default_factory=dict)
    
    def add_token_usage(self, model_name: str, input_tokens: int, output_tokens: int,
                       input_cost_per_1k: float, output_cost_per_1k: float):
        """Add token usage for a specific model."""
        if model_name not in self.model_costs:
            self.model_costs[model_name] = ModelCost(
                model_name=model_name,
                input_cost_per_1k=input_cost_per_1k,
                output_cost_per_1k=output_cost_per_1k
            )
        self.model_costs[model_name].input_tokens += input_tokens
        self.model_costs[model_name].output_tokens += output_tokens
        print(f"[DEBUG] ComponentCosts.add_token_usage: model_name={model_name}, input_tokens={input_tokens}, output_tokens={output_tokens}, model_costs={self.model_costs}")
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost for this component."""
        return sum(cost.total_cost for cost in self.model_costs.values())

class CostTracker:
    """Centralized cost tracking for all components."""
    
    # Default costs per 1K tokens (as of June 2025)
    DEFAULT_COSTS = {
        # OpenAI Models
        "o3": {"input": 0.002, "output": 0.008},  # $2/1M input, $8/1M output
        "o3-mini": {"input": 0.0011, "output": 0.0044},  # $1.10/1M input, $4.40/1M output
        "gpt-4.1": {"input": 0.002, "output": 0.008},  # $2/1M input, $8/1M output
        "gpt-4.1-nano": {"input": 0.0004, "output": 0.0016},  # $0.40/1M input, $1.60/1M output
        "gpt-4o": {"input": 0.0025, "output": 0.01},  # add hyphen variant
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # add hyphen variant
        
        # Anthropic Claude Models
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},  # $3/1M input, $15/1M output
        "claude-3-7-sonnet-20241022": {"input": 0.003, "output": 0.015},  # $3/1M input, $15/1M output
        "claude-3-5-sonnet-20250514": {"input": 0.003, "output": 0.015},  # $3/1M input, $15/1M output
        "claude-3-opus-20250514": {"input": 0.015, "output": 0.075},  # $15/1M input, $75/1M output
        
        # Google Gemini Models
        "gemini-2.5-pro-001": {"input": 0.00125, "output": 0.005},  # $1.25/1M input, $5/1M output
        "gemini-2.5-flash-001": {"input": 0.000075, "output": 0.0003},  # $0.075/1M input, $0.30/1M output
        "gemini-1.5-pro-002": {"input": 0.0005, "output": 0.0015},  # $0.50/1M input, $1.50/1M output
        "gemini-1.5-flash-002": {"input": 0.000075, "output": 0.0003},  # $0.075/1M input, $0.30/1M output
        
        # xAI Grok Models
        "grok-3": {"input": 0.003, "output": 0.015},  # $3/1M input, $15/1M output
        "grok-3-mini": {"input": 0.0003, "output": 0.0005},  # $0.30/1M input, $0.50/1M output
        "grok-3-speed": {"input": 0.005, "output": 0.025},  # $5/1M input, $25/1M output
        "grok-3-mini-speed": {"input": 0.0006, "output": 0.004},  # $0.60/1M input, $4/1M output
        
        # Legacy models (kept for backward compatibility)
        "gpt-4": {"input": 0.0025, "output": 0.01},  # $2.50/1M input, $10/1M output
        "gpt-4-32k": {"input": 0.0025, "output": 0.01},  # $2.50/1M input, $10/1M output
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # $1.50/1M input, $2/1M output
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},  # $3/1M input, $4/1M output
    }
    
    def __init__(self, output_dir: str = "logs/costs"):
        self.components: Dict[str, ComponentCosts] = {
            "agent": ComponentCosts("agent"),
            "emulator": ComponentCosts("emulator"),
            "evaluator": ComponentCosts("evaluator")
        }
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def add_token_usage(self, component: str, model_name: str, 
                       input_tokens: int, output_tokens: int,
                       input_cost_per_1k: Optional[float] = None,
                       output_cost_per_1k: Optional[float] = None):
        """Add token usage for a specific component and model."""
        print(f"[DEBUG] CostTracker.add_token_usage called: component={component}, model_name={model_name}, input_tokens={input_tokens}, output_tokens={output_tokens}, input_cost_per_1k={input_cost_per_1k}, output_cost_per_1k={output_cost_per_1k}")
        if component not in self.components:
            raise ValueError(f"Unknown component: {component}")
        # Get default costs if not provided
        if input_cost_per_1k is None or output_cost_per_1k is None:
            model_costs = self.DEFAULT_COSTS.get(model_name, {"input": 0.0, "output": 0.0})
            input_cost_per_1k = input_cost_per_1k or model_costs["input"]
            output_cost_per_1k = output_cost_per_1k or model_costs["output"]
        self.components[component].add_token_usage(
            model_name, input_tokens, output_tokens,
            input_cost_per_1k, output_cost_per_1k
        )
        
    def get_component_cost(self, component: str) -> float:
        """Get total cost for a specific component."""
        if component not in self.components:
            raise ValueError(f"Unknown component: {component}")
        return self.components[component].total_cost
    
    def get_total_cost(self) -> float:
        """Get total cost across all components."""
        return sum(comp.total_cost for comp in self.components.values())
    
    def save_costs(self, output_file: str) -> None:
        """Save the cost breakdown to a JSON file."""
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert defaultdict to regular dict for JSON serialization
        cost_data = {
            "total_cost": self.get_total_cost(),
            "model_costs": {
                f"{comp_name}_{model_name}": {
                    "input_tokens": model_cost.input_tokens,
                    "output_tokens": model_cost.output_tokens,
                    "input_cost_per_1k": model_cost.input_cost_per_1k,
                    "output_cost_per_1k": model_cost.output_cost_per_1k,
                    "total_cost": model_cost.total_cost
                }
                for comp_name, comp in self.components.items()
                for model_name, model_cost in comp.model_costs.items()
            },
            "token_usage": {
                f"{comp_name}_{model_name}": {
                    "input_tokens": model_cost.input_tokens,
                    "output_tokens": model_cost.output_tokens,
                    "input_cost_per_1k": model_cost.input_cost_per_1k,
                    "output_cost_per_1k": model_cost.output_cost_per_1k
                }
                for comp_name, comp in self.components.items()
                for model_name, model_cost in comp.model_costs.items()
            },
            "component_costs": {
                name: comp.total_cost
                for name, comp in self.components.items()
            }
        }
        print(f"[DEBUG] CostTracker.save_costs: cost_data={cost_data}")
        with open(output_file, "w") as f:
            json.dump(cost_data, f, indent=2) 