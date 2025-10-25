import logging
import os
from typing import Any, Callable, Dict, List, Optional, Mapping

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_anthropic import ChatAnthropic as LangchainChatAnthropic
from langchain_openai import ChatOpenAI as LangchainChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models.openai import convert_dict_to_message
from langchain_community.llms import VLLM as LangchainVLLM
from langchain_openai import OpenAI
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ChatMessage
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from .my_typing import *
from langchain_community.chat_models.anthropic import convert_messages_to_prompt_anthropic

# --- Model Environment Manager ---
class ModelEnvManager:
    """
    Manages environment variables for OpenAI/vLLM switching.
    
    API Priority for GPT models:
    1. OpenAI API (https://api.openai.com/v1) - Official OpenAI service
    2. OpenRouter (https://openrouter.ai/api/v1) - Third-party service providing access to various models
    
    For non-GPT models, the manager handles switching between local vLLM and external APIs.
    """
    _real_openai_api_key = None
    _real_openai_api_base = None
    _real_openrouter_api_key = None
    _real_openrouter_api_base = None
    VLLM_LOCAL_API_BASE = "http://127.0.0.1:8000/v1"
    VLLM_LOCAL_API_KEY = "EMPTY"
    _current_model = None

    @classmethod
    def set_env_for_model(cls, model_name: str):
        # GPT models should use OpenAI API if available, otherwise OpenRouter
        if model_name.lower().startswith("gpt"):
            # Check if we have OpenAI API credentials
            if os.environ.get("OPENAI_API_KEY"):
                # Use OpenAI API directly
                os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
                os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
            elif os.environ.get("OPENROUTER_API_KEY"):
                # Fall back to OpenRouter
                os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
                os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
            else:
                raise ValueError("No OpenAI API key or OpenRouter API key found for GPT models")
            cls._current_model = model_name
            return
            
        # Save real API keys if not already saved
        if cls._real_openai_api_key is None:
            cls._real_openai_api_key = os.environ.get("OPENAI_API_KEY")
        if cls._real_openai_api_base is None:
            cls._real_openai_api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        if cls._real_openrouter_api_key is None:
            cls._real_openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if cls._real_openrouter_api_base is None:
            cls._real_openrouter_api_base = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
            
        # If we're already set for this model, no need to change
        if cls._current_model == model_name:
            return
            
        if cls.is_vllm_model(model_name):
            os.environ["OPENAI_API_BASE"] = cls.VLLM_LOCAL_API_BASE
            os.environ["OPENAI_API_KEY"] = cls.VLLM_LOCAL_API_KEY
        else:
            # Restore real API keys (prefer OpenAI over OpenRouter)
            if cls._real_openrouter_api_key is not None:
                os.environ["OPENAI_API_BASE"] = cls._real_openrouter_api_base
                os.environ["OPENAI_API_KEY"] = cls._real_openrouter_api_key
            elif cls._real_openai_api_key is not None:
                os.environ["OPENAI_API_BASE"] = cls._real_openai_api_base
                os.environ["OPENAI_API_KEY"] = cls._real_openai_api_key
            else:
                os.environ.pop("OPENAI_API_BASE", None)
                os.environ.pop("OPENAI_API_KEY", None)
                
        cls._current_model = model_name

    @staticmethod
    def is_vllm_model(model_name: str) -> bool:
        # vllm_keywords = ["vicuna", "qwen", "llama", "mistral", "phi", "opt", "baichuan", "deepseek", "gemma"]
        vllm_keywords = ["32b"]
        return any(k in model_name.lower() for k in vllm_keywords)

    @staticmethod
    def supports_stop_parameter(model_name: str) -> bool:
        """
        Check if a model supports the stop parameter.
        GPT models (gpt-*) do not support the stop parameter.
        """
        if model_name is None:
            return False
        # GPT models don't support stop parameter
        if model_name.lower().startswith("gpt"):
            return False
        # Local vLLM models generally support stop parameter
        if ModelEnvManager.is_vllm_model(model_name):
            return True
        # For other models (including OpenRouter models), assume they support it
        return True

# --- Model Loader Functions (retain names) ---

def llm_register_args(parser, prefix=None, shortprefix=None, defaults={}):
    model_name = defaults.get("model_name", "gpt-4o-mini")
    temperature = defaults.get("temperature", 0.0)
    max_tokens = defaults.get("max_tokens", None)
    default_retries = 8 if model_name.startswith("claude") else 12
    max_retries = defaults.get("max_retries", default_retries)
    request_timeout = defaults.get("request_timeout", 300)
    if prefix is None:
        prefix = ""
        shortprefix = ""
    else:
        prefix += "-"
        shortprefix = shortprefix or prefix[0]
    parser.add_argument(
        f"--{prefix}model-name",
        f"-{shortprefix}m",
        type=str,
        default=model_name,
        help=f"Model name (OpenAI, Anthropic, or open source)",
    )
    parser.add_argument(
        f"--{prefix}temperature",
        f"-{shortprefix}t",
        type=float,
        default=temperature,
        help="Temperature for sampling",
    )
    parser.add_argument(
        f"--{prefix}max-tokens",
        f"-{shortprefix}mt",
        type=int,
        default=max_tokens,
        help="Max tokens for sampling",
    )
    parser.add_argument(
        f"--{prefix}max-retries",
        f"-{shortprefix}mr",
        type=int,
        default=max_retries,
        help="Max retries for each request",
    )
    parser.add_argument(
        f"--{prefix}request-timeout",
        f"-{shortprefix}rt",
        type=int,
        default=request_timeout,
        help="Timeout for each request",
    )
    # Add tensor_parallel_size for vLLM models (open source)
    parser.add_argument(
        f"--{prefix}tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for vLLM open source models (tensor parallelism)",
    )
    # Add quantization for vLLM models (open source)
    parser.add_argument(
        f"--{prefix}quantization",
        type=str,
        default=None,
        help="Quantization type for vLLM open source models (e.g., awq-int4, int8)",
    )
    # Add GPU memory utilization for vLLM models (open source)
    parser.add_argument(
        f"--{prefix}gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM open source models (0.0 to 1.0)",
    )
    # Add max sequence length for vLLM models (open source)
    parser.add_argument(
        f"--{prefix}max-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length for vLLM open source models",
    )
    parser.add_argument(
        f"--{prefix}vllm-port",
        type=int,
        default=8000,
        help="Port for online vLLM server (OpenAI-compatible API)",
    )

def get_tensor_parallel_size(args, prefix=None):
    # Try to get from args, then env, else default to 1
    if prefix is None:
        prefix = ""
    else:
        prefix = f"{prefix}_"
    arg_name = f"{prefix}tensor_parallel_size"
    if hasattr(args, arg_name):
        val = getattr(args, arg_name)
        if val is not None:
            return val
    # Try environment variable
    val = os.environ.get("TENSOR_PARALLEL_SIZE")
    if val is not None:
        try:
            return int(val)
        except Exception:
            pass
    return 1

# --- Main Loader (now always online for vLLM) ---
def load_openai_llm(model_name: str = "gpt-4o-mini", tensor_parallel_size=None, quantization=None, gpu_memory_utilization=None, max_seq_len=None, vllm_port=8000, **kwargs) -> BaseLanguageModel:
    # Set environment variables for the model (this handles GPT models properly)
    ModelEnvManager.set_env_for_model(model_name)
    
    # Handle GPT models - they should use OpenAI API or OpenRouter
    if model_name.lower().startswith("gpt"):
        # Remove vLLM-specific kwargs for GPT models
        kwargs.pop("tensor_parallel_size", None)
        kwargs.pop("quantization", None)
        kwargs.pop("vllm_kwargs", None)
        kwargs.pop("temperature", None)
        
        llm = ChatOpenAI(model_name=model_name, **kwargs)
        llm.model_name = model_name
        return llm
    
    # OpenAI (non-GPT models)
    if any(model_name.lower().startswith(prefix) for prefix in ["openai", "anthropic", "text-davinci", "gpt4o", "o"]):
        # Remove vllm-specific kwargs
        kwargs.pop("tensor_parallel_size", None)
        kwargs.pop("quantization", None)
        kwargs.pop("vllm_kwargs", None)
        llm = ChatOpenAI(model_name=model_name, **kwargs)
        llm.model_name = model_name
        return llm
    # vLLM (open source, online)
    elif ModelEnvManager.is_vllm_model(model_name):
        api_base = f"http://localhost:{vllm_port}/v1"
        api_key = "EMPTY"
        # Remove irrelevant kwargs
        for k in ["tensor_parallel_size", "quantization", "vllm_kwargs", "gpu_memory_utilization", "max_seq_len"]:
            kwargs.pop(k, None)
        # Qwen3: disable thinking mode and set max_tokens
        if "qwen3" in model_name.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            if "max_tokens" not in kwargs or kwargs["max_tokens"] is None:
                kwargs["max_tokens"] = 2048
        llm = ChatOpenAI(model_name=model_name, openai_api_base=api_base, openai_api_key=api_key, **kwargs)
        return llm
    # Fallback: try OpenAI, but warn
    else:
        print(f"[WARNING] Unknown model type for '{model_name}', defaulting to OpenAI-compatible loading.")
        kwargs.pop("tensor_parallel_size", None)
        kwargs.pop("quantization", None)
        kwargs.pop("vllm_kwargs", None)
        llm = ChatOpenAI(model_name=model_name, **kwargs)
        llm.model_name = model_name
        return llm

# --- Args Loader (update to pass vllm_port) ---
def load_openai_llm_with_args(args, prefix=None, fixed_version=True, **kwargs):
    if prefix is None:
        prefix = ""
    # Compose argument names
    def get_arg(name, default=None):
        arg_name = f"{prefix}_" + name if prefix else name
        return getattr(args, arg_name, default)
    model_name = get_arg("model_name", kwargs.get("model_name"))
    if not model_name:
        raise ValueError(f"Model name for '{prefix or 'model'}' must be specified via --{prefix + '-' if prefix else ''}model-name!")
    temperature = get_arg("temperature", kwargs.get("temperature", 0.0))
    request_timeout = get_arg("request_timeout", kwargs.get("request_timeout", 60))
    max_tokens = get_arg("max_tokens", kwargs.get("max_tokens", None))
    tensor_parallel_size = get_arg("tensor_parallel_size", None)
    quantization = get_arg("quantization", None)
    gpu_memory_utilization = get_arg("gpu_memory_utilization", None)
    max_seq_len = get_arg("max_seq_len", None)
    vllm_port = get_arg("vllm_port", 8000)
    
    # Always check if it's a vLLM model and pass appropriate args
    if ModelEnvManager.is_vllm_model(model_name):
        return load_openai_llm(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            max_tokens=max_tokens,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            max_seq_len=max_seq_len,
            vllm_port=vllm_port,
            **kwargs,
        )
    elif model_name.lower().startswith("gpt"):
        # GPT models should not get vLLM-specific arguments
        return load_openai_llm(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            max_tokens=max_tokens,
            **kwargs,
        )
    else:
        # Remove vLLM-specific args for non-vLLM models
        return load_openai_llm(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            max_tokens=max_tokens,
            **kwargs,
        )

# --- Model Category Utility (keep for compatibility) ---
def get_model_category(llm: BaseLanguageModel):
    if isinstance(llm, ChatOpenAI):
        return "openai"
    elif isinstance(llm, ChatAnthropic):
        return "claude"
    elif isinstance(llm, LangchainVLLM):
        return "vllm"
    elif hasattr(llm, '_llm_type') and llm._llm_type == "mock":
        return "openai"  # Treat mock LLMs as OpenAI for testing
    else:
        raise ValueError(f"Unknown model type: {type(llm)}")

# --- ChatOpenAI Patch (keep for compatibility) ---
class ChatOpenAI(LangchainChatOpenAI):
    def _create_chat_result(self, response, generation_info=None):
        # Handle both dict and object (ChatCompletion) responses
        if isinstance(response, dict):
            choices = response["choices"]
            usage = response.get("usage")
        else:
            choices = response.choices
            usage = getattr(response, "usage", None)
        generations = []
        for res in choices:
            if isinstance(res, dict):
                message_dict = res["message"]
                finish_reason = res.get("finish_reason")
            else:
                message_obj = res.message
                if hasattr(message_obj, "to_dict"):
                    message_dict = message_obj.to_dict()
                else:
                    message_dict = dict(message_obj.__dict__)
                finish_reason = getattr(res, "finish_reason", None)
            message = convert_dict_to_message(message_dict)
            gen = ChatGeneration(
                message=message,
                generation_info=dict(
                    finish_reason=finish_reason,
                ),
            )
            generations.append(gen)
        # Ensure usage is a dict for compatibility
        if usage is not None and not isinstance(usage, dict):
            if hasattr(usage, "to_dict"):
                usage = usage.to_dict()
            else:
                usage = dict(usage.__dict__)
        llm_output = {"token_usage": usage, "model_name": self.model_name}
        if generation_info is not None:
            llm_output["generation_info"] = generation_info
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        # Get the parent's default params
        params = super()._default_params
        
        # For GPT models, remove the stop parameter as it's not supported
        if hasattr(self, 'model_name') and self.model_name and self.model_name.lower().startswith("gpt"):
            if "stop" in params:
                del params["stop"]
        
        return params

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """Get the request payload for the API call."""
        # For GPT models, ignore the stop parameter completely
        if hasattr(self, 'model_name') and self.model_name and self.model_name.lower().startswith("gpt"):
            stop = None
            # Also remove stop from kwargs if it exists
            kwargs.pop("stop", None)
        
        # Call the parent method with the filtered parameters
        return super()._get_request_payload(input_, stop=stop, **kwargs)

    def __init__(self, **kwargs):
        # For GPT models, remove any stop parameter from model_kwargs
        if kwargs.get('model_name', '').lower().startswith('gpt'):
            if 'model_kwargs' in kwargs and 'stop' in kwargs['model_kwargs']:
                del kwargs['model_kwargs']['stop']
            if 'stop' in kwargs:
                del kwargs['stop']
            if 'stop_sequences' in kwargs:
                del kwargs['stop_sequences']
        
        super().__init__(**kwargs)

# --- Anthropic Patch (keep for compatibility) ---
logger = logging.getLogger(__name__)

def _anthropic_create_retry_decorator(llm: ChatOpenAI) -> Callable[[Any], Any]:
    import anthropic
    min_seconds = 1
    max_seconds = 60
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(anthropic.APITimeoutError)
            | retry_if_exception_type(anthropic.APIError)
            | retry_if_exception_type(anthropic.APIConnectionError)
            | retry_if_exception_type(anthropic.RateLimitError)
            | retry_if_exception_type(anthropic.APIConnectionError)
            | retry_if_exception_type(anthropic.APIStatusError)
            | retry_if_exception_type(anthropic.InternalServerError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

class ChatAnthropic(LangchainChatAnthropic):
    max_retries: int = 6
    max_tokens_to_sample: int = 4000
    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        finish_reason = response.stop_reason
        if finish_reason == "max_tokens":
            finish_reason = "length"
        generations = [
            ChatGeneration(
                message=AIMessage(content=response.completion),
                generation_info=dict(
                    finish_reason=finish_reason,
                ),
            )
        ]
        llm_output = {"model_name": response.model}
        return ChatResult(generations=generations, llm_output=llm_output)

def parse_llm_response(
    res: ChatResult, index: int = 0, one_generation_only: bool = True
) -> str:
    print(f"[DEBUG] parse_llm_response: res type={type(res)}, has generations={hasattr(res, 'generations')}")
    if hasattr(res, 'generations'):
        print(f"[DEBUG] generations length: {len(res.generations)}")
        if len(res.generations) > 0:
            print(f"[DEBUG] first generation type: {type(res.generations[0])}")
            if hasattr(res.generations[0], 'generation_info'):
                print(f"[DEBUG] generation_info: {res.generations[0].generation_info}")
            else:
                print(f"[DEBUG] No generation_info attribute")
    
    res = res.generations[0]
    if one_generation_only:
        assert len(res) == 1, res
    res = res[index]
    
    # Handle case where generation_info might be None
    if hasattr(res, 'generation_info') and res.generation_info is not None:
        if res.generation_info.get("finish_reason", None) == "length":
            raise ValueError(f"Discard a response due to length: {res.text}")
    else:
        print(f"[DEBUG] No generation_info or it's None, skipping finish_reason check")
    
    return res.text.strip()
