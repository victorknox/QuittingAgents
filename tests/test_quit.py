import pytest
from langchain_core.language_models import LLM
from toolemu.agents.zero_shot_agent_with_toolkit import ZeroShotAgentWithToolkit, QUIT_ACTION
from toolemu.tools.tool_interface import BaseToolkit
from langchain_core.agents import AgentFinish
from langchain_core.tools import BaseTool
from toolemu.prompts.agent.agent_quit import AGENT_QUIT_SYSTEM_INFO, AGENT_QUIT_FORMAT
from pydantic import Field, PrivateAttr
from procoder.functional import format_multiple_prompts, add_refnames
from toolemu.prompts.agent.shared import AGENT_DUMMY_VARS

class MockTool(BaseTool):
    name = "mock_tool"
    description = "A mock tool for testing"

    def _run(self, query: str) -> str:
        return "Mock tool response"

    def _arun(self, query: str) -> str:
        raise NotImplementedError("Mock tool doesn't support async")

class MockLLM(LLM):
    _last_prompt: str = PrivateAttr(default=None)

    def _call(self, prompt: str, stop=None) -> str:
        print(f"_call called with prompt: {prompt}")
        # Store the prompt for verification
        self._last_prompt = prompt
        # Always return a quit response
        return f"Thought: I should quit because this is a test\n{QUIT_ACTION} Testing quit functionality"

    def predict(self, **kwargs) -> str:
        print(f"predict called with kwargs: {kwargs}")
        # Extract the prompt from kwargs
        prompt = kwargs.get("prompt", "")
        # Store the prompt for verification
        self._last_prompt = prompt
        # Always return a quit response
        return f"Thought: I should quit because this is a test\n{QUIT_ACTION} Testing quit functionality"

    @property
    def _llm_type(self) -> str:
        return "mock"

    @property
    def last_prompt(self) -> str:
        return self._last_prompt

class MockToolkit(BaseToolkit):
    def __init__(self):
        self._name = "mock_toolkit"
        super().__init__()

    @property
    def name(self) -> str:
        return self._name

    def load_tools(self):
        return [MockTool()]

    def create_description(self, detail_level: str) -> str:
        return "Mock toolkit description"

def test_quit_functionality():
    # Create agent with mock LLM and toolkit with one tool
    mock_llm = MockLLM()
    agent = ZeroShotAgentWithToolkit.from_llm_and_toolkits(
        llm=mock_llm,
        toolkits=[MockToolkit()],
        agent_type="quit"  # Use quit type to get quit-specific prompt
    )

    # Test planning with empty intermediate steps
    result = agent.plan(intermediate_steps=[], input="test input")

    # Verify the result is an AgentFinish with [QUIT] prefix
    assert isinstance(result, AgentFinish)
    assert result.return_values["output"].startswith("[QUIT]")
    assert "Testing quit functionality" in result.return_values["output"]

    # Verify that the quit-specific prompt was used
    assert mock_llm.last_prompt is not None
    print(f"Last prompt: {mock_llm.last_prompt}")

    # Format the expected quit prompt
    inputs = dict(toolkit_descriptions="Mock toolkit description", tool_names="mock_tool")
    add_refnames(AGENT_DUMMY_VARS, inputs, include_brackets=False)
    system_info = AGENT_QUIT_SYSTEM_INFO("", "", {"agent": "assistant", "user": "user"})
    assert "You have the ability to quit at any time" in str(system_info)
    assert "If you feel uncertain about the safety implications" in str(AGENT_QUIT_FORMAT) 