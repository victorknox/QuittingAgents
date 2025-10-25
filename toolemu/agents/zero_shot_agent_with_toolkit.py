"""Zero-shot agent with toolkit."""
import re

from langchain.agents.agent import Agent
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import BaseCallbackManager
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


from langchain_core.prompts.chat import BaseStringMessagePromptTemplate

from langchain_core.prompts import HumanMessagePromptTemplate

from langchain_core.prompts import AIMessagePromptTemplate

from langchain_core.prompts import SystemMessagePromptTemplate

from langchain_core.prompts import ChatPromptTemplate


from langchain_core.agents import AgentFinish

from langchain_core.agents import AgentAction

from langchain_core.messages import BaseMessage

from langchain_core.messages import HumanMessage

from langchain_core.messages import AIMessage

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from procoder.functional import (
    add_refnames,
    collect_refnames,
    format_multiple_prompts,
    format_prompt,
)
from toolemu.prompts.agent import (
    AGENT_SYSTEM_INFO,
    AGENT_QUIT_SYSTEM_INFO,
    AGENT_NAIVE_PROMPT,
    AGENT_NAIVE_CLAUDE_PROMPT,
    AGENT_SS_ONLY_PROMPT,
    AGENT_SS_ONLY_CLAUDE_PROMPT,
    AGENT_HELPFUL_SS_PROMPT,
    AGENT_HELPFUL_SS_CLAUDE_PROMPT,
    AGENT_QUIT_PROMPT,
    AGENT_QUIT_CLAUDE_PROMPT,
    AGENT_DUMMY_VARS,
    AGENT_QUIT_SIMPLE_PROMPT,
)
from toolemu.prompts.agent.agent_quit import AGENT_QUIT_SYSTEM_INFO, AGENT_QUIT_PROMPT, AGENT_QUIT_CLAUDE_PROMPT
from toolemu.prompts.agent.shared import AGENT_SYSTEM_INFO
from toolemu.tools.tool_interface import BaseToolkit
from toolemu.utils import get_first_json_object_str
from toolemu.utils.llm import get_model_category
from toolemu.utils.my_typing import *
from toolemu.utils.misc import truncate_to_max_tokens, get_num_tokens

FINAL_ANSWER_ACTION = "Final Answer:"
QUIT_ACTION = "QUIT:"
AGENT_TYPES = ["naive", "ss_only", "helpful_ss", "quit", "specific_quit", "extreme_quit", "simple_quit"]

MAX_TOKENS = 16384


class ZeroShotAgentWithToolkit(ZeroShotAgent):
    cost_tracker: Optional[Any] = None
    def __init__(self, *args, cost_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_tracker = cost_tracker

    @staticmethod
    def get_all_tools(toolkits: Sequence[BaseToolkit]) -> List[BaseTool]:
        """Return all tools available to the agent."""
        all_tools = []
        for toolkit in toolkits:
            all_tools += toolkit.tools
        return all_tools

    @classmethod
    def create_prompt(
        cls,
        toolkits: Sequence[BaseToolkit],
        prompt_type: Optional[str] = "naive",
        input_variables: Optional[List[str]] = None,
        use_chat_format: Optional[bool] = False,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent."""
        toolkit_strings = "\n".join(
            [toolkit.create_description("medium") for toolkit in toolkits]
        )
        tool_names = ", ".join([tool.name for tool in cls.get_all_tools(toolkits)])

        inputs = dict(toolkit_descriptions=toolkit_strings, tool_names=tool_names)

        add_refnames(AGENT_DUMMY_VARS, inputs, include_brackets=False)

        # Use quit-specific system info for quit prompt type
        if prompt_type.startswith("simple_quit"):
            system_info = AGENT_QUIT_SYSTEM_INFO
            prompt_instruction = AGENT_QUIT_SIMPLE_PROMPT
        else:
            system_info = AGENT_QUIT_SYSTEM_INFO if prompt_type.endswith("quit") else AGENT_SYSTEM_INFO
            prompt_instruction = eval(f"AGENT_{prompt_type.upper()}_PROMPT")

        system_info, prompt_instruction = format_multiple_prompts(
            [system_info, prompt_instruction], inputs, include_brackets=[False, True]
        )

        if use_chat_format:
            agent_system_message = SystemMessage(content=system_info)
            agent_instruction_message = HumanMessagePromptTemplate.from_template(
                template=prompt_instruction
            )

            messages = [
                agent_system_message,
                agent_instruction_message,
            ]
            return ChatPromptTemplate.from_messages(messages=messages)
        else:
            template = "\n\n".join([system_info, prompt_instruction])

            if input_variables is None:
                input_variables = ["input", "agent_scratchpad"]
            return PromptTemplate(template=template, input_variables=input_variables)

    @classmethod
    def from_llm_and_toolkits(
        cls,
        llm: BaseLanguageModel,
        toolkits: Sequence[BaseToolkit],
        agent_type: Optional[str] = "naive",
        callback_manager: Optional[BaseCallbackManager] = None,
        use_chat_format: Optional[bool] = False,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        tools = cls.get_all_tools(toolkits)
        cls._validate_tools(tools)

        assert agent_type in AGENT_TYPES, f"agent_type must be one of {AGENT_TYPES}"

        if get_model_category(llm) == "claude":
            prompt_type = agent_type + "_claude"
        else:
            prompt_type = agent_type

        prompt = cls.create_prompt(
            toolkits,
            prompt_type=prompt_type,
            input_variables=input_variables,
            use_chat_format=use_chat_format,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)

    def _fix_text(self, text: str):
        text = text.lstrip()
        if text.startswith(self.llm_prefix):
            text = text[len(self.llm_prefix) :]

        return text.rstrip() + "\n"

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        try:
            result = self._get_action_and_input(text)
        except ValueError:
            # result = "None", "{}"
            result = None  # return None, so that the agent can continue its generation
        return result

    ############# Adapted from Langchain (0.0.141) for compatibility #############

    @property
    def finish_tool_name(self) -> str:
        """Return the name of the finish tool."""
        return "Final Answer"

    @property
    def quit_tool_name(self) -> str:
        """Return the name of the quit tool."""
        return "QUIT"

    def _get_action_and_input(self, llm_output: str) -> Tuple[str, str]:
        """Parse out the action and input from the LLM output.

        Note: if you're specifying a custom prompt for the ZeroShotAgent,
        you will need to ensure that it meets the following Regex requirements.
        The string starting with "Action:" and the following string starting
        with "Action Input:" should be separated by a newline.
        """
        if FINAL_ANSWER_ACTION in llm_output:
            return (
                self.finish_tool_name,
                llm_output.split(FINAL_ANSWER_ACTION)[-1].strip(),
            )
        if QUIT_ACTION in llm_output:
            return (
                self.quit_tool_name,
                llm_output.split(QUIT_ACTION)[-1].strip(),
            )
        # \s matches against tab/newline/whitespace
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # disable checking to avoid exception, let the tool call validate the inputs
        action_input = get_first_json_object_str(
            action_input, enable_check=False, strict=False
        )
        # set strict=False to allow single quote, comments, escape characters, etc
        return action, action_input.strip(" ").strip('"')

    def _get_next_action(self, full_inputs: Dict[str, str]) -> AgentAction:
        # Log and truncate the input prompt if necessary
        if "input" in full_inputs:
            orig = full_inputs["input"]
            orig_tokens = get_num_tokens(orig)
            truncated = truncate_to_max_tokens(orig, MAX_TOKENS)
            truncated_tokens = get_num_tokens(truncated)
            print(f"[INFO] Agent input token count before truncation: {orig_tokens}")
            if orig != truncated:
                print(f"[WARNING] Truncated agent input from {orig_tokens} to {truncated_tokens} tokens to fit max tokens.")
            full_inputs["input"] = truncated
        # Get model name for cost tracking
        model_name = None
        if hasattr(self.llm_chain.llm, 'model_name'):
            model_name = self.llm_chain.llm.model_name
        elif hasattr(self.llm_chain.llm, 'model'):
            model_name = self.llm_chain.llm.model
        elif hasattr(self.llm_chain.llm, '_llm_type'):
            model_name = self.llm_chain.llm._llm_type
        # Track input tokens
        input_tokens = 0
        if model_name:
            input_text = self.llm_chain.prompt.format(**full_inputs)
            input_tokens = get_num_tokens(input_text)
            print(f"[DEBUG] Agent cost tracking: model={model_name}, input_tokens={input_tokens}")
        full_output = self.llm_chain.predict(**full_inputs)
        full_output = self._fix_text(full_output)
        # Track output tokens
        output_tokens = 0
        if model_name:
            output_tokens = get_num_tokens(full_output)
            print(f"[DEBUG] Agent cost tracking: model={model_name}, output_tokens={output_tokens}")
        parsed_output = self._extract_tool_and_input(full_output)
        while parsed_output is None:
            full_output = self._fix_text(full_output)
            full_inputs["agent_scratchpad"] += full_output
            output = self.llm_chain.predict(**full_inputs)
            full_output += output
            # Track additional output tokens
            if model_name:
                output_tokens += get_num_tokens(output)
                print(f"[DEBUG] Agent cost tracking: model={model_name}, additional_output_tokens={get_num_tokens(output)}")
            parsed_output = self._extract_tool_and_input(full_output)
        # Add to cost tracker if available
        if self.cost_tracker and model_name:
            self.cost_tracker.add_token_usage('agent', model_name, input_tokens, output_tokens)
        return AgentAction(
            tool=parsed_output[0], tool_input=parsed_output[1], log=full_output
        )

    async def _aget_next_action(self, full_inputs: Dict[str, str]) -> AgentAction:
        full_output = await self.llm_chain.apredict(**full_inputs)
        parsed_output = self._extract_tool_and_input(full_output)
        while parsed_output is None:
            full_output = self._fix_text(full_output)
            full_inputs["agent_scratchpad"] += full_output
            output = await self.llm_chain.apredict(**full_inputs)
            full_output += output
            parsed_output = self._extract_tool_and_input(full_output)
        return AgentAction(
            tool=parsed_output[0], tool_input=parsed_output[1], log=full_output
        )

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        action = self._get_next_action(full_inputs)
        if action.tool == self.finish_tool_name:
            return AgentFinish({"output": action.tool_input}, action.log)
        if action.tool == self.quit_tool_name:
            # Use QUIT: format consistently
            return AgentFinish({"output": f"QUIT: {action.tool_input}"}, action.log)
        return action

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        action = await self._aget_next_action(full_inputs)
        if action.tool == self.finish_tool_name:
            return AgentFinish({"output": action.tool_input}, action.log)
        if action.tool == self.quit_tool_name:
            # Use QUIT: format consistently
            return AgentFinish({"output": f"QUIT: {action.tool_input}"}, action.log)
        return action
