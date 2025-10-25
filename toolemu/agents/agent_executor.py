"""Virtual agent executor"""

import asyncio
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.agents import (
    AgentExecutor,
    BaseMultiActionAgent,
    BaseSingleActionAgent,
    
    
)

from langchain_core.tools import Tool

from langchain_core.tools import tool
from langchain.agents.agent import ExceptionTool
from langchain_core.language_models import BaseLanguageModel


from langchain_core.callbacks import BaseCallbackHandler

from langchain_core.callbacks import BaseCallbackManager
from langchain.callbacks.manager import (
    
    
    Callbacks,
)

from langchain_core.callbacks import CallbackManagerForChainRun

from langchain_core.callbacks import AsyncCallbackManagerForChainRun


from langchain_core.agents import AgentFinish

from langchain_core.agents import AgentAction

from langchain_core.exceptions import OutputParserException
from langchain_core.tools import BaseTool
from toolemu.tools.tool_interface import BaseToolkit
from toolemu.utils import InvalidTool
from toolemu.utils.my_typing import *

from ..utils.cost_tracker import CostTracker
from ..utils.misc import get_num_tokens


class AgentExecutorWithToolkit(AgentExecutor):
    """Agent executor with toolkits"""

    tool_names: List[str]
    toolkits: List[BaseToolkit]

    @classmethod
    def from_agent_and_toolkits(
        cls,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        toolkits: Sequence[BaseToolkit],
        callbacks: Callbacks = None,
        track_costs: bool = False,
        **kwargs: Any,
    ) -> AgentExecutor:
        """Create from agent and toolkits."""
        tools = agent.get_all_tools(toolkits)
        tool_names = [tool.name for tool in tools]
        executor_class = AgentExecutorWithCostTracking if track_costs else cls
        return executor_class(
            agent=agent,
            tools=tools,
            toolkits=toolkits,
            tool_names=tool_names,
            callbacks=callbacks,
            **kwargs,
        )

    @classmethod
    def from_agent_and_tools(
        cls,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        tools: Sequence[BaseTool],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentExecutor:
        """Replaced by `from_agent_and_toolkits`"""
        raise NotImplementedError("Use `from_agent_and_toolkits` instead")

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Override to add final output log to intermediate steps."""
        # update at langchain==0.0.277
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        # * ====== ToolEmu: begin add ======
        intermediate_steps.append((deepcopy(output), ""))
        # * ====== ToolEmu: end add ======
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    async def _areturn(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Override to add final output log to intermediate steps."""
        # update at langchain==0.0.277
        if run_manager:
            await run_manager.on_agent_finish(
                output, color="green", verbose=self.verbose
            )
        final_output = output.return_values
        # * ====== ToolEmu: begin add ======
        intermediate_steps.append((deepcopy(output), ""))
        # * ====== ToolEmu: end add ======
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    # ToolEmu: below parts are just override to replace InvalidTool with ours, update at langchain==0.0.277
    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override to use custom InvalidTool."""
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]

        # Handle quit responses
        if isinstance(output, AgentAction) and output.tool.startswith("QUIT:"):
            return AgentFinish(
                return_values={"output": output.tool},
                log=output.log,
            )

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                # * ====== ToolEmu: begin overwrite ======
                observation = InvalidTool(available_tools=self.tool_names).run(
                    agent_action.tool,
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
                # * ====== ToolEmu: end overwrite ======
            result.append((agent_action, observation))
        return result

    async def _atake_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override to use custom InvalidTool."""
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = await self.agent.aplan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                await run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = await ExceptionTool().arun(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]

        # Handle quit responses
        if isinstance(output, AgentAction) and output.tool.startswith("QUIT:"):
            return AgentFinish(
                return_values={"output": output.tool},
                log=output.log,
            )

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output

        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output

        async def _aperform_agent_action(
            agent_action: AgentAction,
        ) -> Tuple[AgentAction, str]:
            if run_manager:
                await run_manager.on_agent_action(agent_action, color="green")
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = await tool.arun(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                # * ====== ToolEmu: begin overwrite ======
                observation = await InvalidTool(available_tools=self.tool_names).arun(
                    agent_action.tool,
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
                # * ====== ToolEmu: end overwrite ======
            return agent_action, observation

        # Use asyncio.gather to run multiple tools concurrently
        result = await asyncio.gather(
            *[_aperform_agent_action(agent_action) for agent_action in actions]
        )

        return list(result)

class AgentExecutorWithCostTracking(AgentExecutor):
    """Agent executor with cost tracking capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_tracker = CostTracker()
        
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute the agent with cost tracking."""
        # Track agent costs
        model_name = None
        if hasattr(self.agent.llm_chain.llm, 'model_name'):
            model_name = self.agent.llm_chain.llm.model_name
        elif hasattr(self.agent.llm_chain.llm, 'model'):
            model_name = self.agent.llm_chain.llm.model
        elif hasattr(self.agent.llm_chain.llm, '_llm_type'):
            model_name = self.agent.llm_chain.llm._llm_type
            
        if model_name:
            input_text = self.agent.llm_chain.prompt.format(**inputs)
            input_tokens = get_num_tokens(input_text)
            result = super()._call(inputs, run_manager)
            output_tokens = 0
            if isinstance(result, dict):
                if 'output' in result:
                    output_tokens = get_num_tokens(result['output'])
                elif 'intermediate_steps' in result:
                    for step in result['intermediate_steps']:
                        if isinstance(step, tuple) and len(step) > 0:
                            if hasattr(step[0], 'log'):
                                output_tokens += get_num_tokens(step[0].log)
            print(f"[DEBUG] Agent cost tracking: model={model_name}, input_tokens={input_tokens}, output_tokens={output_tokens}")
            if input_tokens > 0 or output_tokens > 0:
                self.cost_tracker.add_token_usage('agent', model_name, input_tokens, output_tokens)
            return result
        return super()._call(inputs, run_manager)
        
    def _call_emulator(self, action: AgentAction) -> str:
        """Execute emulator with cost tracking."""
        model_name = None
        if hasattr(self.emulator.llm_chain.llm, 'model_name'):
            model_name = self.emulator.llm_chain.llm.model_name
        elif hasattr(self.emulator.llm_chain.llm, 'model'):
            model_name = self.emulator.llm_chain.llm.model
        elif hasattr(self.emulator.llm_chain.llm, '_llm_type'):
            model_name = self.emulator.llm_chain.llm._llm_type
            
        if model_name:
            input_text = self.emulator.llm_chain.prompt.format(action=action)
            input_tokens = get_num_tokens(input_text)
            result = super()._call_emulator(action)
            output_tokens = get_num_tokens(result)
            print(f"[DEBUG] Emulator cost tracking: model={model_name}, input_tokens={input_tokens}, output_tokens={output_tokens}")
            if input_tokens > 0 or output_tokens > 0:
                self.cost_tracker.add_token_usage('emulator', model_name, input_tokens, output_tokens)
            return result
        return super()._call_emulator(action)
        
    def save_costs(self, experiment_name: str) -> str:
        """Save cost information to a file."""
        return self.cost_tracker.save_costs(experiment_name)
        
    def get_total_cost(self) -> float:
        """Get total cost across all components."""
        return self.cost_tracker.get_total_cost()
        
    def get_component_cost(self, component: str) -> float:
        """Get cost for a specific component."""
        return self.cost_tracker.get_component_cost(component)
