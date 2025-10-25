from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseChatModel
from toolemu.agents import (
    SIMULATORS,
    AgentExecutorWithToolkit,
    ZeroShotAgentWithToolkit,
)
from toolemu.tools import get_toolkits_by_names
from toolemu.utils.my_typing import *


def build_agent_executor(
    toolkits: List[str],
    agent_llm: BaseLanguageModel,
    simulator_llm: BaseLanguageModel,
    critiquer_llm: Optional[BaseLanguageModel] = None,
    num_critique_steps: int = 0,
    max_allowed_steps: int = 3,
    agent_type: str = "naive",
    simulator_type: str = "std_thought",
    verbose: bool = True,
    return_intermediate_steps: bool = True,
    max_iterations: int = 15,
    callback_manager: Optional[BaseCallbackManager] = None,
    track_costs: bool = False,
    cost_tracker: Optional[object] = None,
) -> Type[AgentExecutorWithToolkit]:
    toolkits = get_toolkits_by_names(toolkits)
    agent = ZeroShotAgentWithToolkit.from_llm_and_toolkits(
        toolkits=toolkits,
        llm=agent_llm,
        agent_type=agent_type,
        use_chat_format=isinstance(agent_llm, BaseChatModel),
        cost_tracker=cost_tracker,
    )
    executor_class = SIMULATORS[simulator_type]
    executor_kwargs = dict(
            toolkits=toolkits,
            agent=agent,
            verbose=verbose,
        max_iterations=max_iterations,
        callback_manager=callback_manager,
        track_costs=track_costs,
    )
    if cost_tracker is not None:
        executor_kwargs['cost_tracker'] = cost_tracker
    if simulator_type != "normal":
        executor_kwargs.update(dict(
            llm_simulator=simulator_llm,
            llm_critiquer=critiquer_llm,
            num_critique_steps=num_critique_steps,
            max_allowed_steps=max_allowed_steps,
            use_chat_format=isinstance(simulator_llm, BaseChatModel),
            return_intermediate_steps=return_intermediate_steps,
        ))
    agent_executer = executor_class.from_agent_and_toolkits(**executor_kwargs)
    return agent_executer
