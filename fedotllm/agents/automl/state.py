from fedotllm.agents.automl.structured import TabPFNConfig
from fedotllm.agents.base import FedotLLMAgentState
from fedotllm.enviroments import Observation


class AutoMLAgentState(FedotLLMAgentState):
    reflection: str
    tabpfn_config: TabPFNConfig
    skeleton: str
    raw_code: str | None
    code: str | None
    observation: Observation | None
    fix_attempts: int
    metrics: str
    pipeline: str
