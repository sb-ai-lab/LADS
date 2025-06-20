import pandas as pd
from typing import Annotated, Sequence, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from e2b_code_interpreter import Sandbox


class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    task: str
    df: Optional[pd.DataFrame]
    df_train: Optional[pd.DataFrame]
    df_test: Optional[pd.DataFrame]
    df_name: str
    sandbox: Sandbox
    code_improvement_count: int
    current_node: str
    feedback: List[str]
    generated_code: List[str]
    rephrased_plan: str
    code_results: List[str]
    improvements_code: List[str]
    human_understanding: List[str]
