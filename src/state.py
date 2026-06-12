import pandas as pd
from typing import TYPE_CHECKING, Annotated, Any, Sequence, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

if TYPE_CHECKING:
    from e2b_code_interpreter import Sandbox


class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    task: str
    df: Optional[pd.DataFrame]
    df_name: str
    sandbox: Optional[Any]  # e2b_code_interpreter.Sandbox, only used when code_generation_config='e2b'
    code_improvement_count: int
    current_node: str
    feedback: List[str]
    generated_code: str
    rephrased_plan: str
    code_results: str
    test_split: bool
    improvements_code: List[str]
    human_understanding: List[str]
    code_generation_config: str
    train_code: str
    test_code: str
    test_df: Optional[pd.DataFrame]
    test_df_name: str
