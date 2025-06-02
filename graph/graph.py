import re

from langgraph.graph import END, StateGraph, START

from graph.state import AgentState
from graph.code_executor_node import execute_code, execute_code_locally, execute_lightautoml_locally
from graph.llm_nodes import (
    rephraser_agent,
    code_generation_agent,
    validate_solution,
    code_improvement_agent,
    lightautoml_router,
    lightautoml_congig_generator,
    final,
)
from utils.llm_factory import create_llm
from utils.config.loader import load_config

INPUT_NODE = 'input_node'
INPUT_AGENT = "rephraser_agent"
CODE_GENERATOR_AGENT = "code_generator_agent"
CODE_EXECUTOR = "code_executor"
TASK_VALIDATOR = "task_validator"
CODE_IMPROVEMENT_AGENT = "code_improvement_agent"
ANSWER_GENERATOR = "answer_generator"

LIGHTAUTOML_ROUTER_AGENT = "lightautoml_router"
LIGHTAUTOML_CONFIG_GENERATOR_AGENT = "lightautoml_config_generator"
LIGHTAUTOML_LOCAL_EXECUTOR = "lightautoml_local_executor"

PYTHON_REGEX = r"```python-execute(.+?)```"

def code_generation_retry(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if re.findall(PYTHON_REGEX, last_message.content, re.DOTALL | re.MULTILINE):
        return CODE_EXECUTOR
    return TASK_VALIDATOR

def task_validation_retry(state: AgentState) -> str:
    last_message = state['messages'][-1].content
    if last_message == "VALID NO":
        return ANSWER_GENERATOR
    elif last_message == "VALID YES":
        return CODE_IMPROVEMENT_AGENT
    return CODE_GENERATOR_AGENT

def check_number_improvements(state: AgentState) -> str:
    if state['code_improvement_count'] >= 5:
        return ANSWER_GENERATOR
    return CODE_GENERATOR_AGENT
    
def lightautoml_router_func(state: AgentState) -> str:
    if state['messages'][-1].content == "YES":
        return LIGHTAUTOML_CONFIG_GENERATOR_AGENT
    else:
        if len(state['messages'])>2:
            return CODE_GENERATOR_AGENT
        else:
            return INPUT_AGENT

def add_node_name(state: AgentState, node_name: str) -> AgentState:
    state['current_node'] = node_name
    return state

def input_node(state: AgentState) -> AgentState:
    state['task'] = state['messages'][-1].content
    default_state = {
        'code_for_test': [],
        'feedback': [],
        'code_improvement_count': 0,
        'improvements_code': [],
        'human_understanding': []      
    }
    
    for key, value in default_state.items():
        if key not in state:
            state[key] = value
    
    return state

def graph_builder() -> StateGraph:

    config = load_config()

    workflow = StateGraph(AgentState)
    
    nodes = {
        LIGHTAUTOML_LOCAL_EXECUTOR: execute_lightautoml_locally,
        CODE_EXECUTOR: execute_code_locally,
        INPUT_NODE: input_node,
    }

    llm_nodes = {
        INPUT_AGENT: rephraser_agent,
        CODE_GENERATOR_AGENT: code_generation_agent,
        LIGHTAUTOML_ROUTER_AGENT: lightautoml_router,
        LIGHTAUTOML_CONFIG_GENERATOR_AGENT: lightautoml_congig_generator,
        TASK_VALIDATOR: validate_solution,
        CODE_IMPROVEMENT_AGENT: code_improvement_agent,
        ANSWER_GENERATOR: final
    }
    
    for node_name, node_func in nodes.items():
        workflow.add_node(node_name, lambda x, f=node_func, n=node_name: add_node_name(f(x), n))
    for node_name, node_func in llm_nodes.items():
        workflow.add_node(node_name, lambda x, f=node_func, n=node_name: add_node_name(f(x, create_llm(n, config)), n))

    workflow.add_edge(START, INPUT_NODE)
    workflow.add_edge(INPUT_NODE, LIGHTAUTOML_ROUTER_AGENT)
    
    workflow.add_conditional_edges(
        LIGHTAUTOML_ROUTER_AGENT,
        lightautoml_router_func,
        {
            LIGHTAUTOML_CONFIG_GENERATOR_AGENT: LIGHTAUTOML_CONFIG_GENERATOR_AGENT,
            INPUT_AGENT: INPUT_AGENT,
            CODE_GENERATOR_AGENT: CODE_GENERATOR_AGENT
        }
    )
    workflow.add_edge(LIGHTAUTOML_CONFIG_GENERATOR_AGENT, LIGHTAUTOML_LOCAL_EXECUTOR)
    workflow.add_edge(LIGHTAUTOML_LOCAL_EXECUTOR, END)
    workflow.add_edge(INPUT_AGENT, CODE_GENERATOR_AGENT)
    workflow.add_conditional_edges(
        CODE_GENERATOR_AGENT,
        code_generation_retry,
        {CODE_EXECUTOR: CODE_EXECUTOR, TASK_VALIDATOR: TASK_VALIDATOR}
    )
    workflow.add_edge(CODE_EXECUTOR, CODE_GENERATOR_AGENT)
    workflow.add_conditional_edges(
        TASK_VALIDATOR,
        task_validation_retry,
        {ANSWER_GENERATOR: ANSWER_GENERATOR, CODE_IMPROVEMENT_AGENT: CODE_IMPROVEMENT_AGENT, CODE_GENERATOR_AGENT: CODE_GENERATOR_AGENT}
    )
    workflow.add_conditional_edges(
        CODE_IMPROVEMENT_AGENT,
        check_number_improvements,
        {ANSWER_GENERATOR: ANSWER_GENERATOR, CODE_GENERATOR_AGENT: CODE_GENERATOR_AGENT}
    )
    workflow.add_edge(ANSWER_GENERATOR, END)
    try:
        workflow.compile().get_graph(xray=False).draw_mermaid_png(output_file_path='new_graph.png')
    except Exception:
        pass  # skipping graph picture generation
    return workflow.compile()