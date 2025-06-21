import re

from langgraph.graph import END, StateGraph, START

from graph.state import AgentState
from graph.code_executor_node import execute_code_locally, execute_lightautoml_locally
from graph.llm_nodes import (
    input_node,
    rephraser_agent,
    code_generation_agent,
    validate_solution,
    code_improvement_agent,
    lightautoml_router,
    lightautoml_congig_generator,
    feedback_for_code_improvement_agent,
    human_explanation_agent,
    code_router,
    no_code_agent,
    result_summarization_agent,
    final,
)
from utils.llm_factory import create_llm
from utils.config.loader import load_config

INPUT_NODE = "input_node"
INPUT_AGENT = "rephraser_agent"
CODE_GENERATOR_AGENT = "code_generator_agent"
CODE_EXECUTOR = "code_executor"
TASK_VALIDATOR = "task_validator"
CODE_IMPROVEMENT_AGENT = "code_improvement_agent"
ANSWER_GENERATOR = "answer_generator"
HUMAN_EXPLANATION = "human_explanation"
TASK_VALIDATOR_EXPLANATION = "task_validator_explanation"
CODE_IMPROVEMENT_EXPLANATION = "code_improvement_explanation"
FEEDBACK_FOR_CODE_IMPROVEMENT = "feedback_for_code_improvement_agent"
FEEDBACK_FOR_CODE_RESULTS = "feedback_for_code_results_agent"

LIGHTAUTOML_ROUTER_AGENT = "lightautoml_router"
LIGHTAUTOML_CONFIG_GENERATOR_AGENT = "lightautoml_config_generator"
LIGHTAUTOML_LOCAL_EXECUTOR = "lightautoml_local_executor"

CODE_ROUTER = "code_router"
NO_CODE_AGENT = "no_code_agent"
RESULT_SUMMARIZATION_AGENT =  "result_summarization_agent"
RESULT_EXPLANATION =  "result_explanation"


ERROR_REGEX = r"(?:" + "|".join([
    r"Traceback $$most recent call last$$:",
    r"Error:",
    r"Exception:",
    r"ValueError:",
    r"NameError:",
    r"SyntaxError:",
]) + r")"


def code_generation_retry(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if re.findall(ERROR_REGEX, last_message.content, re.DOTALL | re.MULTILINE):
        return CODE_GENERATOR_AGENT 
    return RESULT_SUMMARIZATION_AGENT


def task_validation_retry(state: AgentState) -> str:
    last_message = state['messages'][-1].content
    if "VALID NO" in last_message:
        return TASK_VALIDATOR_EXPLANATION
    elif "VALID YES" in last_message:
        return FEEDBACK_FOR_CODE_IMPROVEMENT
    return CODE_GENERATOR_AGENT


def check_number_improvements(state: AgentState) -> str:
    if state['code_improvement_count'] >= 5:
        return ANSWER_GENERATOR
    return CODE_GENERATOR_AGENT

def code_router_func(state: AgentState) -> str:
    last_message = state['messages'][-1].content
    if "YES" in last_message:
        return LIGHTAUTOML_ROUTER_AGENT
    else:
        return NO_CODE_AGENT

def lightautoml_router_func(state: AgentState) -> str:
    last_message = state['messages'][-1].content
    if "YES" in last_message:
        return LIGHTAUTOML_CONFIG_GENERATOR_AGENT
    else:
        return INPUT_AGENT


def add_node_name(state: AgentState, node_name: str) -> AgentState:
    state['current_node'] = node_name
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
        LIGHTAUTOML_ROUTER_AGENT: lightautoml_router,
        LIGHTAUTOML_CONFIG_GENERATOR_AGENT: lightautoml_congig_generator,
        INPUT_AGENT: rephraser_agent,
        CODE_GENERATOR_AGENT: code_generation_agent,
        TASK_VALIDATOR: validate_solution,
        CODE_IMPROVEMENT_AGENT: code_improvement_agent,
        HUMAN_EXPLANATION: human_explanation_agent,
        TASK_VALIDATOR_EXPLANATION: human_explanation_agent,
        CODE_IMPROVEMENT_EXPLANATION: human_explanation_agent,
        FEEDBACK_FOR_CODE_IMPROVEMENT: feedback_for_code_improvement_agent,
        FEEDBACK_FOR_CODE_RESULTS: feedback_for_code_improvement_agent,
        CODE_ROUTER: code_router,
        NO_CODE_AGENT: no_code_agent,
        RESULT_SUMMARIZATION_AGENT: result_summarization_agent,
        RESULT_EXPLANATION: human_explanation_agent,
        ANSWER_GENERATOR: final,
    }

    for node_name, node_func in nodes.items():
        workflow.add_node(node_name, lambda x, f=node_func, n=node_name: add_node_name(f(x), n))

    for node_name, node_func in llm_nodes.items():
        workflow.add_node(node_name, lambda x, f=node_func, n=node_name: add_node_name(f(x, create_llm(n, config)), n))

    workflow.add_edge(START, INPUT_NODE)
    workflow.add_edge(INPUT_NODE, CODE_ROUTER)
    workflow.add_conditional_edges(
        CODE_ROUTER,
        code_router_func,
        {LIGHTAUTOML_ROUTER_AGENT: LIGHTAUTOML_ROUTER_AGENT,
         NO_CODE_AGENT: NO_CODE_AGENT} 
    ) 
    workflow.add_edge(NO_CODE_AGENT, END)

    workflow.add_conditional_edges(
        LIGHTAUTOML_ROUTER_AGENT,
        lightautoml_router_func,
        {
            LIGHTAUTOML_CONFIG_GENERATOR_AGENT: LIGHTAUTOML_CONFIG_GENERATOR_AGENT,
            INPUT_AGENT: INPUT_AGENT
        }
    )
    workflow.add_edge(LIGHTAUTOML_CONFIG_GENERATOR_AGENT, LIGHTAUTOML_LOCAL_EXECUTOR)
    workflow.add_edge(LIGHTAUTOML_LOCAL_EXECUTOR, END)

    workflow.add_edge(INPUT_AGENT, HUMAN_EXPLANATION)

    workflow.add_edge(HUMAN_EXPLANATION, CODE_GENERATOR_AGENT)
    workflow.add_edge(CODE_GENERATOR_AGENT, CODE_EXECUTOR)
    workflow.add_conditional_edges(
        CODE_EXECUTOR,
        code_generation_retry,
        {RESULT_SUMMARIZATION_AGENT: RESULT_SUMMARIZATION_AGENT, 
         CODE_GENERATOR_AGENT: CODE_GENERATOR_AGENT}
    )
    
    workflow.add_edge(RESULT_SUMMARIZATION_AGENT, RESULT_EXPLANATION)
    workflow.add_edge(RESULT_EXPLANATION, TASK_VALIDATOR)

    workflow.add_conditional_edges(
        TASK_VALIDATOR,
        task_validation_retry,
        {
            TASK_VALIDATOR_EXPLANATION: TASK_VALIDATOR_EXPLANATION,
            FEEDBACK_FOR_CODE_IMPROVEMENT: FEEDBACK_FOR_CODE_IMPROVEMENT,
            CODE_GENERATOR_AGENT: CODE_GENERATOR_AGENT
        }
    )
    
    workflow.add_edge(TASK_VALIDATOR_EXPLANATION, FEEDBACK_FOR_CODE_RESULTS)
    workflow.add_edge(FEEDBACK_FOR_CODE_RESULTS, ANSWER_GENERATOR)
    workflow.add_edge(FEEDBACK_FOR_CODE_IMPROVEMENT, CODE_IMPROVEMENT_AGENT)
    workflow.add_edge(CODE_IMPROVEMENT_AGENT, CODE_IMPROVEMENT_EXPLANATION)

    workflow.add_conditional_edges(
        CODE_IMPROVEMENT_EXPLANATION,
        check_number_improvements,
        {
            ANSWER_GENERATOR: ANSWER_GENERATOR,
            CODE_GENERATOR_AGENT: CODE_GENERATOR_AGENT
        }
    )

    workflow.add_edge(ANSWER_GENERATOR, END)
    try:
        workflow.compile().get_graph(xray=False).draw_mermaid_png(output_file_path='new_graph.png')
    except Exception:
        # skipping graph picture generation
        print(workflow.compile().get_graph().print_ascii())
        pass
    return workflow.compile()