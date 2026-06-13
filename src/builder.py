import re

from langgraph.graph import END, StateGraph, START

from src.state import AgentState
from src.executor import execute_code, execute_train_test, execute_autogluon
from src.nodes import (
    input_node,
    rephraser_agent,
    code_generation_agent,
    validate_solution,
    code_improvement_agent,
    automl_router,
    autogluon_config_generator,
    feedback_for_code_improvement_agent,
    human_explanation_agent,
    train_inference_split,
    check_train_test_inference,
    code_router,
    no_code_agent,
    result_summarization_agent,
    final,
    # interview workflow
    scope_declaration,
    interview_planner,
    interview_question_asker,
    interview_response_handler,
    spec_generator,
    # quality judgment
    model_quality_judge,
    # persistence
    experiment_saver,
)
from src.llm_factory import create_llm
from src.config import load_config

# ── Node name constants ────────────────────────────────────────────────────────

INPUT_NODE = "input_node"
INPUT_AGENT = "rephraser_agent"
CODE_GENERATOR_AGENT = "code_generator_agent"
CODE_EXECUTOR = "code_executor"
TASK_VALIDATOR = "task_validator"
CODE_IMPROVEMENT_AGENT = "code_improvement_agent"
ANSWER_GENERATOR = "answer_generator"
HUMAN_EXPLANATION = "human_explanation_planning"
TASK_VALIDATOR_EXPLANATION = "human_explanation_validator"
CODE_IMPROVEMENT_EXPLANATION = "human_explanation_improvement"
RESULT_EXPLANATION = "human_explanation_results"
FEEDBACK_FOR_CODE_IMPROVEMENT = "feedback_for_code_improvement_agent"
TRAIN_INFERENCE_SPLITTER = "train_inference_splitter"
CHECK_TRAIN_TEST_INFERENCE = "check_train_test_inference"
EXECUTE_TRAIN_TEST = "execute_train_test"

AUTOML_ROUTER_AGENT = "automl_router"
AUTOGLUON_CONFIG_GENERATOR_AGENT = "autogluon_config_generator"
AUTOGLUON_EXECUTOR = "autogluon_executor"
MODEL_QUALITY_JUDGE = "model_quality_judge"

CODE_ROUTER = "code_router"
NO_CODE_AGENT = "no_code_agent"
RESULT_SUMMARIZATION_AGENT = "result_summarization_agent"

SCOPE_DECLARATION = "scope_declaration"
INTERVIEW_PLANNER = "interview_planner"
INTERVIEW_QUESTION_ASKER = "interview_question_asker"
INTERVIEW_RESPONSE_HANDLER = "interview_response_handler"
SPEC_GENERATOR = "spec_generator"

EXPERIMENT_SAVER = "experiment_saver"


ERROR_REGEX = r"(?:" + "|".join([
    r"Traceback $$most recent call last$$:",
    r"Error:",
    r"Exception:",
    r"ValueError:",
    r"NameError:",
    r"SyntaxError:",
]) + r")"


# ── Routing functions ──────────────────────────────────────────────────────────


def interview_check_func(state: AgentState) -> str:
    """Route from input_node: bypass code_router if interview is active."""
    if state.get('interview_active', False):
        return INTERVIEW_RESPONSE_HANDLER
    return CODE_ROUTER


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
    config = load_config()
    if state['code_improvement_count'] >= config.general.max_improvements:
        return TRAIN_INFERENCE_SPLITTER
    return CODE_GENERATOR_AGENT


def code_router_func(state: AgentState) -> str:
    last_message = state['messages'][-1].content
    if "YES" in last_message:
        return AUTOML_ROUTER_AGENT
    return NO_CODE_AGENT


def automl_router_func(state: AgentState) -> str:
    last_message = state['messages'][-1].content.strip().upper()
    if "AUTOGLUON" in last_message:
        return AUTOGLUON_CONFIG_GENERATOR_AGENT
    return SCOPE_DECLARATION


def quality_router_func(state: AgentState) -> str:
    verdict = state.get('model_verdict', 'USABLE')
    if verdict == 'IMPROVABLE':
        return INPUT_AGENT  # enter LLM codegen improvement path
    return EXPERIMENT_SAVER


def interview_loop_func(state: AgentState) -> str:
    idx = state.get('interview_current_idx', 0)
    questions = state.get('interview_questions', [])
    if idx < len(questions):
        return INTERVIEW_QUESTION_ASKER
    return SPEC_GENERATOR


def train_inference_router(state: AgentState) -> str:
    last_message = state['messages'][-1].content
    if "VALID" in last_message:
        return ANSWER_GENERATOR
    return EXECUTE_TRAIN_TEST


def add_node_name(state: AgentState, node_name: str) -> AgentState:
    state['current_node'] = node_name
    return state


# ── Graph builder ──────────────────────────────────────────────────────────────


def graph_builder() -> StateGraph:
    config = load_config()

    workflow = StateGraph(AgentState)

    # Pure Python nodes (no LLM)
    nodes = {
        AUTOGLUON_EXECUTOR: execute_autogluon,
        CODE_EXECUTOR: execute_code,
        EXECUTE_TRAIN_TEST: execute_train_test,
        INPUT_NODE: input_node,
        SCOPE_DECLARATION: scope_declaration,
        INTERVIEW_QUESTION_ASKER: interview_question_asker,
        EXPERIMENT_SAVER: experiment_saver,
    }

    # LLM nodes
    llm_nodes = {
        AUTOML_ROUTER_AGENT: automl_router,
        AUTOGLUON_CONFIG_GENERATOR_AGENT: autogluon_config_generator,
        MODEL_QUALITY_JUDGE: model_quality_judge,
        INPUT_AGENT: rephraser_agent,
        CODE_GENERATOR_AGENT: code_generation_agent,
        TASK_VALIDATOR: validate_solution,
        CODE_IMPROVEMENT_AGENT: code_improvement_agent,
        HUMAN_EXPLANATION: human_explanation_agent,
        TASK_VALIDATOR_EXPLANATION: human_explanation_agent,
        CODE_IMPROVEMENT_EXPLANATION: human_explanation_agent,
        FEEDBACK_FOR_CODE_IMPROVEMENT: feedback_for_code_improvement_agent,
        TRAIN_INFERENCE_SPLITTER: train_inference_split,
        CHECK_TRAIN_TEST_INFERENCE: check_train_test_inference,
        CODE_ROUTER: code_router,
        NO_CODE_AGENT: no_code_agent,
        RESULT_SUMMARIZATION_AGENT: result_summarization_agent,
        RESULT_EXPLANATION: human_explanation_agent,
        ANSWER_GENERATOR: final,
        INTERVIEW_PLANNER: interview_planner,
        INTERVIEW_RESPONSE_HANDLER: interview_response_handler,
        SPEC_GENERATOR: spec_generator,
    }

    for node_name, node_func in nodes.items():
        workflow.add_node(node_name, lambda x, f=node_func, n=node_name: add_node_name(f(x), n))

    for node_name, node_func in llm_nodes.items():
        workflow.add_node(node_name, lambda x, f=node_func, n=node_name: add_node_name(f(x, create_llm(n, config)), n))

    # ── START → input_node ────────────────────────────────────────────────────
    workflow.add_edge(START, INPUT_NODE)

    # ── input_node → [interview check] ───────────────────────────────────────
    workflow.add_conditional_edges(
        INPUT_NODE,
        interview_check_func,
        {CODE_ROUTER: CODE_ROUTER, INTERVIEW_RESPONSE_HANDLER: INTERVIEW_RESPONSE_HANDLER}
    )

    # ── Interview response loop ───────────────────────────────────────────────
    workflow.add_conditional_edges(
        INTERVIEW_RESPONSE_HANDLER,
        interview_loop_func,
        {INTERVIEW_QUESTION_ASKER: INTERVIEW_QUESTION_ASKER, SPEC_GENERATOR: SPEC_GENERATOR}
    )
    workflow.add_edge(INTERVIEW_QUESTION_ASKER, END)  # pause for user input

    # ── Spec generator → LLM codegen path ────────────────────────────────────
    workflow.add_edge(SPEC_GENERATOR, INPUT_AGENT)

    # ── code_router ───────────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        CODE_ROUTER,
        code_router_func,
        {AUTOML_ROUTER_AGENT: AUTOML_ROUTER_AGENT, NO_CODE_AGENT: NO_CODE_AGENT}
    )
    workflow.add_edge(NO_CODE_AGENT, END)

    # ── AutoML routing ────────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        AUTOML_ROUTER_AGENT,
        automl_router_func,
        {
            AUTOGLUON_CONFIG_GENERATOR_AGENT: AUTOGLUON_CONFIG_GENERATOR_AGENT,
            SCOPE_DECLARATION: SCOPE_DECLARATION,
        }
    )

    # ── AutoGluon path ────────────────────────────────────────────────────────
    workflow.add_edge(AUTOGLUON_CONFIG_GENERATOR_AGENT, AUTOGLUON_EXECUTOR)
    workflow.add_edge(AUTOGLUON_EXECUTOR, MODEL_QUALITY_JUDGE)
    workflow.add_conditional_edges(
        MODEL_QUALITY_JUDGE,
        quality_router_func,
        {INPUT_AGENT: INPUT_AGENT, EXPERIMENT_SAVER: EXPERIMENT_SAVER}
    )

    # ── Interview path (from code_router CODEGEN branch) ─────────────────────
    workflow.add_edge(SCOPE_DECLARATION, INTERVIEW_PLANNER)
    workflow.add_conditional_edges(
        INTERVIEW_PLANNER,
        interview_loop_func,
        {INTERVIEW_QUESTION_ASKER: INTERVIEW_QUESTION_ASKER, SPEC_GENERATOR: SPEC_GENERATOR}
    )

    # ── LLM codegen path ──────────────────────────────────────────────────────
    workflow.add_edge(INPUT_AGENT, HUMAN_EXPLANATION)
    workflow.add_edge(HUMAN_EXPLANATION, CODE_GENERATOR_AGENT)
    workflow.add_edge(CODE_GENERATOR_AGENT, CODE_EXECUTOR)
    workflow.add_conditional_edges(
        CODE_EXECUTOR,
        code_generation_retry,
        {RESULT_SUMMARIZATION_AGENT: RESULT_SUMMARIZATION_AGENT, CODE_GENERATOR_AGENT: CODE_GENERATOR_AGENT}
    )

    workflow.add_edge(RESULT_SUMMARIZATION_AGENT, RESULT_EXPLANATION)
    workflow.add_edge(RESULT_EXPLANATION, TASK_VALIDATOR)

    workflow.add_conditional_edges(
        TASK_VALIDATOR,
        task_validation_retry,
        {
            TASK_VALIDATOR_EXPLANATION: TASK_VALIDATOR_EXPLANATION,
            FEEDBACK_FOR_CODE_IMPROVEMENT: FEEDBACK_FOR_CODE_IMPROVEMENT,
            CODE_GENERATOR_AGENT: CODE_GENERATOR_AGENT,
        }
    )

    workflow.add_edge(TASK_VALIDATOR_EXPLANATION, TRAIN_INFERENCE_SPLITTER)
    workflow.add_edge(FEEDBACK_FOR_CODE_IMPROVEMENT, CODE_IMPROVEMENT_AGENT)
    workflow.add_edge(CODE_IMPROVEMENT_AGENT, CODE_IMPROVEMENT_EXPLANATION)

    workflow.add_conditional_edges(
        CODE_IMPROVEMENT_EXPLANATION,
        check_number_improvements,
        {
            TRAIN_INFERENCE_SPLITTER: TRAIN_INFERENCE_SPLITTER,
            CODE_GENERATOR_AGENT: CODE_GENERATOR_AGENT,
        }
    )

    workflow.add_edge(TRAIN_INFERENCE_SPLITTER, EXECUTE_TRAIN_TEST)
    workflow.add_edge(EXECUTE_TRAIN_TEST, CHECK_TRAIN_TEST_INFERENCE)
    workflow.add_conditional_edges(
        CHECK_TRAIN_TEST_INFERENCE,
        train_inference_router,
        {ANSWER_GENERATOR: ANSWER_GENERATOR, EXECUTE_TRAIN_TEST: EXECUTE_TRAIN_TEST}
    )

    workflow.add_edge(ANSWER_GENERATOR, EXPERIMENT_SAVER)
    workflow.add_edge(EXPERIMENT_SAVER, END)

    try:
        workflow.compile().get_graph(xray=False).draw_mermaid_png(output_file_path='new_graph.png')
    except Exception:
        try:
            print(workflow.compile().get_graph().print_ascii())
        except Exception:
            pass
    return workflow.compile()
