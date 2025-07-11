import re
import os
import shutil

from graph.state import AgentState
from graph.prompts import load_prompt

from fedotllm.llm import AIInference
from fedotllm.main import FedotAI


PYTHON_REGEX = r"```python-execute(.+?)```"

# Additional Functions


def construct_user_input(state: AgentState) -> str:
    user_input = f"Задача: {state['task']}\n"
    if "df" in state:
        user_input += f"Превью датасета: {state['df'].head().to_string()}\n"
        user_input += f"Колонки, которые есть в датасете: {state['df'].columns}\n"
    if "df_name" in state:
        user_input += f"Название файла с датасетом: {state['df_name']}\n"
    return user_input


def extract_python_code(text):
    matches = re.findall(PYTHON_REGEX, text, re.DOTALL)
    return matches[0].strip() if matches else None


def find_message_with_code(state: AgentState):
    for i in range(1, len(state['messages'])):
        if re.findall(PYTHON_REGEX, state['messages'][-i].content, re.DOTALL | re.MULTILINE):
            extracted_code = extract_python_code(state['messages'][-i].content)
            break
        else:
            extracted_code = state['messages'][-i].content
    return extracted_code



# Agent


def input_node(state: AgentState) -> AgentState:
    state['task'] = state['messages'][-1].content
    default_state = {
        'code_for_test': [],
        'feedback': [],
        'code_improvement_count': 0,
        'improvements_code': [],
        'human_understanding': [],
        'generated_code': "",
        'code_results': "",
        'rephrased_plan': "",
        'lama': False,
        "test_split": False,
        "test_df": None,
        "test_df_name": "",
    }

    for key, value in default_state.items():
        if key not in state:
            state[key] = value

    return state


def rephraser_agent(state: AgentState, llm):

    user_input = construct_user_input(state)
    prompt_template = load_prompt('rephraser')
    chain = prompt_template | llm
    message = chain.invoke({"user_input": user_input})
    message.content = '\n' + message.content
    state['rephrased_plan'] = message.content.strip()
    return {"messages": message}


def code_router(state: AgentState, llm):

    prompt_template = load_prompt('code_router')
    chain = prompt_template | llm
    response = chain.invoke({"task": state['task']})
    return {"messages": response}


def no_code_agent(state: AgentState, llm):

    prompt_template = load_prompt('no_code')
    chain = prompt_template | llm
    user_input = construct_user_input(state)
    response = chain.invoke({"text": user_input, "history": state['messages']})
    response.content = '\n' + response.content
    return {"messages": response}


def result_explanation_agent(state: AgentState, llm):

    prompt_template = load_prompt('result_explanation')
    chain = prompt_template | llm

    last_two_message = [msg.content for msg in state['messages'][-2:]]
    response = chain.invoke({"text": last_two_message})

    response.content = '\n' + response.content
    return {"messages": response}


def result_summarization_agent(state: AgentState, llm):
    prompt_template = load_prompt('result_summarization')
    chain = prompt_template | llm

    last_two_message = [msg.content for msg in state['messages'][-2:]]
    response = chain.invoke({"text": last_two_message})

    response.content = '\n' + response.content
    return {"messages": response}


def automl_router(state: AgentState, llm):

    prompt_template = load_prompt('automl_router')
    chain = prompt_template | llm
    response = chain.invoke({"task": state['task']})
    return {"messages": response}


def lightautoml_generator(state: AgentState, llm):

    prompt_template = load_prompt('lightautoml_parser')
    chain = prompt_template | llm
    response = chain.invoke({
        "task": state['task'],
        "file_name": state['df_name'],
        "df_columns": state['df'].columns.tolist(),
        "df_head": state['df'].head().to_string()
    })
    response.content = '\n' + response.content.strip()
    return {"messages": response, 'lama': True}


def fedot_generator(state: AgentState, llm) -> str:
    
    output_path = os.path.join(os.getcwd(), 'output')
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    task_path = 'datasets/' #+ state['df_name']

    fedot_ai = FedotAI(
        task_path=task_path,
        inference=AIInference(),
        workspace=output_path,
    )
    output = fedot_ai.ainvoke(message=state['task'])

    # Extract the fedotllm agent message for Code interpretation
    fedotllm_message = output['messages'][-1].content if len(output['messages']) > 1 else "No fedotllm message available"

    current_understanding = state.get('human_understanding', [])
    updated_understanding = current_understanding + [f"**FedotLLM Agent Report:**\n{fedotllm_message}"]

    prompt_template = load_prompt('fedot_parser')
    chain = prompt_template | llm
    response = chain.invoke({"results": output['messages'][1].content})
    
    return {"messages": response, "human_understanding": updated_understanding}


def human_explanation_agent(state: AgentState, llm):

    human_prompts = {
        'rephraser_agent': 'human_explanation_planning',
        'task_validator': 'human_explanation_validator',
        'code_improvement_agent': 'human_explanation_improvement',
        'result_summarization_agent': 'human_explanation_results'
    }

    prompt_template = load_prompt(human_prompts.get(state['current_node'], 'human_explanation'))    
    chain = prompt_template | llm

    last_message = state['messages'][-1].content
    response = chain.invoke({"text": last_message, "history": state['messages']})

    explanation_text = response.content.strip()
    current_understanding = state.get('human_understanding', [])
    updated_understanding = current_understanding + [explanation_text]

    return {
        "messages": response,
        "human_understanding": updated_understanding
    }


def code_generation_agent(state: AgentState, llm):

    prompt_template = load_prompt('code_generator')
    chain = prompt_template | llm
    user_input = construct_user_input(state)
    response = chain.invoke({"user_input": user_input, "history": state['messages']})
    response.content = '\n' + response.content
    return {"messages": response}


def validate_solution(state: AgentState, llm):

    user_input = construct_user_input(state)

    prompt_template = load_prompt('validate_solution')
    chain = prompt_template | llm
    solution = "Код:\n```python-execute" + state["generated_code"] + '\n```'
    solution += "Резульат выполнения кода: " + ''.join(state['code_results'])
    
    message = chain.invoke({"user_input": user_input, "solution": solution, "rephrased_plan": state['rephrased_plan']})

    return {"messages": message}


def feedback_for_code_improvement_agent(state: AgentState, llm_base):

    generated_code = state['generated_code'][-1]
    code_result = state['code_results'][-1] if state['code_results'] else "Нет результатов выполнения кода."

    combined_message = f"Сгенерированный код:\n{generated_code}\n\nРезультат выполнения кода:\n{code_result}"

    user_prompt = load_prompt('output_result_filter')
    chain = user_prompt | llm_base
    response = chain.invoke({"result": combined_message})

    past_feedback = state.get('feedback', [])
    if state.get('improvements_code'):
        latest_improvement = state['improvements_code'][-1]
        past_feedback.append({f"Улучшение {state['code_improvement_count']}": latest_improvement["improve"].content})
    res = {f"Результат {state['code_improvement_count']}": response.content}
    past_feedback.append(res)

    return {"feedback": past_feedback, "messages": response}


def code_improvement_agent(state: AgentState, llm):

    prompt_template = load_prompt('code_improvement')
    user_input = construct_user_input(state)
    feedback = state['feedback'][-1] if state['feedback'] else "Нет предыдущих улучшений."
    code = state['generated_code'][-1]

    chain = prompt_template | llm
    message = chain.invoke({"user_input": user_input, "code": code, "solution": state['generated_code'][-1], "feedback": feedback})

    improvements = state['improvements_code']
    improvements.append({"improve": message})

    return {"messages": message, "code_improvement_count": state['code_improvement_count']+1, "improvements_code": improvements}


def train_inference_split(state: AgentState, llm):
    prompt_template = load_prompt('train_inference_split')
    chain = prompt_template | llm
    response = chain.invoke({"code": state['generated_code'], "train_dataset_name": state['df_name'], "test_dataset_name": state['test_df_name']})

    return {"messages": response, "test_split": True}


def check_train_test_inference(state: AgentState, llm):
    last_message = state['messages'][-1].content
    promt_template = load_prompt('train_test_checker')
    chain = promt_template | llm
    response = chain.invoke({"code_result": last_message, "train_code": state['train_code'], "test_code": state['test_code']})

    return {"messages": response}


def final(state: AgentState, llm):
    prompt_message = load_prompt('output_summarization')
    chain = prompt_message | llm
    message = chain.invoke({"task": state['task'], "feedback": state['feedback']})
    message.content = message.content

    #if state['current_node'] not in ['fedot_config_generator', 'lightautoml_local_executor']:
    os.makedirs('./code', exist_ok=True)
    with open('./code/train.py', 'w', encoding='utf-8') as f:
        f.write(state.get('train_code', ''))
    with open('./code/test.py', 'w', encoding='utf-8') as f:
        f.write(state.get('test_code', ''))
    return {"messages": message}