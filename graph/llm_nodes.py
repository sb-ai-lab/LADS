import os
import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_gigachat.chat_models import GigaChat

from graph.state import AgentState
from graph.prompts import (code_generator_system_prompt, 
                     validate_solution_system_prompt, 
                     validate_solution_user_prompt, 
                     code_improvement_system_prompt,
                     code_improvement_user_prompt,
                     rephraser_system_prompt,
                     rephraser_user_prompt,
                     output_sumarization_system_prompt,
                     output_sumarization_user_prompt,
                     output_result_fillter,
                     lightautoml_router_system_prompt,
                     lightautoml_router_user_prompt,
                     lightautoml_parser_system_prompt,
                     lightautoml_parser_user_prompt,
                     human_explanation_system_prompt,
                     human_explanation_user_prompt
                    )

PYTHON_REGEX = r"```python-execute(.+?)```"

#### Additional Functions
def extract_python_code(text):
    pattern = re.compile(PYTHON_REGEX, re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[0].strip()
    else:
        return None

def find_message_with_code(state: AgentState):
    for i in range(1, len(state['messages'])):
        if re.findall(PYTHON_REGEX, state['messages'][-i].content, re.DOTALL | re.MULTILINE):
            extracted_code = extract_python_code(state['messages'][-i].content)
            break
        else:
            extracted_code = state['messages'][-i].content
            
    return extracted_code


def feedback_for_improvement(state: AgentState, llm_base):
    for i in range(1, len(state['messages'])):
        if "node_name" in state["messages"][-i].additional_kwargs:
            if state["messages"][-i].additional_kwargs['node_name'] == 'CODE_IMPROVEMENT_AGENT':
                state['feedback'].append({f"Улучшение {state['code_improvement_count']}": state["messages"][-i].content})
                break

    last_res = []
    last_res.append(extract_python_code(state['messages'][-4].content))
    last_res.append(state['messages'][-3].content)

    state['feedback'].append({f"Результат {state['code_improvement_count']}": get_result_fillter(last_res, llm_base)})

    return state['feedback']

def get_result_fillter(messege, llm_base):
    user_message = output_result_fillter
    user_message = user_message.format(result=messege)
    llm = llm_base

    response = llm.invoke([{"role": "user", "content": user_message}])
    response = response.content.strip()
    return response

def get_human_explanation(messege, llm_base):
    system_prompt = human_explanation_system_prompt
    user_message = human_explanation_user_prompt
    user_message = user_message.format(text=messege)
    llm = llm_base

    response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
    response = response.content.strip()
    return response



#### Agent 
def rephraser_agent(state: AgentState, llm_base):
    system_prompt = rephraser_system_prompt
    user_message = rephraser_user_prompt
    user_message = user_message.format(task=state['task'], df_head=state['df'].head().to_string(), df_name=state['df_name'])
    llm = llm_base

    response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
    response = response.content.strip()
    message = AIMessage(content=response, additional_kwargs={'node_name': "TASK_REPHRASER"})

    human_respons = get_human_explanation(message.content, llm_base)
    state['human_understanding'].append(human_respons)

    return {"messages": message} 


def code_generation_agent(state: AgentState, llm_base):
    CODE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", code_generator_system_prompt.format(task=state['task'])),
    MessagesPlaceholder("messages")
])
    llm = llm_base
    code_ch = CODE_PROMPT | llm
    response = code_ch.invoke({"messages": state["messages"]})
    return {"messages": response}


def validate_solution(state: AgentState, llm_base):
    system_prompt = validate_solution_system_prompt
    user_message = validate_solution_user_prompt
    user_message = user_message.format(task=state['task'], df_head=state['df'].head().to_string(), solution=state['messages'][-1].content)
    
    code = find_message_with_code(state)
    state['code_for_test'].append({"code": code, "result": state['messages'][-2].content})


    llm = llm_base
    response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
    response = response.content.strip()
    message = AIMessage(content=response, additional_kwargs={'node_name': "TASK_VALIDATOR"})

    human_respons = get_human_explanation(state['messages'][-1].content, llm_base)
    state['human_understanding'].append(human_respons)

    return {"messages": message}


def code_improvement_agent(state: AgentState, llm_base):
    system_prompt = code_improvement_system_prompt
    user_message = code_improvement_user_prompt

    feedback = feedback_for_improvement(state, llm_base)
    code = find_message_with_code(state)
    user_message = user_message.format(task=state['task'], df_head=state['df'].head().to_string(), code = code, solution=state['messages'][-2].content, feedback=feedback)

    llm = llm_base
    response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
    response = response.content.strip()
    message = AIMessage(content=response, additional_kwargs={'node_name': "CODE_IMPROVEMENT_AGENT"})

    human_respons = get_human_explanation(message.content, llm_base)
    state['human_understanding'].append(human_respons)

    state['improvements_code'].append({"improve": message})

    return {"messages": message, "code_improvement_count": state['code_improvement_count']+1}


def output_sumarization(state: AgentState, llm_base):
    system_prompt = output_sumarization_system_prompt
    user_message = output_sumarization_user_prompt
    user_message = user_message.format(task=state['task'], feedback=state["feedback"])

    llm = llm_base

    response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
    response = response.content.strip()
    message = AIMessage(content=response)

    return message


def lightautoml_router(state: AgentState, llm_base):
    system_prompt = lightautoml_router_system_prompt
    user_message = lightautoml_router_user_prompt
    user_message = user_message.format(task=state['task'])

    llm = llm_base

    response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
    return {"messages" : response}

def lightautoml_congig_generator(state: AgentState, llm_base):
    system_prompt = lightautoml_parser_system_prompt
    user_message = lightautoml_parser_user_prompt
    user_message = user_message.format(task=state['task'], file_name=state['df_name'], df_columns=state['df'].columns.tolist(), df_head=state['df'].head().to_string())

    llm = llm_base

    response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
    response = response.content.strip()
    message = AIMessage(content=response)

    return {"messages" : response}

def final(state: AgentState, llm_base):
    message = output_sumarization(state, llm_base)
    return {"messages": message}
