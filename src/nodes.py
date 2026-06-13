import re
import os
import json
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage

from src.state import AgentState
from src.prompts import load_prompt


PYTHON_REGEX = r"```python-execute(.+?)```"

# Additional Functions


def construct_user_input(state: AgentState) -> str:
    user_input = f"Task: {state['task']}\n"
    if "df" in state:
        user_input += f"Dataset preview:\n{state['df'].head().to_string()}\n"
        user_input += f"Dataset columns: {list(state['df'].columns)}\n"
    if "df_name" in state:
        user_input += f"Dataset filename: {state['df_name']}\n"
    return user_input


def extract_python_code(text):
    matches = re.findall(PYTHON_REGEX, text, re.DOTALL)
    return matches[0].strip() if matches else None


# ── Core nodes ────────────────────────────────────────────────────────────────


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
        "test_split": False,
        "test_df": None,
        "test_df_name": "",
        # interview fields
        "interview_active": False,
        "interview_questions": [],
        "interview_current_idx": 0,
        "modeling_spec": "",
        "bioprocess_type": "custom",
        "model_verdict": "",
        "experiment_id": "",
    }

    for key, value in default_state.items():
        if key not in state:
            state[key] = value

    return state


def rephraser_agent(state: AgentState, llm):
    user_input = construct_user_input(state)
    # Inject modeling spec context if available
    if state.get('modeling_spec'):
        user_input += f"\nModeling specification:\n{state['modeling_spec']}\n"
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


def result_summarization_agent(state: AgentState, llm):
    prompt_template = load_prompt('result_summarization')
    chain = prompt_template | llm
    last_two_message = [msg.content for msg in state['messages'][-2:]]
    response = chain.invoke({"text": last_two_message})
    response.content = '\n' + response.content
    return {"messages": response}


def automl_router(state: AgentState, llm):
    df = state.get('df')
    if df is not None:
        shape_str = f"{df.shape[0]} 行 × {df.shape[1]} 列"
        dtypes_str = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
    else:
        shape_str = "未上传数据"
        dtypes_str = "N/A"

    prompt_template = load_prompt('automl_router')
    chain = prompt_template | llm
    response = chain.invoke({
        "task": state['task'],
        "file_name": state.get('df_name') or '未上传',
        "shape": shape_str,
        "dtypes": dtypes_str,
    })
    return {"messages": response}


def autogluon_config_generator(state: AgentState, llm):
    prompt_template = load_prompt('autogluon_config')
    chain = prompt_template | llm
    response = chain.invoke({
        "task": state['task'],
        "file_name": state.get('df_name', 'unknown'),
        "df_columns": list(state['df'].columns) if state.get('df') is not None else [],
        "df_head": state['df'].head().to_string() if state.get('df') is not None else "No data",
    })
    response.content = '\n' + response.content.strip()
    return {"messages": response}


def human_explanation_agent(state: AgentState, llm):
    human_prompts = {
        'rephraser_agent': 'human_explanation_planning',
        'task_validator': 'human_explanation_validator',
        'code_improvement_agent': 'human_explanation_improvement',
        'result_summarization_agent': 'human_explanation_results',
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
        "human_understanding": updated_understanding,
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
    solution = "Code:\n```python-execute" + state["generated_code"] + '\n```'
    solution += "Code execution result: " + ''.join(state['code_results'])

    message = chain.invoke({"user_input": user_input, "solution": solution, "rephrased_plan": state['rephrased_plan']})
    return {"messages": message}


def feedback_for_code_improvement_agent(state: AgentState, llm_base):
    generated_code = state['generated_code'][-1]
    code_result = state['code_results'][-1] if state['code_results'] else "No code execution results available."

    combined_message = f"Generated code:\n{generated_code}\n\nCode execution result:\n{code_result}"

    user_prompt = load_prompt('output_result_filter')
    chain = user_prompt | llm_base
    response = chain.invoke({"result": combined_message})

    past_feedback = state.get('feedback', [])
    if state.get('improvements_code'):
        latest_improvement = state['improvements_code'][-1]
        past_feedback.append({f"Improvement {state['code_improvement_count']}": latest_improvement["improve"].content})
    res = {f"Result {state['code_improvement_count']}": response.content}
    past_feedback.append(res)

    return {"feedback": past_feedback, "messages": response}


def code_improvement_agent(state: AgentState, llm):
    prompt_template = load_prompt('code_improvement')
    user_input = construct_user_input(state)
    feedback = state['feedback'][-1] if state['feedback'] else "No previous improvements."
    code = state['generated_code'][-1]

    chain = prompt_template | llm
    message = chain.invoke({"user_input": user_input, "code": code, "solution": state['generated_code'][-1], "feedback": feedback})

    improvements = state['improvements_code']
    improvements.append({"improve": message})

    return {"messages": message, "code_improvement_count": state['code_improvement_count'] + 1, "improvements_code": improvements}


def train_inference_split(state: AgentState, llm):
    prompt_template = load_prompt('train_inference_split')
    chain = prompt_template | llm
    response = chain.invoke({"code": state['generated_code'], "train_dataset_name": state['df_name'], "test_dataset_name": state['test_df_name']})
    return {"messages": response, "test_split": True}


def check_train_test_inference(state: AgentState, llm):
    last_message = state['messages'][-1].content
    prompt_template = load_prompt('train_test_checker')
    chain = prompt_template | llm
    response = chain.invoke({"code_result": last_message, "train_code": state['train_code'], "test_code": state['test_code']})
    return {"messages": response}


def final(state: AgentState, llm):
    prompt_message = load_prompt('output_summarization')
    chain = prompt_message | llm
    message = chain.invoke({"task": state['task'], "base": state['human_understanding'][1], "feedback": state['feedback']})

    os.makedirs('./code', exist_ok=True)
    with open('./code/train.py', 'w', encoding='utf-8') as f:
        f.write(state.get('train_code', ''))
    with open('./code/test.py', 'w', encoding='utf-8') as f:
        f.write(state.get('test_code', ''))
    return {"messages": message}


# ── Interview workflow nodes ──────────────────────────────────────────────────


def scope_declaration(state: AgentState) -> dict:
    msg = (
        "**在开始之前，说明一下这个工具能帮您做什么**\n\n"
        "✅ **能做的：**\n"
        "- 根据您已有的实验数据，找出规律，预测新样本的检测结果\n"
        "- 评估预测结果的可信度，告诉您结果有多可靠\n"
        "- 导出预测结果文件（CSV），供您直接使用\n\n"
        "❌ **目前暂不支持：**\n"
        "- 实验方案设计（DoE）\n"
        "- 工艺机理分析\n"
        "- 实时数据接入\n\n"
        "接下来，我需要向您了解几个关键信息来确定分析方案，请逐一回答。"
    )
    return {"messages": AIMessage(content=msg)}


def interview_planner(state: AgentState, llm):
    df = state.get('df')
    if df is not None:
        shape_str = f"{df.shape[0]} 行 × {df.shape[1]} 列"
        dtypes_str = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
        df_head = df.head(3).to_string()
    else:
        shape_str = "未上传"
        dtypes_str = "N/A"
        df_head = "N/A"

    prompt_template = load_prompt('interview_planner')
    chain = prompt_template | llm
    response = chain.invoke({
        "task": state['task'],
        "file_name": state.get('df_name') or '未上传',
        "shape": shape_str,
        "dtypes": dtypes_str,
        "df_head": df_head,
    })

    # Parse the JSON question list from the response
    raw = response.content.strip()
    # Extract JSON array from response (may be wrapped in ```json ... ```)
    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    questions = []
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            questions = [{"question": q["question"], "options": q["options"], "answer": ""} for q in parsed]
        except (json.JSONDecodeError, KeyError):
            questions = [{"question": raw, "options": ["请帮我推荐", "其他（请描述）"], "answer": ""}]

    if not questions:
        questions = [{"question": "您希望预测哪个指标？请从数据集列名中选择，或描述您的目标。", "options": ["请帮我推荐", "其他（请描述）"], "answer": ""}]

    return {
        "messages": response,
        "interview_questions": questions,
        "interview_current_idx": 0,
        "interview_active": True,
    }


def interview_question_asker(state: AgentState) -> dict:
    questions = state.get('interview_questions', [])
    idx = state.get('interview_current_idx', 0)

    if idx >= len(questions):
        msg = "所有问题已回答完毕，正在生成建模方案..."
        return {"messages": AIMessage(content=msg)}

    q = questions[idx]
    options_text = "\n".join(f"  {i + 1}. {opt}" for i, opt in enumerate(q["options"]))
    total = len(questions)
    msg = (
        f"**问题 {idx + 1} / {total}**\n\n"
        f"{q['question']}\n\n"
        f"选项：\n{options_text}\n\n"
        f"_请输入选项编号或直接输入您的回答。_"
    )
    return {"messages": AIMessage(content=msg)}


def interview_response_handler(state: AgentState, llm) -> dict:
    questions = state.get('interview_questions', [])
    idx = state.get('interview_current_idx', 0)

    if idx >= len(questions):
        return {"interview_active": False}

    # Latest user message is the answer
    user_answer = state['task']  # input_node sets task = latest message content

    q = questions[idx]
    options = q.get('options', [])

    # Check if user picked "请帮我推荐"
    wants_recommendation = (
        "推荐" in user_answer or
        (user_answer.strip().isdigit() and int(user_answer.strip()) - 1 < len(options)
         and "推荐" in options[int(user_answer.strip()) - 1])
    )

    if wants_recommendation:
        # LLM generates a recommendation
        recommend_prompt = (
            f"你是一位数据科学专家。对于以下问题，请给出一个专业推荐并简要说明理由（1~2句话）。\n\n"
            f"问题：{q['question']}\n"
            f"数据集列名：{list(state['df'].columns) if state.get('df') is not None else '未知'}\n"
            f"建模目标：{state.get('task', '')}"
        )
        from langchain_core.messages import HumanMessage
        rec_response = llm.invoke([HumanMessage(content=recommend_prompt)])
        answer = f"[推荐] {rec_response.content.strip()}"
        reply_msg = AIMessage(content=f"**推荐**：{rec_response.content.strip()}\n\n_已记录此推荐作为您的回答。_")
    else:
        answer = user_answer
        reply_msg = AIMessage(content=f"✓ 已记录：{answer}")

    # Update the question with the answer
    updated_questions = list(questions)
    updated_questions[idx] = {**updated_questions[idx], "answer": answer}
    new_idx = idx + 1
    still_active = new_idx < len(updated_questions)

    return {
        "messages": reply_msg,
        "interview_questions": updated_questions,
        "interview_current_idx": new_idx,
        "interview_active": still_active,
    }


def spec_generator(state: AgentState, llm) -> dict:
    questions = state.get('interview_questions', [])
    qa_pairs = "\n".join(
        f"Q{i + 1}: {q['question']}\nA: {q.get('answer', '未回答')}"
        for i, q in enumerate(questions)
    )

    df = state.get('df')
    df_columns = list(df.columns) if df is not None else []

    prompt_template = load_prompt('spec_generator')
    chain = prompt_template | llm
    response = chain.invoke({
        "task": state['task'],
        "df_columns": df_columns,
        "qa_pairs": qa_pairs,
    })

    spec_text = response.content.strip()
    # Try to extract JSON from response
    json_match = re.search(r'\{.*\}', spec_text, re.DOTALL)
    if json_match:
        spec_json = json_match.group()
    else:
        spec_json = spec_text

    # Detect bioprocess type from spec or task
    task_lower = state.get('task', '').lower()
    spec_lower = spec_json.lower()
    if 'virus' in task_lower or 'lrv' in spec_lower or 'clearance' in task_lower or '清除' in task_lower:
        bioprocess_type = 'virus_clearance'
    elif 'aggregat' in task_lower or '聚集' in task_lower or '聚合' in task_lower:
        bioprocess_type = 'aggregation'
    else:
        bioprocess_type = 'custom'

    summary_msg = (
        f"✅ **建模规格已确定**\n\n"
        f"```json\n{spec_json}\n```\n\n"
        f"正在根据规格启动数据分析..."
    )

    return {
        "messages": AIMessage(content=summary_msg),
        "modeling_spec": spec_json,
        "bioprocess_type": bioprocess_type,
        "interview_active": False,
    }


# ── Quality judgment node ─────────────────────────────────────────────────────


def model_quality_judge(state: AgentState, llm) -> dict:
    df = state.get('df')
    shape_str = f"{df.shape[0]} 行 × {df.shape[1]} 列" if df is not None else "未知"

    # Try to extract target/task_type from modeling_spec
    target = "未知目标"
    task_type = "未知"
    spec_str = state.get('modeling_spec', '')
    if spec_str:
        try:
            spec = json.loads(spec_str)
            target = spec.get('target', target)
            task_type = spec.get('task_type', task_type)
        except (json.JSONDecodeError, TypeError):
            pass

    prompt_template = load_prompt('model_quality_judge')
    chain = prompt_template | llm
    response = chain.invoke({
        "shape": shape_str,
        "target": target,
        "task_type": task_type,
        "code_results": state.get('code_results', '暂无结果'),
        "modeling_spec": spec_str or '未生成规格',
    })

    content = response.content.strip()

    # Extract verdict
    verdict = "USABLE"
    verdict_match = re.search(r"VERDICT:\s*(USABLE|IMPROVABLE|INSUFFICIENT_DATA)", content, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
        # Remove the VERDICT line from display content
        display_content = content[:verdict_match.start()].strip()
    else:
        display_content = content

    display_msg = (
        f"**📊 数据科学评估报告**\n\n{display_content}\n\n"
        f"**结论：** {'✅ 可以使用' if verdict == 'USABLE' else ('⚠️ 建议改进' if verdict == 'IMPROVABLE' else '❌ 数据不足')}"
    )

    return {
        "messages": AIMessage(content=display_msg),
        "model_verdict": verdict,
    }


# ── Experiment persistence node ───────────────────────────────────────────────


def experiment_saver(state: AgentState) -> dict:
    exp_id = state.get('experiment_id') or str(uuid.uuid4())[:8]

    # Determine persistence path from config
    try:
        from src.config import load_config
        cfg = load_config()
        exp_path = Path(cfg.persistence.path) if cfg.persistence else Path("./experiments")
    except Exception:
        exp_path = Path("./experiments")

    exp_path.mkdir(parents=True, exist_ok=True)

    df = state.get('df')
    record = {
        "experiment_id": exp_id,
        "created_at": datetime.now().isoformat(),
        "task": state.get('task', ''),
        "bioprocess_type": state.get('bioprocess_type', 'custom'),
        "modeling_spec": state.get('modeling_spec', ''),
        "interview_log": [
            {"question": q["question"], "answer": q.get("answer", "")}
            for q in state.get('interview_questions', [])
        ],
        "generated_code": state.get('generated_code', ''),
        "code_results": state.get('code_results', ''),
        "model_verdict": state.get('model_verdict', ''),
        "dataset_name": state.get('df_name', ''),
        "dataset_shape": list(df.shape) if df is not None else [],
    }

    file_path = exp_path / f"{exp_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return {"experiment_id": exp_id}
