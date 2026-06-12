import os
import re
import sys
import json
import tempfile
import subprocess

from graph.state import AgentState
from langchain_core.messages import AIMessage

PYTHON_REGEX = r"```python-execute(.+?)```"
JSON_REGEX = r"```json(.+?)```"

matplotlib_setup = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
"""

local_exec_result = """Code execution result:
```
{process_stdout}
```"""

local_exec_error = """Code execution failed with error:
```
{process_stderr}
```
Please fix the error."""

timeout = 600

e2b_exec_error = """Code execution failed with error:
```
{execution_error_traceback}
```
Please fix the error."""

e2b_exec_result = """Code execution result:
```
{logs}
{text_results}
```"""


def execute_e2b_code(sandbox, code: str) -> str:
    result = ''

    execution = sandbox.run_code(code)

    if execution.error:
        result = e2b_exec_error.format(
            execution_error_traceback=execution.error.traceback
        )
    else:
        logs = '\n'.join(execution.logs.stdout)
        text_results = "\n".join([result.text for result in execution.results if result.text])
        result_text = e2b_exec_result.format(
            logs=logs,
            text_results=text_results
        )
        result = result_text.strip()

    return result


def execute_code_locally(code: str) -> str:
    result = ''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(code)
        temp_file.flush()

        try:
            process = subprocess.run(
                [sys.executable, temp_file.name],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if process.returncode == 0:
                result = local_exec_result.format(process_stdout=process.stdout)
            else:
                result = local_exec_error.format(process_stderr=process.stderr)

        except subprocess.TimeoutExpired:
            result = f"Code execution timed out after {timeout} seconds."
        finally:
            os.unlink(temp_file.name)

    return result


def execute_train_test(state: AgentState):
    messages = state['messages']
    last_content = messages[-1].content

    code_blocks = re.findall(PYTHON_REGEX, last_content, re.DOTALL | re.MULTILINE)
    train_code = code_blocks[0].strip() if len(code_blocks) > 0 else ""
    test_code = code_blocks[1].strip() if len(code_blocks) > 1 else ""

    result_train = execute_code_locally(train_code)
    result_test = execute_code_locally(test_code)

    result = AIMessage(content=f"Training code results:\n{result_train}\n\nTest inference results:\n{result_test}")
    return {"messages": result, "train_code": train_code, "test_code": test_code}


def execute_autogluon(state: AgentState) -> dict:
    json_match = re.findall(JSON_REGEX, state['messages'][-1].content, re.DOTALL)
    if not json_match:
        msg = "AutoGluon config not found in the previous message. Please specify target column and task type."
        return {"messages": AIMessage(content=msg)}

    try:
        config = json.loads(json_match[0])
        target = config['target']
        task_type = config.get('task_type', 'binary')
        metric = config.get('metric', 'roc_auc')

        df = state.get('df')
        if df is None:
            return {"messages": AIMessage(content="No dataset loaded. Please upload a dataset first.")}

        from autogluon.tabular import TabularPredictor

        predictor = TabularPredictor(
            label=target,
            problem_type=task_type,
            eval_metric=metric,
        ).fit(df, time_limit=120)

        leaderboard = predictor.leaderboard(silent=True)
        best_score = leaderboard.iloc[0]['score_val']
        best_model = leaderboard.iloc[0]['model']

        report = (
            f"**AutoGluon training complete**\n\n"
            f"- **Target**: `{target}`\n"
            f"- **Task**: {task_type}\n"
            f"- **Metric ({metric})**: {best_score:.4f}\n"
            f"- **Best model**: {best_model}\n\n"
            f"**Leaderboard (top 5)**:\n```\n{leaderboard.head().to_string()}\n```"
        )

        return {"messages": AIMessage(content=report), "code_results": report}

    except ImportError:
        return {"messages": AIMessage(
            content="AutoGluon is not installed. Run `pip install autogluon.tabular` or use the LLM code generation path."
        )}
    except Exception as e:
        return {"messages": AIMessage(content=f"AutoGluon execution failed: {str(e)}")}


def execute_code(state: AgentState):
    messages = state['messages']
    code_blocks = re.findall(PYTHON_REGEX, messages[-1].content, re.DOTALL | re.MULTILINE)
    code_to_execute = "\n".join(code_blocks)
    full_code = matplotlib_setup + code_to_execute + "\nplt.close('all')"
    execution_location = state['code_generation_config']

    if execution_location == 'e2b':
        sandbox = state['sandbox']
        result = execute_e2b_code(sandbox, full_code)
    else:
        result = execute_code_locally(full_code)

    return {"messages": AIMessage(content=result), 'generated_code': code_to_execute, 'code_results': result, 'test_split': False}
