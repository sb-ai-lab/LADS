import os
import re
import sys
import json
import tempfile
import subprocess

from graph.state import AgentState
from langchain_core.messages import AIMessage

lightautoml_template = 'graph/lightautoml_template.py'

PYTHON_REGEX = r"```python-execute(.+?)```"
JSON_REGEX = r"```json(.+?)```"

lightautoml_error = """Code execution of {lightautoml_template} failed with error:
```
{process_err}
```
Please fix the error.
"""

lightautoml_result = """Code execution result of {lightautoml_template}:
```
{process_stdout}
```
"""

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


def execute_lightautoml_locally(state: AgentState):
    messages = state['messages']
    json_block = re.findall(JSON_REGEX, messages[-1].content, re.DOTALL | re.MULTILINE)[0]
    config = json.loads(json_block)
    result = ''
    try:
        process = subprocess.run(
            [
                sys.executable,
                lightautoml_template,
                "--df_name", state['df_name'],
                "--task_type", config['task_type'],
                "--target", config['target'],
                "--task_metric", config['task_metric']
            ],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if process.returncode == 0:
            result_text = lightautoml_result.format(
                lightautoml_template=lightautoml_template,
                process_stdout=process.stdout
            )
            result = result_text.strip()
        else:
            result = lightautoml_error.format(
                lightautoml_template=lightautoml_template,
                process_err=process.stderr
            )

    except subprocess.TimeoutExpired:
        result = f"LightAutoML execution timed out after {timeout} seconds."

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


def execute_code(state: AgentState):
    messages = state['messages']
    code_blocks = re.findall(PYTHON_REGEX, messages[-1].content, re.DOTALL | re.MULTILINE)
    code_to_execute = "\n".join(code_blocks)
    full_code = matplotlib_setup + code_to_execute + "\nplt.close('all')"
    execution_location = state['code_generation_config']

    if state['lama']:
        result = execute_lightautoml_locally(state)
    else:
        if execution_location == 'e2b':
            sandbox = state['sandbox']
            result = execute_e2b_code(sandbox, full_code)

        if execution_location == 'local':
            result = execute_code_locally(full_code)

    return {"messages": AIMessage(content=result), 'generated_code': code_to_execute, 'code_results': result, 'lama': False, 'test_split': False}
