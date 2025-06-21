import subprocess
import sys
import tempfile
import os
import re
import json

from graph.state import AgentState
from langchain_core.messages import AIMessage

lightautoml_template = 'graph/lightautoml_template.py'

PYTHON_REGEX = r"```python-execute(.+?)```"
JSON_REGEX = r"```json(.+?)```"

lightautoml_error = """В результате выполнения блока {lightautoml_template} возникла ошибка:
```
{process_err}
```
Исправь ошибку
"""

lightautoml_result = """Результат выполнения блока {lightautoml_template}:
```
{process_stdout}
```
"""

matplotlib_setup = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
"""

local_exec_result = """Результат выполнения блока {index}:
```
{process_stdout}
```"""

local_exec_error = """В результате выполнения блока {index} возникла ошибка:
```
{process_stderr}
```
Исправь ошибку"""

timeout = 3000

e2b_exec_error = """В результате выполнения блока {index} возникла ошибка:
```
{execution_error_traceback}
```
Исправь ошибку"""

e2b_exec_result = """Результат выполнения блока {index}:
```
{logs}
{text_results}
```"""


def execute_code(state: AgentState):
    messages = state['messages']
    sandbox = state['sandbox']
    code_blocks = re.findall(PYTHON_REGEX, messages[-1].content, re.DOTALL | re.MULTILINE)
    results = []
    executions = []

    for index, code_block in enumerate(code_blocks):
        full_code = matplotlib_setup + code_block + "\nplt.close('all')"
        execution = sandbox.run_code(full_code)
        executions.append(execution)
        
        if execution.error:
            results.append(e2b_exec_error.fromat(
                index=index,
                execution_error_traceback=execution.error.traceback
            ))
            break
        else:
            logs = '\n'.join(execution.logs.stdout)
            text_results = "\n".join([result.text for result in execution.results if result.text])
            result_text = e2b_exec_result.format(
                index=index,
                logs=logs,
                text_results=text_results
            )
            results.append(result_text.strip())
            
    message = AIMessage(content="\n".join(results))
    return {"messages": message}


def execute_code_locally(state: AgentState):
    messages = state['messages']
    code_blocks = re.findall(PYTHON_REGEX, messages[-1].content, re.DOTALL | re.MULTILINE)
    results = []
    executions = []

    for index, code_block in enumerate(code_blocks):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            full_code = matplotlib_setup + code_block + "\nplt.close('all')"
            temp_file.write(full_code)
            temp_file.flush()

            try:
                process = subprocess.run(
                    [sys.executable, temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                execution_result = type('ExecutionResult', (), {
                    'logs': type('Logs', (), {'stdout': [process.stdout] if process.stdout else []}),
                    'results': [],
                    'error': None if process.returncode == 0 else type('Error', (), {
                        'traceback': process.stderr
                    })
                })

                if process.returncode == 0:
                    result_text = local_exec_result.format(
                        index=index,
                        process_stdout=process.stdout
                    )
                    results.append(result_text.strip())
                else:
                    results.append(local_exec_error.format(
                        index=index,
                        process_stderr=process.stderr
                    ))
                    break

                executions.append(execution_result)

            except subprocess.TimeoutExpired:
                results.append(f"Блок {index} превысил время выполнения ({timeout} секунд)")
                break
            finally:
                os.unlink(temp_file.name)

    message = AIMessage(content="\n".join(results))
    result_code = "\n".join(code_blocks)
    old_code = state['generated_code']
    old_code.append(result_code)
    old_results = state['code_results']
    old_results.append(results)
    return {"messages": message, "generated_code": old_code, "code_results": old_results}


def execute_lightautoml_locally(state: AgentState):
    messages = state['messages']
    json_block = re.findall(JSON_REGEX, messages[-1].content, re.DOTALL | re.MULTILINE)[0]
    config = json.loads(json_block)

    results = []
    executions = []

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

        execution_result = type('ExecutionResult', (), {
            'logs': type('Logs', (), {'stdout': [process.stdout] if process.stdout else []}),
            'results': [],
            'error': None if process.returncode == 0 else type('Error', (), {
                'traceback': process.stderr
            })
        })

        if process.returncode == 0:
            result_text = lightautoml_result.format(
                lightautoml_template=lightautoml_template,
                process_stdout=process.stdout
            )
            results.append(result_text.strip())
        else:
            results.append(lightautoml_error.format(
                lightautoml_template=lightautoml_template,
                process_err=process.stderr
            ))

        executions.append(execution_result)

    except subprocess.TimeoutExpired:
        results.append(f"Блок {lightautoml_template} превысил время выполнения ({timeout} секунд)")

    message = AIMessage(content="\n".join(results))
    return {"messages": message}
