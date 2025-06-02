import subprocess
import sys
import tempfile
import os
import re
import json

import re
from typing import List
from graph.state import AgentState
from langchain_core.messages import HumanMessage

lightautoml_template = 'graph/lightautoml_template.py'

PYTHON_REGEX = r"```python-execute(.+?)```"
JSON_REGEX = r"```json(.+?)```"

def extract_text_from_results(results) -> List[str]:
    return "\n".join([result.text for result in results if result.text])

def extract_images_from_results(results) -> List[str]:
    image_descriptions = []
    for result in results:
        if result.png:
            image_descriptions.append("Generated plot")
        elif result.chart:
            image_descriptions.append("Generated chart")
    return "\n".join(image_descriptions) if image_descriptions else ""
    
def execute_code(state: AgentState):
    """Выполнение кода"""
    messages = state['messages']
    sandbox = state['sandbox']
    code_blocks = re.findall(PYTHON_REGEX, messages[-1].content, re.DOTALL | re.MULTILINE)
    results = []
    executions = []

    # Add matplotlib configuration to ensure proper display
    setup_code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
"""
    
    for index, code_block in enumerate(code_blocks):
        # Add matplotlib setup to the beginning of each code block
        full_code = setup_code + code_block + "\nplt.close('all')"  # Close all figures after execution
        execution = sandbox.run_code(full_code)
        executions.append(execution)
        
        if execution.error:
            results.append(f"""В результате выполнения блока {index} возникла ошибка:
```
{execution.error.traceback}
```
Исправь ошибку""")
            break
        else:
            logs = '\n'.join(execution.logs.stdout)
            text_results = extract_text_from_results(execution.results)
            image_results = extract_images_from_results(execution.results)
            result_text = f"""Результат выполнения блока {index}:
```
{logs}
{text_results}
{image_results}
```"""
            results.append(result_text.strip())
            
    message = HumanMessage(content="\n".join(results), additional_kwargs={"executions": executions})
    return {"messages": message}

def execute_code_locally(state: AgentState):
    """Execute Python code locally using subprocess"""
    messages = state['messages']
    code_blocks = re.findall(PYTHON_REGEX, messages[-1].content, re.DOTALL | re.MULTILINE)
    results = []
    executions = []

    setup_code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
"""

    for index, code_block in enumerate(code_blocks):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            full_code = setup_code + code_block + "\nplt.close('all')"
            temp_file.write(full_code)
            temp_file.flush()
            
            try:
                process = subprocess.run(
                    [sys.executable, temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                execution_result = type('ExecutionResult', (), {
                    'logs': type('Logs', (), {'stdout': [process.stdout] if process.stdout else []}),
                    'results': [],
                    'error': None if process.returncode == 0 else type('Error', (), {
                        'traceback': process.stderr
                    })
                })
                
                if process.returncode == 0:
                    result_text = f"""Результат выполнения блока {index}:
```
{process.stdout}
```"""
                    results.append(result_text.strip())
                else:
                    results.append(f"""В результате выполнения блока {index} возникла ошибка:
```
{process.stderr}
```
Исправь ошибку""")
                    break
                    
                executions.append(execution_result)
                
            except subprocess.TimeoutExpired:
                results.append(f"Блок {index} превысил время выполнения (30 секунд)")
                break
            finally:
                os.unlink(temp_file.name)
    
    message = HumanMessage(content="\n".join(results), additional_kwargs={"executions": executions})
    return {"messages": message}


def execute_lightautoml_locally(state: AgentState):
    messages = state['messages']
    json_block = re.findall(JSON_REGEX, messages[-1].content, re.DOTALL | re.MULTILINE)[0]
    config = json.loads(json_block)

    results = []
    executions = []

    # code = open(lightautoml_template, "r").read()

    try:
        process = subprocess.run(
            [
                sys.executable,
                # "-W", "ignore",
                lightautoml_template,
                "--df_name", state['df_name'],
                "--task_type", config['task_type'],
                "--target", config['target'],
                "--task_metric", config['task_metric']
            ],
            capture_output=True,
            text=True,
            timeout=3000
        )
        
        execution_result = type('ExecutionResult', (), {
            'logs': type('Logs', (), {'stdout': [process.stdout] if process.stdout else []}),
            'results': [],
            'error': None if process.returncode == 0 else type('Error', (), {
                'traceback': process.stderr
            })
        })
        
        if process.returncode == 0:
            result_text = f"""Результат выполнения блока {lightautoml_template}:
```
{process.stdout}
```"""
            results.append(result_text.strip())
        else:
            results.append(f"""В результате выполнения блока {lightautoml_template} возникла ошибка:
```
{process.stderr}
```
Исправь ошибку""")
                    
        executions.append(execution_result)
                
    except subprocess.TimeoutExpired:
        results.append(f"Блок {lightautoml_template} превысил время выполнения (3000 секунд)")

    # lf_handler = langfuse_context.get_current_langchain_handler()
    # langfuse_context.score_current_trace(
    #     name = "test-roc-auc",
    #     value = float(process.stdout.split(' ')[-1]),
    #     comment = "check roc_auc!"
    # )
    message = HumanMessage(content="\n".join(results), additional_kwargs={"executions": executions})
    return {"messages": message}
