#!/usr/bin/env python3
"""
CrewAI Runner - Subprocess pour exécuter les crews d'agents

Ce script reçoit des commandes JSON via stdin et retourne des résultats via stdout.
Il utilise CrewAI avec Ollama comme LLM par défaut pour minimiser les coûts.
"""

import sys
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime

# Ensure unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# ===========================================================================
# DEPENDENCIES CHECK
# ===========================================================================

def check_dependencies():
    """Vérifie que les dépendances sont installées"""
    missing = []

    try:
        import crewai
    except ImportError:
        missing.append('crewai')

    try:
        import requests
    except ImportError:
        missing.append('requests')

    if missing:
        print(f"[ERROR] Missing dependencies: {', '.join(missing)}", file=sys.stderr)
        print(f"[ERROR] Install with: pip install {' '.join(missing)}", file=sys.stderr)
        sys.exit(1)

check_dependencies()

from crewai import Agent, Task, Crew, Process, LLM
import requests

# ===========================================================================
# CONFIGURATION
# ===========================================================================

OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')

# ===========================================================================
# HELPERS
# ===========================================================================

def send_response(response: Dict[str, Any]):
    """Envoie une réponse JSON au processus parent"""
    print(json.dumps(response), flush=True)

def send_log(request_id: str, level: str, message: str, agent: str = None, task: str = None):
    """Envoie un log au processus parent"""
    send_response({
        'type': 'LOG',
        'requestId': request_id,
        'payload': {
            'log': {
                'level': level,
                'message': message,
                'agent': agent,
                'task': task,
            }
        }
    })

def send_progress(request_id: str, current_task: str, completed: int, total: int):
    """Envoie une mise à jour de progression"""
    send_response({
        'type': 'PROGRESS',
        'requestId': request_id,
        'payload': {
            'progress': {
                'currentTask': current_task,
                'completedTasks': completed,
                'totalTasks': total,
            }
        }
    })

def send_error(request_id: str, error: str):
    """Envoie une erreur"""
    send_response({
        'type': 'ERROR',
        'requestId': request_id,
        'payload': {
            'error': error
        }
    })

def send_result(request_id: str, result: Dict[str, Any]):
    """Envoie un résultat"""
    send_response({
        'type': 'RESULT',
        'requestId': request_id,
        'payload': {
            'result': result
        }
    })

# ===========================================================================
# OLLAMA INTEGRATION
# ===========================================================================

def check_ollama_connection() -> tuple[bool, List[str]]:
    """Vérifie la connexion à Ollama et retourne les modèles disponibles"""
    try:
        response = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return True, models
        return False, []
    except Exception as e:
        print(f"[WARN] Ollama connection failed: {e}", file=sys.stderr)
        return False, []

def create_llm(config: Dict[str, Any]) -> LLM:
    """Crée un LLM à partir de la configuration"""
    provider = config.get('provider', 'ollama')
    model = config.get('model', 'llama3.2:3b')

    if provider == 'ollama':
        base_url = config.get('baseUrl', OLLAMA_BASE_URL)
        return LLM(
            model=f"ollama/{model}",
            base_url=base_url,
            temperature=config.get('temperature', 0.7),
        )
    elif provider == 'anthropic':
        api_key = config.get('apiKey') or os.environ.get('ANTHROPIC_API_KEY')
        return LLM(
            model=model,
            api_key=api_key,
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('maxTokens', 4096),
        )
    elif provider == 'openai':
        api_key = config.get('apiKey') or os.environ.get('OPENAI_API_KEY')
        return LLM(
            model=model,
            api_key=api_key,
            temperature=config.get('temperature', 0.7),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

# ===========================================================================
# CREW EXECUTION
# ===========================================================================

def execute_crew(request_id: str, crew_config: Dict[str, Any]) -> Dict[str, Any]:
    """Exécute un crew d'agents"""
    start_time = time.time()
    total_tokens = 0
    task_results = []

    try:
        crew_name = crew_config.get('name', 'Unnamed Crew')
        send_log(request_id, 'info', f"Starting crew: {crew_name}")

        # Create agents
        agents_config = crew_config.get('agents', [])
        agents: Dict[str, Agent] = {}

        for agent_config in agents_config:
            agent_name = agent_config.get('name', 'Agent')
            send_log(request_id, 'info', f"Creating agent: {agent_name}", agent=agent_name)

            llm = create_llm(agent_config.get('llm', {'provider': 'ollama', 'model': 'llama3.2:3b'}))

            agent = Agent(
                role=agent_config.get('role', 'Assistant'),
                goal=agent_config.get('goal', 'Help the user'),
                backstory=agent_config.get('backstory', 'A helpful assistant'),
                llm=llm,
                verbose=agent_config.get('verbose', True),
                allow_delegation=agent_config.get('allowDelegation', False),
                max_iter=agent_config.get('maxIterations', 15),
                memory=agent_config.get('memoryEnabled', True),
            )
            agents[agent_name] = agent

        # Create tasks
        tasks_config = crew_config.get('tasks', [])
        tasks: List[Task] = []
        tasks_by_id: Dict[str, Task] = {}
        total_tasks = len(tasks_config)

        for i, task_config in enumerate(tasks_config):
            task_id = task_config.get('id', f'task_{i}')
            agent_name = task_config.get('agent')

            if agent_name not in agents:
                raise ValueError(f"Unknown agent '{agent_name}' for task '{task_id}'")

            send_log(request_id, 'info', f"Creating task: {task_id}", task=task_id)

            # Handle context (dependencies on previous tasks)
            context = []
            for ctx_id in task_config.get('context', []):
                if ctx_id in tasks_by_id:
                    context.append(tasks_by_id[ctx_id])

            task = Task(
                description=task_config.get('description', ''),
                expected_output=task_config.get('expectedOutput', ''),
                agent=agents[agent_name],
                context=context if context else None,
                async_execution=task_config.get('asyncExecution', False),
                human_input=task_config.get('humanInput', False),
            )

            tasks.append(task)
            tasks_by_id[task_id] = task

        # Create crew
        process_type = crew_config.get('process', 'sequential')
        process = Process.sequential if process_type == 'sequential' else Process.hierarchical

        crew_kwargs = {
            'agents': list(agents.values()),
            'tasks': tasks,
            'process': process,
            'verbose': crew_config.get('verbose', True),
            'memory': crew_config.get('memory', True),
        }

        # For hierarchical process, add manager LLM
        if process == Process.hierarchical and crew_config.get('managerLlm'):
            crew_kwargs['manager_llm'] = create_llm(crew_config['managerLlm'])

        if crew_config.get('maxRpm'):
            crew_kwargs['max_rpm'] = crew_config['maxRpm']

        crew = Crew(**crew_kwargs)

        send_log(request_id, 'info', f"Executing crew with {len(agents)} agents and {len(tasks)} tasks")

        # Execute
        result = crew.kickoff()

        # Process results
        end_time = time.time()
        execution_time_ms = int((end_time - start_time) * 1000)

        # Build task results
        for i, task_config in enumerate(tasks_config):
            task_id = task_config.get('id', f'task_{i}')
            task_results.append({
                'taskId': task_id,
                'success': True,
                'output': str(result) if i == len(tasks_config) - 1 else 'Completed',
                'executionTimeMs': execution_time_ms // len(tasks_config),
            })
            send_progress(request_id, task_id, i + 1, total_tasks)

        return {
            'success': True,
            'crewName': crew_name,
            'finalOutput': str(result),
            'taskResults': task_results,
            'totalTokensUsed': total_tokens,
            'totalExecutionTimeMs': execution_time_ms,
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] Crew execution failed: {error_msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        return {
            'success': False,
            'crewName': crew_config.get('name', 'Unknown'),
            'finalOutput': None,
            'taskResults': task_results,
            'totalTokensUsed': total_tokens,
            'totalExecutionTimeMs': int((time.time() - start_time) * 1000),
            'error': error_msg,
        }

# ===========================================================================
# MAIN LOOP
# ===========================================================================

def handle_request(request: Dict[str, Any]):
    """Gère une requête entrante"""
    request_type = request.get('type')
    request_id = request.get('requestId', 'unknown')

    if request_type == 'CHECK_HEALTH':
        connected, models = check_ollama_connection()
        send_response({
            'type': 'HEALTH',
            'requestId': request_id,
            'payload': {
                'health': {
                    'status': 'ok' if connected else 'error',
                    'ollamaConnected': connected,
                    'availableModels': models,
                }
            }
        })

    elif request_type == 'EXECUTE_CREW':
        payload = request.get('payload', {})
        crew_config = payload.get('crew')

        if not crew_config:
            send_error(request_id, 'Missing crew configuration')
            return

        result = execute_crew(request_id, crew_config)
        send_result(request_id, result)

    elif request_type == 'LIST_MODELS':
        connected, models = check_ollama_connection()
        send_response({
            'type': 'RESULT',
            'requestId': request_id,
            'payload': {
                'result': {
                    'success': connected,
                    'models': models,
                }
            }
        })

    elif request_type == 'STOP':
        print("[INFO] Received stop command, shutting down...", file=sys.stderr)
        sys.exit(0)

    else:
        send_error(request_id, f"Unknown request type: {request_type}")

def main():
    """Main loop - lit les commandes JSON depuis stdin"""
    print("[INFO] CrewAI Runner started", file=sys.stderr)
    print(f"[INFO] Ollama URL: {OLLAMA_BASE_URL}", file=sys.stderr)

    # Check Ollama connection at startup
    connected, models = check_ollama_connection()
    if connected:
        print(f"[INFO] Ollama connected, available models: {', '.join(models)}", file=sys.stderr)
    else:
        print("[WARN] Ollama not connected, will use cloud LLMs as fallback", file=sys.stderr)

    # Main loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            handle_request(request)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

if __name__ == '__main__':
    main()
