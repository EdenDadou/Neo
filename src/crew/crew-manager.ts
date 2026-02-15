/**
 * CrewManager - Gestionnaire de l'intégration CrewAI
 *
 * Gère la communication avec le subprocess Python CrewAI
 * et orchestre l'exécution des crews d'agents.
 */

import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';
import { LLM_PRESETS } from './types';
import type {
  CrewConfig,
  CrewExecutionResult,
  CrewIPCRequest,
  CrewIPCResponse,
  LLMConfig,
} from './types';

// ESM compatibility for __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ===========================================================================
// TYPES INTERNES
// ===========================================================================

interface PendingRequest {
  resolve: (result: CrewExecutionResult) => void;
  reject: (error: Error) => void;
  timeout: NodeJS.Timeout;
  logs: string[];
}

type CrewManagerEvent =
  | { type: 'crew_started'; crewName: string }
  | { type: 'crew_completed'; result: CrewExecutionResult }
  | { type: 'crew_failed'; crewName: string; error: string }
  | { type: 'task_progress'; crewName: string; currentTask: string; progress: number }
  | { type: 'agent_log'; agent: string; message: string }
  | { type: 'subprocess_error'; error: string }
  | { type: 'subprocess_ready' }
  | { type: 'subprocess_stopped' };

// ===========================================================================
// CREW MANAGER
// ===========================================================================

export class CrewManager extends EventEmitter {
  private pythonProcess: ChildProcess | null = null;
  private pendingRequests: Map<string, PendingRequest> = new Map();
  private isReady = false;
  private buffer = '';

  // Configuration
  private readonly pythonPath: string;
  private readonly crewScriptPath: string;
  private readonly defaultTimeout: number;
  private readonly ollamaBaseUrl: string;

  constructor(options?: {
    pythonPath?: string;
    crewScriptPath?: string;
    defaultTimeout?: number;
    ollamaBaseUrl?: string;
  }) {
    super();

    this.pythonPath = options?.pythonPath || 'python3';
    this.crewScriptPath = options?.crewScriptPath || path.join(__dirname, 'python', 'crew_runner.py');
    this.defaultTimeout = options?.defaultTimeout || 300000; // 5 minutes par défaut
    this.ollamaBaseUrl = options?.ollamaBaseUrl || 'http://localhost:11434';
  }

  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================

  /**
   * Démarre le subprocess Python CrewAI
   */
  async start(): Promise<void> {
    if (this.pythonProcess) {
      console.log('[CrewManager] Already running');
      return;
    }

    // Vérifier que le script Python existe
    if (!fs.existsSync(this.crewScriptPath)) {
      throw new Error(`CrewAI script not found at: ${this.crewScriptPath}`);
    }

    return new Promise((resolve, reject) => {
      console.log('[CrewManager] Starting Python subprocess...');

      this.pythonProcess = spawn(this.pythonPath, [this.crewScriptPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          OLLAMA_BASE_URL: this.ollamaBaseUrl,
          PYTHONUNBUFFERED: '1',
        },
      });

      // Handle stdout (JSON responses)
      this.pythonProcess.stdout?.on('data', (data: Buffer) => {
        this.handleStdout(data.toString());
      });

      // Handle stderr (logs)
      this.pythonProcess.stderr?.on('data', (data: Buffer) => {
        const message = data.toString().trim();
        if (message) {
          console.log(`[CrewAI] ${message}`);
        }
      });

      // Handle process exit
      this.pythonProcess.on('exit', (code) => {
        console.log(`[CrewManager] Python process exited with code ${code}`);
        this.pythonProcess = null;
        this.isReady = false;
        this.emit('event', { type: 'subprocess_stopped' } as CrewManagerEvent);

        // Reject all pending requests
        for (const [requestId, pending] of this.pendingRequests) {
          clearTimeout(pending.timeout);
          pending.reject(new Error('CrewAI subprocess stopped unexpectedly'));
          this.pendingRequests.delete(requestId);
        }
      });

      // Handle errors
      this.pythonProcess.on('error', (error) => {
        console.error('[CrewManager] Process error:', error);
        reject(error);
      });

      // Wait for ready signal or timeout
      const readyTimeout = setTimeout(() => {
        reject(new Error('CrewAI subprocess failed to start within 30 seconds'));
      }, 30000);

      const checkHealth = async () => {
        try {
          await this.checkHealth();
          clearTimeout(readyTimeout);
          this.isReady = true;
          this.emit('event', { type: 'subprocess_ready' } as CrewManagerEvent);
          console.log('[CrewManager] ✅ Python subprocess ready');
          resolve();
        } catch {
          // Retry after 1 second
          setTimeout(checkHealth, 1000);
        }
      };

      // Start health check after process starts
      setTimeout(checkHealth, 2000);
    });
  }

  /**
   * Arrête le subprocess Python
   */
  async stop(): Promise<void> {
    if (!this.pythonProcess) {
      return;
    }

    console.log('[CrewManager] Stopping Python subprocess...');

    // Send stop command
    this.sendRequest({ type: 'STOP', requestId: 'stop' });

    // Wait a bit for graceful shutdown
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Force kill if still running
    if (this.pythonProcess) {
      this.pythonProcess.kill('SIGTERM');
      this.pythonProcess = null;
    }

    this.isReady = false;
  }

  // ===========================================================================
  // COMMUNICATION
  // ===========================================================================

  /**
   * Envoie une requête au subprocess Python
   */
  private sendRequest(request: CrewIPCRequest): void {
    if (!this.pythonProcess?.stdin) {
      throw new Error('Python subprocess not running');
    }

    const json = JSON.stringify(request) + '\n';
    this.pythonProcess.stdin.write(json);
  }

  /**
   * Gère les données reçues du subprocess
   */
  private handleStdout(data: string): void {
    this.buffer += data;

    // Process complete JSON lines
    const lines = this.buffer.split('\n');
    this.buffer = lines.pop() || ''; // Keep incomplete line in buffer

    for (const line of lines) {
      if (!line.trim()) continue;

      try {
        const response: CrewIPCResponse = JSON.parse(line);
        this.handleResponse(response);
      } catch (error) {
        console.warn('[CrewManager] Failed to parse response:', line);
      }
    }
  }

  /**
   * Gère une réponse du subprocess
   */
  private handleResponse(response: CrewIPCResponse): void {
    const pending = this.pendingRequests.get(response.requestId);

    switch (response.type) {
      case 'RESULT':
        if (pending && response.payload.result) {
          clearTimeout(pending.timeout);
          pending.resolve(response.payload.result);
          this.pendingRequests.delete(response.requestId);
          this.emit('event', {
            type: 'crew_completed',
            result: response.payload.result,
          } as CrewManagerEvent);
        }
        break;

      case 'ERROR':
        if (pending) {
          clearTimeout(pending.timeout);
          pending.reject(new Error(response.payload.error || 'Unknown error'));
          this.pendingRequests.delete(response.requestId);
          this.emit('event', {
            type: 'crew_failed',
            crewName: 'unknown',
            error: response.payload.error || 'Unknown error',
          } as CrewManagerEvent);
        }
        break;

      case 'LOG':
        if (response.payload.log) {
          const { level, message, agent } = response.payload.log;
          console.log(`[CrewAI:${level}] ${agent ? `[${agent}] ` : ''}${message}`);

          if (pending) {
            pending.logs.push(message);
          }

          if (agent) {
            this.emit('event', {
              type: 'agent_log',
              agent,
              message,
            } as CrewManagerEvent);
          }
        }
        break;

      case 'PROGRESS':
        if (response.payload.progress) {
          const { currentTask, completedTasks, totalTasks } = response.payload.progress;
          this.emit('event', {
            type: 'task_progress',
            crewName: 'unknown',
            currentTask,
            progress: completedTasks / totalTasks,
          } as CrewManagerEvent);
        }
        break;

      case 'HEALTH':
        // Health check response is handled separately
        break;
    }
  }

  // ===========================================================================
  // PUBLIC API
  // ===========================================================================

  /**
   * Vérifie la santé du subprocess et la connexion Ollama
   */
  async checkHealth(): Promise<{
    status: 'ok' | 'error';
    ollamaConnected: boolean;
    availableModels: string[];
  }> {
    const requestId = `health_${Date.now()}`;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Health check timeout'));
      }, 10000);

      // Temporary handler for health response
      const handler = (data: string) => {
        try {
          const response: CrewIPCResponse = JSON.parse(data.trim());
          if (response.requestId === requestId && response.type === 'HEALTH') {
            clearTimeout(timeout);
            this.pythonProcess?.stdout?.off('data', handler);
            resolve(response.payload.health || {
              status: 'error',
              ollamaConnected: false,
              availableModels: [],
            });
          }
        } catch {
          // Ignore parse errors
        }
      };

      this.pythonProcess?.stdout?.on('data', handler);
      this.sendRequest({ type: 'CHECK_HEALTH', requestId });
    });
  }

  /**
   * Exécute un Crew d'agents
   */
  async executeCrew(
    config: CrewConfig,
    options?: { timeout?: number }
  ): Promise<CrewExecutionResult> {
    if (!this.isReady) {
      throw new Error('CrewManager not ready. Call start() first.');
    }

    const requestId = `crew_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    const timeout = options?.timeout || this.defaultTimeout;

    console.log(`[CrewManager] Executing crew "${config.name}" with ${config.agents.length} agents and ${config.tasks.length} tasks`);

    this.emit('event', {
      type: 'crew_started',
      crewName: config.name,
    } as CrewManagerEvent);

    return new Promise((resolve, reject) => {
      const timeoutHandle = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        reject(new Error(`Crew execution timeout after ${timeout}ms`));
      }, timeout);

      this.pendingRequests.set(requestId, {
        resolve,
        reject,
        timeout: timeoutHandle,
        logs: [],
      });

      this.sendRequest({
        type: 'EXECUTE_CREW',
        requestId,
        payload: {
          crew: config,
          timeout,
        },
      });
    });
  }

  /**
   * Liste les modèles Ollama disponibles
   */
  async listOllamaModels(): Promise<string[]> {
    const health = await this.checkHealth();
    return health.availableModels;
  }

  /**
   * Retourne les presets LLM disponibles
   */
  getLLMPresets(): typeof LLM_PRESETS {
    return LLM_PRESETS;
  }

  /**
   * Sélectionne le meilleur LLM pour une tâche donnée
   * Priorise les modèles locaux (Ollama) pour économiser
   */
  async selectBestLLM(
    taskComplexity: 'simple' | 'moderate' | 'complex',
    requiresCode: boolean = false
  ): Promise<LLMConfig> {
    // Vérifier quels modèles Ollama sont disponibles
    let availableModels: string[] = [];
    try {
      availableModels = await this.listOllamaModels();
    } catch {
      console.warn('[CrewManager] Could not list Ollama models, will use cloud fallback if needed');
    }

    // Prioriser les modèles locaux
    if (requiresCode && availableModels.includes('codellama:13b')) {
      return LLM_PRESETS.OLLAMA_CODE;
    }

    switch (taskComplexity) {
      case 'simple':
        if (availableModels.some(m => m.includes('llama3.2') || m.includes('phi'))) {
          return LLM_PRESETS.OLLAMA_FAST;
        }
        // Fallback to cloud (cheapest)
        return LLM_PRESETS.CLAUDE_HAIKU;

      case 'moderate':
        if (availableModels.some(m => m.includes('llama3.2:8b') || m.includes('mistral'))) {
          return LLM_PRESETS.OLLAMA_BALANCED;
        }
        return LLM_PRESETS.CLAUDE_HAIKU;

      case 'complex':
        if (availableModels.some(m => m.includes('mixtral') || m.includes('70b'))) {
          return LLM_PRESETS.OLLAMA_SMART;
        }
        // For complex tasks, cloud might be necessary
        return LLM_PRESETS.CLAUDE_SONNET;
    }
  }

  /**
   * Retourne si le manager est prêt
   */
  isRunning(): boolean {
    return this.isReady;
  }
}

// ===========================================================================
// SINGLETON
// ===========================================================================

let crewManagerInstance: CrewManager | null = null;

export function getCrewManager(): CrewManager {
  if (!crewManagerInstance) {
    crewManagerInstance = new CrewManager();
  }
  return crewManagerInstance;
}
