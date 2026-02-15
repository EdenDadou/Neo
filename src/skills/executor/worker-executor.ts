/**
 * WorkerExecutor - Exécution des skills dans des Worker Threads isolés
 *
 * Sécurité:
 * - Chaque skill s'exécute dans un Worker Thread séparé
 * - Timeout strict pour éviter les boucles infinies
 * - Memory limit pour éviter les fuites mémoire
 * - Communication via messages (pas d'accès direct)
 */

import { Worker } from 'worker_threads';
import { EventEmitter } from 'events';
import * as path from 'path';
import { fileURLToPath } from 'url';

// ESM compatibility for __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import type {
  SkillDefinition,
  CapabilityGrant,
  WorkerMessage,
  WorkerResponse,
  SerializedCapabilities,
  CapabilityType,
} from '../types';
import { SKILL_DEFAULTS } from '../types';

// ===========================================================================
// TYPES
// ===========================================================================

interface ExecutionRequest {
  executionId: string;
  skill: SkillDefinition;
  input: Record<string, unknown>;
  capabilities: CapabilityGrant;
  timeout: number;
}

interface PendingExecution {
  request: ExecutionRequest;
  worker: Worker;
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timeoutHandle: NodeJS.Timeout;
  startTime: number;
}

interface PooledWorker {
  worker: Worker;
  busy: boolean;
  createdAt: Date;
  lastUsedAt: Date;
  executionCount: number;
}

// ===========================================================================
// WORKER EXECUTOR
// ===========================================================================

export class WorkerExecutor extends EventEmitter {
  private workerPool: PooledWorker[] = [];
  private pendingExecutions: Map<string, PendingExecution> = new Map();
  private skillManager: SkillManagerInterface;
  private isRunning = false;
  private idleCheckInterval: NodeJS.Timeout | null = null;

  private readonly maxWorkers = SKILL_DEFAULTS.MAX_WORKERS;
  private readonly minWorkers = SKILL_DEFAULTS.MIN_WORKERS;
  private readonly workerIdleTimeout = SKILL_DEFAULTS.WORKER_IDLE_TIMEOUT_MS;

  constructor(skillManager: SkillManagerInterface) {
    super();
    this.skillManager = skillManager;
  }

  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================

  /**
   * Démarrer le pool de workers
   */
  async start(): Promise<void> {
    if (this.isRunning) return;

    // Créer les workers minimum
    for (let i = 0; i < this.minWorkers; i++) {
      this.createWorker();
    }

    // Vérifier périodiquement les workers inactifs
    this.idleCheckInterval = setInterval(() => {
      this.cleanIdleWorkers();
    }, 30000);

    this.isRunning = true;
    console.log(`[WorkerExecutor] Pool démarré avec ${this.minWorkers} worker(s)`);
  }

  /**
   * Arrêter le pool de workers
   */
  async stop(): Promise<void> {
    if (!this.isRunning) return;

    // Arrêter le check des workers inactifs
    if (this.idleCheckInterval) {
      clearInterval(this.idleCheckInterval);
      this.idleCheckInterval = null;
    }

    // Annuler toutes les exécutions en cours
    for (const [executionId, pending] of this.pendingExecutions) {
      clearTimeout(pending.timeoutHandle);
      pending.reject(new Error('Executor arrêté'));
      this.pendingExecutions.delete(executionId);
    }

    // Terminer tous les workers
    for (const pooled of this.workerPool) {
      await this.terminateWorker(pooled);
    }
    this.workerPool = [];

    this.isRunning = false;
    console.log('[WorkerExecutor] Pool arrêté');
  }

  // ===========================================================================
  // EXECUTION
  // ===========================================================================

  /**
   * Exécuter un skill dans un worker
   */
  async execute(request: ExecutionRequest): Promise<unknown> {
    if (!this.isRunning) {
      throw new Error('WorkerExecutor non démarré');
    }

    return new Promise((resolve, reject) => {
      // Obtenir un worker disponible
      const worker = this.getAvailableWorker();
      if (!worker) {
        reject(new Error('Aucun worker disponible'));
        return;
      }

      // Configurer le timeout
      const timeoutHandle = setTimeout(() => {
        this.handleTimeout(request.executionId);
      }, request.timeout);

      // Enregistrer l'exécution
      const pending: PendingExecution = {
        request,
        worker: worker.worker,
        resolve,
        reject,
        timeoutHandle,
        startTime: Date.now(),
      };
      this.pendingExecutions.set(request.executionId, pending);

      // Marquer le worker comme occupé
      worker.busy = true;

      // Sérialiser les capabilities pour le worker
      const serializedCapabilities = this.serializeCapabilities(request.capabilities);

      // Envoyer le message au worker
      const message: WorkerMessage = {
        type: 'EXECUTE_SKILL',
        executionId: request.executionId,
        payload: {
          code: request.skill.code,
          input: request.input,
          capabilities: serializedCapabilities,
          timeout: request.timeout,
        },
      };

      worker.worker.postMessage(message);
    });
  }

  // ===========================================================================
  // WORKER MANAGEMENT
  // ===========================================================================

  /**
   * Créer un nouveau worker
   */
  private createWorker(): PooledWorker {
    const workerPath = path.join(__dirname, 'skill-worker.ts');

    const worker = new Worker(workerPath, {
      // Utiliser tsx pour exécuter le TypeScript
      execArgv: ['--import', 'tsx'],
      // Limiter les ressources
      resourceLimits: {
        maxOldGenerationSizeMb: SKILL_DEFAULTS.MEMORY_LIMIT_MB,
        maxYoungGenerationSizeMb: SKILL_DEFAULTS.MEMORY_LIMIT_MB / 4,
      },
    });

    const pooled: PooledWorker = {
      worker,
      busy: false,
      createdAt: new Date(),
      lastUsedAt: new Date(),
      executionCount: 0,
    };

    // Gérer les messages du worker
    worker.on('message', (response: WorkerResponse) => {
      this.handleWorkerMessage(pooled, response);
    });

    // Gérer les erreurs
    worker.on('error', (error) => {
      this.handleWorkerError(pooled, error);
    });

    // Gérer la fermeture
    worker.on('exit', (code) => {
      this.handleWorkerExit(pooled, code);
    });

    this.workerPool.push(pooled);

    console.log(`[WorkerExecutor] Worker créé (total: ${this.workerPool.length})`);
    return pooled;
  }

  /**
   * Obtenir un worker disponible
   */
  private getAvailableWorker(): PooledWorker | null {
    // Chercher un worker libre
    const available = this.workerPool.find(w => !w.busy);
    if (available) {
      return available;
    }

    // Créer un nouveau worker si on n'a pas atteint la limite
    if (this.workerPool.length < this.maxWorkers) {
      return this.createWorker();
    }

    return null;
  }

  /**
   * Nettoyer les workers inactifs
   */
  private cleanIdleWorkers(): void {
    const now = Date.now();

    const toRemove: PooledWorker[] = [];

    for (const pooled of this.workerPool) {
      // Garder au moins le minimum de workers
      if (this.workerPool.length - toRemove.length <= this.minWorkers) {
        break;
      }

      // Si le worker est inactif depuis trop longtemps
      if (!pooled.busy && now - pooled.lastUsedAt.getTime() > this.workerIdleTimeout) {
        toRemove.push(pooled);
      }
    }

    for (const pooled of toRemove) {
      this.terminateWorker(pooled);
      const index = this.workerPool.indexOf(pooled);
      if (index !== -1) {
        this.workerPool.splice(index, 1);
      }
    }

    if (toRemove.length > 0) {
      console.log(`[WorkerExecutor] ${toRemove.length} worker(s) inactif(s) supprimé(s)`);
    }
  }

  /**
   * Terminer un worker
   */
  private async terminateWorker(pooled: PooledWorker): Promise<void> {
    try {
      await pooled.worker.terminate();
    } catch (error) {
      console.error('[WorkerExecutor] Erreur terminaison worker:', error);
    }
  }

  // ===========================================================================
  // MESSAGE HANDLING
  // ===========================================================================

  /**
   * Gérer un message du worker
   */
  private async handleWorkerMessage(pooled: PooledWorker, response: WorkerResponse): Promise<void> {
    const pending = this.pendingExecutions.get(response.executionId);

    if (response.type === 'LOG') {
      // Log du skill
      const { level, message } = response.payload.log!;
      console.log(`[Skill:${response.executionId}] [${level}] ${message}`);
      return;
    }

    if (response.type === 'CAPABILITY_REQUEST') {
      // Le skill demande à utiliser une capability
      await this.handleCapabilityRequest(pooled, response);
      return;
    }

    if (response.type === 'PONG') {
      // Réponse au ping
      return;
    }

    // RESULT ou ERROR
    if (!pending) {
      console.warn(`[WorkerExecutor] Exécution inconnue: ${response.executionId}`);
      return;
    }

    // Nettoyer
    clearTimeout(pending.timeoutHandle);
    this.pendingExecutions.delete(response.executionId);

    // Libérer le worker
    pooled.busy = false;
    pooled.lastUsedAt = new Date();
    pooled.executionCount++;

    if (response.type === 'RESULT') {
      pending.resolve(response.payload.output);
    } else if (response.type === 'ERROR') {
      const error = new Error(response.payload.error?.message || 'Erreur inconnue');
      (error as Error & { code?: string }).code = response.payload.error?.code;
      pending.reject(error);
    }
  }

  /**
   * Gérer une demande de capability du worker
   */
  private async handleCapabilityRequest(
    pooled: PooledWorker,
    response: WorkerResponse
  ): Promise<void> {
    const pending = this.pendingExecutions.get(response.executionId);
    if (!pending) return;

    const { type, method, args } = response.payload.capabilityRequest!;

    try {
      // Déléguer au SkillManager
      const result = await this.skillManager.handleCapabilityRequest(
        pending.request.skill.id,
        type,
        method,
        args
      );

      // Renvoyer le résultat au worker
      pooled.worker.postMessage({
        type: 'CAPABILITY_RESPONSE',
        executionId: response.executionId,
        payload: { result },
      });
    } catch (error) {
      // Renvoyer l'erreur au worker
      pooled.worker.postMessage({
        type: 'CAPABILITY_RESPONSE',
        executionId: response.executionId,
        payload: {
          error: {
            message: error instanceof Error ? error.message : String(error),
            code: 'CAPABILITY_ERROR',
          },
        },
      });
    }
  }

  /**
   * Gérer une erreur du worker
   */
  private handleWorkerError(pooled: PooledWorker, error: Error): void {
    console.error('[WorkerExecutor] Erreur worker:', error.message);

    // Rejeter toutes les exécutions sur ce worker
    for (const [executionId, pending] of this.pendingExecutions) {
      if (pending.worker === pooled.worker) {
        clearTimeout(pending.timeoutHandle);
        pending.reject(error);
        this.pendingExecutions.delete(executionId);
      }
    }

    // Remplacer le worker
    const index = this.workerPool.indexOf(pooled);
    if (index !== -1) {
      this.workerPool.splice(index, 1);
      this.createWorker();
    }
  }

  /**
   * Gérer la fermeture d'un worker
   */
  private handleWorkerExit(pooled: PooledWorker, code: number): void {
    if (code !== 0) {
      console.warn(`[WorkerExecutor] Worker terminé avec code ${code}`);
    }

    // Rejeter les exécutions en cours sur ce worker
    for (const [executionId, pending] of this.pendingExecutions) {
      if (pending.worker === pooled.worker) {
        clearTimeout(pending.timeoutHandle);
        pending.reject(new Error(`Worker terminé (code: ${code})`));
        this.pendingExecutions.delete(executionId);
      }
    }

    // Retirer du pool
    const index = this.workerPool.indexOf(pooled);
    if (index !== -1) {
      this.workerPool.splice(index, 1);

      // Remplacer si on est en dessous du minimum
      if (this.isRunning && this.workerPool.length < this.minWorkers) {
        this.createWorker();
      }
    }
  }

  /**
   * Gérer un timeout
   */
  private handleTimeout(executionId: string): void {
    const pending = this.pendingExecutions.get(executionId);
    if (!pending) return;

    console.warn(`[WorkerExecutor] Timeout: ${executionId}`);

    // Nettoyer
    this.pendingExecutions.delete(executionId);

    // Rejeter
    pending.reject(new Error(`Timeout après ${pending.request.timeout}ms`));

    // Terminer et remplacer le worker (il pourrait être bloqué)
    const pooledIndex = this.workerPool.findIndex(p => p.worker === pending.worker);
    if (pooledIndex !== -1) {
      const pooled = this.workerPool[pooledIndex];
      this.terminateWorker(pooled);
      this.workerPool.splice(pooledIndex, 1);
      this.createWorker();
    }
  }

  // ===========================================================================
  // HELPERS
  // ===========================================================================

  /**
   * Sérialiser les capabilities pour le worker
   */
  private serializeCapabilities(grant: CapabilityGrant): SerializedCapabilities {
    const allowed: CapabilityType[] = grant.capabilities.map(c => c.type);
    const config: Record<CapabilityType, Record<string, unknown> | undefined> = {
      web_fetch: undefined,
      memory_read: undefined,
      memory_write: undefined,
      browser: undefined,
      file_read: undefined,
      llm_call: undefined,
    };

    for (const cap of grant.capabilities) {
      config[cap.type] = cap.options;
    }

    return { allowed, config };
  }

  /**
   * Obtenir les statistiques du pool
   */
  getStats(): {
    totalWorkers: number;
    busyWorkers: number;
    pendingExecutions: number;
    isRunning: boolean;
  } {
    return {
      totalWorkers: this.workerPool.length,
      busyWorkers: this.workerPool.filter(w => w.busy).length,
      pendingExecutions: this.pendingExecutions.size,
      isRunning: this.isRunning,
    };
  }
}

// ===========================================================================
// INTERFACE FOR SKILL MANAGER
// ===========================================================================

interface SkillManagerInterface {
  handleCapabilityRequest(
    skillId: string,
    capabilityType: CapabilityType,
    method: string,
    args: unknown[]
  ): Promise<unknown>;
}
