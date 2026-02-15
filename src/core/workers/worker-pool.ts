/**
 * WorkerPool - Pool de Workers managé par Brain
 *
 * Gère un pool dynamique de WorkerAgents:
 * - Spawn de workers selon la charge
 * - Queue de tâches avec priorités
 * - Auto-scaling (min/max workers)
 * - Recyclage des workers après X tâches
 */

import { EventEmitter } from 'events';
import { randomUUID } from 'crypto';
import { WorkerAgent, WorkerTask, WorkerResult, WorkerTaskType } from './worker-agent';

// ===========================================================================
// TYPES
// ===========================================================================

export interface PoolConfig {
  minWorkers: number;      // Minimum de workers actifs
  maxWorkers: number;      // Maximum de workers
  workerIdleTimeout: number;  // ms avant de terminer un worker inactif
  maxTasksPerWorker: number;  // Recycler worker après X tâches
  defaultTaskTimeout: number; // Timeout par défaut pour les tâches
}

export interface QueuedTask {
  task: WorkerTask;
  resolve: (result: WorkerResult) => void;
  reject: (error: Error) => void;
  queuedAt: Date;
}

// ===========================================================================
// WORKER POOL
// ===========================================================================

export class WorkerPool extends EventEmitter {
  private workers: Map<string, WorkerAgent> = new Map();
  private taskQueue: QueuedTask[] = [];
  private config: PoolConfig;
  private isRunning = false;
  private processInterval: NodeJS.Timeout | null = null;
  private idleCheckInterval: NodeJS.Timeout | null = null;

  // Stats
  private totalTasksProcessed = 0;
  private totalTasksFailed = 0;

  constructor(config?: Partial<PoolConfig>) {
    super();
    this.config = {
      minWorkers: config?.minWorkers ?? 2,
      maxWorkers: config?.maxWorkers ?? 10,
      workerIdleTimeout: config?.workerIdleTimeout ?? 60000, // 1 min
      maxTasksPerWorker: config?.maxTasksPerWorker ?? 100,
      defaultTaskTimeout: config?.defaultTaskTimeout ?? 30000, // 30s
    };
  }

  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================

  /**
   * Démarrer le pool
   */
  start(): void {
    if (this.isRunning) return;

    this.isRunning = true;

    // Créer les workers minimum
    for (let i = 0; i < this.config.minWorkers; i++) {
      this.spawnWorker();
    }

    // Démarrer le processing de la queue
    this.processInterval = setInterval(() => {
      this.processQueue();
    }, 100); // Vérifier toutes les 100ms

    // Vérifier les workers inactifs
    this.idleCheckInterval = setInterval(() => {
      this.cleanIdleWorkers();
    }, 30000); // Toutes les 30s

    console.log(`[WorkerPool] 🚀 Pool démarré avec ${this.config.minWorkers} worker(s)`);
  }

  /**
   * Arrêter le pool
   */
  async stop(): Promise<void> {
    if (!this.isRunning) return;

    this.isRunning = false;

    // Arrêter les intervals
    if (this.processInterval) {
      clearInterval(this.processInterval);
      this.processInterval = null;
    }
    if (this.idleCheckInterval) {
      clearInterval(this.idleCheckInterval);
      this.idleCheckInterval = null;
    }

    // Rejeter les tâches en queue
    for (const queued of this.taskQueue) {
      queued.reject(new Error('Pool arrêté'));
    }
    this.taskQueue = [];

    // Terminer tous les workers
    for (const worker of this.workers.values()) {
      worker.terminate();
    }
    this.workers.clear();

    console.log('[WorkerPool] 🛑 Pool arrêté');
  }

  // ===========================================================================
  // TASK SUBMISSION
  // ===========================================================================

  /**
   * Soumettre une tâche au pool
   * Retourne une Promise qui se résout quand la tâche est terminée
   */
  submit(
    type: WorkerTaskType,
    payload: unknown,
    options?: {
      priority?: 'low' | 'normal' | 'high' | 'critical';
      timeout?: number;
    }
  ): Promise<WorkerResult> {
    return new Promise((resolve, reject) => {
      if (!this.isRunning) {
        reject(new Error('Pool non démarré'));
        return;
      }

      const task: WorkerTask = {
        id: randomUUID(),
        type,
        priority: options?.priority || 'normal',
        payload,
        timeout: options?.timeout || this.config.defaultTaskTimeout,
        createdAt: new Date(),
      };

      const queued: QueuedTask = {
        task,
        resolve,
        reject,
        queuedAt: new Date(),
      };

      // Insérer selon la priorité
      this.insertByPriority(queued);

      console.log(`[WorkerPool] 📥 Tâche ajoutée: ${type} (priority: ${task.priority}, queue: ${this.taskQueue.length})`);

      // Trigger un processing immédiat
      this.processQueue();
    });
  }

  /**
   * Soumettre plusieurs tâches en parallèle
   */
  submitBatch(
    tasks: Array<{
      type: WorkerTaskType;
      payload: unknown;
      priority?: 'low' | 'normal' | 'high' | 'critical';
      timeout?: number;
    }>
  ): Promise<WorkerResult[]> {
    return Promise.all(
      tasks.map(t => this.submit(t.type, t.payload, {
        priority: t.priority,
        timeout: t.timeout,
      }))
    );
  }

  /**
   * Insérer une tâche selon sa priorité
   */
  private insertByPriority(queued: QueuedTask): void {
    const priorityOrder = { critical: 0, high: 1, normal: 2, low: 3 };
    const taskPriority = priorityOrder[queued.task.priority];

    // Trouver la position d'insertion
    let insertIndex = this.taskQueue.length;
    for (let i = 0; i < this.taskQueue.length; i++) {
      const existingPriority = priorityOrder[this.taskQueue[i].task.priority];
      if (taskPriority < existingPriority) {
        insertIndex = i;
        break;
      }
    }

    this.taskQueue.splice(insertIndex, 0, queued);
  }

  // ===========================================================================
  // QUEUE PROCESSING
  // ===========================================================================

  /**
   * Traiter la queue de tâches
   */
  private processQueue(): void {
    if (this.taskQueue.length === 0) return;

    // Trouver un worker disponible
    let availableWorker = this.getAvailableWorker();

    // Spawner un nouveau worker si besoin et possible
    if (!availableWorker && this.workers.size < this.config.maxWorkers) {
      availableWorker = this.spawnWorker();
    }

    if (!availableWorker) {
      // Tous les workers sont occupés, attendre
      return;
    }

    // Prendre la première tâche de la queue
    const queued = this.taskQueue.shift();
    if (!queued) return;

    // Exécuter la tâche
    availableWorker.execute(queued.task)
      .then(result => {
        this.totalTasksProcessed++;
        queued.resolve(result);
        this.emit('task_completed', result);

        // Recycler le worker si nécessaire
        const stats = availableWorker!.getStats();
        if (stats.completedTasks >= this.config.maxTasksPerWorker) {
          this.recycleWorker(availableWorker!.id);
        }
      })
      .catch(error => {
        this.totalTasksFailed++;
        queued.reject(error);
        this.emit('task_failed', { taskId: queued.task.id, error });
      });
  }

  // ===========================================================================
  // WORKER MANAGEMENT
  // ===========================================================================

  /**
   * Spawner un nouveau worker
   */
  private spawnWorker(): WorkerAgent {
    const workerNumber = this.workers.size + 1;
    const worker = new WorkerAgent(`Worker-${workerNumber}`);

    // Écouter les événements du worker
    worker.on('task_started', (data) => this.emit('worker_task_started', data));
    worker.on('task_completed', (data) => this.emit('worker_task_completed', data));
    worker.on('task_failed', (data) => this.emit('worker_task_failed', data));

    this.workers.set(worker.id, worker);

    console.log(`[WorkerPool] ➕ Worker spawné: ${worker.name} (total: ${this.workers.size})`);

    return worker;
  }

  /**
   * Obtenir un worker disponible
   */
  private getAvailableWorker(): WorkerAgent | null {
    for (const worker of this.workers.values()) {
      if (worker.isAvailable()) {
        return worker;
      }
    }
    return null;
  }

  /**
   * Nettoyer les workers inactifs
   */
  private cleanIdleWorkers(): void {
    if (this.workers.size <= this.config.minWorkers) return;

    const toRemove: string[] = [];

    for (const [id, worker] of this.workers) {
      const stats = worker.getStats();

      // Garder les workers minimum
      if (this.workers.size - toRemove.length <= this.config.minWorkers) {
        break;
      }

      // Worker inactif depuis trop longtemps
      if (
        worker.isAvailable() &&
        stats.uptimeMs > this.config.workerIdleTimeout &&
        stats.completedTasks === 0
      ) {
        toRemove.push(id);
      }
    }

    for (const id of toRemove) {
      this.terminateWorker(id);
    }

    if (toRemove.length > 0) {
      console.log(`[WorkerPool] 🧹 ${toRemove.length} worker(s) inactif(s) terminé(s)`);
    }
  }

  /**
   * Recycler un worker (terminer et en créer un nouveau)
   */
  private recycleWorker(workerId: string): void {
    const worker = this.workers.get(workerId);
    if (!worker) return;

    console.log(`[WorkerPool] ♻️ Recyclage worker: ${worker.name}`);

    this.terminateWorker(workerId);

    // Créer un nouveau worker si on est sous le minimum
    if (this.workers.size < this.config.minWorkers) {
      this.spawnWorker();
    }
  }

  /**
   * Terminer un worker
   */
  private terminateWorker(workerId: string): void {
    const worker = this.workers.get(workerId);
    if (!worker) return;

    worker.terminate();
    this.workers.delete(workerId);
  }

  // ===========================================================================
  // STATS & INFO
  // ===========================================================================

  getStats(): {
    isRunning: boolean;
    totalWorkers: number;
    availableWorkers: number;
    busyWorkers: number;
    queueLength: number;
    totalTasksProcessed: number;
    totalTasksFailed: number;
    successRate: number;
    workers: Array<ReturnType<WorkerAgent['getStats']>>;
  } {
    const workerStats = Array.from(this.workers.values()).map(w => w.getStats());
    const availableWorkers = workerStats.filter(w => w.status === 'idle').length;
    const total = this.totalTasksProcessed + this.totalTasksFailed;

    return {
      isRunning: this.isRunning,
      totalWorkers: this.workers.size,
      availableWorkers,
      busyWorkers: this.workers.size - availableWorkers,
      queueLength: this.taskQueue.length,
      totalTasksProcessed: this.totalTasksProcessed,
      totalTasksFailed: this.totalTasksFailed,
      successRate: total > 0 ? this.totalTasksProcessed / total : 1,
      workers: workerStats,
    };
  }

  getQueueStatus(): {
    length: number;
    byPriority: Record<string, number>;
    oldestTaskAge: number | null;
  } {
    const byPriority: Record<string, number> = {
      critical: 0,
      high: 0,
      normal: 0,
      low: 0,
    };

    for (const queued of this.taskQueue) {
      byPriority[queued.task.priority]++;
    }

    const oldestTaskAge = this.taskQueue.length > 0
      ? Date.now() - this.taskQueue[0].queuedAt.getTime()
      : null;

    return {
      length: this.taskQueue.length,
      byPriority,
      oldestTaskAge,
    };
  }
}

// ===========================================================================
// SINGLETON
// ===========================================================================

let workerPoolInstance: WorkerPool | null = null;

export function getWorkerPool(config?: Partial<PoolConfig>): WorkerPool {
  if (!workerPoolInstance) {
    workerPoolInstance = new WorkerPool(config);
  }
  return workerPoolInstance;
}
