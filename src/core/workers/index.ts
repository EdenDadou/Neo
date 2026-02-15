/**
 * Workers Module
 *
 * Système de workers pour exécution non-bloquante des tâches.
 * Brain délègue tout travail aux workers pour rester disponible.
 */

export { WorkerAgent } from './worker-agent';
export type {
  WorkerTask,
  WorkerResult,
  WorkerTaskType,
  WorkerStatus,
} from './worker-agent';

export { WorkerPool, getWorkerPool } from './worker-pool';
export type { PoolConfig } from './worker-pool';
