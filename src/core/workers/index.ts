/**
 * Workers Module
 *
 * Système de workers pour exécution non-bloquante des tâches.
 * Brain délègue tout travail aux workers pour rester disponible.
 */

export {
  WorkerAgent,
  WorkerTask,
  WorkerResult,
  WorkerTaskType,
  WorkerStatus,
} from './worker-agent';

export {
  WorkerPool,
  PoolConfig,
  getWorkerPool,
} from './worker-pool';
