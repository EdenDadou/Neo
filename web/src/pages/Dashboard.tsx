/**
 * Dashboard - Vue d'ensemble des agents et modèles Neo
 *
 * Affiche en temps réel:
 * - État des agents (Vox, Brain, Memory)
 * - Modèles utilisés
 * - Statistiques du WorkerPool
 * - Métriques de tokens/coûts
 */

import { useState, useEffect } from 'react';
import { api } from '../lib/api';

// ===========================================================================
// TYPES
// ===========================================================================

interface AgentStatus {
  name: string;
  role: string;
  isRunning: boolean;
  isAlive: boolean;
  lastHeartbeat: string;
  uptimeMs: number;
}

interface WorkerStats {
  id: string;
  name: string;
  status: 'idle' | 'working' | 'completed' | 'failed' | 'terminated';
  completedTasks: number;
  failedTasks: number;
  successRate: number;
  uptimeMs: number;
}

interface PoolStats {
  isRunning: boolean;
  totalWorkers: number;
  availableWorkers: number;
  busyWorkers: number;
  queueLength: number;
  totalTasksProcessed: number;
  totalTasksFailed: number;
  successRate: number;
  workers: WorkerStats[];
}

interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  tier: string;
  isAvailable: boolean;
  tasksHandled: number;
}

interface TokenStats {
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCost: number;
  byModel: Record<string, { tokens: number; cost: number }>;
}

interface DashboardData {
  agents: AgentStatus[];
  pool: PoolStats;
  models: ModelInfo[];
  tokens: TokenStats;
  uptime: number;
}

// ===========================================================================
// COMPONENTS
// ===========================================================================

function StatusBadge({ status }: { status: 'online' | 'offline' | 'warning' }) {
  const colors = {
    online: 'bg-green-500',
    offline: 'bg-red-500',
    warning: 'bg-yellow-500',
  };

  return (
    <span className={`inline-block w-3 h-3 rounded-full ${colors[status]} animate-pulse`} />
  );
}

function formatUptime(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}j ${hours % 24}h`;
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
}

function formatCost(cost: number): string {
  if (cost < 0.01) return `$${(cost * 100).toFixed(2)}c`;
  return `$${cost.toFixed(4)}`;
}

function AgentCard({ agent }: { agent: AgentStatus }) {
  const status = agent.isRunning && agent.isAlive ? 'online' : agent.isRunning ? 'warning' : 'offline';

  const icons: Record<string, string> = {
    vox: '🎙️',
    brain: '🧠',
    memory: '💾',
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl">{icons[agent.role] || '🤖'}</span>
          <h3 className="text-lg font-semibold text-white">{agent.name}</h3>
        </div>
        <StatusBadge status={status} />
      </div>
      <div className="space-y-1 text-sm text-gray-400">
        <p>Role: <span className="text-gray-300">{agent.role.toUpperCase()}</span></p>
        <p>Uptime: <span className="text-gray-300">{formatUptime(agent.uptimeMs)}</span></p>
        <p>Heartbeat: <span className="text-gray-300">{new Date(agent.lastHeartbeat).toLocaleTimeString()}</span></p>
      </div>
    </div>
  );
}

function WorkerPoolCard({ pool }: { pool: PoolStats }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <span>⚙️</span> Worker Pool
        </h3>
        <StatusBadge status={pool.isRunning ? 'online' : 'offline'} />
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-cyan-400">{pool.totalWorkers}</div>
          <div className="text-xs text-gray-500">Workers</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-400">{pool.availableWorkers}</div>
          <div className="text-xs text-gray-500">Disponibles</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-yellow-400">{pool.busyWorkers}</div>
          <div className="text-xs text-gray-500">Occupés</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-400">{pool.queueLength}</div>
          <div className="text-xs text-gray-500">En queue</div>
        </div>
      </div>

      <div className="border-t border-gray-700 pt-3 mt-3">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Tâches traitées:</span>
          <span className="text-white">{pool.totalTasksProcessed.toLocaleString()}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Taux de succès:</span>
          <span className={pool.successRate > 0.95 ? 'text-green-400' : 'text-yellow-400'}>
            {(pool.successRate * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Worker list */}
      {pool.workers.length > 0 && (
        <div className="border-t border-gray-700 pt-3 mt-3">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Workers actifs</h4>
          <div className="space-y-1">
            {pool.workers.map((worker) => (
              <div key={worker.id} className="flex items-center justify-between text-xs">
                <span className="text-gray-300">{worker.name}</span>
                <span className={`px-2 py-0.5 rounded ${
                  worker.status === 'idle' ? 'bg-green-900 text-green-300' :
                  worker.status === 'working' ? 'bg-yellow-900 text-yellow-300' :
                  'bg-gray-700 text-gray-400'
                }`}>
                  {worker.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ModelsCard({ models }: { models: ModelInfo[] }) {
  const groupedByProvider = models.reduce((acc, model) => {
    if (!acc[model.provider]) acc[model.provider] = [];
    acc[model.provider].push(model);
    return acc;
  }, {} as Record<string, ModelInfo[]>);

  const providerIcons: Record<string, string> = {
    anthropic: '🟣',
    groq: '🟢',
    ollama: '🦙',
    deepseek: '🔵',
    mistral: '🟠',
    huggingface: '🤗',
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
        <span>🤖</span> Modèles disponibles
      </h3>

      <div className="space-y-4">
        {Object.entries(groupedByProvider).map(([provider, providerModels]) => (
          <div key={provider}>
            <div className="flex items-center gap-2 mb-2">
              <span>{providerIcons[provider] || '⚪'}</span>
              <span className="text-sm font-medium text-gray-300 uppercase">{provider}</span>
            </div>
            <div className="space-y-1 pl-6">
              {providerModels.map((model) => (
                <div key={model.id} className="flex items-center justify-between text-sm">
                  <span className={model.isAvailable ? 'text-gray-300' : 'text-gray-500 line-through'}>
                    {model.name}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      model.tier === 'free' ? 'bg-green-900 text-green-300' :
                      model.tier === 'cheap' ? 'bg-blue-900 text-blue-300' :
                      'bg-purple-900 text-purple-300'
                    }`}>
                      {model.tier}
                    </span>
                    {model.isAvailable && (
                      <StatusBadge status="online" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TokensCard({ tokens }: { tokens: TokenStats }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
        <span>📊</span> Utilisation des tokens
      </h3>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center">
          <div className="text-xl font-bold text-blue-400">
            {(tokens.totalInputTokens / 1000).toFixed(1)}K
          </div>
          <div className="text-xs text-gray-500">Input</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-bold text-green-400">
            {(tokens.totalOutputTokens / 1000).toFixed(1)}K
          </div>
          <div className="text-xs text-gray-500">Output</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-bold text-yellow-400">
            {formatCost(tokens.totalCost)}
          </div>
          <div className="text-xs text-gray-500">Coût total</div>
        </div>
      </div>

      {Object.keys(tokens.byModel).length > 0 && (
        <div className="border-t border-gray-700 pt-3 mt-3">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Par modèle</h4>
          <div className="space-y-1">
            {Object.entries(tokens.byModel).map(([model, stats]) => (
              <div key={model} className="flex items-center justify-between text-xs">
                <span className="text-gray-300 truncate flex-1">{model}</span>
                <span className="text-gray-400 ml-2">
                  {(stats.tokens / 1000).toFixed(1)}K • {formatCost(stats.cost)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ===========================================================================
// MAIN COMPONENT
// ===========================================================================

export function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchData = async () => {
    try {
      const response = await api.get<DashboardData>('/api/dashboard');
      setData(response);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur de chargement');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();

    // Refresh toutes les 5 secondes
    const interval = setInterval(fetchData, 5000);

    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cyan-500" />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="bg-red-900/20 border border-red-500 rounded-lg p-4 text-red-400">
        <h3 className="font-semibold mb-2">Erreur de connexion</h3>
        <p>{error}</p>
        <button
          onClick={fetchData}
          className="mt-3 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded"
        >
          Réessayer
        </button>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <span className="text-3xl">🧠</span>
            Neo Dashboard
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            Uptime: {formatUptime(data.uptime)}
          </p>
        </div>
        <div className="text-right text-sm text-gray-500">
          {lastUpdate && (
            <p>Dernière MAJ: {lastUpdate.toLocaleTimeString()}</p>
          )}
          {error && (
            <p className="text-yellow-500">⚠️ Connexion instable</p>
          )}
        </div>
      </div>

      {/* Agents Grid */}
      <div>
        <h2 className="text-lg font-semibold text-gray-300 mb-3">Agents</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {data.agents.map((agent) => (
            <AgentCard key={agent.role} agent={agent} />
          ))}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <WorkerPoolCard pool={data.pool} />
        <ModelsCard models={data.models} />
        <TokensCard tokens={data.tokens} />
      </div>

      {/* Quick Actions */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Actions rapides</h3>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => api.post('/api/system/restart')}
            className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded flex items-center gap-2"
          >
            🔄 Redémarrer
          </button>
          <button
            onClick={() => api.post('/api/memory/backup')}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded flex items-center gap-2"
          >
            💾 Sauvegarder mémoire
          </button>
          <button
            onClick={() => api.post('/api/cache/clear')}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded flex items-center gap-2"
          >
            🧹 Vider cache
          </button>
        </div>
      </div>
    </div>
  );
}
