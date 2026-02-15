/**
 * CORE - Point d'entrée du système multi-agents
 *
 * Architecture à 3 agents :
 * - VOX : Interface utilisateur
 * - MEMORY : Gestion mémoire, skills, learning loop
 * - BRAIN : Orchestrateur intelligent
 */

import { VoxAgent } from './vox';
import { MemoryAgent } from './memory';
import { BrainAgent } from './brain';
import { messageBus } from './message-bus';
import { EventEmitter } from 'events';
import { getTokenManager, TokenManager } from './token-manager';
import { getModelRouter, ModelRouter } from './models';
import { getWorkerPool } from './workers';

export interface CoreConfig {
  model?: string;
  debug?: boolean;
}

export class Core extends EventEmitter {
  private vox: VoxAgent;
  private memory: MemoryAgent;
  private brain: BrainAgent;
  private tokenManager: TokenManager;
  private modelRouter: ModelRouter;
  private startTime: Date = new Date();
  private isRunning = false;

  constructor(config: CoreConfig = {}) {
    super();

    // Initialiser le TokenManager (conscience des ressources)
    this.tokenManager = getTokenManager();
    this.modelRouter = getModelRouter();

    // Initialiser les agents
    this.vox = new VoxAgent(
      config.model ? { model: config.model } : undefined
    );
    this.memory = new MemoryAgent(
      config.model ? { model: config.model } : undefined
    );
    this.brain = new BrainAgent(
      config.model ? { model: config.model } : undefined
    );

    // Écouter les réponses pour les émettre vers l'extérieur
    this.setupEventForwarding();

    if (config.debug) {
      this.enableDebugLogging();
    }
  }

  /**
   * Démarrer le système
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      console.log('[Core] Système déjà démarré');
      return;
    }

    console.log('[Core] ═══════════════════════════════════════');
    console.log('[Core]        DÉMARRAGE DU SYSTÈME           ');
    console.log('[Core] ═══════════════════════════════════════');

    // Démarrer les agents
    this.memory.start(); // Memory d'abord pour charger l'état
    this.brain.start();
    this.vox.start();

    // Charger l'état persisté
    await this.memory.load();

    this.isRunning = true;

    // Initialiser le ModelRouter et détecter les modèles disponibles
    const modelRouter = getModelRouter();
    const availableModels = await modelRouter.detectAvailableModels();

    console.log('[Core] ✅ Système opérationnel');
    console.log('[Core]    - Vox: Interface utilisateur');
    console.log('[Core]    - Memory: Mémoire & Learning');
    console.log('[Core]    - Brain: Orchestration');
    console.log(`[Core]    - Modèles: ${availableModels.length} disponibles`);
    console.log(`[Core]    - TokenManager: Actif (limite: ${this.tokenManager.getQuickStats().remainingTokens.toLocaleString()} tokens/jour)`);
    console.log('[Core] ═══════════════════════════════════════');
  }

  /**
   * Arrêter le système
   */
  async stop(): Promise<void> {
    if (!this.isRunning) return;

    console.log('[Core] Arrêt du système...');

    // Persister l'état avant arrêt
    await this.memory.persist();

    // Arrêter les agents
    this.vox.stop();
    this.brain.stop();
    this.memory.stop();

    this.isRunning = false;

    console.log('[Core] Système arrêté');
  }

  /**
   * Envoyer un message utilisateur
   */
  async chat(message: string): Promise<void> {
    if (!this.isRunning) {
      throw new Error('Le système n\'est pas démarré. Appelez start() d\'abord.');
    }

    await this.vox.receiveUserInput(message);
  }

  /**
   * Stocker une information en mémoire
   */
  async remember(
    type: 'fact' | 'preference' | 'skill',
    content: string,
    tags?: string[]
  ): Promise<string> {
    return this.memory.store(type, content, { tags });
  }

  /**
   * Rechercher dans la mémoire
   */
  async recall(query: string, limit = 5): Promise<unknown[]> {
    return this.memory.search(query, { limit });
  }

  /**
   * Créer une tâche
   */
  createTask(
    title: string,
    description: string,
    priority?: number
  ): unknown {
    return this.memory.createTask(title, description, { priority });
  }

  /**
   * Enregistrer un feedback/correction
   */
  async correct(
    originalResponse: string,
    correction: string,
    feedback: string
  ): Promise<void> {
    await this.memory.recordLearning(
      'user_correction',
      'Correction utilisateur',
      originalResponse,
      feedback,
      correction
    );
  }

  /**
   * Obtenir les métriques du système
   * Inclut maintenant les stats de ressources (tokens, coûts)
   */
  getMetrics(): {
    brain: ReturnType<BrainAgent['getMetrics']>;
    conversation: { turns: number };
    memory: ReturnType<MemoryAgent['getStats']>;
    resources: {
      quick: ReturnType<TokenManager['getQuickStats']>;
      plan: ReturnType<TokenManager['getResourcePlan']>;
    };
  } {
    return {
      brain: this.brain.getMetrics(),
      conversation: {
        turns: this.vox.getConversationHistory().length,
      },
      memory: this.memory.getStats(),
      resources: {
        quick: this.tokenManager.getQuickStats(),
        plan: this.tokenManager.getResourcePlan(),
      },
    };
  }

  /**
   * Obtenir un rapport détaillé d'usage des ressources
   */
  getResourceReport(period: 'today' | 'week' | 'month' | 'all' = 'today'): ReturnType<TokenManager['getUsageReport']> {
    return this.tokenManager.getUsageReport(period);
  }

  /**
   * Configurer les limites quotidiennes de ressources
   */
  setResourceLimits(limits: { maxTokens?: number; maxCost?: number; warningThreshold?: number }): void {
    this.tokenManager.setDailyLimits(limits);
  }

  /**
   * Vérifier la santé du système (Règle 2: Neo ne s'éteint jamais)
   */
  getHealthStatus(): {
    isHealthy: boolean;
    agents: {
      vox: ReturnType<VoxAgent['getHealthStatus']>;
      memory: ReturnType<MemoryAgent['getHealthStatus']>;
      brain: ReturnType<BrainAgent['getHealthStatus']>;
    };
    unhealthyAgents: string[];
  } {
    const voxStatus = this.vox.getHealthStatus();
    const memoryStatus = this.memory.getHealthStatus();
    const brainStatus = this.brain.getHealthStatus();

    const unhealthyAgents: string[] = [];
    if (!voxStatus.isAlive) unhealthyAgents.push('vox');
    if (!memoryStatus.isAlive) unhealthyAgents.push('memory');
    if (!brainStatus.isAlive) unhealthyAgents.push('brain');

    return {
      isHealthy: unhealthyAgents.length === 0,
      agents: {
        vox: voxStatus,
        memory: memoryStatus,
        brain: brainStatus,
      },
      unhealthyAgents,
    };
  }

  /**
   * Tenter de récupérer un agent qui ne répond plus
   */
  async recoverAgent(agentName: 'vox' | 'memory' | 'brain'): Promise<boolean> {
    console.log(`[Core] ⚠️ Tentative de récupération de ${agentName}...`);

    try {
      const agent = this[agentName];
      agent.stop();

      // Attendre un peu
      await new Promise(resolve => setTimeout(resolve, 1000));

      agent.start();

      // Recharger l'état si c'est Memory
      if (agentName === 'memory') {
        await this.memory.load();
      }

      console.log(`[Core] ✅ ${agentName} récupéré avec succès`);
      return true;
    } catch (error) {
      console.error(`[Core] ❌ Échec de récupération de ${agentName}:`, error);
      return false;
    }
  }

  /**
   * Réinitialiser la conversation (pas la mémoire)
   */
  resetConversation(): void {
    this.vox.resetConversation();
  }

  /**
   * Obtenir les données pour le dashboard
   */
  getDashboardData(): {
    agents: Array<{
      name: string;
      role: string;
      isRunning: boolean;
      isAlive: boolean;
      lastHeartbeat: string;
      uptimeMs: number;
    }>;
    pool: ReturnType<typeof getWorkerPool>['getStats'] extends () => infer R ? R : never;
    models: Array<{
      id: string;
      name: string;
      provider: string;
      tier: string;
      isAvailable: boolean;
      tasksHandled: number;
    }>;
    tokens: {
      totalInputTokens: number;
      totalOutputTokens: number;
      totalCost: number;
      byModel: Record<string, { tokens: number; cost: number }>;
    };
    uptime: number;
  } {
    // Agents status
    const healthStatus = this.getHealthStatus();
    const agents = [
      {
        name: 'Vox',
        role: 'vox',
        isRunning: healthStatus.agents.vox.isRunning,
        isAlive: healthStatus.agents.vox.isAlive,
        lastHeartbeat: healthStatus.agents.vox.lastHeartbeat.toISOString(),
        uptimeMs: healthStatus.agents.vox.uptimeMs,
      },
      {
        name: 'Brain',
        role: 'brain',
        isRunning: healthStatus.agents.brain.isRunning,
        isAlive: healthStatus.agents.brain.isAlive,
        lastHeartbeat: healthStatus.agents.brain.lastHeartbeat.toISOString(),
        uptimeMs: healthStatus.agents.brain.uptimeMs,
      },
      {
        name: 'Memory',
        role: 'memory',
        isRunning: healthStatus.agents.memory.isRunning,
        isAlive: healthStatus.agents.memory.isAlive,
        lastHeartbeat: healthStatus.agents.memory.lastHeartbeat.toISOString(),
        uptimeMs: healthStatus.agents.memory.uptimeMs,
      },
    ];

    // Worker pool stats
    const workerPool = getWorkerPool();
    const pool = workerPool.getStats();

    // Models info
    const availableModels = this.modelRouter.getAvailableModels();
    const models = availableModels.map(m => ({
      id: m.id,
      name: m.name,
      provider: m.provider,
      tier: m.tier,
      isAvailable: true,
      tasksHandled: 0, // TODO: track this in ModelRouter
    }));

    // Token stats
    const quickStats = this.tokenManager.getQuickStats();
    const tokens = {
      totalInputTokens: quickStats.todayTokens,
      totalOutputTokens: Math.floor(quickStats.todayTokens * 0.3), // Estimation
      totalCost: quickStats.todayCost,
      byModel: {} as Record<string, { tokens: number; cost: number }>,
    };

    // Uptime
    const uptime = Date.now() - this.startTime.getTime();

    return {
      agents,
      pool,
      models,
      tokens,
      uptime,
    };
  }

  /**
   * Sauvegarder la mémoire
   */
  backupMemory(): string {
    return this.memory.backup();
  }

  /**
   * Vider le cache
   */
  clearCache(): void {
    this.memory.clearCache();
  }

  /**
   * Configurer le forwarding des événements
   */
  private setupEventForwarding(): void {
    // Track emitted messages to avoid duplicates
    const emittedMessages = new Set<string>();

    const emitOnce = (message: string) => {
      const hash = message.substring(0, 100);
      if (emittedMessages.has(hash)) return;
      emittedMessages.add(hash);
      // Clean old entries after 5 seconds
      setTimeout(() => emittedMessages.delete(hash), 5000);
      this.emit('response', message);
    };

    // Écouter les broadcasts (Vox envoie via broadcast)
    messageBus.on('broadcast', (message) => {
      if (message.type === 'response_ready') {
        const payload = message.payload as { target?: string; message?: string };
        if (payload.target === 'user' && payload.message) {
          emitOnce(payload.message);
        }
      }
    });

    // Écouter les messages directs pour typing et erreurs
    messageBus.on('message', (message) => {
      if (message.type === 'typing') {
        this.emit('typing');
      }
      if (message.type === 'error') {
        const payload = message.payload as { message?: string };
        this.emit('error', { message: payload.message || 'Unknown error' });
      }
    });
  }

  /**
   * Activer le logging de debug
   */
  private enableDebugLogging(): void {
    messageBus.on('message', (message) => {
      console.log(
        `[DEBUG] ${message.from} → ${message.to}: ${message.type}`,
        JSON.stringify(message.payload).substring(0, 100)
      );
    });
  }
}

// Exports
export { VoxAgent } from './vox';
export { MemoryAgent } from './memory';
export { BrainAgent } from './brain';
export { messageBus } from './message-bus';
export { TokenManager, getTokenManager } from './token-manager';
export { ModelRouter, getModelRouter } from './models';
export * from './types';
