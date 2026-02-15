/**
 * TOKEN MANAGER - Gestion centralisée des ressources de Neo
 *
 * Règle: Neo doit être conscient de ses ressources et faire aussi bien pour moins cher
 *
 * Responsabilités:
 * 1. Tracker l'usage des tokens par agent et par modèle
 * 2. Appliquer les limites quotidiennes
 * 3. Reporter les coûts en temps réel
 * 4. Suggérer des optimisations
 * 5. Permettre à Neo de "planifier" ses ressources
 */

import { ModelRouter, ModelInfo, getModelRouter } from './models';
import { PersistenceLayer } from './memory/persistence';

// ===========================================================================
// TYPES
// ===========================================================================

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cost: number;
  model: string;
  provider: string;
  timestamp: Date;
}

export interface AgentUsageStats {
  agentName: string;
  totalCalls: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCost: number;
  averageTokensPerCall: number;
  lastCall: Date | null;
}

export interface DailyLimits {
  maxTokens: number;
  maxCost: number;       // En USD
  warningThreshold: number; // % avant alerte (ex: 0.8 = 80%)
}

export interface ResourcePlan {
  remainingTokens: number;
  remainingBudget: number;
  hoursUntilReset: number;
  recommendedTier: 'free' | 'cheap' | 'standard' | 'premium';
  warnings: string[];
  canAfford: {
    simpleChat: number;    // Nombre estimé de conversations simples
    complexTask: number;   // Nombre estimé de tâches complexes
    synthesis: number;     // Nombre estimé de cycles de synthèse
  };
}

export interface UsageReport {
  period: 'today' | 'week' | 'month' | 'all';
  startDate: Date;
  endDate: Date;
  totalTokens: number;
  totalCost: number;
  byAgent: Record<string, AgentUsageStats>;
  byModel: Record<string, { calls: number; tokens: number; cost: number }>;
  byTier: Record<string, { calls: number; tokens: number; cost: number }>;
  trends: {
    averageDailyCost: number;
    averageDailyTokens: number;
    peakHour: number;
    mostUsedModel: string;
    mostExpensiveAgent: string;
  };
}

// ===========================================================================
// TOKEN MANAGER
// ===========================================================================

export class TokenManager {
  private modelRouter: ModelRouter;
  private persistence: PersistenceLayer | null = null;

  // Usage tracking in-memory (aussi sauvegardé en DB)
  private usageLog: TokenUsage[] = [];
  private agentStats: Map<string, AgentUsageStats> = new Map();

  // Limites
  private dailyLimits: DailyLimits = {
    maxTokens: 1_000_000,      // 1M tokens/jour par défaut
    maxCost: 10.0,             // $10/jour par défaut
    warningThreshold: 0.8,
  };

  // Tracking du jour courant
  private todayUsage = {
    date: new Date().toISOString().split('T')[0],
    tokens: 0,
    cost: 0,
    calls: 0,
  };

  // Estimations de coûts par type d'opération (en tokens)
  private static readonly OPERATION_ESTIMATES = {
    simpleChat: { input: 500, output: 300 },
    complexTask: { input: 2000, output: 1500 },
    synthesis: { input: 3000, output: 500 },
    contextEnrichment: { input: 800, output: 400 },
    factCheck: { input: 600, output: 200 },
  };

  constructor(persistence?: PersistenceLayer) {
    this.modelRouter = getModelRouter();
    this.persistence = persistence || null;

    // Charger les stats depuis la persistence si disponible
    this.loadFromPersistence();

    // Reset quotidien
    this.scheduleDailyReset();
  }

  // ===========================================================================
  // TRACKING
  // ===========================================================================

  /**
   * Enregistrer une utilisation de tokens
   */
  recordUsage(
    agentName: string,
    usage: {
      inputTokens: number;
      outputTokens: number;
      model: string;
      provider: string;
      cost?: number;
    }
  ): void {
    const totalTokens = usage.inputTokens + usage.outputTokens;
    const modelInfo = this.getModelInfo(usage.model);

    // Calculer le coût si non fourni
    const cost = usage.cost ?? this.calculateCost(totalTokens, modelInfo);

    const record: TokenUsage = {
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      totalTokens,
      cost,
      model: usage.model,
      provider: usage.provider,
      timestamp: new Date(),
    };

    // Ajouter au log
    this.usageLog.push(record);

    // Garder seulement les 1000 derniers pour la mémoire
    if (this.usageLog.length > 1000) {
      this.usageLog = this.usageLog.slice(-1000);
    }

    // Mettre à jour les stats par agent
    this.updateAgentStats(agentName, record);

    // Mettre à jour le compteur quotidien
    this.updateDailyUsage(record);

    // Vérifier les limites
    this.checkLimits();

    // Sauvegarder périodiquement
    this.saveToPersistence();

    console.log(`[TokenManager] ${agentName}: ${totalTokens} tokens ($${cost.toFixed(4)})`);
  }

  /**
   * Mettre à jour les stats d'un agent
   */
  private updateAgentStats(agentName: string, usage: TokenUsage): void {
    const existing = this.agentStats.get(agentName) || {
      agentName,
      totalCalls: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      totalCost: 0,
      averageTokensPerCall: 0,
      lastCall: null,
    };

    existing.totalCalls++;
    existing.totalInputTokens += usage.inputTokens;
    existing.totalOutputTokens += usage.outputTokens;
    existing.totalCost += usage.cost;
    existing.averageTokensPerCall =
      (existing.totalInputTokens + existing.totalOutputTokens) / existing.totalCalls;
    existing.lastCall = usage.timestamp;

    this.agentStats.set(agentName, existing);
  }

  /**
   * Mettre à jour l'usage quotidien
   */
  private updateDailyUsage(usage: TokenUsage): void {
    const today = new Date().toISOString().split('T')[0];

    // Reset si nouveau jour
    if (this.todayUsage.date !== today) {
      this.resetDailyUsage();
    }

    this.todayUsage.tokens += usage.totalTokens;
    this.todayUsage.cost += usage.cost;
    this.todayUsage.calls++;
  }

  /**
   * Reset quotidien
   */
  private resetDailyUsage(): void {
    console.log('[TokenManager] Reset quotidien');
    this.todayUsage = {
      date: new Date().toISOString().split('T')[0],
      tokens: 0,
      cost: 0,
      calls: 0,
    };
  }

  /**
   * Programmer le reset quotidien
   */
  private scheduleDailyReset(): void {
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setDate(tomorrow.getDate() + 1);
    tomorrow.setHours(0, 0, 0, 0);

    const msUntilMidnight = tomorrow.getTime() - now.getTime();

    setTimeout(() => {
      this.resetDailyUsage();
      this.scheduleDailyReset(); // Replanifier pour le lendemain
    }, msUntilMidnight);
  }

  // ===========================================================================
  // LIMITES ET ALERTES
  // ===========================================================================

  /**
   * Configurer les limites quotidiennes
   */
  setDailyLimits(limits: Partial<DailyLimits>): void {
    this.dailyLimits = { ...this.dailyLimits, ...limits };
    console.log(`[TokenManager] Limites mises à jour: ${this.dailyLimits.maxTokens} tokens, $${this.dailyLimits.maxCost}/jour`);
  }

  /**
   * Vérifier les limites et émettre des alertes
   */
  private checkLimits(): void {
    const warnings: string[] = [];

    // Vérifier tokens
    const tokenUsagePercent = this.todayUsage.tokens / this.dailyLimits.maxTokens;
    if (tokenUsagePercent >= 1.0) {
      warnings.push(`LIMITE TOKENS ATTEINTE: ${this.todayUsage.tokens}/${this.dailyLimits.maxTokens}`);
    } else if (tokenUsagePercent >= this.dailyLimits.warningThreshold) {
      warnings.push(`Alerte tokens: ${Math.round(tokenUsagePercent * 100)}% utilisés`);
    }

    // Vérifier coût
    const costUsagePercent = this.todayUsage.cost / this.dailyLimits.maxCost;
    if (costUsagePercent >= 1.0) {
      warnings.push(`LIMITE BUDGET ATTEINTE: $${this.todayUsage.cost.toFixed(2)}/$${this.dailyLimits.maxCost}`);
    } else if (costUsagePercent >= this.dailyLimits.warningThreshold) {
      warnings.push(`Alerte budget: ${Math.round(costUsagePercent * 100)}% utilisé ($${this.todayUsage.cost.toFixed(2)})`);
    }

    // Logger les warnings
    for (const warning of warnings) {
      console.warn(`[TokenManager] ⚠️ ${warning}`);
    }
  }

  /**
   * Vérifier si une opération est autorisée (dans les limites)
   */
  canAfford(estimatedTokens: number, estimatedCost?: number): {
    allowed: boolean;
    reason?: string;
    suggestedAlternative?: string;
  } {
    const projectedTokens = this.todayUsage.tokens + estimatedTokens;
    const projectedCost = this.todayUsage.cost + (estimatedCost || 0);

    if (projectedTokens > this.dailyLimits.maxTokens) {
      return {
        allowed: false,
        reason: `Dépasserait la limite de tokens (${projectedTokens}/${this.dailyLimits.maxTokens})`,
        suggestedAlternative: 'Utiliser un modèle plus léger ou attendre demain',
      };
    }

    if (projectedCost > this.dailyLimits.maxCost) {
      return {
        allowed: false,
        reason: `Dépasserait le budget ($${projectedCost.toFixed(2)}/$${this.dailyLimits.maxCost})`,
        suggestedAlternative: 'Utiliser un modèle gratuit ou moins cher',
      };
    }

    return { allowed: true };
  }

  // ===========================================================================
  // PLANIFICATION DES RESSOURCES
  // ===========================================================================

  /**
   * Obtenir un plan de ressources pour le reste de la journée
   */
  getResourcePlan(): ResourcePlan {
    const now = new Date();
    const endOfDay = new Date(now);
    endOfDay.setHours(23, 59, 59, 999);
    const hoursRemaining = (endOfDay.getTime() - now.getTime()) / (1000 * 60 * 60);

    const remainingTokens = Math.max(0, this.dailyLimits.maxTokens - this.todayUsage.tokens);
    const remainingBudget = Math.max(0, this.dailyLimits.maxCost - this.todayUsage.cost);

    // Calculer combien d'opérations on peut encore faire
    const estimates = TokenManager.OPERATION_ESTIMATES;
    const avgTokensPerSimpleChat = estimates.simpleChat.input + estimates.simpleChat.output;
    const avgTokensPerComplexTask = estimates.complexTask.input + estimates.complexTask.output;
    const avgTokensPerSynthesis = estimates.synthesis.input + estimates.synthesis.output;

    // Recommander un tier basé sur le budget restant
    let recommendedTier: 'free' | 'cheap' | 'standard' | 'premium';
    if (remainingBudget < 0.5) {
      recommendedTier = 'free';
    } else if (remainingBudget < 2) {
      recommendedTier = 'cheap';
    } else if (remainingBudget < 5) {
      recommendedTier = 'standard';
    } else {
      recommendedTier = 'premium';
    }

    const warnings: string[] = [];

    if (remainingTokens < 50000) {
      warnings.push('Tokens limités - privilégier les modèles gratuits');
    }
    if (remainingBudget < 1) {
      warnings.push('Budget limité - utiliser Ollama ou Groq (gratuits)');
    }
    if (hoursRemaining < 2 && remainingBudget > this.dailyLimits.maxCost * 0.5) {
      warnings.push('Budget sous-utilisé - possibilité d\'utiliser des modèles premium');
    }

    return {
      remainingTokens,
      remainingBudget,
      hoursUntilReset: hoursRemaining,
      recommendedTier,
      warnings,
      canAfford: {
        simpleChat: Math.floor(remainingTokens / avgTokensPerSimpleChat),
        complexTask: Math.floor(remainingTokens / avgTokensPerComplexTask),
        synthesis: Math.floor(remainingTokens / avgTokensPerSynthesis),
      },
    };
  }

  /**
   * Suggérer le meilleur modèle pour une tâche en fonction des ressources
   */
  suggestModelForTask(
    task: 'simple_chat' | 'code' | 'reasoning' | 'creative' | 'factual'
  ): { model: ModelInfo | null; reason: string } {
    const plan = this.getResourcePlan();

    // Obtenir les estimations pour ce type de tâche
    const estimates = this.getTaskEstimates(task);
    const affordCheck = this.canAfford(estimates.totalTokens);

    if (!affordCheck.allowed) {
      // Chercher un modèle gratuit
      const freeModel = this.modelRouter.selectModel({
        task,
        maxTier: 'free',
        preferSpeed: true,
      });

      return {
        model: freeModel,
        reason: `Budget limité - ${affordCheck.reason}. Modèle gratuit recommandé.`,
      };
    }

    // Sélectionner selon le tier recommandé
    const model = this.modelRouter.selectModel({
      task,
      maxTier: plan.recommendedTier,
      preferSpeed: task === 'simple_chat',
    });

    return {
      model,
      reason: `Tier ${plan.recommendedTier} recommandé (reste $${plan.remainingBudget.toFixed(2)})`,
    };
  }

  /**
   * Obtenir les estimations de tokens pour un type de tâche
   */
  private getTaskEstimates(task: string): { inputTokens: number; outputTokens: number; totalTokens: number } {
    const estimates = TokenManager.OPERATION_ESTIMATES;

    switch (task) {
      case 'simple_chat':
        return {
          inputTokens: estimates.simpleChat.input,
          outputTokens: estimates.simpleChat.output,
          totalTokens: estimates.simpleChat.input + estimates.simpleChat.output,
        };
      case 'code':
      case 'reasoning':
      case 'creative':
        return {
          inputTokens: estimates.complexTask.input,
          outputTokens: estimates.complexTask.output,
          totalTokens: estimates.complexTask.input + estimates.complexTask.output,
        };
      default:
        return {
          inputTokens: estimates.simpleChat.input,
          outputTokens: estimates.simpleChat.output,
          totalTokens: estimates.simpleChat.input + estimates.simpleChat.output,
        };
    }
  }

  // ===========================================================================
  // REPORTING
  // ===========================================================================

  /**
   * Générer un rapport d'usage
   */
  getUsageReport(period: 'today' | 'week' | 'month' | 'all' = 'today'): UsageReport {
    const now = new Date();
    let startDate: Date;

    switch (period) {
      case 'today':
        startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        break;
      case 'week':
        startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        break;
      case 'month':
        startDate = new Date(now.getFullYear(), now.getMonth(), 1);
        break;
      case 'all':
        startDate = new Date(0);
        break;
    }

    // Filtrer les logs pour la période
    const periodLogs = this.usageLog.filter(log => log.timestamp >= startDate);

    // Calculer les totaux
    let totalTokens = 0;
    let totalCost = 0;
    const byModel: Record<string, { calls: number; tokens: number; cost: number }> = {};
    const byTier: Record<string, { calls: number; tokens: number; cost: number }> = {};
    const hourCounts: number[] = new Array(24).fill(0);

    for (const log of periodLogs) {
      totalTokens += log.totalTokens;
      totalCost += log.cost;

      // Par modèle
      if (!byModel[log.model]) {
        byModel[log.model] = { calls: 0, tokens: 0, cost: 0 };
      }
      byModel[log.model].calls++;
      byModel[log.model].tokens += log.totalTokens;
      byModel[log.model].cost += log.cost;

      // Par tier
      const modelInfo = this.getModelInfo(log.model);
      const tier = modelInfo?.tier || 'unknown';
      if (!byTier[tier]) {
        byTier[tier] = { calls: 0, tokens: 0, cost: 0 };
      }
      byTier[tier].calls++;
      byTier[tier].tokens += log.totalTokens;
      byTier[tier].cost += log.cost;

      // Par heure
      hourCounts[log.timestamp.getHours()]++;
    }

    // Trouver le modèle le plus utilisé
    let mostUsedModel = '';
    let maxCalls = 0;
    for (const [model, stats] of Object.entries(byModel)) {
      if (stats.calls > maxCalls) {
        maxCalls = stats.calls;
        mostUsedModel = model;
      }
    }

    // Trouver l'agent le plus coûteux
    let mostExpensiveAgent = '';
    let maxAgentCost = 0;
    for (const [agent, stats] of this.agentStats) {
      if (stats.totalCost > maxAgentCost) {
        maxAgentCost = stats.totalCost;
        mostExpensiveAgent = agent;
      }
    }

    // Calculer les moyennes
    const daysDiff = Math.max(1, Math.ceil((now.getTime() - startDate.getTime()) / (24 * 60 * 60 * 1000)));

    return {
      period,
      startDate,
      endDate: now,
      totalTokens,
      totalCost,
      byAgent: Object.fromEntries(this.agentStats),
      byModel,
      byTier,
      trends: {
        averageDailyCost: totalCost / daysDiff,
        averageDailyTokens: totalTokens / daysDiff,
        peakHour: hourCounts.indexOf(Math.max(...hourCounts)),
        mostUsedModel,
        mostExpensiveAgent,
      },
    };
  }

  /**
   * Obtenir un résumé rapide pour le dashboard
   */
  getQuickStats(): {
    todayTokens: number;
    todayCost: number;
    todayCalls: number;
    remainingBudget: number;
    remainingTokens: number;
    recommendedTier: string;
  } {
    const plan = this.getResourcePlan();

    return {
      todayTokens: this.todayUsage.tokens,
      todayCost: this.todayUsage.cost,
      todayCalls: this.todayUsage.calls,
      remainingBudget: plan.remainingBudget,
      remainingTokens: plan.remainingTokens,
      recommendedTier: plan.recommendedTier,
    };
  }

  // ===========================================================================
  // HELPERS
  // ===========================================================================

  /**
   * Obtenir les infos d'un modèle
   */
  private getModelInfo(modelId: string): ModelInfo | undefined {
    return this.modelRouter.getAvailableModels().find(m => m.id === modelId);
  }

  /**
   * Calculer le coût pour un nombre de tokens
   */
  private calculateCost(tokens: number, modelInfo?: ModelInfo): number {
    if (!modelInfo) return 0;
    return (tokens / 1000) * modelInfo.costPer1kTokens;
  }

  // ===========================================================================
  // PERSISTENCE
  // ===========================================================================

  /**
   * Charger les stats depuis la persistence
   */
  private loadFromPersistence(): void {
    if (!this.persistence) return;

    try {
      const savedStats = this.persistence.getPersonality('token_stats');
      if (savedStats) {
        const parsed = JSON.parse(savedStats);

        // Restaurer les stats par agent
        if (parsed.agentStats) {
          this.agentStats = new Map(Object.entries(parsed.agentStats));
        }

        // Restaurer l'usage quotidien si c'est le même jour
        if (parsed.todayUsage && parsed.todayUsage.date === new Date().toISOString().split('T')[0]) {
          this.todayUsage = parsed.todayUsage;
        }

        console.log('[TokenManager] Stats restaurées depuis la persistence');
      }
    } catch (error) {
      console.error('[TokenManager] Erreur chargement stats:', error);
    }
  }

  /**
   * Sauvegarder les stats dans la persistence
   */
  private saveToPersistence(): void {
    if (!this.persistence) return;

    try {
      const toSave = {
        agentStats: Object.fromEntries(this.agentStats),
        todayUsage: this.todayUsage,
        savedAt: new Date().toISOString(),
      };

      this.persistence.setPersonality('token_stats', JSON.stringify(toSave));
    } catch (error) {
      console.error('[TokenManager] Erreur sauvegarde stats:', error);
    }
  }

  /**
   * Définir la persistence (peut être appelé après construction)
   */
  setPersistence(persistence: PersistenceLayer): void {
    this.persistence = persistence;
    this.loadFromPersistence();
  }
}

// ===========================================================================
// SINGLETON
// ===========================================================================

let tokenManagerInstance: TokenManager | null = null;

export function getTokenManager(persistence?: PersistenceLayer): TokenManager {
  if (!tokenManagerInstance) {
    tokenManagerInstance = new TokenManager(persistence);
  } else if (persistence && !tokenManagerInstance['persistence']) {
    tokenManagerInstance.setPersistence(persistence);
  }
  return tokenManagerInstance;
}
