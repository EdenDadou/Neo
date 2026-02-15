/**
 * WorkerAgent - Agent d'exécution des tâches
 *
 * Les Workers sont spawnnés par Brain pour exécuter des tâches spécifiques:
 * - Appels LLM (thinking, analysis)
 * - Exécution de skills
 * - Tâches de code/recherche
 * - Opérations longues
 *
 * Brain reste libre d'orchestrer pendant que les Workers bossent
 */

import { randomUUID } from 'crypto';
import { EventEmitter } from 'events';
import { getModelRouter, ModelRouter } from '../models';
import { getTokenManager, TokenManager } from '../token-manager';

// ===========================================================================
// TYPES
// ===========================================================================

export type WorkerTaskType =
  | 'llm_call'        // Appel LLM simple
  | 'llm_reasoning'   // Appel LLM pour raisonnement complexe
  | 'skill_execute'   // Exécution d'un skill
  | 'web_search'      // Recherche web
  | 'code_analysis'   // Analyse de code
  | 'custom';         // Tâche personnalisée

export interface WorkerTask {
  id: string;
  type: WorkerTaskType;
  priority: 'low' | 'normal' | 'high' | 'critical';
  payload: unknown;
  timeout: number;  // ms
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
}

export interface WorkerResult {
  taskId: string;
  success: boolean;
  output?: unknown;
  error?: {
    message: string;
    code: string;
    recoverable: boolean;
  };
  metrics: {
    executionTimeMs: number;
    tokensUsed?: number;
    cost?: number;
  };
}

export type WorkerStatus = 'idle' | 'working' | 'completed' | 'failed' | 'terminated';

// ===========================================================================
// WORKER AGENT
// ===========================================================================

export class WorkerAgent extends EventEmitter {
  readonly id: string;
  readonly name: string;
  private status: WorkerStatus = 'idle';
  private currentTask: WorkerTask | null = null;
  private modelRouter: ModelRouter;
  private tokenManager: TokenManager;
  private createdAt: Date;
  private completedTasks = 0;
  private failedTasks = 0;

  constructor(name?: string) {
    super();
    this.id = randomUUID();
    this.name = name || `Worker-${this.id.slice(0, 8)}`;
    this.modelRouter = getModelRouter();
    this.tokenManager = getTokenManager();
    this.createdAt = new Date();

    console.log(`[${this.name}] 🔧 Worker créé`);
  }

  // ===========================================================================
  // EXECUTION
  // ===========================================================================

  /**
   * Exécuter une tâche
   * Retourne une Promise qui se résout quand la tâche est terminée
   */
  async execute(task: WorkerTask): Promise<WorkerResult> {
    if (this.status === 'working') {
      throw new Error(`Worker ${this.name} déjà occupé`);
    }

    this.status = 'working';
    this.currentTask = task;
    task.startedAt = new Date();

    console.log(`[${this.name}] ⚡ Début tâche: ${task.type} (${task.id.slice(0, 8)})`);
    this.emit('task_started', { workerId: this.id, task });

    const startTime = Date.now();

    try {
      // Timeout wrapper
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => {
          reject(new Error(`Timeout après ${task.timeout}ms`));
        }, task.timeout);
      });

      // Exécution selon le type
      const executionPromise = this.executeTask(task);

      const output = await Promise.race([executionPromise, timeoutPromise]);

      const executionTimeMs = Date.now() - startTime;
      task.completedAt = new Date();
      this.status = 'completed';
      this.completedTasks++;
      this.currentTask = null;

      const result: WorkerResult = {
        taskId: task.id,
        success: true,
        output,
        metrics: {
          executionTimeMs,
          tokensUsed: (output as { tokensUsed?: number })?.tokensUsed,
          cost: (output as { cost?: number })?.cost,
        },
      };

      console.log(`[${this.name}] ✅ Tâche terminée en ${executionTimeMs}ms`);
      this.emit('task_completed', { workerId: this.id, result });

      // Revenir à idle pour la prochaine tâche
      this.status = 'idle';

      return result;
    } catch (error) {
      const executionTimeMs = Date.now() - startTime;
      this.status = 'failed';
      this.failedTasks++;
      this.currentTask = null;

      const errorMessage = error instanceof Error ? error.message : String(error);
      const isTimeout = errorMessage.includes('Timeout');

      const result: WorkerResult = {
        taskId: task.id,
        success: false,
        error: {
          message: errorMessage,
          code: isTimeout ? 'TIMEOUT' : 'EXECUTION_ERROR',
          recoverable: !isTimeout,
        },
        metrics: {
          executionTimeMs,
        },
      };

      console.log(`[${this.name}] ❌ Tâche échouée: ${errorMessage}`);
      this.emit('task_failed', { workerId: this.id, result });

      // Revenir à idle malgré l'échec
      this.status = 'idle';

      return result;
    }
  }

  /**
   * Dispatcher l'exécution selon le type de tâche
   */
  private async executeTask(task: WorkerTask): Promise<unknown> {
    switch (task.type) {
      case 'llm_call':
        return this.executeLLMCall(task.payload as LLMCallPayload);

      case 'llm_reasoning':
        return this.executeLLMReasoning(task.payload as LLMReasoningPayload);

      case 'skill_execute':
        return this.executeSkill(task.payload as SkillExecutePayload);

      case 'web_search':
        return this.executeWebSearch(task.payload as WebSearchPayload);

      case 'code_analysis':
        return this.executeCodeAnalysis(task.payload as CodeAnalysisPayload);

      case 'custom':
        return this.executeCustom(task.payload as CustomPayload);

      default:
        throw new Error(`Type de tâche inconnu: ${task.type}`);
    }
  }

  // ===========================================================================
  // TASK IMPLEMENTATIONS
  // ===========================================================================

  /**
   * Appel LLM simple
   */
  private async executeLLMCall(payload: LLMCallPayload): Promise<unknown> {
    const { prompt, systemPrompt, model, maxTokens, temperature } = payload;

    // Sélectionner le modèle optimal si non spécifié
    const selectedModel = model || this.selectOptimalModel('simple_chat');

    const response = await this.modelRouter.complete(selectedModel, {
      messages: [{ role: 'user', content: prompt }],
      systemPrompt,
      maxTokens: maxTokens || 2048,
      temperature: temperature || 0.7,
    });

    // Track usage
    this.tokenManager.recordUsage(this.name, {
      inputTokens: Math.floor(response.tokensUsed * 0.7),
      outputTokens: Math.floor(response.tokensUsed * 0.3),
      model: selectedModel,
      provider: response.provider,
      cost: response.cost,
    });

    return {
      content: response.content,
      model: response.model,
      tokensUsed: response.tokensUsed,
      cost: response.cost,
    };
  }

  /**
   * Appel LLM pour raisonnement complexe
   */
  private async executeLLMReasoning(payload: LLMReasoningPayload): Promise<unknown> {
    const { prompt, context, systemPrompt, steps } = payload;

    // Pour le raisonnement, on utilise un meilleur modèle
    const selectedModel = this.selectOptimalModel('reasoning');

    // Construire le prompt de raisonnement
    const reasoningPrompt = `
${systemPrompt || 'Tu es un assistant de raisonnement logique.'}

CONTEXTE:
${context || 'Aucun contexte supplémentaire.'}

PROBLÈME:
${prompt}

${steps ? `ÉTAPES À SUIVRE:\n${steps.map((s, i) => `${i + 1}. ${s}`).join('\n')}` : ''}

Raisonne étape par étape et fournis une réponse structurée.
`;

    const response = await this.modelRouter.complete(selectedModel, {
      messages: [{ role: 'user', content: reasoningPrompt }],
      maxTokens: 4096,
      temperature: 0.3, // Plus déterministe pour le raisonnement
    });

    this.tokenManager.recordUsage(this.name, {
      inputTokens: Math.floor(response.tokensUsed * 0.7),
      outputTokens: Math.floor(response.tokensUsed * 0.3),
      model: selectedModel,
      provider: response.provider,
      cost: response.cost,
    });

    return {
      reasoning: response.content,
      model: response.model,
      tokensUsed: response.tokensUsed,
      cost: response.cost,
    };
  }

  /**
   * Exécution d'un skill (délégué au SkillManager externe)
   */
  private async executeSkill(payload: SkillExecutePayload): Promise<unknown> {
    const { skillManager, skillId, input } = payload;

    if (!skillManager) {
      throw new Error('SkillManager non fourni');
    }

    const result = await skillManager.executeSkill({
      skillId,
      input,
    });

    return result;
  }

  /**
   * Recherche web
   */
  private async executeWebSearch(payload: WebSearchPayload): Promise<unknown> {
    const { query, maxResults, searchService } = payload;

    if (!searchService) {
      throw new Error('Service de recherche non fourni');
    }

    const results = await searchService.search(query, { maxResults: maxResults || 5 });

    return {
      query,
      results,
      count: results.length,
    };
  }

  /**
   * Analyse de code
   */
  private async executeCodeAnalysis(payload: CodeAnalysisPayload): Promise<unknown> {
    const { code, language, analysisType } = payload;

    const prompt = `
Analyse ce code ${language || ''}:

\`\`\`${language || ''}
${code}
\`\`\`

Type d'analyse demandée: ${analysisType || 'général'}

Fournis une analyse structurée en JSON:
{
  "summary": "Résumé du code",
  "complexity": "low|medium|high",
  "issues": ["problème1", "problème2"],
  "suggestions": ["suggestion1", "suggestion2"],
  "security": ["risque1"] // si applicable
}
`;

    const selectedModel = this.selectOptimalModel('code');

    const response = await this.modelRouter.complete(selectedModel, {
      messages: [{ role: 'user', content: prompt }],
      maxTokens: 2048,
      temperature: 0.2,
    });

    // Parser le JSON de la réponse
    try {
      const jsonMatch = response.content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return {
          analysis: JSON.parse(jsonMatch[0]),
          raw: response.content,
          model: response.model,
        };
      }
    } catch {
      // Fallback si pas de JSON valide
    }

    return {
      analysis: null,
      raw: response.content,
      model: response.model,
    };
  }

  /**
   * Tâche personnalisée (fonction fournie)
   */
  private async executeCustom(payload: CustomPayload): Promise<unknown> {
    const { handler, args } = payload;

    if (typeof handler !== 'function') {
      throw new Error('Handler personnalisé invalide');
    }

    return handler(...(args || []));
  }

  // ===========================================================================
  // HELPERS
  // ===========================================================================

  /**
   * Sélectionner le modèle optimal pour une tâche
   */
  private selectOptimalModel(
    task: 'simple_chat' | 'code' | 'reasoning' | 'creative' | 'factual'
  ): string {
    // Essayer de trouver un modèle gratuit ou pas cher d'abord
    for (const maxTier of ['free', 'cheap', 'standard'] as const) {
      const model = this.modelRouter.selectModel({
        task,
        maxTier,
        preferSpeed: task === 'simple_chat',
      });

      if (model) {
        return model.id;
      }
    }

    // Fallback vers Claude Haiku
    return 'claude-3-5-haiku-20241022';
  }

  /**
   * Terminer le worker
   */
  terminate(): void {
    this.status = 'terminated';
    this.removeAllListeners();
    console.log(`[${this.name}] 🛑 Worker terminé`);
  }

  // ===========================================================================
  // GETTERS
  // ===========================================================================

  getStatus(): WorkerStatus {
    return this.status;
  }

  getCurrentTask(): WorkerTask | null {
    return this.currentTask;
  }

  getStats(): {
    id: string;
    name: string;
    status: WorkerStatus;
    completedTasks: number;
    failedTasks: number;
    successRate: number;
    uptimeMs: number;
  } {
    const total = this.completedTasks + this.failedTasks;
    return {
      id: this.id,
      name: this.name,
      status: this.status,
      completedTasks: this.completedTasks,
      failedTasks: this.failedTasks,
      successRate: total > 0 ? this.completedTasks / total : 1,
      uptimeMs: Date.now() - this.createdAt.getTime(),
    };
  }

  isAvailable(): boolean {
    return this.status === 'idle';
  }
}

// ===========================================================================
// PAYLOAD TYPES
// ===========================================================================

interface LLMCallPayload {
  prompt: string;
  systemPrompt?: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
}

interface LLMReasoningPayload {
  prompt: string;
  context?: string;
  systemPrompt?: string;
  steps?: string[];
}

interface SkillExecutePayload {
  skillManager: {
    executeSkill(input: { skillId: string; input: Record<string, unknown> }): Promise<unknown>;
  };
  skillId: string;
  input: Record<string, unknown>;
}

interface WebSearchPayload {
  query: string;
  maxResults?: number;
  searchService: {
    search(query: string, options: { maxResults: number }): Promise<unknown[]>;
  };
}

interface CodeAnalysisPayload {
  code: string;
  language?: string;
  analysisType?: 'security' | 'performance' | 'style' | 'bugs' | 'general';
}

interface CustomPayload {
  handler: (...args: unknown[]) => Promise<unknown>;
  args?: unknown[];
}
