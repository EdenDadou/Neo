/**
 * Base Agent - Classe abstraite pour tous les agents
 *
 * Intègre maintenant:
 * - Heartbeat (Règle 2: Neo ne s'éteint jamais)
 * - TokenManager (conscience des ressources)
 * - ModelRouter (utilisation du bon modèle)
 */

import Anthropic from '@anthropic-ai/sdk';
import type { AgentConfig, AgentMessage, AgentRole } from './types';
import { messageBus } from './message-bus';
import { getTokenManager, TokenManager } from './token-manager';
import { getModelRouter, ModelRouter } from './models';

export abstract class BaseAgent {
  protected config: AgentConfig;
  protected client: Anthropic;
  protected isRunning = false;

  // Heartbeat (Règle 2: Neo ne s'éteint jamais)
  private lastHeartbeat: Date = new Date();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private static readonly HEARTBEAT_INTERVAL_MS = 30000; // 30 secondes
  private static readonly HEARTBEAT_TIMEOUT_MS = 60000; // 1 minute sans heartbeat = problème

  // Resource management (conscience des ressources)
  protected tokenManager: TokenManager;
  protected modelRouter: ModelRouter;

  constructor(config: AgentConfig) {
    this.config = config;
    // Initialize Anthropic client with API key from environment
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (apiKey && apiKey !== 'test-key-for-structure-check' && apiKey !== 'your_api_key_here') {
      this.client = new Anthropic({ apiKey });
    } else {
      // Create a placeholder client - will fail on use but won't crash on init
      // The modelRouter will handle fallbacks
      this.client = null as unknown as Anthropic;
      console.warn(`[${config.name}] ⚠️ No valid ANTHROPIC_API_KEY - Claude direct calls will fail`);
    }
    this.tokenManager = getTokenManager();
    this.modelRouter = getModelRouter();
  }

  /**
   * Démarrer l'agent et s'abonner au message bus
   */
  start(): void {
    if (this.isRunning) return;

    this.isRunning = true;
    messageBus.subscribe(this.config.role, this.handleMessage.bind(this));
    this.startHeartbeat();
    this.onStart();

    console.log(`[${this.config.name}] Agent démarré`);
  }

  /**
   * Arrêter l'agent
   */
  stop(): void {
    if (!this.isRunning) return;

    this.isRunning = false;
    this.stopHeartbeat();
    messageBus.unsubscribe(this.config.role, this.handleMessage.bind(this));
    this.onStop();

    console.log(`[${this.config.name}] Agent arrêté`);
  }

  // ===========================================================================
  // HEARTBEAT (Règle 2: Neo ne s'éteint jamais)
  // ===========================================================================

  /**
   * Démarrer le heartbeat
   */
  private startHeartbeat(): void {
    this.pulse();
    this.heartbeatInterval = setInterval(() => {
      this.pulse();
    }, BaseAgent.HEARTBEAT_INTERVAL_MS);
  }

  /**
   * Arrêter le heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  /**
   * Enregistrer un battement de coeur
   */
  protected pulse(): void {
    this.lastHeartbeat = new Date();
  }

  /**
   * Vérifier si l'agent est en vie (heartbeat récent)
   */
  isAlive(): boolean {
    const timeSinceLastHeartbeat = Date.now() - this.lastHeartbeat.getTime();
    return timeSinceLastHeartbeat < BaseAgent.HEARTBEAT_TIMEOUT_MS;
  }

  /**
   * Obtenir le statut de santé de l'agent
   */
  getHealthStatus(): {
    name: string;
    role: AgentRole;
    isRunning: boolean;
    isAlive: boolean;
    lastHeartbeat: Date;
    uptimeMs: number;
  } {
    return {
      name: this.config.name,
      role: this.config.role,
      isRunning: this.isRunning,
      isAlive: this.isAlive(),
      lastHeartbeat: this.lastHeartbeat,
      uptimeMs: this.isRunning ? Date.now() - this.lastHeartbeat.getTime() : 0,
    };
  }

  /**
   * Envoyer un message à un autre agent
   */
  protected send(
    to: AgentRole | 'broadcast',
    type: AgentMessage['type'],
    payload: unknown
  ): string {
    return messageBus.send({
      from: this.config.role,
      to,
      type,
      payload,
    });
  }

  /**
   * Envoyer et attendre une réponse
   */
  protected async sendAndWait<T>(
    to: AgentRole,
    type: AgentMessage['type'],
    payload: unknown,
    timeoutMs?: number
  ): Promise<T> {
    return messageBus.sendAndWait<T>(
      {
        from: this.config.role,
        to,
        type,
        payload,
      },
      timeoutMs
    );
  }

  /**
   * Répondre à un message
   */
  protected reply(originalMessage: AgentMessage, payload: unknown): void {
    messageBus.reply(originalMessage, payload);
  }

  /**
   * Appeler le LLM avec le système prompt de l'agent
   * Track automatiquement l'usage des tokens
   */
  protected async think(
    userMessage: string,
    additionalContext?: string
  ): Promise<string> {
    // If no Anthropic client, try to use modelRouter
    if (!this.client) {
      return this.thinkOptimized(userMessage, 'simple_chat', additionalContext);
    }

    const systemPrompt = additionalContext
      ? `${this.config.systemPrompt}\n\n--- CONTEXTE ADDITIONNEL ---\n${additionalContext}`
      : this.config.systemPrompt;

    try {
      const response = await this.client.messages.create({
        model: this.config.model,
        max_tokens: this.config.maxTokens,
        temperature: this.config.temperature,
        system: systemPrompt,
        messages: [
          {
            role: 'user',
            content: userMessage,
          },
        ],
      });

      // Track token usage
      this.tokenManager.recordUsage(this.config.name, {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
        model: this.config.model,
        provider: 'anthropic',
      });

      const textBlock = response.content.find((block) => block.type === 'text');
      return textBlock ? textBlock.text : '';
    } catch (error) {
      console.error(`[${this.config.name}] Erreur think:`, error);
      throw error;
    }
  }

  /**
   * Appeler le LLM avec sélection automatique du modèle optimal
   * Utilise le ModelRouter pour choisir le modèle le moins cher adapté à la tâche
   */
  protected async thinkOptimized(
    userMessage: string,
    task: 'simple_chat' | 'code' | 'reasoning' | 'creative' | 'factual' = 'simple_chat',
    additionalContext?: string
  ): Promise<string> {
    const systemPrompt = additionalContext
      ? `${this.config.systemPrompt}\n\n--- CONTEXTE ADDITIONNEL ---\n${additionalContext}`
      : this.config.systemPrompt;

    // Demander au TokenManager le meilleur modèle
    const { model, reason } = this.tokenManager.suggestModelForTask(task);

    if (model) {
      console.log(`[${this.config.name}] Modèle optimisé: ${model.name} (${reason})`);

      try {
        const response = await this.modelRouter.complete(model.id, {
          messages: [{ role: 'user', content: userMessage }],
          systemPrompt,
          maxTokens: this.config.maxTokens,
          temperature: this.config.temperature,
        });

        // Track token usage
        this.tokenManager.recordUsage(this.config.name, {
          inputTokens: Math.floor(response.tokensUsed * 0.7), // Estimation
          outputTokens: Math.floor(response.tokensUsed * 0.3),
          model: model.id,
          provider: model.provider,
          cost: response.cost,
        });

        return response.content;
      } catch (error) {
        console.warn(`[${this.config.name}] Fallback vers Claude: ${error}`);
        // Fallback vers la méthode standard
        return this.think(userMessage, additionalContext);
      }
    }

    // Pas de modèle alternatif disponible, utiliser Claude
    return this.think(userMessage, additionalContext);
  }

  /**
   * Appeler le LLM avec conversation multi-tours
   * Track automatiquement l'usage des tokens
   */
  protected async thinkWithHistory(
    messages: Array<{ role: 'user' | 'assistant'; content: string }>,
    additionalContext?: string
  ): Promise<string> {
    const systemPrompt = additionalContext
      ? `${this.config.systemPrompt}\n\n--- CONTEXTE ADDITIONNEL ---\n${additionalContext}`
      : this.config.systemPrompt;

    const response = await this.client.messages.create({
      model: this.config.model,
      max_tokens: this.config.maxTokens,
      temperature: this.config.temperature,
      system: systemPrompt,
      messages,
    });

    // Track token usage
    this.tokenManager.recordUsage(this.config.name, {
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
      model: this.config.model,
      provider: 'anthropic',
    });

    const textBlock = response.content.find((block) => block.type === 'text');
    return textBlock ? textBlock.text : '';
  }

  /**
   * Obtenir les stats de ressources de cet agent
   */
  getResourceStats(): {
    quickStats: ReturnType<TokenManager['getQuickStats']>;
    plan: ReturnType<TokenManager['getResourcePlan']>;
  } {
    return {
      quickStats: this.tokenManager.getQuickStats(),
      plan: this.tokenManager.getResourcePlan(),
    };
  }

  /**
   * Handler de message - à implémenter par chaque agent
   */
  protected abstract handleMessage(message: AgentMessage): Promise<void>;

  /**
   * Hook appelé au démarrage
   */
  protected onStart(): void {}

  /**
   * Hook appelé à l'arrêt
   */
  protected onStop(): void {}
}
