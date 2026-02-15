/**
 * MODEL REGISTRY - Gestion intelligente des modèles LLM
 *
 * Stratégie de coût:
 * 1. GRATUIT: Ollama local, Groq (limité), HuggingFace Inference
 * 2. TRÈS PAS CHER: Mistral, DeepSeek, Together AI
 * 3. STANDARD: Claude Haiku, GPT-4o-mini
 * 4. PREMIUM: Claude Sonnet/Opus (pour tâches complexes)
 *
 * BRAIN choisit le modèle selon la complexité de la tâche
 */

import Anthropic from '@anthropic-ai/sdk';
import { getAnthropicAuthType } from '../../utils/env';

// ===========================================================================
// TYPES
// ===========================================================================

export type ModelTier = 'free' | 'cheap' | 'standard' | 'premium';

export type ModelProvider =
  | 'ollama'      // Local, gratuit
  | 'groq'        // Gratuit (rate limited)
  | 'huggingface' // Gratuit (rate limited)
  | 'mistral'     // Très pas cher
  | 'deepseek'    // Très pas cher
  | 'together'    // Pas cher
  | 'openrouter'  // Agrégateur multi-modèles
  | 'anthropic';  // Claude (standard/premium)

export interface ModelInfo {
  id: string;
  name: string;
  provider: ModelProvider;
  tier: ModelTier;
  contextWindow: number;
  costPer1kTokens: number;  // En USD, 0 = gratuit
  capabilities: ModelCapability[];
  speed: 'fast' | 'medium' | 'slow';
  quality: 'basic' | 'good' | 'excellent';
}

export type ModelCapability =
  | 'chat'
  | 'code'
  | 'reasoning'
  | 'creative'
  | 'multilingual'
  | 'long_context'
  | 'function_calling'
  | 'vision';

export interface ModelConfig {
  ollamaUrl?: string;
  groqApiKey?: string;
  mistralApiKey?: string;
  deepseekApiKey?: string;
  togetherApiKey?: string;
  openrouterApiKey?: string;
  anthropicApiKey?: string;
  huggingfaceApiKey?: string;
}

export interface CompletionRequest {
  messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }>;
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
}

export interface CompletionResponse {
  content: string;
  model: string;
  provider: ModelProvider;
  tokensUsed: number;
  cost: number;
}

// ===========================================================================
// MODEL REGISTRY
// ===========================================================================

const MODELS: ModelInfo[] = [
  // === GRATUIT (Tier: free) ===
  {
    id: 'llama3.2:3b',
    name: 'Llama 3.2 3B (Ollama)',
    provider: 'ollama',
    tier: 'free',
    contextWindow: 8192,
    costPer1kTokens: 0,
    capabilities: ['chat', 'multilingual'],
    speed: 'fast',
    quality: 'basic',
  },
  {
    id: 'llama3.1:8b',
    name: 'Llama 3.1 8B (Ollama)',
    provider: 'ollama',
    tier: 'free',
    contextWindow: 8192,
    costPer1kTokens: 0,
    capabilities: ['chat', 'code', 'reasoning', 'multilingual'],
    speed: 'medium',
    quality: 'good',
  },
  {
    id: 'mistral:7b',
    name: 'Mistral 7B (Ollama)',
    provider: 'ollama',
    tier: 'free',
    contextWindow: 8192,
    costPer1kTokens: 0,
    capabilities: ['chat', 'code', 'multilingual'],
    speed: 'medium',
    quality: 'good',
  },
  {
    id: 'llama-3.1-8b-instant',
    name: 'Llama 3.1 8B (Groq)',
    provider: 'groq',
    tier: 'free',
    contextWindow: 8192,
    costPer1kTokens: 0,
    capabilities: ['chat', 'code', 'reasoning'],
    speed: 'fast',
    quality: 'good',
  },
  {
    id: 'mixtral-8x7b-32768',
    name: 'Mixtral 8x7B (Groq)',
    provider: 'groq',
    tier: 'free',
    contextWindow: 32768,
    costPer1kTokens: 0,
    capabilities: ['chat', 'code', 'reasoning', 'long_context'],
    speed: 'fast',
    quality: 'good',
  },
  {
    id: 'mistralai/Mistral-7B-Instruct-v0.3',
    name: 'Mistral 7B (HuggingFace)',
    provider: 'huggingface',
    tier: 'free',
    contextWindow: 8192,
    costPer1kTokens: 0,
    capabilities: ['chat', 'code', 'multilingual'],
    speed: 'medium',
    quality: 'good',
  },
  {
    id: 'microsoft/Phi-3-mini-4k-instruct',
    name: 'Phi-3 Mini (HuggingFace)',
    provider: 'huggingface',
    tier: 'free',
    contextWindow: 4096,
    costPer1kTokens: 0,
    capabilities: ['chat', 'code', 'reasoning'],
    speed: 'fast',
    quality: 'good',
  },
  {
    id: 'google/gemma-2-9b-it',
    name: 'Gemma 2 9B (HuggingFace)',
    provider: 'huggingface',
    tier: 'free',
    contextWindow: 8192,
    costPer1kTokens: 0,
    capabilities: ['chat', 'reasoning', 'multilingual'],
    speed: 'medium',
    quality: 'good',
  },

  // === TRÈS PAS CHER (Tier: cheap) ===
  {
    id: 'deepseek-chat',
    name: 'DeepSeek Chat',
    provider: 'deepseek',
    tier: 'cheap',
    contextWindow: 32768,
    costPer1kTokens: 0.0001,  // $0.1/M tokens
    capabilities: ['chat', 'code', 'reasoning', 'long_context'],
    speed: 'fast',
    quality: 'good',
  },
  {
    id: 'mistral-small-latest',
    name: 'Mistral Small',
    provider: 'mistral',
    tier: 'cheap',
    contextWindow: 32768,
    costPer1kTokens: 0.0002,
    capabilities: ['chat', 'code', 'function_calling'],
    speed: 'fast',
    quality: 'good',
  },
  {
    id: 'meta-llama/Llama-3.1-8B-Instruct-Turbo',
    name: 'Llama 3.1 8B (Together)',
    provider: 'together',
    tier: 'cheap',
    contextWindow: 8192,
    costPer1kTokens: 0.0002,
    capabilities: ['chat', 'code'],
    speed: 'fast',
    quality: 'good',
  },

  // === STANDARD (Tier: standard) ===
  {
    id: 'claude-haiku-4-5-20250515',
    name: 'Claude Haiku 4.5',
    provider: 'anthropic',
    tier: 'standard',
    contextWindow: 200000,
    costPer1kTokens: 0.001,
    capabilities: ['chat', 'code', 'reasoning', 'long_context', 'multilingual'],
    speed: 'fast',
    quality: 'good',
  },
  {
    id: 'mistral-large-latest',
    name: 'Mistral Large',
    provider: 'mistral',
    tier: 'standard',
    contextWindow: 32768,
    costPer1kTokens: 0.002,
    capabilities: ['chat', 'code', 'reasoning', 'function_calling'],
    speed: 'medium',
    quality: 'excellent',
  },

  // === PREMIUM (Tier: premium) ===
  {
    id: 'claude-sonnet-4-20250514',
    name: 'Claude Sonnet 4',
    provider: 'anthropic',
    tier: 'premium',
    contextWindow: 200000,
    costPer1kTokens: 0.003,
    capabilities: ['chat', 'code', 'reasoning', 'creative', 'long_context', 'multilingual', 'function_calling'],
    speed: 'medium',
    quality: 'excellent',
  },
  {
    id: 'claude-opus-4-20250514',
    name: 'Claude Opus 4',
    provider: 'anthropic',
    tier: 'premium',
    contextWindow: 200000,
    costPer1kTokens: 0.015,
    capabilities: ['chat', 'code', 'reasoning', 'creative', 'long_context', 'multilingual', 'function_calling'],
    speed: 'slow',
    quality: 'excellent',
  },
];

// ===========================================================================
// MODEL ROUTER
// ===========================================================================

export class ModelRouter {
  private config: ModelConfig;
  private anthropicClient: Anthropic | null = null;
  private availableModels: ModelInfo[] = [];
  private usageStats: Map<string, { calls: number; tokens: number; cost: number }> = new Map();

  constructor(config?: Partial<ModelConfig>) {
    this.config = {
      ollamaUrl: config?.ollamaUrl || process.env.OLLAMA_URL || 'http://localhost:11434',
      groqApiKey: config?.groqApiKey || process.env.GROQ_API_KEY,
      mistralApiKey: config?.mistralApiKey || process.env.MISTRAL_API_KEY,
      deepseekApiKey: config?.deepseekApiKey || process.env.DEEPSEEK_API_KEY,
      togetherApiKey: config?.togetherApiKey || process.env.TOGETHER_API_KEY,
      openrouterApiKey: config?.openrouterApiKey || process.env.OPENROUTER_API_KEY,
      anthropicApiKey: config?.anthropicApiKey || process.env.ANTHROPIC_API_KEY,
      huggingfaceApiKey: config?.huggingfaceApiKey || process.env.HUGGINGFACE_API_KEY,
    };

    this.initializeClients();
  }

  private initializeClients(): void {
    // Anthropic
    if (this.config.anthropicApiKey) {
      this.anthropicClient = new Anthropic({ apiKey: this.config.anthropicApiKey });
    }
  }

  /**
   * Détecter les modèles disponibles selon les API keys configurées
   */
  async detectAvailableModels(): Promise<ModelInfo[]> {
    this.availableModels = [];

    // Vérifier Ollama local
    try {
      const response = await fetch(`${this.config.ollamaUrl}/api/tags`, {
        signal: AbortSignal.timeout(2000),
      });
      if (response.ok) {
        const data = await response.json() as { models?: Array<{ name: string }> };
        const localModels = data.models?.map(m => m.name) || [];

        for (const model of MODELS.filter(m => m.provider === 'ollama')) {
          const baseId = model.id.split(':')[0];
          if (localModels.some(lm => lm.startsWith(baseId))) {
            this.availableModels.push(model);
          }
        }
        console.log(`[ModelRouter] Ollama: ${localModels.length} modèles locaux`);
      }
    } catch {
      console.log('[ModelRouter] Ollama non disponible');
    }

    // Vérifier les API keys
    if (this.config.groqApiKey) {
      this.availableModels.push(...MODELS.filter(m => m.provider === 'groq'));
      console.log('[ModelRouter] Groq: disponible (gratuit)');
    }

    if (this.config.deepseekApiKey) {
      this.availableModels.push(...MODELS.filter(m => m.provider === 'deepseek'));
      console.log('[ModelRouter] DeepSeek: disponible');
    }

    if (this.config.mistralApiKey) {
      this.availableModels.push(...MODELS.filter(m => m.provider === 'mistral'));
      console.log('[ModelRouter] Mistral: disponible');
    }

    if (this.config.togetherApiKey) {
      this.availableModels.push(...MODELS.filter(m => m.provider === 'together'));
      console.log('[ModelRouter] Together: disponible');
    }

    // Anthropic Claude - PRIORITAIRE si configuré
    if (this.config.anthropicApiKey &&
        this.config.anthropicApiKey !== 'test-key-for-structure-check' &&
        this.config.anthropicApiKey !== 'your_api_key_here') {
      this.availableModels.push(...MODELS.filter(m => m.provider === 'anthropic'));
      console.log('[ModelRouter] Anthropic Claude: disponible (PRIORITAIRE)');
    }

    if (this.config.huggingfaceApiKey) {
      this.availableModels.push(...MODELS.filter(m => m.provider === 'huggingface'));
      console.log('[ModelRouter] HuggingFace: disponible (gratuit)');
    }

    // Trier: Claude en premier (qualité), puis par tier et qualité
    // Neo préfère Claude quand disponible, les modèles locaux sont des fallbacks
    this.availableModels.sort((a, b) => {
      // Anthropic toujours en premier (c'est le modèle principal)
      if (a.provider === 'anthropic' && b.provider !== 'anthropic') return -1;
      if (b.provider === 'anthropic' && a.provider !== 'anthropic') return 1;

      // Pour les modèles Anthropic, trier par qualité (Sonnet > Haiku)
      if (a.provider === 'anthropic' && b.provider === 'anthropic') {
        const qualityOrder = { basic: 0, good: 1, excellent: 2 };
        return qualityOrder[b.quality] - qualityOrder[a.quality];
      }

      // Pour les autres, trier par tier (gratuit d'abord) puis qualité
      const tierOrder = { free: 0, cheap: 1, standard: 2, premium: 3 };
      if (tierOrder[a.tier] !== tierOrder[b.tier]) {
        return tierOrder[a.tier] - tierOrder[b.tier];
      }
      const qualityOrder = { basic: 0, good: 1, excellent: 2 };
      return qualityOrder[b.quality] - qualityOrder[a.quality];
    });

    console.log(`[ModelRouter] ${this.availableModels.length} modèles disponibles`);
    return this.availableModels;
  }

  /**
   * Sélectionner le meilleur modèle pour une tâche
   * PRIORITÉ: Claude (Anthropic) quand disponible, sinon fallback vers modèles locaux/gratuits
   */
  selectModel(options: {
    task: 'simple_chat' | 'code' | 'reasoning' | 'creative' | 'factual';
    maxTier?: ModelTier;
    requiredCapabilities?: ModelCapability[];
    preferSpeed?: boolean;
    preferClaude?: boolean; // Default true - prefer Claude when available
  }): ModelInfo | null {
    const { task, maxTier = 'premium', requiredCapabilities = [], preferSpeed = false, preferClaude = true } = options;

    // Mapping tâche → capacités requises
    const taskCapabilities: Record<string, ModelCapability[]> = {
      simple_chat: ['chat'],
      code: ['chat', 'code'],
      reasoning: ['chat', 'reasoning'],
      creative: ['chat', 'creative'],
      factual: ['chat'],
    };

    const neededCapabilities = [...taskCapabilities[task], ...requiredCapabilities];
    const tierOrder = { free: 0, cheap: 1, standard: 2, premium: 3 };
    const maxTierValue = tierOrder[maxTier];

    // Filtrer les modèles compatibles
    let candidates = this.availableModels.filter(model => {
      // Vérifier le tier
      if (tierOrder[model.tier] > maxTierValue) return false;

      // Vérifier les capacités
      for (const cap of neededCapabilities) {
        if (!model.capabilities.includes(cap)) return false;
      }

      return true;
    });

    if (candidates.length === 0) {
      console.warn(`[ModelRouter] Aucun modèle trouvé pour: ${task} (maxTier: ${maxTier})`);
      return this.availableModels[0] || null;  // Fallback
    }

    // Si on préfère Claude et qu'il est disponible, le mettre en premier
    if (preferClaude) {
      const claudeModels = candidates.filter(m => m.provider === 'anthropic');
      if (claudeModels.length > 0) {
        // Préférer Haiku pour simple_chat (rapide et pas cher), Sonnet pour le reste
        if (task === 'simple_chat') {
          const haiku = claudeModels.find(m => m.id.includes('haiku'));
          if (haiku) return haiku;
        }
        // Pour les autres tâches, préférer Sonnet
        const sonnet = claudeModels.find(m => m.id.includes('sonnet'));
        if (sonnet) return sonnet;
        // Fallback vers n'importe quel Claude
        return claudeModels[0];
      }
    }

    // Pas de Claude disponible, trier selon préférence
    if (preferSpeed) {
      const speedOrder = { fast: 0, medium: 1, slow: 2 };
      candidates.sort((a, b) => speedOrder[a.speed] - speedOrder[b.speed]);
    } else {
      const qualityOrder = { basic: 0, good: 1, excellent: 2 };
      candidates.sort((a, b) => qualityOrder[b.quality] - qualityOrder[a.quality]);
    }

    return candidates[0];
  }

  /**
   * Appeler un modèle
   */
  async complete(
    modelId: string,
    request: CompletionRequest
  ): Promise<CompletionResponse> {
    const model = this.availableModels.find(m => m.id === modelId) ||
                  MODELS.find(m => m.id === modelId);

    if (!model) {
      throw new Error(`Model not found: ${modelId}`);
    }

    let response: CompletionResponse;

    switch (model.provider) {
      case 'ollama':
        response = await this.completeOllama(model, request);
        break;
      case 'groq':
        response = await this.completeGroq(model, request);
        break;
      case 'deepseek':
        response = await this.completeDeepSeek(model, request);
        break;
      case 'mistral':
        response = await this.completeMistral(model, request);
        break;
      case 'anthropic':
        response = await this.completeAnthropic(model, request);
        break;
      case 'huggingface':
        response = await this.completeHuggingFace(model, request);
        break;
      default:
        throw new Error(`Provider not implemented: ${model.provider}`);
    }

    // Mettre à jour les stats
    this.updateUsageStats(modelId, response.tokensUsed, response.cost);

    return response;
  }

  // === PROVIDERS ===

  private async completeOllama(model: ModelInfo, request: CompletionRequest): Promise<CompletionResponse> {
    const messages = request.systemPrompt
      ? [{ role: 'system' as const, content: request.systemPrompt }, ...request.messages]
      : request.messages;

    const response = await fetch(`${this.config.ollamaUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model.id,
        messages,
        stream: false,
        options: {
          temperature: request.temperature || 0.7,
          num_predict: request.maxTokens || 2048,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama error: ${response.status}`);
    }

    const data = await response.json() as {
      message: { content: string };
      eval_count?: number;
    };

    return {
      content: data.message.content,
      model: model.id,
      provider: 'ollama',
      tokensUsed: data.eval_count || 0,
      cost: 0,
    };
  }

  private async completeGroq(model: ModelInfo, request: CompletionRequest): Promise<CompletionResponse> {
    if (!this.config.groqApiKey) {
      throw new Error('Groq API key not configured');
    }

    const messages = request.systemPrompt
      ? [{ role: 'system' as const, content: request.systemPrompt }, ...request.messages]
      : request.messages;

    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.groqApiKey}`,
      },
      body: JSON.stringify({
        model: model.id,
        messages,
        max_tokens: request.maxTokens || 2048,
        temperature: request.temperature || 0.7,
      }),
    });

    if (!response.ok) {
      throw new Error(`Groq error: ${response.status}`);
    }

    const data = await response.json() as {
      choices: Array<{ message: { content: string } }>;
      usage: { total_tokens: number };
    };

    return {
      content: data.choices[0].message.content,
      model: model.id,
      provider: 'groq',
      tokensUsed: data.usage.total_tokens,
      cost: 0,  // Gratuit
    };
  }

  private async completeDeepSeek(model: ModelInfo, request: CompletionRequest): Promise<CompletionResponse> {
    if (!this.config.deepseekApiKey) {
      throw new Error('DeepSeek API key not configured');
    }

    const messages = request.systemPrompt
      ? [{ role: 'system' as const, content: request.systemPrompt }, ...request.messages]
      : request.messages;

    const response = await fetch('https://api.deepseek.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.deepseekApiKey}`,
      },
      body: JSON.stringify({
        model: model.id,
        messages,
        max_tokens: request.maxTokens || 2048,
        temperature: request.temperature || 0.7,
      }),
    });

    if (!response.ok) {
      throw new Error(`DeepSeek error: ${response.status}`);
    }

    const data = await response.json() as {
      choices: Array<{ message: { content: string } }>;
      usage: { total_tokens: number };
    };

    const tokensUsed = data.usage.total_tokens;

    return {
      content: data.choices[0].message.content,
      model: model.id,
      provider: 'deepseek',
      tokensUsed,
      cost: (tokensUsed / 1000) * model.costPer1kTokens,
    };
  }

  private async completeMistral(model: ModelInfo, request: CompletionRequest): Promise<CompletionResponse> {
    if (!this.config.mistralApiKey) {
      throw new Error('Mistral API key not configured');
    }

    const messages = request.systemPrompt
      ? [{ role: 'system' as const, content: request.systemPrompt }, ...request.messages]
      : request.messages;

    const response = await fetch('https://api.mistral.ai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.mistralApiKey}`,
      },
      body: JSON.stringify({
        model: model.id,
        messages,
        max_tokens: request.maxTokens || 2048,
        temperature: request.temperature || 0.7,
      }),
    });

    if (!response.ok) {
      throw new Error(`Mistral error: ${response.status}`);
    }

    const data = await response.json() as {
      choices: Array<{ message: { content: string } }>;
      usage: { total_tokens: number };
    };

    const tokensUsed = data.usage.total_tokens;

    return {
      content: data.choices[0].message.content,
      model: model.id,
      provider: 'mistral',
      tokensUsed,
      cost: (tokensUsed / 1000) * model.costPer1kTokens,
    };
  }

  private async completeAnthropic(model: ModelInfo, request: CompletionRequest): Promise<CompletionResponse> {
    const authType = getAnthropicAuthType();

    // Pour les tokens OAuth, utiliser l'API directement avec Bearer auth
    if (authType === 'oauth') {
      return this.completeAnthropicOAuth(model, request);
    }

    // Pour les clés API standard, utiliser le SDK
    if (!this.anthropicClient) {
      throw new Error('Anthropic client not initialized');
    }

    const response = await this.anthropicClient.messages.create({
      model: model.id,
      max_tokens: request.maxTokens || 2048,
      system: request.systemPrompt,
      messages: request.messages.map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      })),
    });

    const content = response.content[0].type === 'text' ? response.content[0].text : '';
    const tokensUsed = response.usage.input_tokens + response.usage.output_tokens;

    return {
      content,
      model: model.id,
      provider: 'anthropic',
      tokensUsed,
      cost: (tokensUsed / 1000) * model.costPer1kTokens,
    };
  }

  /**
   * Appel Anthropic avec OAuth token (Bearer auth)
   * Pour les abonnements Claude Pro/Max
   */
  private async completeAnthropicOAuth(model: ModelInfo, request: CompletionRequest): Promise<CompletionResponse> {
    const oauthToken = this.config.anthropicApiKey;

    if (!oauthToken) {
      throw new Error('OAuth token not configured');
    }

    const messages = request.messages.map(m => ({
      role: m.role as 'user' | 'assistant',
      content: m.content,
    }));

    const body: Record<string, unknown> = {
      model: model.id,
      max_tokens: request.maxTokens || 2048,
      messages,
    };

    if (request.systemPrompt) {
      body.system = request.systemPrompt;
    }

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${oauthToken}`,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Anthropic OAuth error: ${response.status} - ${errorText}`);
    }

    const data = await response.json() as {
      content: Array<{ type: string; text?: string }>;
      usage: { input_tokens: number; output_tokens: number };
    };

    const content = data.content[0]?.type === 'text' ? data.content[0].text || '' : '';
    const tokensUsed = data.usage.input_tokens + data.usage.output_tokens;

    return {
      content,
      model: model.id,
      provider: 'anthropic',
      tokensUsed,
      cost: 0, // OAuth/subscription = pas de coût API direct
    };
  }

  private async completeHuggingFace(model: ModelInfo, request: CompletionRequest): Promise<CompletionResponse> {
    if (!this.config.huggingfaceApiKey) {
      throw new Error('HuggingFace API key not configured');
    }

    // Construire le prompt au format chat
    let prompt = '';
    if (request.systemPrompt) {
      prompt += `<|system|>\n${request.systemPrompt}\n`;
    }
    for (const msg of request.messages) {
      prompt += `<|${msg.role}|>\n${msg.content}\n`;
    }
    prompt += '<|assistant|>\n';

    const response = await fetch(`https://api-inference.huggingface.co/models/${model.id}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.huggingfaceApiKey}`,
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          max_new_tokens: request.maxTokens || 1024,
          temperature: request.temperature || 0.7,
          return_full_text: false,
          do_sample: true,
        },
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HuggingFace error: ${response.status} - ${errorText}`);
    }

    const data = await response.json() as Array<{ generated_text: string }> | { error: string };

    if ('error' in data) {
      throw new Error(`HuggingFace error: ${data.error}`);
    }

    const generatedText = Array.isArray(data) ? data[0]?.generated_text || '' : '';

    // Estimation des tokens (approximatif)
    const tokensUsed = Math.ceil((prompt.length + generatedText.length) / 4);

    return {
      content: generatedText.trim(),
      model: model.id,
      provider: 'huggingface',
      tokensUsed,
      cost: 0,  // Gratuit
    };
  }

  // === STATS ===

  private updateUsageStats(modelId: string, tokens: number, cost: number): void {
    const current = this.usageStats.get(modelId) || { calls: 0, tokens: 0, cost: 0 };
    this.usageStats.set(modelId, {
      calls: current.calls + 1,
      tokens: current.tokens + tokens,
      cost: current.cost + cost,
    });
  }

  getUsageStats(): Map<string, { calls: number; tokens: number; cost: number }> {
    return new Map(this.usageStats);
  }

  getTotalCost(): number {
    let total = 0;
    this.usageStats.forEach(stats => {
      total += stats.cost;
    });
    return total;
  }

  getAvailableModels(): ModelInfo[] {
    return [...this.availableModels];
  }

  getModelsByTier(tier: ModelTier): ModelInfo[] {
    return this.availableModels.filter(m => m.tier === tier);
  }
}

// Singleton
let routerInstance: ModelRouter | null = null;

export function getModelRouter(config?: Partial<ModelConfig>): ModelRouter {
  if (!routerInstance) {
    routerInstance = new ModelRouter(config);
  }
  return routerInstance;
}
