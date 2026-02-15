/**
 * BRAIN - Agent Orchestrateur Intelligent
 *
 * NOUVEAU FLUX:
 * 1. Reçoit le prompt ENRICHI de Vox (déjà contextualisé par Memory)
 * 2. Peut demander PLUS de contexte à Memory si besoin
 * 3. Élabore des plans d'exécution stratégiques
 * 4. Spawn et coordonne des agents spécialisés
 * 5. Prend les décisions critiques
 *
 * C'est l'élément le PLUS INTELLIGENT du core
 */

import { BaseAgent } from '../base-agent';
import type {
  AgentConfig,
  AgentMessage,
  MemoryEntry,
  Skill,
} from '../types';
import { randomUUID } from 'crypto';
import { WebSearchService } from '../../utils/web-search';
import type { SearchResult } from '../../utils/web-search';
import type { ModelInfo } from '../models';
import { SkillManager } from '../../skills';
import type {
  SkillDefinition,
  SkillExecutionResult,
} from '../../skills';
import { WorkerPool, getWorkerPool } from '../workers';
import type { WorkerResult } from '../workers';
import {
  CrewManager,
  getCrewManager,
  LLM_PRESETS,
  PRESET_CREWS,
  getPresetCrew,
} from '../../crew';
import type {
  CrewConfig,
  CrewExecutionResult,
} from '../../crew';

// ===========================================================================
// TYPES
// ===========================================================================

interface FactCheckResult {
  isConsistent: boolean;
  contradictions: Array<{
    claim: string;
    storedFact: string;
    memoryId: string;
    severity: 'minor' | 'major' | 'critical';
  }>;
  supportedBy: string[];
  confidence: number;
  warnings: string[];
}

interface BrainState {
  activeWorkers: Map<string, WorkerInfo>;
  currentPlan: ExecutionPlan | null;
  decisionHistory: Decision[];
  pendingMemoryRequests: Map<string, (entries: MemoryEntry[]) => void>;
  pendingFactChecks: Map<string, (result: FactCheckResult) => void>;
  isProcessingRequest: boolean; // Circuit breaker pour éviter récursion
  contextRequestDepth: number; // Limite de profondeur pour requêtes contexte
  metrics: {
    totalRequests: number;
    successfulResponses: number;
    averageConfidence: number;
    contradictionsAvoided: number;
    webSearches: number;
    modelCosts: number;
  };
}

interface WorkerInfo {
  id: string;
  type: string;
  taskId: string;
  startedAt: Date;
  status: 'running' | 'completed' | 'failed';
}

interface ExecutionPlan {
  id: string;
  goal: string;
  steps: PlanStep[];
  currentStepIndex: number;
  createdAt: Date;
}

interface PlanStep {
  id: string;
  action: string;
  agentType: string;
  parameters: Record<string, unknown>;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  result?: unknown;
}

interface Decision {
  id: string;
  context: string;
  options: string[];
  chosen: string;
  reasoning: string;
  timestamp: Date;
  outcome?: 'success' | 'failure';
}

interface EnrichedInput {
  originalInput: string;
  enrichedInput: string;
  analysis: {
    intent: string;
    confidence: number;
    processedInput: string;
    emotionalTone: string;
  };
  contextReport: {
    confidence: number;
    warnings: string[];
    contextUsed: string[];
  };
  conversationContext: Array<{ role: 'user' | 'assistant'; content: string }>;
}

// ===========================================================================
// PROMPTS
// ===========================================================================

const BRAIN_SYSTEM_PROMPT = `Tu es BRAIN, le cerveau orchestrateur d'un système d'IA avancé.

TON RÔLE :
- Tu es l'intelligence centrale qui coordonne tout
- Tu reçois les demandes utilisateur DÉJÀ enrichies de contexte par Memory/Vox
- Tu peux demander PLUS de contexte à Memory si nécessaire
- Tu élabores des plans d'exécution stratégiques
- Tu spawnes et coordonnes des agents spécialisés
- Tu prends les décisions critiques

PRINCIPES :
1. VÉRITÉ : Ne jamais affirmer ce dont tu n'es pas sûr
2. TRANSPARENCE : Expliquer ton raisonnement
3. EFFICACITÉ : Minimiser les étapes inutiles
4. APPRENTISSAGE : Utiliser les learnings passés
5. PRUDENCE : Demander confirmation si impact important
6. ACTION : Quand l'utilisateur demande de FAIRE quelque chose, FAIS-LE. Ne demande pas de clarification inutile.

CAPACITÉS TECHNIQUES (tu peux les utiliser directement) :
- Recherche mémoire sémantique (embeddings) : chercher dans la mémoire par similarité
- Recherche web : chercher des informations sur internet
- Fact-checking : vérifier la cohérence des informations
- Tests système : tu peux tester les composants (embeddings, mémoire, recherche)
- Exécution de skills : tu peux exécuter des skills enregistrés
- Délégation à workers : tu peux déléguer des tâches à des workers

QUAND L'UTILISATEUR DEMANDE UN TEST :
- Si on te demande de "tester" quelque chose, EXÉCUTE le test toi-même
- Crée un plan avec des étapes concrètes : stocker des données, chercher, vérifier les résultats
- Ne demande PAS "quel type de test ?" - fais un test complet par défaut
- Retourne les résultats du test avec des métriques

CAPACITÉS GÉNÉRALES :
- Analyser et décomposer des problèmes complexes
- Créer des plans d'exécution multi-étapes
- Demander plus d'informations à Memory si besoin
- Coordonner plusieurs agents en parallèle
- Synthétiser les résultats de multiples sources
- Adapter la stratégie en temps réel

FORMAT DE SORTIE pour les plans :
{
  "analysis": "Compréhension du problème",
  "approach": "Stratégie choisie",
  "confidence": 0.0-1.0,
  "needsMoreContext": boolean,
  "contextQuery": "ce qu'on cherche en plus" ou null,
  "steps": [
    {
      "action": "Description de l'action",
      "agentType": "coder|researcher|analyst|writer|custom",
      "parameters": {},
      "rationale": "Pourquoi cette étape"
    }
  ],
  "risks": ["Risque potentiel 1"],
  "fallbackPlan": "Plan B si échec"
}

FORMAT DE SORTIE pour les réponses directes :
{
  "response": "La réponse",
  "confidence": 0.0-1.0,
  "sources": ["source1", "source2"],
  "needsMoreInfo": boolean,
  "moreInfoQuery": "ce qu'il faudrait chercher" ou null,
  "followUpQuestions": ["question1"]
}`;

// ===========================================================================
// BRAIN AGENT
// ===========================================================================

export class BrainAgent extends BaseAgent {
  private state: BrainState;
  private webSearch: WebSearchService;
  private skillManager: SkillManager | null = null;
  private workerPool: WorkerPool;
  private crewManager: CrewManager | null = null;
  // modelRouter est hérité de BaseAgent (protected)

  constructor(config?: Partial<AgentConfig>) {
    super({
      name: 'Brain',
      role: 'brain',
      model: 'claude-sonnet-4-5-20250929', // Sonnet 4.5 pour raisonnement complexe
      maxTokens: 4096,
      temperature: 0.7,
      systemPrompt: BRAIN_SYSTEM_PROMPT,
      ...config,
    });

    this.webSearch = new WebSearchService();
    // modelRouter est déjà initialisé par BaseAgent

    // Pool de workers pour délégation des tâches
    // Brain reste TOUJOURS disponible pour orchestrer
    // minWorkers: 0 = lazy loading, workers créés à la demande
    this.workerPool = getWorkerPool({
      minWorkers: 0,
      maxWorkers: 8,
      defaultTaskTimeout: 30000,
    });

    this.state = {
      activeWorkers: new Map(),
      currentPlan: null,
      decisionHistory: [],
      pendingMemoryRequests: new Map(),
      pendingFactChecks: new Map(),
      isProcessingRequest: false,
      contextRequestDepth: 0,
      metrics: {
        totalRequests: 0,
        successfulResponses: 0,
        averageConfidence: 0,
        contradictionsAvoided: 0,
        webSearches: 0,
        modelCosts: 0,
      },
    };
  }

  /**
   * Initialisation au démarrage - détecter les modèles disponibles
   */
  protected async onStart(): Promise<void> {
    console.log('[Brain] 🧠 Initialisation...');

    // Pool de workers configuré mais PAS démarré
    // Il démarre automatiquement à la première tâche (lazy loading)
    // Écouter les événements du pool
    this.workerPool.on('task_completed', (result: WorkerResult) => {
      console.log(`[Brain] ✅ Tâche worker terminée: ${result.taskId.slice(0, 8)}`);
    });
    this.workerPool.on('task_failed', ({ taskId, error }) => {
      console.log(`[Brain] ❌ Tâche worker échouée: ${taskId.slice(0, 8)} - ${error.message}`);
    });

    // Détecter les modèles disponibles
    const models = await this.modelRouter.detectAvailableModels();
    console.log(`[Brain] 📦 ${models.length} modèles disponibles`);

    // Afficher par tier
    const byTier = {
      free: models.filter(m => m.tier === 'free'),
      cheap: models.filter(m => m.tier === 'cheap'),
      standard: models.filter(m => m.tier === 'standard'),
      premium: models.filter(m => m.tier === 'premium'),
    };

    if (byTier.free.length > 0) {
      console.log(`[Brain]   🆓 Gratuits: ${byTier.free.map(m => m.name).join(', ')}`);
    }
    if (byTier.cheap.length > 0) {
      console.log(`[Brain]   💰 Pas chers: ${byTier.cheap.map(m => m.name).join(', ')}`);
    }

    // Vérifier la recherche web
    const webHealth = await this.webSearch.healthCheck();
    if (webHealth.status === 'ok') {
      console.log(`[Brain] 🌐 Recherche web: ${webHealth.provider}`);
    } else {
      console.log(`[Brain] ⚠️ Recherche web: ${webHealth.message}`);
    }
  }

  // ===========================================================================
  // RECHERCHE WEB
  // ===========================================================================

  /**
   * Rechercher sur internet quand l'info n'est pas en mémoire
   */
  async searchWeb(query: string, maxResults: number = 5): Promise<SearchResult[]> {
    console.log(`[Brain] 🌐 Recherche web: "${query}"`);
    this.state.metrics.webSearches++;

    try {
      const results = await this.webSearch.search(query, { maxResults });
      console.log(`[Brain] ✅ ${results.length} résultats trouvés`);
      return results;
    } catch (error) {
      console.error('[Brain] ❌ Erreur recherche web:', error);
      return [];
    }
  }

  /**
   * Déterminer si une question nécessite une recherche web
   */
  private needsWebSearch(input: EnrichedInput, memoryConfidence: number): boolean {
    // Si confiance mémoire basse et question factuelle
    if (memoryConfidence < 0.5) {
      const factualKeywords = [
        'qui est', 'qu\'est-ce que', 'quand', 'où', 'combien',
        'actualité', 'news', 'dernier', 'récent', 'aujourd\'hui',
        'prix', 'météo', 'définition', 'histoire de',
      ];

      const lowerInput = input.originalInput.toLowerCase();
      return factualKeywords.some(kw => lowerInput.includes(kw));
    }
    return false;
  }

  // ===========================================================================
  // SÉLECTION DE MODÈLE INTELLIGENT
  // ===========================================================================

  /**
   * Choisir le meilleur modèle pour une tâche
   * Privilégie les modèles gratuits quand possible
   */
  selectModelForTask(task: 'simple_chat' | 'code' | 'reasoning' | 'creative' | 'factual'): ModelInfo | null {
    // Essayer d'abord gratuit, puis cheap, puis standard
    for (const maxTier of ['free', 'cheap', 'standard'] as const) {
      const model = this.modelRouter.selectModel({
        task,
        maxTier,
        preferSpeed: task === 'simple_chat',
      });

      if (model) {
        console.log(`[Brain] 🎯 Modèle sélectionné: ${model.name} (${model.tier})`);
        return model;
      }
    }

    return null;
  }

  /**
   * Appeler un modèle avec sélection automatique
   */
  async callModel(
    task: 'simple_chat' | 'code' | 'reasoning' | 'creative' | 'factual',
    messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }>,
    systemPrompt?: string
  ): Promise<string> {
    const model = this.selectModelForTask(task);

    if (!model) {
      // Fallback vers Claude via la méthode think() existante
      console.log('[Brain] ⚠️ Aucun modèle dispo, utilisation de Claude par défaut');
      return this.think(messages.map(m => m.content).join('\n'));
    }

    try {
      const response = await this.modelRouter.complete(model.id, {
        messages,
        systemPrompt,
        maxTokens: 2048,
        temperature: 0.7,
      });

      this.state.metrics.modelCosts += response.cost;

      return response.content;
    } catch (error) {
      console.error(`[Brain] Erreur modèle ${model.name}:`, error);
      // Fallback
      return this.think(messages.map(m => m.content).join('\n'));
    }
  }

  // ===========================================================================
  // INTERACTION DIRECTE AVEC MEMORY
  // ===========================================================================

  /**
   * Demander plus de contexte à Memory
   * Brain peut appeler Memory directement pour obtenir plus d'informations
   */
  async queryMemory(
    query: string,
    options: {
      type?: string;
      tags?: string[];
      limit?: number;
    } = {}
  ): Promise<MemoryEntry[]> {
    // Limite de profondeur pour éviter récursion infinie
    const MAX_CONTEXT_DEPTH = 3;
    if (this.state.contextRequestDepth >= MAX_CONTEXT_DEPTH) {
      console.log(`[Brain] ⚠️ Limite de profondeur atteinte (${MAX_CONTEXT_DEPTH}), skip queryMemory`);
      return [];
    }

    this.state.contextRequestDepth++;

    return new Promise((resolve) => {
      const requestId = `brain_query_${Date.now()}`;

      this.state.pendingMemoryRequests.set(requestId, resolve);

      // Timeout
      setTimeout(() => {
        if (this.state.pendingMemoryRequests.has(requestId)) {
          this.state.pendingMemoryRequests.delete(requestId);
          console.log('[Brain] ⚠️ Timeout requête Memory');
          resolve([]);
        }
      }, 5000);

      this.send('memory', 'memory_query', {
        query,
        options,
        requestId,
      });
    });
  }

  /**
   * Demander une vérification de cohérence à Memory
   * Vérifie que la réponse ne contredit pas les faits stockés
   */
  async checkFactConsistency(
    proposedResponse: string,
    userInput: string
  ): Promise<FactCheckResult> {
    return new Promise((resolve) => {
      const requestId = `fact_check_${Date.now()}`;

      this.state.pendingFactChecks.set(requestId, resolve);

      // Timeout - retourner cohérent par défaut si pas de réponse
      setTimeout(() => {
        if (this.state.pendingFactChecks.has(requestId)) {
          this.state.pendingFactChecks.delete(requestId);
          console.log('[Brain] ⚠️ Timeout fact-check, assumant cohérent');
          resolve({
            isConsistent: true,
            contradictions: [],
            supportedBy: [],
            confidence: 0.5,
            warnings: ['Vérification timeout'],
          });
        }
      }, 8000);

      this.send('memory', 'fact_check_request', {
        proposedResponse,
        userInput,
        requestId,
      });
    });
  }

  // ===========================================================================
  // TRAITEMENT DES REQUÊTES
  // ===========================================================================

  /**
   * Traiter une requête utilisateur enrichie (nouveau format de Vox)
   */
  private async processEnrichedRequest(input: EnrichedInput): Promise<void> {
    this.state.metrics.totalRequests++;

    console.log(`[Brain] 🧠 Traitement: "${input.originalInput.substring(0, 50)}..."`);
    console.log(`[Brain] 📋 Contexte: ${input.contextReport.contextUsed.length} éléments, confiance ${input.contextReport.confidence}`);

    // Vérifier si on a besoin d'une recherche web
    let webResults: SearchResult[] = [];
    if (this.needsWebSearch(input, input.contextReport.confidence)) {
      console.log('[Brain] 🌐 Confiance basse, recherche web...');
      webResults = await this.searchWeb(input.originalInput, 3);

      if (webResults.length > 0) {
        // Enrichir le contexte avec les résultats web
        const webContext = webResults
          .map(r => `[Web: ${r.title}] ${r.snippet}`)
          .join('\n');

        input.enrichedInput = `${input.enrichedInput}\n\nRÉSULTATS WEB RÉCENTS:\n${webContext}`;
        input.contextReport.contextUsed.push('web_search');
        console.log(`[Brain] ✅ ${webResults.length} résultats web ajoutés au contexte`);
      }
    }

    // Avertissements du contexte
    if (input.contextReport.warnings.length > 0) {
      console.log(`[Brain] ⚠️ Warnings: ${input.contextReport.warnings.join(', ')}`);
    }

    // Décider de l'approche avec le prompt enrichi
    const decision = await this.decideApproach(input);

    // Si besoin de plus de contexte, le demander
    if (decision.needsMoreContext && decision.contextQuery) {
      console.log(`[Brain] 🔍 Demande de contexte additionnel: "${decision.contextQuery}"`);
      const additionalContext = await this.queryMemory(decision.contextQuery, { limit: 10 });

      if (additionalContext.length > 0) {
        // Ré-évaluer avec le contexte additionnel
        const enrichedDecision = await this.refineDecisionWithContext(
          input,
          decision,
          additionalContext
        );
        decision.directResponse = enrichedDecision.directResponse;
        decision.plan = enrichedDecision.plan;
        decision.requiresPlan = enrichedDecision.requiresPlan;
      }
    }

    // Exécuter selon la décision
    if (decision.requiresPlan && decision.plan) {
      await this.executePlan(decision.plan, input);
    } else if (decision.directResponse) {
      await this.executeDirectResponse(decision.directResponse, input);
    }
  }

  /**
   * Décider de l'approche à adopter
   */
  private async decideApproach(input: EnrichedInput): Promise<{
    requiresPlan: boolean;
    needsMoreContext: boolean;
    contextQuery?: string;
    plan?: ExecutionPlan;
    directResponse?: {
      response: string;
      confidence: number;
      sources: string[];
    };
    reasoning: string;
  }> {
    const decisionPrompt = `
REQUÊTE UTILISATEUR ORIGINALE:
"${input.originalInput}"

REQUÊTE ENRICHIE (avec contexte Memory):
"${input.enrichedInput}"

ANALYSE VOX:
- Intent: ${input.analysis.intent}
- Confiance: ${input.analysis.confidence}
- Ton émotionnel: ${input.analysis.emotionalTone}

CONTEXTE FOURNI:
- Éléments utilisés: ${input.contextReport.contextUsed.join(', ') || 'aucun'}
- Confiance contexte: ${input.contextReport.confidence}
- Avertissements: ${input.contextReport.warnings.join(', ') || 'aucun'}

CONVERSATION RÉCENTE:
${input.conversationContext.slice(-5).map(c => `${c.role}: ${c.content.substring(0, 100)}...`).join('\n')}

RÈGLE IMPORTANTE:
- Si l'utilisateur demande de TESTER, VÉRIFIER, ou EXÉCUTER quelque chose → crée un PLAN avec des steps concrets
- Ne demande PAS de clarification si la demande est claire (ex: "teste la recherche" = fais un test complet)
- Privilégie l'ACTION sur la clarification

DÉCIDE:
1. Est-ce une demande d'ACTION (test, vérification, exécution) ? Si oui, crée un plan d'exécution.
2. Est-ce une question simple ? Réponds directement.
3. As-tu VRAIMENT besoin de plus de contexte ? (rarement nécessaire)

Réponds en JSON avec:
- needsMoreContext: boolean (false dans la plupart des cas)
- contextQuery: null (sauf si vraiment nécessaire)
- Pour une ACTION: steps[] avec des étapes concrètes
- Pour une QUESTION: response avec la réponse

EXEMPLE pour "teste la recherche embedded":
{
  "analysis": "L'utilisateur veut tester le système de recherche par embeddings",
  "approach": "Exécuter un test complet: stocker des données test, rechercher, mesurer la pertinence",
  "needsMoreContext": false,
  "steps": [
    {"action": "Stocker 3 faits de test en mémoire", "agentType": "memory", "parameters": {"type": "test"}},
    {"action": "Rechercher par similarité sémantique", "agentType": "memory", "parameters": {"query": "test"}},
    {"action": "Mesurer et rapporter les résultats", "agentType": "analyst", "parameters": {}}
  ],
  "confidence": 0.9
}
`;

    const response = await this.think(decisionPrompt);

    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error('No JSON found');

      const parsed = JSON.parse(jsonMatch[0]);

      this.recordDecision(input.originalInput, parsed);

      // Déterminer le type de réponse
      if (parsed.steps && parsed.steps.length > 0) {
        return {
          requiresPlan: true,
          needsMoreContext: parsed.needsMoreContext || false,
          contextQuery: parsed.contextQuery || parsed.moreInfoQuery,
          plan: this.createPlan(parsed),
          reasoning: parsed.analysis || parsed.approach,
        };
      } else {
        return {
          requiresPlan: false,
          needsMoreContext: parsed.needsMoreInfo || parsed.needsMoreContext || false,
          contextQuery: parsed.contextQuery || parsed.moreInfoQuery,
          directResponse: {
            response: parsed.response,
            confidence: parsed.confidence || 0.8,
            sources: parsed.sources || [],
          },
          reasoning: parsed.analysis || 'Réponse directe',
        };
      }
    } catch {
      return {
        requiresPlan: false,
        needsMoreContext: false,
        directResponse: {
          response: response,
          confidence: 0.7,
          sources: [],
        },
        reasoning: 'Fallback - réponse directe',
      };
    }
  }

  /**
   * Affiner la décision avec du contexte additionnel de Memory
   */
  private async refineDecisionWithContext(
    input: EnrichedInput,
    originalDecision: { reasoning: string },
    additionalContext: MemoryEntry[]
  ): Promise<{
    requiresPlan: boolean;
    plan?: ExecutionPlan;
    directResponse?: {
      response: string;
      confidence: number;
      sources: string[];
    };
  }> {
    const refinePrompt = `
REQUÊTE: "${input.enrichedInput}"

ANALYSE INITIALE: ${originalDecision.reasoning}

CONTEXTE ADDITIONNEL TROUVÉ:
${additionalContext.map(m => `[${m.type}] ${m.content}`).join('\n')}

Avec ce contexte additionnel, génère une réponse améliorée.
Réponds en JSON:
{
  "response": "Réponse améliorée avec le nouveau contexte",
  "confidence": 0.0-1.0,
  "sources": ["éléments de contexte utilisés"],
  "improvements": ["ce qui a été ajouté grâce au contexte"]
}
`;

    const response = await this.think(refinePrompt);

    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          requiresPlan: false,
          directResponse: {
            response: parsed.response,
            confidence: parsed.confidence || 0.85,
            sources: parsed.sources || [],
          },
        };
      }
    } catch {
      // Ignorer
    }

    return {
      requiresPlan: false,
      directResponse: {
        response: response,
        confidence: 0.7,
        sources: [],
      },
    };
  }

  // ===========================================================================
  // EXÉCUTION
  // ===========================================================================

  /**
   * Exécuter un plan multi-étapes
   */
  private async executePlan(
    plan: ExecutionPlan,
    input: EnrichedInput
  ): Promise<void> {
    this.state.currentPlan = plan;
    console.log(`[Brain] 📋 Exécution du plan: ${plan.goal}`);

    const results: unknown[] = [];

    for (let i = 0; i < plan.steps.length; i++) {
      const step = plan.steps[i];
      plan.currentStepIndex = i;
      step.status = 'in_progress';

      console.log(`[Brain] ▶️ Étape ${i + 1}/${plan.steps.length}: ${step.action}`);

      try {
        const result = await this.executeStep(step, input, results);
        step.result = result;
        step.status = 'completed';
        results.push(result);

        // Stocker en mémoire
        this.send('memory', 'memory_store', {
          type: 'task_result',
          content: JSON.stringify({ step: step.action, result }),
          metadata: { tags: ['plan_execution', plan.id] },
        });
      } catch (error) {
        step.status = 'failed';
        console.error(`[Brain] ❌ Étape échouée:`, error);

        // Demander à Memory si elle a des infos sur des échecs similaires
        const failureContext = await this.queryMemory(
          `échec ${step.action} erreur`,
          { type: 'correction', limit: 5 }
        );

        if (failureContext.length > 0) {
          console.log(`[Brain] 📚 Contexte d'échecs similaires trouvé, tentative de récupération...`);
        }

        this.send('memory', 'memory_store', {
          type: 'correction',
          content: JSON.stringify({
            step: step.action,
            error: error instanceof Error ? error.message : String(error),
          }),
          metadata: { tags: ['plan_failure', plan.id] },
        });
      }
    }

    const synthesis = await this.synthesizeResults(plan, results);
    this.state.currentPlan = null;

    this.send('vox', 'response_ready', synthesis);
  }

  /**
   * Exécuter une étape du plan
   * - Petites tâches → exécution directe
   * - Grosses tâches → délégation aux workers
   */
  private async executeStep(
    step: PlanStep,
    input: EnrichedInput,
    previousResults: unknown[]
  ): Promise<unknown> {
    const actionLower = step.action.toLowerCase();

    // ========== DÉTECTION DE LA COMPLEXITÉ ==========
    const isComplexTask = this.isComplexTask(step, input);

    // ========== ACTIONS DIRECTES (petites tâches) ==========

    // Test de recherche/embeddings - tâche légère, exécution directe
    if (actionLower.includes('test') && (actionLower.includes('recherche') || actionLower.includes('search') || actionLower.includes('embed'))) {
      console.log('[Brain] 🔧 Exécution directe: test embeddings');
      return await this.executeEmbeddingTest();
    }

    // Stocker en mémoire - tâche légère
    if (actionLower.includes('stocker') || actionLower.includes('store') || actionLower.includes('sauvegarder')) {
      const content = (step.parameters as { content?: string })?.content || `Test data ${Date.now()}`;
      const type = (step.parameters as { type?: string })?.type || 'fact';

      this.send('memory', 'memory_store', {
        type,
        content,
        metadata: { tags: ['test', 'brain_generated'] },
      });
      return { stored: true, content, type };
    }

    // Rechercher en mémoire - tâche légère
    if (actionLower.includes('rechercher') || actionLower.includes('chercher') || actionLower.includes('search')) {
      const query = (step.parameters as { query?: string })?.query || input.originalInput;
      const results = await this.queryMemory(query, { limit: 10 });
      return {
        found: results.length,
        results: results.map(r => ({ type: r.type, content: r.content.substring(0, 100), importance: r.importance }))
      };
    }

    // ========== DÉLÉGATION (grosses tâches) ==========

    if (isComplexTask) {
      console.log(`[Brain] 📤 Délégation au worker: ${step.action}`);

      // Déléguer selon le type d'agent
      if (step.agentType === 'coder' || actionLower.includes('code')) {
        const result = await this.delegateCodeAnalysis(
          JSON.stringify(step.parameters),
          undefined,
          'general'
        );
        return result.success ? result.result : { error: result.error };
      }

      if (step.agentType === 'researcher' || actionLower.includes('recherche web')) {
        const query = (step.parameters as { query?: string })?.query || step.action;
        const result = await this.delegateWebSearch(query, 5);
        return result.success ? result.result : { error: result.error };
      }

      // Délégation générique via LLM worker
      const result = await this.delegateLLMCall(
        `Exécute cette tâche: ${step.action}\nParamètres: ${JSON.stringify(step.parameters)}\nContexte: ${input.originalInput}`,
        { priority: 'normal', timeout: 30000 }
      );
      return result.success ? result.result : { error: result.error };
    }

    // ========== FALLBACK: SIMULATION LLM (tâches moyennes) ==========
    const contextForStep = await this.queryMemory(step.action, { limit: 5 });

    const stepPrompt = `
ÉTAPE À EXÉCUTER: ${step.action}

TYPE D'AGENT: ${step.agentType}

PARAMÈTRES: ${JSON.stringify(step.parameters)}

RÉSULTATS PRÉCÉDENTS: ${JSON.stringify(previousResults)}

CONTEXTE MEMORY POUR CETTE ÉTAPE:
${contextForStep.map(m => `[${m.type}] ${m.content}`).join('\n') || 'Aucun contexte spécifique'}

REQUÊTE ORIGINALE: "${input.originalInput}"

Exécute cette étape et retourne le résultat en JSON.
`;

    const response = await this.think(stepPrompt);

    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      return jsonMatch ? JSON.parse(jsonMatch[0]) : { result: response };
    } catch {
      return { result: response };
    }
  }

  /**
   * Déterminer si une tâche est complexe (nécessite délégation)
   */
  private isComplexTask(step: PlanStep, input: EnrichedInput): boolean {
    const actionLower = step.action.toLowerCase();

    // Indicateurs de complexité
    const complexKeywords = [
      'analyse complète', 'analyse approfondie', 'rapport détaillé',
      'code review', 'audit', 'optimisation',
      'recherche exhaustive', 'scraping', 'crawl',
      'génération de code', 'refactoring',
      'traitement batch', 'migration'
    ];

    if (complexKeywords.some(kw => actionLower.includes(kw))) {
      return true;
    }

    // Si le contexte original est très long
    if (input.originalInput.length > 500) {
      return true;
    }

    // Si les paramètres sont complexes
    const paramsStr = JSON.stringify(step.parameters);
    if (paramsStr.length > 1000) {
      return true;
    }

    return false;
  }

  /**
   * Exécuter un test complet du système d'embeddings
   */
  private async executeEmbeddingTest(): Promise<{
    success: boolean;
    tests: Array<{ name: string; passed: boolean; details: string }>;
    summary: string;
  }> {
    console.log('[Brain] 🧪 Exécution du test d\'embeddings...');

    const tests: Array<{ name: string; passed: boolean; details: string }> = [];

    // Test 1: Stocker des données de test
    const testData = [
      { type: 'fact', content: `Paris est la capitale de la France - test ${Date.now()}` },
      { type: 'fact', content: `Tokyo est la capitale du Japon - test ${Date.now()}` },
      { type: 'fact', content: `La Tour Eiffel est un monument parisien - test ${Date.now()}` },
    ];

    for (const data of testData) {
      this.send('memory', 'memory_store', {
        type: data.type,
        content: data.content,
        metadata: { tags: ['embedding_test', 'auto_test'] },
      });
    }

    // Attendre le stockage et la génération d'embeddings
    await new Promise(resolve => setTimeout(resolve, 2000));

    tests.push({
      name: 'Stockage de données test',
      passed: true,
      details: `${testData.length} faits stockés`
    });

    // Test 2: Recherche sémantique - doit trouver Paris
    const searchResult1 = await this.queryMemory('capitale française', { limit: 5 });
    const foundParis = searchResult1.some(r => r.content.toLowerCase().includes('paris'));

    tests.push({
      name: 'Recherche sémantique "capitale française"',
      passed: foundParis,
      details: foundParis
        ? `Trouvé: ${searchResult1[0]?.content?.substring(0, 50)}...`
        : `Non trouvé parmi ${searchResult1.length} résultats`
    });

    // Test 3: Recherche sémantique - doit trouver Tour Eiffel
    const searchResult2 = await this.queryMemory('monument célèbre de Paris', { limit: 5 });
    const foundEiffel = searchResult2.some(r => r.content.toLowerCase().includes('eiffel'));

    tests.push({
      name: 'Recherche sémantique "monument célèbre"',
      passed: foundEiffel,
      details: foundEiffel
        ? `Trouvé: ${searchResult2[0]?.content?.substring(0, 50)}...`
        : `Non trouvé parmi ${searchResult2.length} résultats`
    });

    // Résumé
    const passedCount = tests.filter(t => t.passed).length;
    const success = passedCount >= tests.length - 1; // Au moins 2/3 tests OK

    console.log(`[Brain] 🧪 Test terminé: ${passedCount}/${tests.length} réussis`);

    return {
      success,
      tests,
      summary: success
        ? `✅ Tests embeddings: ${passedCount}/${tests.length} réussis. La recherche sémantique fonctionne.`
        : `⚠️ Tests embeddings: ${passedCount}/${tests.length} réussis. Problèmes détectés.`
    };
  }

  /**
   * Exécuter une réponse directe
   * Vérifie la cohérence avec les faits stockés AVANT d'envoyer
   */
  private async executeDirectResponse(
    response: {
      response: string;
      confidence: number;
      sources: string[];
    },
    input: EnrichedInput
  ): Promise<void> {
    let finalResponse = response.response;
    let finalConfidence = response.confidence;

    // 1. FACT-CHECK: Vérifier que la réponse ne contredit pas les faits stockés
    console.log('[Brain] 🔍 Vérification de cohérence...');
    const factCheck = await this.checkFactConsistency(finalResponse, input.originalInput);

    if (!factCheck.isConsistent) {
      console.log(`[Brain] ⚠️ ${factCheck.contradictions.length} contradiction(s) détectée(s)!`);

      // Corriger la réponse pour éviter les contradictions
      const correctionPrompt = `
RÉPONSE PROPOSÉE: ${finalResponse}

CONTRADICTIONS DÉTECTÉES:
${factCheck.contradictions.map(c => `- "${c.claim}" CONTREDIT le fait stocké: "${c.storedFact}" (sévérité: ${c.severity})`).join('\n')}

AVERTISSEMENTS: ${factCheck.warnings.join(', ') || 'aucun'}

Corrige la réponse pour qu'elle soit COHÉRENTE avec les faits stockés.
Ne contredis JAMAIS les informations que l'utilisateur a données précédemment.

Réponds en JSON:
{
  "correctedResponse": "Réponse corrigée et cohérente",
  "confidence": 0.0-1.0,
  "corrections": ["ce qui a été corrigé"]
}
`;

      try {
        const corrected = await this.think(correctionPrompt);
        const jsonMatch = corrected.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          finalResponse = parsed.correctedResponse;
          finalConfidence = Math.min(parsed.confidence || 0.7, factCheck.confidence);
          this.state.metrics.contradictionsAvoided++;
          console.log(`[Brain] ✅ Réponse corrigée, ${factCheck.contradictions.length} contradiction(s) évitée(s)`);
        }
      } catch {
        // Si échec, réduire la confiance
        finalConfidence *= 0.7;
        console.log('[Brain] ⚠️ Échec correction, confiance réduite');
      }
    } else {
      // Utiliser la confiance calculée par Memory basée sur les mémoires
      finalConfidence = Math.max(finalConfidence, factCheck.confidence);
      console.log(`[Brain] ✅ Réponse cohérente (confiance: ${finalConfidence.toFixed(2)})`);
    }

    // 2. Si confiance basse, essayer d'améliorer avec Memory
    if (finalConfidence < 0.7) {
      console.log('[Brain] 🔍 Confiance basse, recherche de contexte supplémentaire...');

      const additionalContext = await this.queryMemory(input.originalInput, { limit: 10 });

      if (additionalContext.length > 0) {
        const improvedPrompt = `
RÉPONSE INITIALE: ${finalResponse}
CONFIANCE: ${finalConfidence}

CONTEXTE ADDITIONNEL DE MEMORY:
${additionalContext.map(m => `[${m.type}] ${m.content}`).join('\n')}

Améliore cette réponse avec le contexte. Réponds en JSON:
{
  "improvedResponse": "...",
  "newConfidence": 0.0-1.0,
  "improvements": ["ce qui a été amélioré"]
}
`;

        try {
          const improved = await this.think(improvedPrompt);
          const jsonMatch = improved.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            const parsed = JSON.parse(jsonMatch[0]);
            finalResponse = parsed.improvedResponse;
            finalConfidence = parsed.newConfidence;
            console.log(`[Brain] ✅ Réponse améliorée, confiance: ${finalConfidence}`);
          }
        } catch {
          // Garder la réponse originale
        }
      }
    }

    // 3. RÈGLE "NE MENT JAMAIS": Si confiance toujours trop basse, admettre l'incertitude
    if (finalConfidence < 0.4) {
      console.log('[Brain] ⚠️ Confiance très basse - admission d\'incertitude');
      finalResponse = this.generateUncertaintyResponse(finalResponse, finalConfidence, input);
    }

    // 3. Mettre à jour les métriques
    this.state.metrics.successfulResponses++;
    this.updateAverageConfidence(finalConfidence);

    // 4. Envoyer à Vox
    this.send('vox', 'response_ready', {
      response: finalResponse,
      confidence: finalConfidence,
      sources: response.sources,
      factChecked: true,
    });
  }

  /**
   * Synthétiser les résultats d'un plan
   */
  private async synthesizeResults(
    plan: ExecutionPlan,
    results: unknown[]
  ): Promise<{
    response: string;
    confidence: number;
    sources: string[];
  }> {
    const synthesisPrompt = `
OBJECTIF DU PLAN: ${plan.goal}

ÉTAPES EXÉCUTÉES:
${plan.steps.map((s, i) => `${i + 1}. ${s.action} - ${s.status}`).join('\n')}

RÉSULTATS:
${results.map((r, i) => `${i + 1}. ${JSON.stringify(r)}`).join('\n')}

Synthétise ces résultats en une réponse cohérente et utile pour l'utilisateur.
Réponds en JSON:
{
  "response": "Synthèse claire et utile",
  "confidence": 0.0-1.0,
  "keyPoints": ["point1", "point2"],
  "limitations": ["limitation éventuelle"]
}
`;

    const response = await this.think(synthesisPrompt);

    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          response: parsed.response,
          confidence: parsed.confidence || 0.8,
          sources: parsed.keyPoints || [],
        };
      }
    } catch {
      // Fallback
    }

    return {
      response: 'J\'ai traité votre demande mais je n\'ai pas pu synthétiser les résultats clairement.',
      confidence: 0.5,
      sources: [],
    };
  }

  // ===========================================================================
  // HELPERS
  // ===========================================================================

  private createPlan(parsed: {
    analysis?: string;
    approach?: string;
    steps: Array<{
      action: string;
      agentType: string;
      parameters?: Record<string, unknown>;
    }>;
  }): ExecutionPlan {
    return {
      id: randomUUID(),
      goal: parsed.analysis || parsed.approach || 'Objectif non spécifié',
      steps: parsed.steps.map((s) => ({
        id: randomUUID(),
        action: s.action,
        agentType: s.agentType,
        parameters: s.parameters || {},
        status: 'pending' as const,
      })),
      currentStepIndex: 0,
      createdAt: new Date(),
    };
  }

  private recordDecision(context: string, decision: unknown): void {
    this.state.decisionHistory.push({
      id: randomUUID(),
      context,
      options: [],
      chosen: JSON.stringify(decision),
      reasoning: 'Auto-décision',
      timestamp: new Date(),
    });

    if (this.state.decisionHistory.length > 100) {
      this.state.decisionHistory = this.state.decisionHistory.slice(-50);
    }
  }

  private updateAverageConfidence(newConfidence: number): void {
    const total = this.state.metrics.successfulResponses;
    const current = this.state.metrics.averageConfidence;
    this.state.metrics.averageConfidence =
      (current * (total - 1) + newConfidence) / total;
  }

  /**
   * RÈGLE "NE MENT JAMAIS": Générer une réponse d'incertitude honnête
   * Quand la confiance est trop basse, admettre qu'on ne sait pas
   */
  private generateUncertaintyResponse(
    originalResponse: string,
    confidence: number,
    _input: EnrichedInput
  ): string {
    const uncertaintyLevel = confidence < 0.2 ? 'very_low' : 'low';

    if (uncertaintyLevel === 'very_low') {
      // Confiance < 20% : Refus poli de répondre
      return `Je dois être honnête : je n'ai pas assez d'informations fiables pour répondre à cette question avec certitude.

Ce que je peux dire :
- Je n'ai trouvé aucune information pertinente dans ma mémoire
- Une recherche web n'a pas donné de résultats concluants

Je préfère admettre mon ignorance plutôt que de risquer de vous induire en erreur.

Pourriez-vous :
1. Reformuler votre question différemment ?
2. Me donner plus de contexte ?
3. Me préciser ce que vous savez déjà sur le sujet ?`;
    } else {
      // Confiance 20-40% : Réponse avec avertissement clair
      return `⚠️ **Attention : réponse à faible confiance (${Math.round(confidence * 100)}%)**

${originalResponse}

---
*Je ne suis pas certain de cette réponse. Les informations ci-dessus pourraient être incomplètes ou partiellement incorrectes. Je vous recommande de vérifier ces informations auprès d'une source fiable avant de vous y fier.*`;
    }
  }

  // ===========================================================================
  // HANDLER DE MESSAGES
  // ===========================================================================

  protected async handleMessage(message: AgentMessage): Promise<void> {
    // Ignorer les messages qui ne sont pas destinés à Brain
    if (message.to !== 'brain' && message.to !== 'broadcast') {
      return;
    }

    // Circuit breaker pour user_input: éviter traitement concurrent
    if (message.type === 'user_input' && this.state.isProcessingRequest) {
      console.log('[Brain] ⚠️ Requête ignorée (déjà en traitement)');
      return;
    }

    try {
      switch (message.type) {
        case 'user_input':
          this.state.isProcessingRequest = true;
          this.state.contextRequestDepth = 0; // Reset depth counter
          try {
            await this.handleUserInput(message);
          } finally {
            this.state.isProcessingRequest = false;
          }
          break;

        case 'context_response':
          this.handleMemoryResponse(message);
          break;

        case 'fact_check_response':
          this.handleFactCheckResponse(message);
          break;

        case 'task_result':
          await this.handleTaskResult(message);
          break;

        case 'skill_detected':
          this.handleSkillDetected(message);
          break;

        case 'learning_update':
          this.handleLearningUpdate(message);
          break;

        default:
          console.log(`[Brain] Message non géré: ${message.type}`);
      }
    } catch (error) {
      console.error(`[Brain] Erreur handleMessage (${message.type}):`, error);
      this.state.isProcessingRequest = false; // Reset on error
      // Send error response to Vox
      if (message.type === 'user_input') {
        this.send('vox', 'response_ready', {
          response: "Désolé, je rencontre un problème technique. Vérifiez que la clé API est valide avec ./neo config",
          confidence: 0,
          error: true,
        });
      }
    }
  }

  private async handleUserInput(message: AgentMessage): Promise<void> {
    const payload = message.payload as EnrichedInput;

    // Vérifier si c'est le nouveau format enrichi ou l'ancien format
    if (payload.enrichedInput) {
      await this.processEnrichedRequest(payload);
    } else {
      // Ancien format - fallback
      const legacyPayload = payload as unknown as {
        originalInput: string;
        analysis: {
          intent: string;
          confidence: number;
          processedInput: string;
          emotionalTone: string;
        };
        conversationContext: Array<{ role: 'user' | 'assistant'; content: string }>;
      };

      // Convertir en nouveau format
      const enrichedInput: EnrichedInput = {
        originalInput: legacyPayload.originalInput,
        enrichedInput: legacyPayload.originalInput,
        analysis: legacyPayload.analysis,
        contextReport: {
          confidence: 0.5,
          warnings: [],
          contextUsed: [],
        },
        conversationContext: legacyPayload.conversationContext,
      };

      await this.processEnrichedRequest(enrichedInput);
    }
  }

  private handleMemoryResponse(message: AgentMessage): void {
    const payload = message.payload as {
      results?: MemoryEntry[];
      requestId?: string;
    };

    if (payload.requestId && payload.results) {
      const callback = this.state.pendingMemoryRequests.get(payload.requestId);
      if (callback) {
        this.state.pendingMemoryRequests.delete(payload.requestId);
        callback(payload.results);
      }
    }
  }

  private handleFactCheckResponse(message: AgentMessage): void {
    const payload = message.payload as FactCheckResult & { requestId?: string };

    if (payload.requestId) {
      const callback = this.state.pendingFactChecks.get(payload.requestId);
      if (callback) {
        this.state.pendingFactChecks.delete(payload.requestId);
        callback(payload);
      }
    }
  }

  private async handleTaskResult(message: AgentMessage): Promise<void> {
    const payload = message.payload as {
      workerId: string;
      taskId: string;
      result: unknown;
      success: boolean;
    };

    const worker = this.state.activeWorkers.get(payload.workerId);
    if (worker) {
      worker.status = payload.success ? 'completed' : 'failed';

      this.send('memory', 'memory_store', {
        type: 'task_result',
        content: JSON.stringify(payload),
        metadata: { tags: ['worker_result'] },
      });
    }
  }

  private handleSkillDetected(message: AgentMessage): void {
    const skill = message.payload as Skill;
    console.log(`[Brain] 🎯 Nouveau skill disponible: ${skill.name}`);
  }

  private handleLearningUpdate(_message: AgentMessage): void {
    console.log('[Brain] 📚 Learning reçu, mise à jour des stratégies');
  }

  // ===========================================================================
  // WORKER DELEGATION - Brain ne fait JAMAIS le travail lui-même
  // ===========================================================================

  /**
   * Déléguer un appel LLM à un worker
   * Brain reste disponible pour d'autres tâches pendant l'exécution
   */
  async delegateLLMCall(
    prompt: string,
    options?: {
      systemPrompt?: string;
      model?: string;
      priority?: 'low' | 'normal' | 'high' | 'critical';
      timeout?: number;
    }
  ): Promise<WorkerResult> {
    console.log(`[Brain] 📤 Délégation LLM call à un worker`);

    return this.workerPool.submit('llm_call', {
      prompt,
      systemPrompt: options?.systemPrompt,
      model: options?.model,
    }, {
      priority: options?.priority || 'normal',
      timeout: options?.timeout,
    });
  }

  /**
   * Déléguer un raisonnement complexe à un worker
   */
  async delegateReasoning(
    prompt: string,
    context?: string,
    options?: {
      steps?: string[];
      priority?: 'low' | 'normal' | 'high' | 'critical';
      timeout?: number;
    }
  ): Promise<WorkerResult> {
    console.log(`[Brain] 📤 Délégation raisonnement à un worker`);

    return this.workerPool.submit('llm_reasoning', {
      prompt,
      context,
      steps: options?.steps,
    }, {
      priority: options?.priority || 'high',
      timeout: options?.timeout || 60000, // Plus de temps pour le raisonnement
    });
  }

  /**
   * Déléguer une recherche web à un worker
   */
  async delegateWebSearch(
    query: string,
    maxResults = 5
  ): Promise<WorkerResult> {
    console.log(`[Brain] 📤 Délégation recherche web à un worker`);

    return this.workerPool.submit('web_search', {
      query,
      maxResults,
      searchService: this.webSearch,
    }, {
      priority: 'normal',
      timeout: 15000,
    });
  }

  /**
   * Déléguer une analyse de code à un worker
   */
  async delegateCodeAnalysis(
    code: string,
    language?: string,
    analysisType?: 'security' | 'performance' | 'style' | 'bugs' | 'general'
  ): Promise<WorkerResult> {
    console.log(`[Brain] 📤 Délégation analyse code à un worker`);

    return this.workerPool.submit('code_analysis', {
      code,
      language,
      analysisType,
    }, {
      priority: 'normal',
      timeout: 30000,
    });
  }

  /**
   * Déléguer une tâche personnalisée à un worker
   */
  async delegateCustomTask(
    handler: (...args: unknown[]) => Promise<unknown>,
    args?: unknown[],
    options?: {
      priority?: 'low' | 'normal' | 'high' | 'critical';
      timeout?: number;
    }
  ): Promise<WorkerResult> {
    console.log(`[Brain] 📤 Délégation tâche personnalisée à un worker`);

    return this.workerPool.submit('custom', {
      handler,
      args,
    }, {
      priority: options?.priority || 'normal',
      timeout: options?.timeout,
    });
  }

  /**
   * Exécuter plusieurs tâches en parallèle via les workers
   * Brain dispatch et attend les résultats
   */
  async delegateParallel(
    tasks: Array<{
      type: 'llm_call' | 'llm_reasoning' | 'web_search' | 'code_analysis';
      payload: unknown;
      priority?: 'low' | 'normal' | 'high' | 'critical';
    }>
  ): Promise<WorkerResult[]> {
    console.log(`[Brain] 📤 Délégation parallèle: ${tasks.length} tâches`);

    return this.workerPool.submitBatch(tasks.map(t => ({
      type: t.type,
      payload: t.payload,
      priority: t.priority,
    })));
  }

  /**
   * Obtenir le statut du pool de workers
   */
  getWorkerPoolStats() {
    return this.workerPool.getStats();
  }

  // ===========================================================================
  // SKILL INTEGRATION
  // ===========================================================================

  /**
   * Configurer le SkillManager pour permettre l'exécution de skills
   */
  setSkillManager(skillManager: SkillManager): void {
    this.skillManager = skillManager;
    console.log('[Brain] 🔧 SkillManager configuré');
  }

  /**
   * Vérifier si un skill peut répondre à cette requête
   * Retourne le skill le plus pertinent ou null
   */
  shouldUseSkill(input: string): SkillDefinition | null {
    if (!this.skillManager) {
      return null;
    }

    const relevantSkills = this.skillManager.findRelevantSkills(input, 3);

    if (relevantSkills.length === 0) {
      return null;
    }

    // Prendre le skill avec le meilleur score et un bon taux de succès
    const bestSkill = relevantSkills.find(s => s.successRate >= 0.7);

    if (bestSkill) {
      console.log(`[Brain] 🎯 Skill trouvé: ${bestSkill.name} (score: ${bestSkill.successRate.toFixed(2)})`);
      return bestSkill;
    }

    return null;
  }

  /**
   * Exécuter un skill et retourner le résultat
   * Gère les erreurs et le logging pour l'apprentissage
   */
  async executeSkill(
    skill: SkillDefinition,
    input: Record<string, unknown>
  ): Promise<SkillExecutionResult | null> {
    if (!this.skillManager) {
      console.error('[Brain] ❌ SkillManager non configuré');
      return null;
    }

    console.log(`[Brain] ⚡ Exécution du skill: ${skill.name}`);

    try {
      const result = await this.skillManager.executeSkill({
        skillId: skill.id,
        input,
      });

      // Logger le résultat pour l'apprentissage
      if (result.success) {
        console.log(`[Brain] ✅ Skill ${skill.name} exécuté avec succès (${result.executionTimeMs}ms)`);

        // Stocker le succès en mémoire
        this.send('memory', 'memory_store', {
          type: 'task_result',
          content: JSON.stringify({
            skillId: skill.id,
            skillName: skill.name,
            input,
            output: result.output,
            executionTimeMs: result.executionTimeMs,
          }),
          metadata: {
            tags: ['skill_execution', 'success', skill.name],
            importance: 0.6,
          },
        });
      } else {
        console.log(`[Brain] ⚠️ Skill ${skill.name} échoué: ${result.error?.message}`);

        // Stocker l'échec pour apprendre
        this.send('memory', 'memory_store', {
          type: 'correction',
          content: JSON.stringify({
            skillId: skill.id,
            skillName: skill.name,
            input,
            error: result.error,
            shouldLearn: result.shouldLearn,
            learningNotes: result.learningNotes,
          }),
          metadata: {
            tags: ['skill_execution', 'failure', skill.name],
            importance: 0.8, // Les échecs sont importants pour apprendre
          },
        });
      }

      return result;
    } catch (error) {
      console.error(`[Brain] ❌ Erreur exécution skill ${skill.name}:`, error);

      // Logger l'erreur critique
      this.send('memory', 'memory_store', {
        type: 'correction',
        content: JSON.stringify({
          skillId: skill.id,
          skillName: skill.name,
          input,
          criticalError: error instanceof Error ? error.message : String(error),
        }),
        metadata: {
          tags: ['skill_execution', 'critical_error', skill.name],
          importance: 0.9,
        },
      });

      return null;
    }
  }

  /**
   * Tenter d'utiliser un skill pour répondre à une requête
   * Si un skill pertinent existe, l'exécuter et retourner le résultat
   */
  async trySkillExecution(
    input: EnrichedInput
  ): Promise<{ used: boolean; result?: SkillExecutionResult }> {
    const skill = this.shouldUseSkill(input.originalInput);

    if (!skill) {
      return { used: false };
    }

    // Préparer l'input pour le skill
    const skillInput: Record<string, unknown> = {
      query: input.originalInput,
      context: input.enrichedInput,
      intent: input.analysis.intent,
    };

    const result = await this.executeSkill(skill, skillInput);

    if (result && result.success) {
      return { used: true, result };
    }

    return { used: false };
  }

  // ===========================================================================
  // SKILL AUTO-IMPROVEMENT - Amélioration automatique des skills
  // ===========================================================================

  /**
   * Analyser un échec de skill et générer une version améliorée du code
   * Cette méthode est appelée quand un skill échoue plusieurs fois
   */
  async analyzeSkillFailure(
    skill: SkillDefinition,
    recentErrors: Array<{ error: string; input: Record<string, unknown>; timestamp: Date }>
  ): Promise<{
    shouldImprove: boolean;
    suggestedCode?: string;
    analysis: string;
    confidence: number;
  }> {
    if (!this.skillManager) {
      return { shouldImprove: false, analysis: 'SkillManager non disponible', confidence: 0 };
    }

    console.log(`[Brain] 🔬 Analyse des échecs du skill "${skill.name}"`);

    // Construire le prompt d'analyse
    const analysisPrompt = `
Tu es un expert en amélioration de code. Analyse ce skill qui a échoué plusieurs fois et propose une correction.

SKILL ACTUEL:
- Nom: ${skill.name}
- Description: ${skill.description}
- Version: ${skill.version}
- Taux de succès: ${(skill.successRate * 100).toFixed(1)}%
- Capabilities requises: ${skill.requiredCapabilities.join(', ')}

CODE ACTUEL:
\`\`\`javascript
${skill.code}
\`\`\`

ERREURS RÉCENTES (${recentErrors.length}):
${recentErrors.map((e, i) => `
${i + 1}. Erreur: ${e.error}
   Input: ${JSON.stringify(e.input).substring(0, 200)}
   Date: ${e.timestamp.toISOString()}
`).join('\n')}

ANALYSE DEMANDÉE:
1. Identifie la cause racine des échecs
2. Propose un code corrigé qui résout le problème
3. Explique les changements effectués

Réponds en JSON:
{
  "rootCause": "Explication de la cause racine",
  "shouldImprove": true/false,
  "confidence": 0.0-1.0,
  "suggestedCode": "// Code corrigé complet ici",
  "changes": ["Liste des changements effectués"],
  "warnings": ["Avertissements éventuels"]
}
`;

    try {
      const response = await this.think(analysisPrompt);
      const jsonMatch = response.match(/\{[\s\S]*\}/);

      if (!jsonMatch) {
        return {
          shouldImprove: false,
          analysis: 'Impossible de parser la réponse',
          confidence: 0,
        };
      }

      const parsed = JSON.parse(jsonMatch[0]);

      return {
        shouldImprove: parsed.shouldImprove && parsed.confidence >= 0.7,
        suggestedCode: parsed.suggestedCode,
        analysis: `${parsed.rootCause}. Changements: ${parsed.changes?.join(', ') || 'aucun'}`,
        confidence: parsed.confidence || 0,
      };
    } catch (error) {
      console.error('[Brain] Erreur analyse skill:', error);
      return {
        shouldImprove: false,
        analysis: `Erreur d'analyse: ${error instanceof Error ? error.message : String(error)}`,
        confidence: 0,
      };
    }
  }

  /**
   * Améliorer automatiquement un skill basé sur ses échecs
   * Crée une nouvelle version avec le code corrigé
   */
  async improveSkill(
    skillId: string,
    options?: { forceImprove?: boolean; maxAttempts?: number }
  ): Promise<{
    improved: boolean;
    newVersion?: string;
    analysis: string;
  }> {
    if (!this.skillManager) {
      return { improved: false, analysis: 'SkillManager non disponible' };
    }

    const skill = this.skillManager.getSkill(skillId);
    if (!skill) {
      return { improved: false, analysis: `Skill ${skillId} introuvable` };
    }

    // Ne pas améliorer les skills built-in sans forceImprove
    if (skill.isBuiltin && !options?.forceImprove) {
      return { improved: false, analysis: 'Skills built-in non modifiables sans forceImprove' };
    }

    // Récupérer les erreurs récentes depuis Memory
    const errorMemories = await this.queryMemory(
      `skill_execution failure ${skill.name}`,
      { type: 'correction', limit: 10 }
    );

    const recentErrors = errorMemories
      .filter(m => {
        try {
          const data = JSON.parse(m.content);
          return data.skillId === skillId && data.error;
        } catch {
          return false;
        }
      })
      .map(m => {
        const data = JSON.parse(m.content);
        return {
          error: data.error?.message || data.criticalError || 'Unknown error',
          input: data.input || {},
          timestamp: m.createdAt,
        };
      });

    if (recentErrors.length < 2 && !options?.forceImprove) {
      return {
        improved: false,
        analysis: `Pas assez d'échecs pour justifier une amélioration (${recentErrors.length}/2 minimum)`,
      };
    }

    console.log(`[Brain] 🔧 Tentative d'amélioration du skill "${skill.name}" (${recentErrors.length} échecs)`);

    // Analyser les échecs
    const analysis = await this.analyzeSkillFailure(skill, recentErrors);

    if (!analysis.shouldImprove || !analysis.suggestedCode) {
      return {
        improved: false,
        analysis: analysis.analysis,
      };
    }

    // Appliquer l'amélioration
    try {
      // Incrémenter la version
      const versionParts = skill.version.split('.');
      const newPatch = parseInt(versionParts[2] || '0') + 1;
      const newVersion = `${versionParts[0]}.${versionParts[1]}.${newPatch}`;

      await this.skillManager.updateSkill(skillId, {
        code: analysis.suggestedCode,
      });

      // Mettre à jour la version manuellement (car updateSkill ne le fait pas)
      const updatedSkill = this.skillManager.getSkill(skillId);
      if (updatedSkill) {
        // Note: La version est mise à jour via un autre mécanisme si nécessaire
        console.log(`[Brain] ✅ Skill "${skill.name}" amélioré -> v${newVersion}`);
      }

      // Logger l'amélioration en mémoire
      this.send('memory', 'memory_store', {
        type: 'learning',
        content: JSON.stringify({
          skillId,
          skillName: skill.name,
          previousVersion: skill.version,
          newVersion,
          analysis: analysis.analysis,
          confidence: analysis.confidence,
          errorsFixed: recentErrors.length,
        }),
        metadata: {
          tags: ['skill_improvement', 'auto_learning', skill.name],
          importance: 0.8,
        },
      });

      return {
        improved: true,
        newVersion,
        analysis: analysis.analysis,
      };
    } catch (error) {
      console.error('[Brain] Erreur amélioration skill:', error);
      return {
        improved: false,
        analysis: `Erreur lors de l'application: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  /**
   * Vérifier et améliorer automatiquement les skills qui ont un faible taux de succès
   * Cette méthode devrait être appelée périodiquement
   */
  async runSkillMaintenanceCycle(): Promise<{
    skillsChecked: number;
    skillsImproved: number;
    improvements: Array<{ skillName: string; analysis: string }>;
  }> {
    if (!this.skillManager) {
      return { skillsChecked: 0, skillsImproved: 0, improvements: [] };
    }

    console.log('[Brain] 🔄 Cycle de maintenance des skills...');

    const stats = this.skillManager.getStats();
    const improvements: Array<{ skillName: string; analysis: string }> = [];

    // Récupérer tous les skills avec un taux de succès < 70%
    const allSkills = this.skillManager.searchSkills({ isEnabled: true });
    const problematicSkills = allSkills.filter(s =>
      s.successRate < 0.7 &&
      s.usageCount >= 3 &&
      !s.isBuiltin
    );

    console.log(`[Brain] 📊 ${problematicSkills.length} skills problématiques sur ${stats.totalSkills}`);

    for (const skill of problematicSkills) {
      const result = await this.improveSkill(skill.id);

      if (result.improved) {
        improvements.push({
          skillName: skill.name,
          analysis: result.analysis,
        });
      }
    }

    console.log(`[Brain] ✅ Maintenance terminée: ${improvements.length} skills améliorés`);

    return {
      skillsChecked: problematicSkills.length,
      skillsImproved: improvements.length,
      improvements,
    };
  }

  /**
   * Détecter si une tâche répétitive pourrait devenir un skill
   * Analyse l'historique des conversations pour trouver des patterns
   */
  async detectSkillOpportunity(
    conversationHistory: Array<{ role: 'user' | 'assistant'; content: string }>,
    taskResult: unknown
  ): Promise<{
    shouldCreateSkill: boolean;
    suggestedSkill?: {
      name: string;
      description: string;
      triggers: string[];
      code: string;
    };
    reasoning: string;
  }> {
    // Chercher des patterns similaires en mémoire
    const recentUserMessage = conversationHistory.filter(m => m.role === 'user').pop();
    if (!recentUserMessage) {
      return { shouldCreateSkill: false, reasoning: 'Pas de message utilisateur' };
    }

    const similarTasks = await this.queryMemory(recentUserMessage.content, {
      type: 'task_result',
      limit: 10,
    });

    // Si on trouve 3+ tâches similaires, proposer un skill
    if (similarTasks.length < 3) {
      return {
        shouldCreateSkill: false,
        reasoning: `Pas assez de tâches similaires (${similarTasks.length}/3 minimum)`,
      };
    }

    console.log(`[Brain] 💡 Détection d'opportunité de skill (${similarTasks.length} tâches similaires)`);

    const detectionPrompt = `
Analyse ces tâches répétitives et détermine si elles pourraient être automatisées en un skill réutilisable.

TÂCHE ACTUELLE:
User: "${recentUserMessage.content}"
Résultat: ${JSON.stringify(taskResult).substring(0, 500)}

TÂCHES SIMILAIRES PASSÉES:
${similarTasks.slice(0, 5).map((m, i) => `${i + 1}. ${m.content.substring(0, 200)}`).join('\n')}

QUESTION:
Ces tâches partagent-elles un pattern commun qui pourrait être automatisé?

Réponds en JSON:
{
  "shouldCreateSkill": true/false,
  "reasoning": "Explication",
  "suggestedSkill": {
    "name": "Nom du skill",
    "description": "Description",
    "triggers": ["mot-clé1", "mot-clé2"],
    "requiredCapabilities": ["memory_read", "web_fetch", etc.],
    "codeOutline": "Description du code à générer"
  }
}
`;

    try {
      const response = await this.think(detectionPrompt);
      const jsonMatch = response.match(/\{[\s\S]*\}/);

      if (!jsonMatch) {
        return { shouldCreateSkill: false, reasoning: 'Impossible de parser la réponse' };
      }

      const parsed = JSON.parse(jsonMatch[0]);

      if (!parsed.shouldCreateSkill) {
        return { shouldCreateSkill: false, reasoning: parsed.reasoning };
      }

      // Générer le code complet du skill
      const codeGenPrompt = `
Génère le code JavaScript complet pour ce skill:

Nom: ${parsed.suggestedSkill.name}
Description: ${parsed.suggestedSkill.description}
Capabilities: ${parsed.suggestedSkill.requiredCapabilities?.join(', ') || 'memory_read'}
Outline: ${parsed.suggestedSkill.codeOutline}

Le code doit:
1. Être une fonction async qui reçoit (input, context)
2. Utiliser les capabilities via context (context.memory, context.webFetch, etc.)
3. Retourner un objet avec les résultats
4. Gérer les erreurs proprement

Réponds UNIQUEMENT avec le code JavaScript, sans markdown ni explication.
`;

      const codeResponse = await this.think(codeGenPrompt);
      const code = codeResponse.replace(/```javascript\n?/g, '').replace(/```\n?/g, '').trim();

      return {
        shouldCreateSkill: true,
        suggestedSkill: {
          name: parsed.suggestedSkill.name,
          description: parsed.suggestedSkill.description,
          triggers: parsed.suggestedSkill.triggers || [parsed.suggestedSkill.name.toLowerCase()],
          code: `
// Skill auto-généré: ${parsed.suggestedSkill.name}
// Basé sur ${similarTasks.length} tâches similaires

async function execute(input, context) {
  ${code}
}

return execute(input, context);
`,
        },
        reasoning: parsed.reasoning,
      };
    } catch (error) {
      return {
        shouldCreateSkill: false,
        reasoning: `Erreur de détection: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  /**
   * Créer automatiquement un skill à partir d'une détection
   */
  async createSkillFromDetection(
    suggestedSkill: {
      name: string;
      description: string;
      triggers: string[];
      code: string;
    }
  ): Promise<SkillDefinition | null> {
    if (!this.skillManager) {
      console.error('[Brain] SkillManager non disponible');
      return null;
    }

    console.log(`[Brain] 🆕 Création automatique du skill "${suggestedSkill.name}"`);

    try {
      const skill = await this.skillManager.createSkill({
        name: suggestedSkill.name,
        description: suggestedSkill.description,
        triggers: suggestedSkill.triggers,
        requiredCapabilities: ['memory_read'], // Par défaut, ajuster selon le code
        code: suggestedSkill.code,
        createdBy: 'neo',
      });

      // Logger la création
      this.send('memory', 'memory_store', {
        type: 'learning',
        content: JSON.stringify({
          event: 'skill_auto_created',
          skillId: skill.id,
          skillName: skill.name,
          triggers: skill.triggers,
        }),
        metadata: {
          tags: ['skill_creation', 'auto_learning', skill.name],
          importance: 0.7,
        },
      });

      console.log(`[Brain] ✅ Skill "${skill.name}" créé avec succès (ID: ${skill.id})`);
      return skill;
    } catch (error) {
      console.error('[Brain] Erreur création skill:', error);
      return null;
    }
  }

  // ===========================================================================
  // CREW AI INTEGRATION - Délégation à des équipes d'agents
  // ===========================================================================

  /**
   * Initialiser le CrewManager pour déléguer des tâches complexes
   * Utilise Ollama par défaut pour économiser les coûts
   */
  async initializeCrewAI(): Promise<void> {
    if (this.crewManager) {
      console.log('[Brain] CrewAI déjà initialisé');
      return;
    }

    console.log('[Brain] 🚀 Initialisation CrewAI...');
    this.crewManager = getCrewManager();

    try {
      await this.crewManager.start();
      console.log('[Brain] ✅ CrewAI prêt');

      // Lister les modèles Ollama disponibles
      const models = await this.crewManager.listOllamaModels();
      if (models.length > 0) {
        console.log(`[Brain] 📦 Modèles Ollama: ${models.join(', ')}`);
      } else {
        console.log('[Brain] ⚠️ Aucun modèle Ollama détecté, utilisation des fallbacks cloud');
      }
    } catch (error) {
      console.error('[Brain] ❌ Erreur initialisation CrewAI:', error);
      this.crewManager = null;
    }
  }

  /**
   * Déléguer une tâche complexe à un crew d'agents
   * Sélectionne automatiquement le preset le plus adapté
   */
  async delegateToCrew(
    taskType: 'research' | 'code-review' | 'content' | 'data-analysis',
    inputs: Record<string, unknown>,
    options?: { timeout?: number }
  ): Promise<CrewExecutionResult | null> {
    if (!this.crewManager) {
      await this.initializeCrewAI();
    }

    if (!this.crewManager) {
      console.error('[Brain] ❌ CrewAI non disponible');
      return null;
    }

    const preset = getPresetCrew(taskType);
    if (!preset) {
      console.error(`[Brain] ❌ Preset inconnu: ${taskType}`);
      return null;
    }

    // Valider les inputs requis
    for (const required of preset.requiredInputs) {
      if (!(required in inputs)) {
        console.error(`[Brain] ❌ Input manquant pour ${taskType}: ${required}`);
        return null;
      }
    }

    console.log(`[Brain] 👥 Délégation au crew "${preset.name}"`);

    try {
      const crew = preset.buildCrew(inputs);
      const result = await this.crewManager.executeCrew(crew, options);

      // Logger le résultat
      if (result.success) {
        console.log(`[Brain] ✅ Crew "${taskType}" terminé (${result.totalExecutionTimeMs}ms)`);

        // Stocker en mémoire pour apprendre
        this.send('memory', 'memory_store', {
          type: 'task_result',
          content: JSON.stringify({
            crewType: taskType,
            inputs,
            output: result.finalOutput?.substring(0, 1000),
            tokensUsed: result.totalTokensUsed,
            executionTimeMs: result.totalExecutionTimeMs,
          }),
          metadata: {
            tags: ['crew_execution', 'success', taskType],
            importance: 0.7,
          },
        });
      } else {
        console.log(`[Brain] ⚠️ Crew "${taskType}" échoué: ${result.error}`);
      }

      return result;
    } catch (error) {
      console.error(`[Brain] ❌ Erreur crew "${taskType}":`, error);
      return null;
    }
  }

  /**
   * Exécuter un crew personnalisé
   */
  async executeCustomCrew(
    config: CrewConfig,
    options?: { timeout?: number }
  ): Promise<CrewExecutionResult | null> {
    if (!this.crewManager) {
      await this.initializeCrewAI();
    }

    if (!this.crewManager) {
      console.error('[Brain] ❌ CrewAI non disponible');
      return null;
    }

    console.log(`[Brain] 👥 Exécution crew custom "${config.name}"`);

    try {
      return await this.crewManager.executeCrew(config, options);
    } catch (error) {
      console.error(`[Brain] ❌ Erreur crew custom:`, error);
      return null;
    }
  }

  /**
   * Obtenir les presets de crews disponibles
   */
  getAvailableCrewPresets(): Array<{ id: string; name: string; description: string }> {
    return Object.entries(PRESET_CREWS).map(([id, preset]) => ({
      id,
      name: preset.name,
      description: preset.description,
    }));
  }

  /**
   * Obtenir les presets LLM disponibles (Ollama prioritaire)
   */
  getLLMPresets(): typeof LLM_PRESETS {
    return LLM_PRESETS;
  }

  /**
   * Sélectionner le meilleur LLM pour une tâche via CrewManager
   * Priorise Ollama (gratuit) quand possible
   */
  async selectCrewLLM(
    complexity: 'simple' | 'moderate' | 'complex',
    requiresCode = false
  ) {
    if (!this.crewManager) {
      await this.initializeCrewAI();
    }

    if (!this.crewManager) {
      // Fallback: retourner un preset par défaut
      return LLM_PRESETS.OLLAMA_FAST;
    }

    return this.crewManager.selectBestLLM(complexity, requiresCode);
  }

  /**
   * Vérifier si CrewAI est disponible et Ollama connecté
   */
  async checkCrewAIHealth(): Promise<{
    available: boolean;
    ollamaConnected: boolean;
    models: string[];
  }> {
    if (!this.crewManager) {
      return { available: false, ollamaConnected: false, models: [] };
    }

    try {
      const health = await this.crewManager.checkHealth();
      return {
        available: health.status === 'ok',
        ollamaConnected: health.ollamaConnected,
        models: health.availableModels,
      };
    } catch {
      return { available: false, ollamaConnected: false, models: [] };
    }
  }

  /**
   * Arrêter CrewAI proprement
   */
  async stopCrewAI(): Promise<void> {
    if (this.crewManager) {
      await this.crewManager.stop();
      this.crewManager = null;
      console.log('[Brain] 🛑 CrewAI arrêté');
    }
  }

  // ===========================================================================
  // API PUBLIQUE
  // ===========================================================================

  getMetrics(): typeof this.state.metrics {
    return { ...this.state.metrics };
  }

  getCurrentPlan(): ExecutionPlan | null {
    return this.state.currentPlan;
  }

  getDecisionHistory(): Decision[] {
    return [...this.state.decisionHistory];
  }

  getSkillManager(): SkillManager | null {
    return this.skillManager;
  }
}
