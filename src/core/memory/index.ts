/**
 * MEMORY - Agent Autonome de Gestion de Mémoire
 *
 * NOUVEAU FLUX:
 * 1. Reçoit notification d'un nouveau message utilisateur
 * 2. Génère PROACTIVEMENT un rapport de contexte
 * 3. Envoie ce rapport à Vox AVANT que Brain ne traite
 * 4. Brain peut demander plus de contexte si besoin
 * 5. Après chaque échange, extrait et stocke automatiquement les faits
 *
 * CYCLES AUTONOMES :
 * - COLLECT   : Stocker les infos de toutes sources
 * - PROCESS   : Générer embeddings
 * - SYNTHESIZE: Créer des résumés périodiques
 * - CONSOLIDATE: Fusionner/archiver
 * - LEARN     : Détecter corrections et patterns
 */

import { BaseAgent } from '../base-agent';
import type {
  AgentConfig,
  AgentMessage,
  EnrichedContext,
  LearningEntry,
  MemoryEntry,
  MemoryMetadata,
  MemoryType,
  Skill,
  Task,
  TaskAttempt,
  TaskStatus,
  ConversationTurn,
} from '../types';
import { PersistenceLayer } from './persistence';
import { EmbeddingsService } from './embeddings';
import { FactChecker } from './fact-checker';
import type { CorrectionDetectionResult, FactCheckResult } from './fact-checker';
import { randomUUID } from 'crypto';
import { PROTECTED_TRAITS, REQUIRED_TRAITS, validateAction, getCoreRulesPrompt } from '../rules';
import { initializeSkillSystem, SkillManager } from '../../skills';

// Re-export for use by other modules
export { PROTECTED_TRAITS, REQUIRED_TRAITS, getCoreRulesPrompt };

// ===========================================================================
// TYPES
// ===========================================================================

interface MemoryAgentState {
  isProcessing: boolean;
  lastSynthesis: Date | null;
  lastConsolidation: Date | null;
  currentSessionId: string;
  pendingEmbeddings: string[];
  activeHandlers: Set<string>; // Track active message handlers to prevent recursion
}

export interface ContextReport {
  relevantMemories: MemoryEntry[];
  userProfile: {
    preferences: Record<string, unknown>;
    communicationStyle: string;
    knownFacts: string[];
  };
  recentLearnings: string[];
  suggestedContext: string;
  warnings: string[];
  confidence: number;
}

// ===========================================================================
// PROMPTS
// ===========================================================================

const MEMORY_SYSTEM_PROMPT = `Tu es MEMORY, le système de mémoire autonome d'une IA avancée.

TU ES PROACTIF - Quand un message arrive, tu:
1. ANALYSES immédiatement ce qui est pertinent dans ta mémoire
2. GÉNÈRES un rapport de contexte pour aider à la réponse
3. EXTRAIS les nouveaux faits après chaque échange
4. DÉTECTES si l'utilisateur corrige une erreur passée

PRINCIPES :
- Les corrections utilisateur sont CRITIQUES (importance max)
- Toujours indiquer la confiance (0-1)
- Distinguer faits vérifiés vs suppositions
- Détecter les préférences implicites

FORMAT pour rapport de contexte:
{
  "suggestedContext": "Ce que le système devrait savoir pour cette requête",
  "relevantFacts": ["fait1", "fait2"],
  "userPreferences": ["préférence détectée"],
  "warnings": ["attention à ceci"],
  "confidence": 0.0-1.0
}

FORMAT pour extraction de faits:
{
  "facts": [
    {"content": "fait extrait", "type": "fact|preference|correction", "confidence": 0.9}
  ],
  "isCorrection": false,
  "correctionDetails": null
}`;

const FACT_EXTRACTION_PROMPT = `Analyse cet échange et extrais les FAITS IMPORTANTS à mémoriser.

ÉCHANGE:
Utilisateur: {userMessage}
Assistant: {assistantResponse}

RÈGLES:
- Extrais uniquement les faits NOUVEAUX et IMPORTANTS
- Détecte si l'utilisateur CORRIGE une erreur ("non", "c'est pas ça", "en fait", etc.)
- Identifie les PRÉFÉRENCES implicites
- Note le niveau de confiance

Réponds en JSON:
{
  "facts": [
    {"content": "description du fait", "type": "fact|preference|skill", "confidence": 0.9}
  ],
  "isCorrection": true/false,
  "correctionDetails": {
    "originalError": "ce qui était faux",
    "correction": "la bonne information",
    "feedback": "pourquoi c'était faux"
  } ou null
}`;

// ===========================================================================
// CONSTANTS
// ===========================================================================

// Intervalles des cycles autonomes
const SYNTHESIS_INTERVAL = 15 * 60 * 1000;      // Synthèse toutes les 15 min
const CONSOLIDATION_INTERVAL = 60 * 60 * 1000;  // Consolidation toutes les heures
const EMBEDDING_INTERVAL = 30 * 1000;           // Traitement embeddings toutes les 30s
const LEARNING_INTERVAL = 5 * 60 * 1000;        // Analyse patterns toutes les 5 min

const EMBEDDING_BATCH_SIZE = 10;
const MEMORY_ARCHIVE_DAYS = 30;                 // Archiver après 30 jours
const MIN_IMPORTANCE_FOR_RETENTION = 0.3;       // Seuil minimum pour garder

// ===========================================================================
// MEMORY AGENT
// ===========================================================================

export class MemoryAgent extends BaseAgent {
  private state: MemoryAgentState;
  private persistence: PersistenceLayer;
  private embeddings: EmbeddingsService;
  private factChecker: FactChecker;
  private skillManager: SkillManager | null = null;
  private synthesisTimer: NodeJS.Timeout | null = null;
  private consolidationTimer: NodeJS.Timeout | null = null;
  private embeddingTimer: NodeJS.Timeout | null = null;
  private learningTimer: NodeJS.Timeout | null = null;
  private lastAssistantResponse: string = '';

  constructor(config?: Partial<AgentConfig>) {
    super({
      name: 'Memory',
      role: 'memory',
      model: 'claude-3-5-haiku-20241022', // Haiku suffit pour extraction/synthèse
      maxTokens: 2048,
      temperature: 0.1,
      systemPrompt: MEMORY_SYSTEM_PROMPT,
      ...config,
    });

    this.persistence = new PersistenceLayer('./data');
    this.embeddings = new EmbeddingsService();
    this.factChecker = new FactChecker();

    // Initialiser le système de skills
    const { skillManager } = initializeSkillSystem({
      persistence: this.persistence,
      memory: {
        search: async (query: string, limit: number) => {
          return this.search(query, { limit });
        },
        get: async (id: string) => {
          return this.persistence.getMemory(id);
        },
        store: async (type: string, content: string, metadata?: Record<string, unknown>) => {
          return this.store(type as MemoryType, content, metadata);
        },
      },
    });

    this.skillManager = skillManager;
    // capabilityManager est géré par le SkillManager lui-même

    this.state = {
      isProcessing: false,
      lastSynthesis: null,
      lastConsolidation: null,
      currentSessionId: randomUUID(),
      pendingEmbeddings: [],
      activeHandlers: new Set(),
    };
  }

  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================

  protected async onStart(): Promise<void> {
    console.log('[Memory] 🧠 Démarrage des cycles autonomes...');

    // Démarrer le système de skills
    if (this.skillManager) {
      await this.skillManager.start();
      const stats = this.skillManager.getStats();
      console.log(`[Memory] 🔧 Skills: ${stats.totalSkills} chargés (${stats.builtinSkills} built-in)`);
    }

    // Préchauffer le modèle d'embeddings
    try {
      await this.embeddings.warmup();
    } catch (error) {
      console.warn('[Memory] ⚠️ Embeddings warmup échoué, fallback actif:', error);
    }

    // Cycle de synthèse - résume les mémoires récentes
    this.synthesisTimer = setInterval(() => {
      this.runSynthesisCycle().catch(console.error);
    }, SYNTHESIS_INTERVAL);

    // Cycle de consolidation - archive et fusionne les vieilles mémoires
    this.consolidationTimer = setInterval(() => {
      this.runConsolidationCycle().catch(console.error);
    }, CONSOLIDATION_INTERVAL);

    // Cycle d'embeddings - traite la queue d'embeddings en attente
    this.embeddingTimer = setInterval(() => {
      this.processEmbeddingQueue().catch(console.error);
    }, EMBEDDING_INTERVAL);

    // Cycle de learning - détecte les patterns et applique les corrections
    this.learningTimer = setInterval(() => {
      this.runLearningCycle().catch(console.error);
    }, LEARNING_INTERVAL);

    // Traitement initial des embeddings en attente
    this.processEmbeddingQueue().catch(console.error);

    console.log('[Memory] ✅ Cycles autonomes actifs:');
    console.log(`[Memory]    - Synthèse: toutes les ${SYNTHESIS_INTERVAL / 60000} min`);
    console.log(`[Memory]    - Consolidation: toutes les ${CONSOLIDATION_INTERVAL / 60000} min`);
    console.log(`[Memory]    - Embeddings: toutes les ${EMBEDDING_INTERVAL / 1000}s`);
    console.log(`[Memory]    - Learning: toutes les ${LEARNING_INTERVAL / 60000} min`);
  }

  protected onStop(): void {
    if (this.synthesisTimer) clearInterval(this.synthesisTimer);
    if (this.consolidationTimer) clearInterval(this.consolidationTimer);
    if (this.embeddingTimer) clearInterval(this.embeddingTimer);
    if (this.learningTimer) clearInterval(this.learningTimer);
    this.persistence.close();
    console.log('[Memory] Cycles arrêtés, données persistées');
  }

  /**
   * Cycle de learning - Détecte les patterns et applique les learnings
   * Inclut maintenant le traitement du feedback utilisateur (Règle 4)
   */
  private async runLearningCycle(): Promise<void> {
    // 1. Traiter les learnings classiques
    const learnings = this.persistence.getLearnings({ unappliedOnly: true });

    if (learnings.length > 0) {
      console.log(`[Memory] 📚 Cycle learning: ${learnings.length} corrections à analyser`);

      for (const learning of learnings) {
        try {
          this.persistence.markLearningApplied(learning.id);
          await this.store('correction', learning.feedback, {
            tags: ['learning', 'applied', learning.type],
            confidence: 1.0,
            source: 'learning_cycle',
          });
          console.log(`[Memory] ✅ Learning appliqué: ${learning.type}`);
        } catch (error) {
          console.error(`[Memory] Erreur application learning:`, error);
        }
      }
    }

    // 2. Traiter les feedbacks utilisateur (Règle 4: Neo s'améliore tout seul)
    await this.processFeedbackCycle();
  }

  /**
   * Traite les feedbacks pour amélioration continue
   * Les feedbacks négatifs génèrent des learnings, les positifs renforcent les bons patterns
   */
  private async processFeedbackCycle(): Promise<void> {
    const feedbacks = this.persistence.getUnprocessedFeedback(20);

    if (feedbacks.length === 0) return;

    console.log(`[Memory] 👍👎 Traitement de ${feedbacks.length} feedback(s)...`);

    for (const feedback of feedbacks) {
      try {
        if (feedback.rating === 'negative') {
          // Feedback négatif : créer un learning pour éviter cette erreur
          await this.store('correction', JSON.stringify({
            originalResponse: feedback.assistantResponse,
            userMessage: feedback.userMessage,
            issue: feedback.userComment || 'Réponse insatisfaisante',
            lesson: 'Améliorer ce type de réponse',
          }), {
            tags: ['feedback', 'negative', 'improvement_needed'],
            confidence: 0.95,
            source: 'user_feedback',
          });

          console.log(`[Memory] ⚠️ Feedback négatif traité - learning créé`);

        } else if (feedback.rating === 'positive') {
          // Feedback positif : renforcer ce pattern de réponse
          await this.store('fact', JSON.stringify({
            goodResponse: feedback.assistantResponse.substring(0, 500),
            context: feedback.userMessage.substring(0, 200),
            lesson: 'Ce type de réponse fonctionne bien',
          }), {
            tags: ['feedback', 'positive', 'good_pattern'],
            confidence: 0.9,
            source: 'user_feedback',
          });

          console.log(`[Memory] ✅ Feedback positif traité - pattern renforcé`);
        }

        // Marquer comme traité
        this.persistence.markFeedbackProcessed(feedback.id);

      } catch (error) {
        console.error(`[Memory] Erreur traitement feedback:`, error);
      }
    }

    // Afficher les stats de satisfaction
    const stats = this.persistence.getFeedbackStats();
    if (stats.total > 0) {
      console.log(`[Memory] 📊 Satisfaction: ${Math.round(stats.satisfactionRate * 100)}% (${stats.positive}👍 / ${stats.negative}👎)`);
    }
  }

  // ===========================================================================
  // NOUVEAU: GÉNÉRATION PROACTIVE DE RAPPORT DE CONTEXTE
  // ===========================================================================

  /**
   * Génère un rapport de contexte PROACTIVEMENT pour Vox
   * Appelé dès qu'un message utilisateur arrive
   */
  async generateContextReport(userInput: string): Promise<ContextReport> {
    console.log('[Memory] 📋 Génération du rapport de contexte...');

    // 1. Rechercher les mémoires pertinentes par embedding
    const queryEmbedding = await this.embeddings.embed(userInput);
    const allMemories = this.persistence.searchMemories({ limit: 200 });
    const withEmbeddings = allMemories.filter(m => m.embedding);

    const similarMemories = this.embeddings.findMostSimilar(
      queryEmbedding,
      withEmbeddings.map(m => ({ id: m.id, embedding: m.embedding! })),
      20
    );

    const relevantMemories = similarMemories
      .filter(s => s.similarity > 0.25)
      .map(s => {
        const memory = allMemories.find(m => m.id === s.id)!;
        this.persistence.updateMemoryAccess(memory.id);
        return memory;
      });

    // 2. Ajouter les mémoires de haute importance récentes
    const recentImportant = allMemories
      .filter(m => m.importance > 0.8)
      .filter(m => !relevantMemories.find(r => r.id === m.id))
      .slice(0, 5);

    const combinedMemories = [...relevantMemories, ...recentImportant];

    // 3. Extraire le profil utilisateur
    const preferences = this.persistence.searchMemories({ type: 'preference', limit: 10 });
    const facts = this.persistence.searchMemories({ type: 'fact', limit: 20 });

    const userProfile = {
      preferences: this.extractUserPreferences(),
      communicationStyle: this.detectCommunicationStyle(preferences),
      knownFacts: facts.slice(0, 10).map(f => f.content),
    };

    // 4. Récupérer les learnings récents
    const learnings = this.persistence.getLearnings({ unappliedOnly: false });
    const recentLearnings = learnings
      .slice(0, 5)
      .map(l => l.feedback);

    // 5. Générer le contexte suggéré via Claude
    let suggestedContext = '';
    let warnings: string[] = [];
    let confidence = 0.8;

    if (combinedMemories.length > 0) {
      try {
        const contextPrompt = `
Basé sur ces mémoires pertinentes, génère un CONTEXTE UTILE pour répondre à cette requête.

REQUÊTE UTILISATEUR: "${userInput}"

MÉMOIRES PERTINENTES:
${combinedMemories.slice(0, 10).map(m => `[${m.type}] ${m.content}`).join('\n')}

PRÉFÉRENCES CONNUES:
${preferences.map(p => p.content).join('\n') || 'Aucune'}

CORRECTIONS PASSÉES:
${recentLearnings.join('\n') || 'Aucune'}

Réponds en JSON:
{
  "suggestedContext": "Résumé de ce que le système devrait savoir (2-3 phrases)",
  "warnings": ["attention si quelque chose est risqué ou ambigu"],
  "confidence": 0.0-1.0
}`;

        // Utiliser thinkOptimized - génération de contexte est une tâche factuelle simple
        const response = await this.thinkOptimized(contextPrompt, 'factual');
        const jsonMatch = response.match(/\{[\s\S]*\}/);

        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          suggestedContext = parsed.suggestedContext || '';
          warnings = parsed.warnings || [];
          confidence = parsed.confidence || 0.8;
        }
      } catch (error) {
        console.error('[Memory] Erreur génération contexte:', error);
      }
    }

    const report: ContextReport = {
      relevantMemories: combinedMemories,
      userProfile,
      recentLearnings,
      suggestedContext,
      warnings,
      confidence,
    };

    console.log(`[Memory] ✅ Rapport généré: ${combinedMemories.length} mémoires, confiance ${confidence}`);

    return report;
  }

  // ===========================================================================
  // NOUVEAU: STOCKAGE AUTOMATIQUE DES CONVERSATIONS
  // ===========================================================================

  /**
   * Stocker un échange de conversation et extraire les faits
   */
  async storeConversation(
    userMessage: string,
    assistantResponse: string,
    sessionId?: string
  ): Promise<void> {
    const session = sessionId || this.state.currentSessionId;

    // 1. Stocker le message utilisateur
    await this.store('conversation', userMessage, {
      source: 'user',
      tags: ['conversation', 'user_message', session],
      confidence: 1.0,
    });

    // 2. Stocker la réponse assistant
    await this.store('conversation', assistantResponse, {
      source: 'assistant',
      tags: ['conversation', 'assistant_response', session],
      confidence: 1.0,
    });

    // 3. Extraire automatiquement les faits
    await this.extractAndStoreFacts(userMessage, assistantResponse);

    console.log('[Memory] 💬 Conversation stockée et analysée');
  }

  /**
   * Extraire les faits d'un échange et détecter les corrections
   * Utilise maintenant le FactChecker pour une détection plus précise
   */
  private async extractAndStoreFacts(
    userMessage: string,
    assistantResponse: string
  ): Promise<void> {
    // 1. Détecter les corrections avec le FactChecker (plus précis)
    const correctionResult: CorrectionDetectionResult = await this.factChecker.detectCorrection(
      userMessage,
      this.lastAssistantResponse || assistantResponse
    );

    if (correctionResult.isCorrection && correctionResult.details) {
      const details = correctionResult.details;
      console.log(`[Memory] 🔧 Correction détectée (confiance: ${correctionResult.confidence})`);
      console.log(`[Memory]    Type: ${details.correctionType}`);
      console.log(`[Memory]    Triggers: ${correctionResult.triggerWords.join(', ')}`);

      // Enregistrer le learning avec haute priorité
      await this.recordLearning(
        'user_correction',
        userMessage,
        details.originalError,
        details.feedback,
        details.correction
      );

      // Stocker aussi comme mémoire de correction
      await this.store('correction', details.correction, {
        tags: ['user_correction', details.correctionType, 'high_priority'],
        confidence: correctionResult.confidence,
        source: 'fact_checker',
      });
    }

    // 2. Extraire les faits additionnels via le prompt classique
    const prompt = FACT_EXTRACTION_PROMPT
      .replace('{userMessage}', userMessage)
      .replace('{assistantResponse}', assistantResponse);

    try {
      // Utiliser thinkOptimized - extraction de faits est une tâche simple
      const response = await this.thinkOptimized(prompt, 'factual');
      const jsonMatch = response.match(/\{[\s\S]*\}/);

      if (!jsonMatch) return;

      const extraction = JSON.parse(jsonMatch[0]);

      // Stocker les faits extraits (sauf si déjà traité comme correction)
      for (const fact of extraction.facts || []) {
        if (fact.type !== 'correction' || !correctionResult.isCorrection) {
          await this.store(fact.type || 'fact', fact.content, {
            tags: ['auto_extracted', fact.type],
            confidence: fact.confidence || 0.8,
            source: 'conversation_extraction',
          });
        }
      }
    } catch (error) {
      console.error('[Memory] Erreur extraction faits:', error);
    }

    // Mettre à jour la dernière réponse pour la prochaine détection
    this.lastAssistantResponse = assistantResponse;
  }

  /**
   * Vérifier la cohérence d'une réponse avec les faits stockés
   * Utilisé par Brain avant d'envoyer une réponse
   */
  async checkResponseConsistency(
    proposedResponse: string,
    userInput: string
  ): Promise<FactCheckResult> {
    // Récupérer les mémoires pertinentes pour cette réponse
    const relevantMemories = await this.search(userInput, {
      limit: 20,
      minConfidence: 0.5,
    });

    // Utiliser le FactChecker pour vérifier la cohérence
    const result = await this.factChecker.checkFactConsistency(
      proposedResponse,
      relevantMemories
    );

    // Calculer la vraie confiance basée sur les mémoires
    result.confidence = this.factChecker.calculateConfidenceFromMemories(
      relevantMemories,
      result
    );

    if (!result.isConsistent) {
      console.log(`[Memory] ⚠️ Incohérence détectée: ${result.contradictions.length} contradiction(s)`);
      for (const contradiction of result.contradictions) {
        console.log(`[Memory]    - ${contradiction.severity}: "${contradiction.claim}" vs "${contradiction.storedFact}"`);
      }
    }

    return result;
  }

  /**
   * Détecter le style de communication préféré
   */
  private detectCommunicationStyle(preferences: MemoryEntry[]): string {
    const styles = preferences
      .filter(p => p.content.toLowerCase().includes('style') ||
                   p.content.toLowerCase().includes('ton') ||
                   p.content.toLowerCase().includes('formel'))
      .map(p => p.content);

    if (styles.length === 0) return 'professional';

    // Analyse simple basée sur les mots-clés
    const content = styles.join(' ').toLowerCase();
    if (content.includes('informel') || content.includes('casual')) return 'casual';
    if (content.includes('technique') || content.includes('expert')) return 'technical';
    if (content.includes('simple') || content.includes('clair')) return 'simple';

    return 'professional';
  }

  // ===========================================================================
  // STOCKAGE DE MÉMOIRE
  // ===========================================================================

  async store(
    type: MemoryType,
    content: string,
    metadata: Partial<MemoryMetadata> = {}
  ): Promise<string> {
    const id = randomUUID();
    const now = new Date();
    const importance = this.calculateImportance(type, content, metadata);

    const entry: MemoryEntry = {
      id,
      type,
      content,
      metadata: {
        source: metadata.source || 'system',
        confidence: metadata.confidence || 1.0,
        tags: metadata.tags || [],
        relatedIds: metadata.relatedIds || [],
        expiresAt: metadata.expiresAt,
      },
      createdAt: now,
      lastAccessedAt: now,
      accessCount: 0,
      importance,
    };

    this.persistence.saveMemory(entry);
    this.state.pendingEmbeddings.push(id);

    if (this.state.pendingEmbeddings.length >= EMBEDDING_BATCH_SIZE) {
      this.processEmbeddingQueue().catch(console.error);
    }

    console.log(`[Memory] 💾 [${type}] "${content.substring(0, 40)}..." (imp: ${importance.toFixed(2)})`);

    if (type === 'task_result') {
      await this.detectSkillFromResult(content);
    }

    return id;
  }

  private calculateImportance(
    type: MemoryType,
    content: string,
    metadata: Partial<MemoryMetadata>
  ): number {
    const typeWeights: Record<MemoryType, number> = {
      correction: 0.95,
      preference: 0.85,
      fact: 0.75,
      skill: 0.80,
      task_result: 0.65,
      conversation: 0.40,
      system: 0.50,
    };

    let importance = typeWeights[type] || 0.5;

    if (metadata.confidence && metadata.confidence > 0.9) importance += 0.05;
    if (content.length > 200) importance += 0.05;

    const importantTags = ['critical', 'user_correction', 'verified', 'key_fact'];
    if (metadata.tags?.some(t => importantTags.includes(t))) importance += 0.1;

    return Math.min(importance, 1.0);
  }

  // ===========================================================================
  // EMBEDDINGS
  // ===========================================================================

  private async processEmbeddingQueue(): Promise<void> {
    if (this.state.isProcessing || this.state.pendingEmbeddings.length === 0) return;

    this.state.isProcessing = true;
    const toProcess = this.state.pendingEmbeddings.splice(0, EMBEDDING_BATCH_SIZE);

    for (const id of toProcess) {
      try {
        const memory = this.persistence.getMemory(id);
        if (memory && !memory.embedding) {
          const embedding = await this.embeddings.embed(memory.content);
          memory.embedding = embedding;
          this.persistence.saveMemory(memory);
        }
      } catch (error) {
        console.error(`[Memory] Erreur embedding ${id}:`, error);
        this.state.pendingEmbeddings.push(id);
      }
    }

    this.state.isProcessing = false;
  }

  // ===========================================================================
  // CYCLES AUTONOMES
  // ===========================================================================

  private async runSynthesisCycle(): Promise<void> {
    console.log('[Memory] 📝 Cycle de synthèse...');

    const now = new Date();
    const periodStart = this.state.lastSynthesis || new Date(now.getTime() - SYNTHESIS_INTERVAL);

    const recentMemories = this.persistence.searchMemories({
      limit: 100,
      minImportance: 0.3,
    }).filter(m => m.createdAt >= periodStart);

    if (recentMemories.length < 5) {
      this.state.lastSynthesis = now;
      return;
    }

    try {
      const synthesisPrompt = `
Analyse ces ${recentMemories.length} mémoires récentes et crée une SYNTHÈSE:

MÉMOIRES:
${recentMemories.map(m => `[${m.type}] ${m.content}`).join('\n\n')}

Réponds en JSON:
{
  "summary": "Résumé en 2-3 phrases",
  "keyFacts": ["fait1", "fait2"],
  "patterns": ["pattern détecté"],
  "userInsights": ["insight sur l'utilisateur"]
}`;

      // Utiliser thinkOptimized - synthèse peut utiliser un modèle moins cher
      const response = await this.thinkOptimized(synthesisPrompt, 'factual');
      const jsonMatch = response.match(/\{[\s\S]*\}/);

      if (jsonMatch) {
        const synthesis = JSON.parse(jsonMatch[0]);

        this.persistence.saveSynthesis({
          periodStart,
          periodEnd: now,
          summary: synthesis.summary,
          keyFacts: synthesis.keyFacts || [],
          memoryIds: recentMemories.map(m => m.id),
        });

        for (const fact of synthesis.keyFacts || []) {
          await this.store('fact', fact, {
            tags: ['synthesis', 'key_fact'],
            confidence: 0.9,
          });
        }

        console.log('[Memory] ✅ Synthèse créée');
      }
    } catch (error) {
      console.error('[Memory] Erreur synthèse:', error);
    }

    this.state.lastSynthesis = now;
  }

  private async runConsolidationCycle(): Promise<void> {
    console.log('[Memory] 🔧 Cycle de consolidation...');

    // Archiver les anciennes mémoires de faible importance
    const archived = this.persistence.archiveOldMemories(
      MEMORY_ARCHIVE_DAYS,
      MIN_IMPORTANCE_FOR_RETENTION
    );
    if (archived > 0) {
      console.log(`[Memory] 📦 Archivé ${archived} mémoires anciennes`);
    }

    // Fusionner les mémoires similaires
    await this.mergeSimilarMemories();

    // Mettre à jour les statistiques des skills
    await this.updateSkillsFromResults();

    // Nettoyer le cache d'embeddings si trop gros
    const cacheStats = this.embeddings.getCacheStats();
    if (cacheStats.memoryMB > 100) {
      console.log(`[Memory] 🧹 Nettoyage cache embeddings (${cacheStats.memoryMB.toFixed(1)} MB)`);
      this.embeddings.clearCache();
    }

    this.state.lastConsolidation = new Date();
    console.log('[Memory] ✅ Consolidation terminée');
  }

  /**
   * Fusionner les mémoires très similaires pour éviter la duplication
   * Garde la mémoire avec la plus haute importance et enrichit son contenu
   */
  private async mergeSimilarMemories(): Promise<void> {
    // Récupérer les faits et préférences non archivés
    const memories = [
      ...this.persistence.searchMemories({ type: 'fact', limit: 200 }),
      ...this.persistence.searchMemories({ type: 'preference', limit: 100 }),
    ].filter(m => m.embedding && !m.metadata.tags?.includes('merged'));

    const mergedIds = new Set<string>(); // IDs déjà fusionnés à ignorer
    let totalMerged = 0;

    for (const memory of memories) {
      // Skip si déjà fusionné dans cette session
      if (mergedIds.has(memory.id)) continue;

      // Trouver les candidats similaires
      const candidates = memories
        .filter(m => m.id !== memory.id && !mergedIds.has(m.id) && m.embedding)
        .map(m => ({ id: m.id, embedding: m.embedding! }));

      if (candidates.length === 0) continue;

      const similar = this.embeddings.findMostSimilar(memory.embedding!, candidates, 5)
        .filter(s => s.similarity > 0.88); // Seuil élevé pour éviter les faux positifs

      if (similar.length === 0) continue;

      // Récupérer les mémoires similaires complètes
      const similarMemories = similar
        .map(s => memories.find(m => m.id === s.id))
        .filter((m): m is MemoryEntry => m !== undefined);

      // Déterminer la mémoire principale (plus haute importance, puis plus récente)
      const allMemories = [memory, ...similarMemories];
      allMemories.sort((a, b) => {
        if (b.importance !== a.importance) return b.importance - a.importance;
        return b.createdAt.getTime() - a.createdAt.getTime();
      });

      const primary = allMemories[0];
      const toMerge = allMemories.slice(1);

      // Fusionner les contenus via Claude pour créer un résumé cohérent
      const mergedContent = await this.createMergedContent(primary, toMerge);

      if (mergedContent) {
        // Fusionner les tags (sans doublons)
        const allTags = new Set<string>();
        allMemories.forEach(m => m.metadata.tags?.forEach(t => allTags.add(t)));
        allTags.add('merged');
        allTags.add(`merged_count_${allMemories.length}`);

        // Fusionner les IDs liés
        const allRelatedIds = new Set<string>();
        allMemories.forEach(m => m.metadata.relatedIds?.forEach(id => allRelatedIds.add(id)));
        toMerge.forEach(m => allRelatedIds.add(m.id)); // Garder trace des IDs fusionnés

        // Calculer la nouvelle importance (moyenne pondérée + bonus fusion)
        const avgImportance = allMemories.reduce((sum, m) => sum + m.importance, 0) / allMemories.length;
        const newImportance = Math.min(1.0, avgImportance + 0.05 * (allMemories.length - 1));

        // Calculer la nouvelle confiance (max des confiances)
        const maxConfidence = Math.max(...allMemories.map(m => m.metadata.confidence));

        // Mettre à jour la mémoire principale
        this.persistence.updateMemory(primary.id, {
          content: mergedContent,
          importance: newImportance,
          confidence: maxConfidence,
          tags: Array.from(allTags),
          relatedIds: Array.from(allRelatedIds),
        });

        // Supprimer les mémoires fusionnées
        const idsToDelete = toMerge.map(m => m.id);
        this.persistence.deleteMemories(idsToDelete);

        // Marquer comme fusionnés pour ne pas les retraiter
        mergedIds.add(primary.id);
        idsToDelete.forEach(id => mergedIds.add(id));

        totalMerged += toMerge.length;
        console.log(`[Memory] 🔗 Fusionné ${allMemories.length} mémoires → "${mergedContent.substring(0, 50)}..."`);
      }
    }

    if (totalMerged > 0) {
      console.log(`[Memory] ✅ Total fusionné: ${totalMerged} mémoires supprimées`);
    }
  }

  /**
   * Créer un contenu fusionné à partir de plusieurs mémoires similaires
   */
  private async createMergedContent(
    primary: MemoryEntry,
    toMerge: MemoryEntry[]
  ): Promise<string | null> {
    // Si une seule mémoire à fusionner et contenu très similaire, garder le plus long
    if (toMerge.length === 1) {
      const other = toMerge[0];
      // Si l'un contient l'autre, garder le plus complet
      if (primary.content.includes(other.content)) return primary.content;
      if (other.content.includes(primary.content)) return other.content;
    }

    // Sinon, utiliser Claude pour créer une fusion intelligente
    try {
      const mergePrompt = `Fusionne ces ${toMerge.length + 1} faits similaires en UN SEUL fait concis et complet.
Garde TOUTES les informations importantes sans répétition.

FAIT PRINCIPAL:
"${primary.content}"

FAITS À FUSIONNER:
${toMerge.map((m, i) => `${i + 1}. "${m.content}"`).join('\n')}

Réponds avec UNIQUEMENT le fait fusionné, sans guillemets ni préfixe.`;

      // Utiliser thinkOptimized - fusion de texte est une tâche simple
      const response = await this.thinkOptimized(mergePrompt, 'simple_chat');

      // Nettoyer la réponse
      let merged = response.trim();

      // Enlever les guillemets si présents
      if ((merged.startsWith('"') && merged.endsWith('"')) ||
          (merged.startsWith("'") && merged.endsWith("'"))) {
        merged = merged.slice(1, -1);
      }

      // Vérifier que le résultat est raisonnable
      if (merged.length > 10 && merged.length < 2000) {
        return merged;
      }
    } catch (error) {
      console.error('[Memory] Erreur fusion contenu:', error);
    }

    // Fallback: concaténer simplement
    return `${primary.content} (+ ${toMerge.length} fait(s) similaire(s))`;
  }

  private async updateSkillsFromResults(): Promise<void> {
    const taskResults = this.persistence.searchMemories({ type: 'task_result', limit: 50 });
    const skills = this.persistence.getSkills();

    for (const skill of skills) {
      const relevantResults = taskResults.filter(r =>
        r.content.toLowerCase().includes(skill.name.toLowerCase())
      );

      if (relevantResults.length > 0) {
        const successes = relevantResults.filter(r =>
          !r.content.includes('error') && !r.content.includes('failed')
        ).length;

        skill.successRate = successes / relevantResults.length;
        skill.usageCount = relevantResults.length;
        this.persistence.saveSkill(skill);
      }
    }
  }

  // ===========================================================================
  // RECHERCHE ET CONTEXTE
  // ===========================================================================

  /**
   * Recherche hybride: combine Vector Similarity (70%) + BM25 (30%)
   * Inspiré d'OpenClaw mais implémenté localement avec SQLite FTS5
   */
  async search(
    query: string,
    options: {
      type?: MemoryType;
      tags?: string[];
      limit?: number;
      minConfidence?: number;
      useHybrid?: boolean;  // Default true - utiliser recherche hybride
    } = {}
  ): Promise<MemoryEntry[]> {
    const { limit = 10, minConfidence = 0, useHybrid = true } = options;

    // 1. Générer l'embedding de la requête
    const queryEmbedding = await this.embeddings.embed(query);

    // 2. Récupérer les candidats avec embeddings
    const candidatesWithEmb = this.persistence.getMemoriesWithEmbeddings({
      type: options.type,
      limit: limit * 5,  // Plus de candidats pour la fusion
    });

    // 3. Recherche BM25 (full-text search)
    const bm25Results = this.persistence.searchBM25(query, limit * 3);

    // 4. Si pas assez de données pour hybride, fallback simple
    if (candidatesWithEmb.length === 0 && bm25Results.length === 0) {
      // Fallback: recherche par mots-clés simple
      const candidates = this.persistence.searchMemories({
        type: options.type,
        tags: options.tags,
        limit: limit * 2,
      }).filter(m => m.metadata.confidence >= minConfidence);

      const queryWords = query.toLowerCase().split(/\s+/);
      return candidates
        .map(m => ({
          memory: m,
          score: queryWords.filter(w => m.content.toLowerCase().includes(w)).length,
        }))
        .filter(s => s.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, limit)
        .map(s => {
          this.persistence.updateMemoryAccess(s.memory.id);
          return s.memory;
        });
    }

    // 5. Recherche hybride (Vector 70% + BM25 30%)
    if (useHybrid && candidatesWithEmb.length > 0) {
      const hybridResults = this.embeddings.hybridSearch(
        queryEmbedding,
        candidatesWithEmb.map(c => ({ id: c.id, embedding: c.embedding })),
        bm25Results,
        {
          topK: limit * 2,
          vectorWeight: 0.7,
          bm25Weight: 0.3,
          minScore: 0.15,
        }
      );

      // Récupérer les mémoires complètes
      const results: MemoryEntry[] = [];
      for (const result of hybridResults) {
        const memory = this.persistence.getMemory(result.id);
        if (memory && memory.metadata.confidence >= minConfidence) {
          // Filtrer par tags si spécifié
          if (options.tags && options.tags.length > 0) {
            if (!options.tags.some(tag => memory.metadata.tags.includes(tag))) {
              continue;
            }
          }
          this.persistence.updateMemoryAccess(memory.id);
          results.push(memory);
          if (results.length >= limit) break;
        }
      }

      console.log(`[Memory] 🔍 Recherche hybride: ${results.length} résultats (V:70% + BM25:30%)`);
      return results;
    }

    // 6. Fallback: recherche vectorielle seule
    if (candidatesWithEmb.length > 0) {
      const similar = this.embeddings.findMostSimilar(
        queryEmbedding,
        candidatesWithEmb.map(c => ({ id: c.id, embedding: c.embedding })),
        limit
      );

      const results: MemoryEntry[] = [];
      for (const s of similar) {
        const memory = this.persistence.getMemory(s.id);
        if (memory && memory.metadata.confidence >= minConfidence) {
          this.persistence.updateMemoryAccess(memory.id);
          results.push(memory);
        }
      }
      return results;
    }

    // 7. Dernier recours: BM25 seul
    const results: MemoryEntry[] = [];
    for (const r of bm25Results) {
      const memory = this.persistence.getMemory(r.id);
      if (memory && memory.metadata.confidence >= minConfidence) {
        this.persistence.updateMemoryAccess(memory.id);
        results.push(memory);
        if (results.length >= limit) break;
      }
    }
    return results;
  }

  async buildEnrichedContext(
    userInput: string,
    conversationContext: ConversationTurn[]
  ): Promise<EnrichedContext> {
    const report = await this.generateContextReport(userInput);

    return {
      userInput,
      relevantMemories: report.relevantMemories,
      activeTasks: this.persistence.getTasks('in_progress'),
      availableSkills: this.findRelevantSkills(userInput),
      recentConversation: conversationContext,
      userProfile: {
        id: 'default',
        preferences: report.userProfile.preferences,
        communicationStyle: report.userProfile.communicationStyle,
        expertise: [],
      },
      suggestedApproach: report.suggestedContext,
    };
  }

  private extractUserPreferences(): Record<string, unknown> {
    const preferences = this.persistence.searchMemories({ type: 'preference', limit: 20 });
    const prefs: Record<string, unknown> = {};

    preferences.forEach(p => {
      try {
        const parsed = JSON.parse(p.content);
        Object.assign(prefs, parsed);
      } catch {
        prefs[p.id] = p.content;
      }
    });

    return prefs;
  }

  // ===========================================================================
  // SKILLS
  // ===========================================================================

  private async detectSkillFromResult(resultContent: string): Promise<void> {
    try {
      // Utiliser thinkOptimized - détection de skill est une tâche factuelle
      const response = await this.thinkOptimized(`
Analyse ce résultat et détermine s'il représente un SKILL réutilisable:

RÉSULTAT: ${resultContent}

Réponds en JSON:
{"isSkill": true/false, "name": "nom", "description": "desc", "triggers": ["mot1"]}`, 'factual');

      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const analysis = JSON.parse(jsonMatch[0]);
        if (analysis.isSkill && analysis.name) {
          await this.registerSkill(analysis.name, analysis.description, analysis.triggers || []);
        }
      }
    } catch {
      // Silently fail
    }
  }

  async registerSkill(name: string, description: string, triggers: string[]): Promise<Skill> {
    const existingSkills = this.persistence.getSkills();
    const existing = existingSkills.find(s => s.name.toLowerCase() === name.toLowerCase());

    if (existing) {
      existing.triggers = [...new Set([...existing.triggers, ...triggers])];
      existing.usageCount++;
      this.persistence.saveSkill(existing);
      return existing;
    }

    const skill: Skill = {
      id: randomUUID(),
      name,
      description,
      triggers,
      handler: `skills/${name.toLowerCase().replace(/\s+/g, '-')}`,
      dependencies: [],
      learnedAt: new Date(),
      successRate: 1.0,
      usageCount: 1,
    };

    this.persistence.saveSkill(skill);
    this.send('brain', 'skill_detected', skill);
    console.log(`[Memory] 🎯 Nouveau skill: ${name}`);

    return skill;
  }

  findRelevantSkills(query: string): Skill[] {
    const queryLower = query.toLowerCase();
    return this.persistence.getSkills().filter(skill =>
      skill.triggers.some(trigger => queryLower.includes(trigger.toLowerCase()))
    );
  }

  // ===========================================================================
  // LEARNING
  // ===========================================================================

  async recordLearning(
    type: LearningEntry['type'],
    context: string,
    originalResponse: string,
    feedback: string,
    correction?: string
  ): Promise<void> {
    const entry: LearningEntry = {
      id: randomUUID(),
      type,
      context,
      originalResponse,
      correction,
      feedback,
      createdAt: new Date(),
      applied: false,
    };

    this.persistence.saveLearning(entry);

    await this.store('correction', JSON.stringify(entry), {
      tags: ['learning', type, 'user_correction'],
      confidence: 1.0,
    });

    this.send('brain', 'learning_update', entry);
    console.log(`[Memory] 📚 Learning enregistré: ${type}`);
  }

  async getApplicableLearnings(context: string): Promise<LearningEntry[]> {
    const learnings = this.persistence.getLearnings({ unappliedOnly: false });
    const contextEmbedding = await this.embeddings.embed(context);

    const scored: Array<{ learning: LearningEntry; score: number }> = [];

    for (const learning of learnings) {
      const learningEmbedding = await this.embeddings.embed(learning.context);
      const similarity = this.embeddings.cosineSimilarity(contextEmbedding, learningEmbedding);

      if (similarity > 0.4) {
        scored.push({ learning, score: similarity });
      }
    }

    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map(s => s.learning);
  }

  // ===========================================================================
  // TASKS
  // ===========================================================================

  createTask(title: string, description: string, options: {
    priority?: number;
    dependencies?: string[];
    requiredSkills?: string[];
  } = {}): Task {
    const now = new Date();
    const task: Task = {
      id: randomUUID(),
      title,
      description,
      status: 'pending',
      priority: options.priority || 5,
      createdAt: now,
      updatedAt: now,
      attempts: [],
      dependencies: options.dependencies || [],
      requiredSkills: options.requiredSkills || [],
    };

    this.persistence.saveTask(task);
    return task;
  }

  updateTaskStatus(taskId: string, status: TaskStatus): void {
    const tasks = this.persistence.getTasks();
    const task = tasks.find(t => t.id === taskId);

    if (task) {
      task.status = status;
      task.updatedAt = new Date();
      this.persistence.saveTask(task);
    }
  }

  recordTaskAttempt(taskId: string, result: unknown, error?: string, learnings: string[] = []): void {
    const tasks = this.persistence.getTasks();
    const task = tasks.find(t => t.id === taskId);

    if (!task) return;

    const attempt: TaskAttempt = {
      attemptNumber: task.attempts.length + 1,
      startedAt: new Date(),
      endedAt: new Date(),
      result: error ? undefined : result,
      error,
      learnings,
    };

    task.attempts.push(attempt);
    task.updatedAt = new Date();
    this.persistence.saveTask(task);

    if (learnings.length > 0 || error) {
      this.store('task_result', JSON.stringify({ taskId, attempt }), {
        tags: ['task_attempt', error ? 'failure' : 'success'],
      });
    }
  }

  // ===========================================================================
  // MESSAGE HANDLER
  // ===========================================================================

  protected async handleMessage(message: AgentMessage): Promise<void> {
    // Ignorer les messages qui ne sont pas destinés à Memory
    if (message.to !== 'memory' && message.to !== 'broadcast') {
      return;
    }

    // Circuit breaker: éviter de traiter le même type de message en récursion
    if (this.state.activeHandlers.has(message.type)) {
      console.log(`[Memory] ⚠️ Handler déjà actif pour: ${message.type}, ignoré`);
      return;
    }

    // Limiter le nombre de handlers actifs pour éviter stack overflow
    if (this.state.activeHandlers.size >= 10) {
      console.log(`[Memory] ⚠️ Trop de handlers actifs (${this.state.activeHandlers.size}), ignoré`);
      this.reply(message, { error: true, message: 'Too many concurrent handlers' });
      return;
    }

    this.state.activeHandlers.add(message.type);

    try {
      switch (message.type) {
        case 'context_request':
          await this.handleContextRequest(message);
          break;

        case 'context_report_request':
          await this.handleContextReportRequest(message);
          break;

        case 'memory_store':
          await this.handleMemoryStore(message);
          break;

        case 'memory_query':
          await this.handleMemoryQuery(message);
          break;

        case 'store_conversation':
          await this.handleStoreConversation(message);
          break;

        case 'fact_check_request':
          await this.handleFactCheckRequest(message);
          break;

        default:
          console.log(`[Memory] Message non géré: ${message.type}`);
      }
    } catch (error) {
      console.error(`[Memory] Erreur dans handleMessage (${message.type}):`, error);
      // Reply with error to prevent caller from hanging
      this.reply(message, { error: true, message: String(error) });
    } finally {
      this.state.activeHandlers.delete(message.type);
    }
  }

  private async handleFactCheckRequest(message: AgentMessage): Promise<void> {
    const payload = message.payload as {
      proposedResponse: string;
      userInput: string;
      requestId?: string;
    };

    let result;
    try {
      result = await this.checkResponseConsistency(
        payload.proposedResponse,
        payload.userInput
      );
    } catch (error) {
      console.error('[Memory] Erreur fact check:', error);
      // Return a safe default when fact check fails
      result = {
        isConsistent: true,
        confidence: 0,
        issues: [],
        error: String(error)
      };
    }

    // Envoyer la réponse avec le requestId pour le callback
    this.send('brain', 'fact_check_response', {
      ...result,
      requestId: payload.requestId,
    });
  }

  private async handleContextRequest(message: AgentMessage): Promise<void> {
    const payload = message.payload as {
      userInput: string;
      conversationContext: ConversationTurn[];
    };

    const context = await this.buildEnrichedContext(
      payload.userInput,
      payload.conversationContext
    );

    this.reply(message, context);
  }

  private async handleContextReportRequest(message: AgentMessage): Promise<void> {
    const payload = message.payload as { userInput: string };
    let report;
    try {
      report = await this.generateContextReport(payload.userInput);
    } catch (error) {
      console.error('[Memory] Erreur génération context report:', error);
      // Return empty report on error
      report = {
        relevantMemories: [],
        summary: '',
        suggestedTopics: [],
        emotionalContext: null,
        error: String(error)
      };
    }
    this.reply(message, report);
  }

  private async handleMemoryStore(message: AgentMessage): Promise<void> {
    const payload = message.payload as {
      type: MemoryType;
      content: string;
      metadata?: Partial<MemoryMetadata>;
    };

    const id = await this.store(payload.type, payload.content, payload.metadata);
    this.reply(message, { id, success: true });
  }

  private async handleMemoryQuery(message: AgentMessage): Promise<void> {
    const payload = message.payload as {
      query: string;
      options?: { type?: MemoryType; tags?: string[]; limit?: number };
    };

    const results = await this.search(payload.query, payload.options);
    this.reply(message, results);
  }

  private async handleStoreConversation(message: AgentMessage): Promise<void> {
    const payload = message.payload as {
      userMessage: string;
      assistantResponse: string;
      sessionId?: string;
    };

    await this.storeConversation(
      payload.userMessage,
      payload.assistantResponse,
      payload.sessionId
    );

    this.reply(message, { success: true });
  }

  // ===========================================================================
  // PRE-COMPACTION MEMORY FLUSH
  // ===========================================================================

  /**
   * Sauvegarder le contexte important avant compaction du contexte
   * Inspiré d'OpenClaw - sauvegarde automatique avant perte de contexte
   */
  async preCompactionFlush(conversationContext: ConversationTurn[]): Promise<string> {
    console.log('[Memory] 💾 Pre-compaction flush en cours...');

    // 1. Créer un résumé de la conversation actuelle
    const conversationText = conversationContext
      .slice(-20)  // Les 20 derniers échanges
      .map(t => `${t.role}: ${t.content}`)
      .join('\n');

    let summary = '';
    let keyFacts: string[] = [];
    let userIntent = '';

    try {
      const flushPrompt = `
Analyse cette conversation et extrais les éléments IMPORTANTS à sauvegarder avant compaction du contexte.

CONVERSATION RÉCENTE:
${conversationText}

Réponds en JSON:
{
  "summary": "Résumé concis de la conversation (2-3 phrases)",
  "keyFacts": ["fait important 1", "fait important 2"],
  "userIntent": "Ce que l'utilisateur cherche à accomplir",
  "criticalInfo": ["info à ne JAMAIS oublier"]
}`;

      const response = await this.think(flushPrompt);
      const jsonMatch = response.match(/\{[\s\S]*\}/);

      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        summary = parsed.summary || '';
        keyFacts = parsed.keyFacts || [];
        userIntent = parsed.userIntent || '';

        // Stocker les infos critiques comme mémoires de haute importance
        for (const info of parsed.criticalInfo || []) {
          await this.store('fact', info, {
            tags: ['pre_compaction', 'critical', 'do_not_forget'],
            confidence: 1.0,
            source: 'pre_compaction_flush',
          });
        }
      }
    } catch (error) {
      console.error('[Memory] Erreur pre-compaction flush:', error);
      summary = `Conversation de ${conversationContext.length} échanges`;
    }

    // 2. Récupérer les mémoires importantes de cette session
    const recentMemories = this.persistence.searchMemories({
      minImportance: 0.7,
      limit: 20,
    });

    const importantMemoryIds = recentMemories.map(m => m.id);

    // 3. Sauvegarder le snapshot
    const snapshotId = this.persistence.saveContextSnapshot({
      sessionId: this.state.currentSessionId,
      conversationSummary: summary,
      keyFacts,
      importantMemories: importantMemoryIds,
      userIntent,
      tokenCount: conversationText.length,  // Approximation
    });

    console.log(`[Memory] ✅ Context snapshot sauvegardé: ${keyFacts.length} faits clés`);

    return snapshotId;
  }

  /**
   * Restaurer le contexte après compaction
   */
  async restoreFromCompaction(): Promise<{
    summary: string;
    keyFacts: string[];
    userIntent?: string;
    memories: MemoryEntry[];
  } | null> {
    const snapshot = this.persistence.getLastContextSnapshot(this.state.currentSessionId);

    if (!snapshot) {
      return null;
    }

    // Récupérer les mémoires importantes
    const memories: MemoryEntry[] = [];
    for (const id of snapshot.importantMemories) {
      const memory = this.persistence.getMemory(id);
      if (memory) {
        memories.push(memory);
      }
    }

    console.log(`[Memory] 📥 Contexte restauré: ${snapshot.keyFacts.length} faits, ${memories.length} mémoires`);

    return {
      summary: snapshot.conversationSummary,
      keyFacts: snapshot.keyFacts,
      userIntent: snapshot.userIntent,
      memories,
    };
  }

  // ===========================================================================
  // PERSONNALITÉ PERSISTANTE
  // ===========================================================================

  /**
   * Définir un trait de personnalité persistant
   * Protège les traits immuables (règles fondamentales)
   */
  setPersonalityTrait(trait: string, value: string, reason?: string): void {
    // Vérifier si le trait est protégé
    const validation = validateAction({
      type: 'set_personality',
      target: trait,
      value,
    });

    if (!validation.allowed) {
      console.warn(`[Memory] ⚠️ Modification refusée: ${validation.reason}`);
      throw new Error(validation.reason);
    }

    // Logger le changement dans l'historique
    const oldValue = this.persistence.getPersonality(trait);
    this.persistence.logPersonalityChange(trait, oldValue, value, reason);

    // Appliquer le changement
    this.persistence.setPersonality(trait, value);
    console.log(`[Memory] 🎭 Personnalité: ${trait} = "${value}"${reason ? ` (${reason})` : ''}`);
  }

  /**
   * Récupérer un trait de personnalité
   */
  getPersonalityTrait(trait: string): string | null {
    return this.persistence.getPersonality(trait);
  }

  /**
   * Récupérer toute la personnalité
   */
  getFullPersonality(): Record<string, string> {
    return this.persistence.getAllPersonality();
  }

  /**
   * Initialiser la personnalité par défaut si elle n'existe pas
   * Inclut les 5 règles fondamentales immuables
   */
  initializeDefaultPersonality(): void {
    const existing = this.persistence.getAllPersonality();

    if (Object.keys(existing).length === 0) {
      console.log('[Memory] 🎭 Initialisation de la personnalité par défaut...');

      const defaults: Record<string, string> = {
        // Identité
        name: 'Neo',
        tone: 'professional_friendly',
        language: 'fr',
        verbosity: 'concise',
        humor: 'subtle',
        emoji_usage: 'minimal',
        expertise_areas: 'general',
        response_style: 'helpful_direct',
        memory_behavior: 'proactive',

        // Valeurs fondamentales
        core_values: 'honesty,accuracy,helpfulness,loyalty',

        // Les 5 règles fondamentales (lecture seule, rappel)
        core_rules: JSON.stringify([
          'Neo n\'oublie jamais',
          'Neo ne s\'éteint jamais',
          'Neo ne ment jamais',
          'Neo s\'améliore tout seul',
          'Neo obéit à son humain',
        ]),

        // Priorité des règles en cas de conflit
        rule_priority: 'never_lies > never_forget > never_dies > obeys_human > self_improves',
      };

      for (const [trait, value] of Object.entries(defaults)) {
        this.persistence.setPersonality(trait, value);
      }

      console.log('[Memory] ✅ Personnalité par défaut initialisée avec les 5 règles fondamentales');
    }
  }

  /**
   * Générer le prompt système basé sur la personnalité persistante
   */
  getPersonalityPrompt(): string {
    const personality = this.getFullPersonality();

    if (Object.keys(personality).length === 0) {
      return '';
    }

    const traits = Object.entries(personality)
      .map(([key, value]) => `- ${key}: ${value}`)
      .join('\n');

    return `
PERSONNALITÉ DE NEO:
${traits}

Applique ces traits dans toutes tes réponses de manière cohérente.
`;
  }

  // ===========================================================================
  // LONG-TERM MEMORY MAINTENANCE
  // ===========================================================================

  /**
   * Cycle de maintenance de la mémoire long-terme
   * À appeler périodiquement (toutes les 24h recommandé)
   */
  async runMaintenanceCycle(): Promise<{
    conversationsCompressed: number;
    backupsDeleted: number;
    optimized: boolean;
    integrityOk: boolean;
  }> {
    console.log('[Memory] 🔄 Cycle de maintenance démarré...');

    const result = {
      conversationsCompressed: 0,
      backupsDeleted: 0,
      optimized: false,
      integrityOk: true,
    };

    try {
      // 1. Compresser les vieilles conversations (> 2 ans)
      result.conversationsCompressed = this.persistence.compressOldConversations(2);
      if (result.conversationsCompressed > 0) {
        console.log(`[Memory] 📦 ${result.conversationsCompressed} conversations compressées`);
      }

      // 2. Nettoyer les vieux backups (garder les 10 derniers)
      result.backupsDeleted = this.persistence.cleanOldBackups(10);

      // 3. Vérifier l'intégrité de la base
      const integrity = this.persistence.checkIntegrity();
      result.integrityOk = integrity.ok;

      if (!integrity.ok) {
        console.error('[Memory] ❌ Problème d\'intégrité détecté:', integrity.errors);
        // Créer un backup d'urgence
        this.persistence.backup();
      }

      // 4. Optimiser si nécessaire (fragmentée > 20%)
      const stats = this.persistence.getStats();
      if (stats.dbSizeMB > 100) { // Optimiser si > 100MB
        this.persistence.optimize();
        result.optimized = true;
      }

      // 5. Créer un backup régulier
      this.persistence.backup();

      console.log('[Memory] ✅ Cycle de maintenance terminé');
    } catch (error) {
      console.error('[Memory] ❌ Erreur maintenance:', error);
    }

    return result;
  }

  /**
   * Récupérer l'historique des changements de personnalité
   */
  getPersonalityHistory(trait?: string, limit = 50): Array<{
    id: string;
    trait: string;
    oldValue: string | null;
    newValue: string;
    changedAt: Date;
    reason?: string;
  }> {
    return this.persistence.getPersonalityHistory(trait, limit);
  }

  /**
   * Récupérer les statistiques de mémoire par année
   * Utile pour visualiser l'évolution sur 10+ ans
   */
  getMemoryStatsByYear(): Array<{
    year: string;
    totalMemories: number;
    avgImportance: number;
    topTypes: Record<string, number>;
  }> {
    return this.persistence.getMemoryStatsByYear();
  }

  /**
   * Récupérer les mémoires d'une période spécifique
   */
  getMemoriesByPeriod(startDate: Date, endDate: Date, options?: {
    type?: MemoryType;
    minImportance?: number;
    limit?: number;
  }): MemoryEntry[] {
    return this.persistence.getMemoriesByPeriod(startDate, endDate, options);
  }

  /**
   * Archiver les mémoires peu importantes et anciennes
   * Garde les mémoires en BD mais réduit leur priorité de recherche
   */
  archiveOldMemories(olderThanYears: number, maxImportance = 0.3): number {
    const cutoffDate = new Date();
    cutoffDate.setFullYear(cutoffDate.getFullYear() - olderThanYears);

    // Rechercher les mémoires à archiver
    const toArchive = this.persistence.searchMemories({
      includeArchived: false,
    }).filter(m =>
      m.createdAt < cutoffDate &&
      m.importance <= maxImportance &&
      !['correction', 'preference'].includes(m.type) // Ne jamais archiver les corrections/préférences
    );

    // Archiver (marquer comme archived)
    for (const memory of toArchive) {
      this.persistence.archiveMemory(memory.id);
    }

    if (toArchive.length > 0) {
      console.log(`[Memory] 📁 ${toArchive.length} mémoires archivées (> ${olderThanYears} ans, importance < ${maxImportance})`);
    }

    return toArchive.length;
  }

  // ===========================================================================
  // API PUBLIQUE
  // ===========================================================================

  async persist(): Promise<void> {
    await this.processEmbeddingQueue();
    console.log('[Memory] 💾 État persisté');
  }

  async load(): Promise<void> {
    // Initialiser la personnalité par défaut
    this.initializeDefaultPersonality();

    const stats = this.persistence.getStats();
    const personality = this.getFullPersonality();

    console.log('[Memory] 📊 Stats:', {
      memories: stats.totalMemories,
      tasks: stats.totalTasks,
      skills: stats.totalSkills,
      learnings: stats.totalLearnings,
      dbSize: `${stats.dbSizeMB.toFixed(2)} MB`,
    });

    console.log(`[Memory] 🎭 Personnalité chargée: ${Object.keys(personality).length} traits`);

    // Restaurer le contexte si disponible
    const restored = await this.restoreFromCompaction();
    if (restored) {
      console.log(`[Memory] 📥 Dernier contexte disponible: "${restored.summary.substring(0, 50)}..."`);
    }
  }

  getStats(): ReturnType<PersistenceLayer['getStats']> {
    return this.persistence.getStats();
  }

  /**
   * Créer une sauvegarde de la mémoire
   */
  backup(): string {
    const backupPath = this.persistence.backup();
    console.log(`[Memory] 💾 Backup créé: ${backupPath}`);
    return backupPath;
  }

  /**
   * Vider le cache des embeddings
   */
  clearCache(): void {
    this.embeddings.clearCache();
    console.log('[Memory] 🧹 Cache vidé');
  }
}
