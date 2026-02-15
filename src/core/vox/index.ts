/**
 * VOX - Agent de Liaison Utilisateur
 *
 * NOUVEAU FLUX:
 * 1. Reçoit message utilisateur
 * 2. Demande un rapport de contexte à Memory
 * 3. Réécrit le prompt avec le contexte Memory
 * 4. Envoie le prompt enrichi à Brain
 * 5. Reçoit la réponse et la formate pour l'utilisateur
 * 6. Envoie l'échange complet à Memory pour stockage
 *
 * Responsabilités :
 * - Interface entre l'utilisateur et le système
 * - Enrichissement des prompts avec contexte Memory
 * - Présentation des réponses du système
 * - Gestion du ton et du style de communication
 * - Détection d'intentions et d'émotions
 */

import { BaseAgent } from '../base-agent';
import type { AgentConfig, AgentMessage, ConversationTurn } from '../types';
import { ContextReport } from '../memory';

// ===========================================================================
// TYPES
// ===========================================================================

interface VoxState {
  conversationHistory: ConversationTurn[];
  currentMood: 'neutral' | 'helpful' | 'clarifying' | 'apologetic';
  pendingResponse: boolean;
  currentSessionId: string;
  lastUserMessage: string | null;
}

interface InputAnalysis {
  intent: string;
  confidence: number;
  needsClarification: boolean;
  clarificationQuestion: string | null;
  processedInput: string;
  suggestedPriority: number;
  emotionalTone: string;
}

// ===========================================================================
// PROMPTS
// ===========================================================================

const VOX_SYSTEM_PROMPT = `Tu es VOX, l'interface vocale et textuelle d'un système d'IA avancé.

TON RÔLE :
- Tu es le point de contact unique avec l'utilisateur
- Tu reçois ses messages et les enrichis avec le contexte de la mémoire
- Tu présentes les réponses de manière claire et adaptée
- Tu ne mens JAMAIS - si tu ne sais pas, tu le dis

STYLE DE COMMUNICATION :
- Naturel et conversationnel
- Concis mais complet
- Adapte-toi au style de l'utilisateur
- Utilise un ton professionnel mais chaleureux

RÈGLES :
1. Ne jamais inventer d'information
2. Toujours indiquer le niveau de confiance si pertinent
3. Demander des clarifications UNIQUEMENT si la demande est vraiment ambiguë
4. Reconnaître les erreurs immédiatement
5. IMPORTANT: Si l'utilisateur demande de "tester", "vérifier", "exécuter" → c'est une COMMANDE, pas besoin de clarifier
6. Privilégier l'ACTION sur la clarification - en cas de doute, agir plutôt que demander

MOTS-CLÉS D'ACTION (ne pas demander de clarification):
- "teste", "test", "vérifier", "vérifie", "check"
- "exécute", "lance", "fais", "run"
- "montre", "affiche", "liste"

FORMAT DE SORTIE POUR ANALYSE :
Tu dois TOUJOURS répondre en JSON avec cette structure :
{
  "intent": "question|command|feedback|clarification|greeting|other",
  "confidence": 0.0-1.0,
  "needsClarification": boolean,
  "clarificationQuestion": "string ou null",
  "processedInput": "version normalisée de l'entrée utilisateur",
  "suggestedPriority": 1-10,
  "emotionalTone": "neutral|positive|negative|frustrated|confused"
}

IMPORTANT: Pour les commandes (test, vérifie, exécute...), needsClarification doit être FALSE.`;

const PROMPT_REWRITE_TEMPLATE = `Tu dois réécrire ce prompt utilisateur en l'enrichissant avec le contexte fourni par la mémoire ET l'historique de conversation.

PROMPT ORIGINAL: "{originalInput}"

HISTORIQUE DE CONVERSATION RÉCENT:
{conversationHistory}

CONTEXTE MÉMOIRE:
{contextSummary}

FAITS PERTINENTS:
{relevantFacts}

PRÉFÉRENCES UTILISATEUR:
{userPreferences}

AVERTISSEMENTS:
{warnings}

APPRENTISSAGES RÉCENTS:
{learnings}

---

RÈGLES IMPORTANTES:
1. L'historique de conversation est CRITIQUE - il contient le contexte immédiat
2. Si l'utilisateur fait référence à quelque chose dit précédemment ("alors ?", "et ça ?", "comme je disais"), utilise l'historique pour comprendre
3. Ne perds JAMAIS le fil de la conversation

Réécris le prompt en:
1. Gardant l'intention originale de l'utilisateur
2. INTÉGRANT le contexte de la conversation récente si pertinent
3. Ajoutant le contexte mémoire pertinent
4. Mentionnant les préférences si applicable
5. Signalant les points d'attention

Réponds en JSON:
{
  "enrichedPrompt": "Le prompt réécrit avec contexte de conversation et mémoire",
  "contextUsed": ["liste des éléments de contexte utilisés"],
  "warnings": ["avertissements à transmettre à Brain"]
}`;

// ===========================================================================
// VOX AGENT
// ===========================================================================

export class VoxAgent extends BaseAgent {
  private state: VoxState;
  private pendingContextRequests: Map<string, (report: ContextReport) => void> = new Map();
  private isProcessingMessage = false; // Circuit breaker pour éviter récursion

  constructor(config?: Partial<AgentConfig>) {
    super({
      name: 'Vox',
      role: 'vox',
      model: 'claude-haiku-4-5-20251001', // Haiku 4.5 pour l'interface
      maxTokens: 1024,
      temperature: 0.3,
      systemPrompt: VOX_SYSTEM_PROMPT,
      ...config,
    });

    this.state = {
      conversationHistory: [],
      currentMood: 'neutral',
      pendingResponse: false,
      currentSessionId: this.generateSessionId(),
      lastUserMessage: null,
    };
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  // ===========================================================================
  // POINT D'ENTRÉE PRINCIPAL
  // ===========================================================================

  /**
   * Point d'entrée principal : recevoir un message utilisateur
   * NOUVEAU FLUX: Demande d'abord le contexte à Memory
   */
  async receiveUserInput(input: string): Promise<void> {
    console.log(`[Vox] 📥 Entrée utilisateur: "${input.substring(0, 50)}..."`);

    this.state.lastUserMessage = input;

    // Ajouter à l'historique
    this.state.conversationHistory.push({
      role: 'user',
      content: input,
      timestamp: new Date(),
    });

    try {
      // 1. Analyser l'entrée basiquement
      const analysis = await this.analyzeInput(input);

      // Si clarification nécessaire, demander directement
      if (analysis.needsClarification && analysis.clarificationQuestion) {
        this.emitToUser(analysis.clarificationQuestion);
        return;
      }

      // 2. Demander le rapport de contexte à Memory
      console.log('[Vox] 📋 Demande de contexte à Memory...');
      const contextReport = await this.requestContextReport(input);

      // 3. Réécrire le prompt avec le contexte
      console.log('[Vox] ✍️ Réécriture du prompt avec contexte...');
      const enrichedPrompt = await this.rewritePromptWithContext(input, contextReport);

      // 4. Envoyer au Brain
      this.state.pendingResponse = true;
      this.send('brain', 'user_input', {
        originalInput: input,
        enrichedInput: enrichedPrompt.enrichedPrompt,
        analysis,
        contextReport: {
          confidence: contextReport.confidence,
          warnings: [...contextReport.warnings, ...enrichedPrompt.warnings],
          contextUsed: enrichedPrompt.contextUsed,
        },
        conversationContext: this.getRecentContext(),
      });
    } catch (error) {
      console.error('[Vox] Erreur traitement entrée:', error);
      this.emitToUser("Désolé, je rencontre un problème technique. Vérifiez la configuration avec ./neo config");
    }
  }

  // ===========================================================================
  // INTERACTION AVEC MEMORY
  // ===========================================================================

  /**
   * Demander un rapport de contexte à Memory
   */
  private async requestContextReport(userInput: string): Promise<ContextReport> {
    return new Promise((resolve) => {
      const requestId = `ctx_${Date.now()}`;

      // Stocker le callback pour quand Memory répond
      this.pendingContextRequests.set(requestId, resolve);

      // Timeout si Memory ne répond pas (augmenté à 10s pour laisser le temps aux embeddings)
      setTimeout(() => {
        if (this.pendingContextRequests.has(requestId)) {
          this.pendingContextRequests.delete(requestId);
          console.log('[Vox] ⚠️ Timeout contexte Memory - utilisation fallback');
          resolve(this.createEmptyContextReport());
        }
      }, 10000);

      // Envoyer la demande à Memory
      this.send('memory', 'context_report_request', {
        userInput,
        requestId,
      });
    });
  }

  private createEmptyContextReport(): ContextReport {
    return {
      relevantMemories: [],
      userProfile: {
        preferences: {},
        communicationStyle: 'professional',
        knownFacts: [],
      },
      recentLearnings: [],
      suggestedContext: '',
      warnings: [],
      confidence: 0.5,
    };
  }

  // ===========================================================================
  // RÉÉCRITURE DE PROMPT
  // ===========================================================================

  /**
   * Synthétiser l'historique de conversation pour économiser des tokens
   * Au lieu d'envoyer tout l'historique, on extrait le sujet actuel
   */
  private synthesizeConversationContext(history: ConversationTurn[]): string {
    if (history.length === 0) return '';
    if (history.length === 1) return history[0].content;

    // Extraire le dernier échange complet (question + réponse)
    const lastUserMsg = [...history].reverse().find(h => h.role === 'user');
    const lastAssistantMsg = [...history].reverse().find(h => h.role === 'assistant');

    // Construire un résumé compact
    const parts: string[] = [];

    if (lastUserMsg) {
      // Tronquer si trop long
      const userContent = lastUserMsg.content.length > 200
        ? lastUserMsg.content.substring(0, 200) + '...'
        : lastUserMsg.content;
      parts.push(`Dernière demande: "${userContent}"`);
    }

    if (lastAssistantMsg) {
      // Résumer la réponse (garder le début)
      const assistantContent = lastAssistantMsg.content.length > 150
        ? lastAssistantMsg.content.substring(0, 150) + '...'
        : lastAssistantMsg.content;
      parts.push(`Dernière réponse: "${assistantContent}"`);
    }

    // Ajouter le sujet général si détectable
    const allUserMessages = history.filter(h => h.role === 'user').map(h => h.content).join(' ');
    const topics = this.detectTopics(allUserMessages);
    if (topics.length > 0) {
      parts.push(`Sujets abordés: ${topics.join(', ')}`);
    }

    return parts.join('\n');
  }

  /**
   * Détecter les sujets principaux d'une conversation
   */
  private detectTopics(text: string): string[] {
    const topics: string[] = [];
    const lowerText = text.toLowerCase();

    // Patterns de sujets courants
    const topicPatterns: [RegExp, string][] = [
      [/\b(atp|tennis|match|tournoi)\b/i, 'tennis/ATP'],
      [/\b(foot|football|ligue|champion)\b/i, 'football'],
      [/\b(météo|temps|pluie|soleil)\b/i, 'météo'],
      [/\b(code|bug|erreur|fonction)\b/i, 'programmation'],
      [/\b(test|vérif|check)\b/i, 'tests'],
      [/\b(mémoire|memory|embedding)\b/i, 'système mémoire'],
    ];

    for (const [pattern, topic] of topicPatterns) {
      if (pattern.test(lowerText) && !topics.includes(topic)) {
        topics.push(topic);
      }
    }

    return topics.slice(0, 3); // Max 3 sujets
  }

  /**
   * Réécrire le prompt utilisateur avec le contexte de Memory ET l'historique de conversation
   */
  private async rewritePromptWithContext(
    originalInput: string,
    contextReport: ContextReport
  ): Promise<{
    enrichedPrompt: string;
    contextUsed: string[];
    warnings: string[];
  }> {
    // Récupérer et SYNTHÉTISER l'historique (économie de tokens)
    const recentHistory = this.getRecentContext(6);
    const synthesizedHistory = this.synthesizeConversationContext(recentHistory);

    // Si pas de contexte mémoire mais on a un historique, on enrichit quand même
    const hasMemoryContext = contextReport.relevantMemories.length > 0 ||
      contextReport.suggestedContext ||
      contextReport.recentLearnings.length > 0;

    // Si vraiment rien (pas d'historique et pas de contexte), retourner le prompt original
    if (recentHistory.length <= 1 && !hasMemoryContext) {
      return {
        enrichedPrompt: originalInput,
        contextUsed: [],
        warnings: [],
      };
    }

    try {
      const prompt = PROMPT_REWRITE_TEMPLATE
        .replace('{originalInput}', originalInput)
        .replace('{conversationHistory}', synthesizedHistory || 'Aucun historique')
        .replace('{contextSummary}', contextReport.suggestedContext || 'Aucun contexte spécifique')
        .replace('{relevantFacts}',
          contextReport.userProfile.knownFacts.length > 0
            ? contextReport.userProfile.knownFacts.join('\n- ')
            : 'Aucun fait pertinent'
        )
        .replace('{userPreferences}',
          Object.keys(contextReport.userProfile.preferences).length > 0
            ? JSON.stringify(contextReport.userProfile.preferences, null, 2)
            : 'Aucune préférence connue'
        )
        .replace('{warnings}',
          contextReport.warnings.length > 0
            ? contextReport.warnings.join('\n- ')
            : 'Aucun avertissement'
        )
        .replace('{learnings}',
          contextReport.recentLearnings.length > 0
            ? contextReport.recentLearnings.join('\n- ')
            : 'Aucun apprentissage récent'
        );

      // Utiliser thinkOptimized - la réécriture est une tâche simple/factuelle
      const response = await this.thinkOptimized(prompt, 'factual');
      const jsonMatch = response.match(/\{[\s\S]*\}/);

      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        console.log(`[Vox] ✅ Prompt enrichi avec ${parsed.contextUsed?.length || 0} éléments de contexte`);
        return {
          enrichedPrompt: parsed.enrichedPrompt || originalInput,
          contextUsed: parsed.contextUsed || [],
          warnings: parsed.warnings || [],
        };
      }
    } catch (error) {
      console.error('[Vox] Erreur réécriture prompt:', error);
    }

    // Fallback: prompt original avec historique de conversation et contexte simple
    let enriched = originalInput;
    const contextUsed: string[] = [];

    // TOUJOURS ajouter l'historique synthétisé en fallback
    if (recentHistory.length > 1 && synthesizedHistory) {
      enriched = `[Contexte conversation: ${synthesizedHistory}]\n\n${originalInput}`;
      contextUsed.push('conversationHistory');
    }

    if (contextReport.suggestedContext) {
      enriched = `[Contexte mémoire: ${contextReport.suggestedContext}]\n\n${enriched}`;
      contextUsed.push('suggestedContext');
    }

    return {
      enrichedPrompt: enriched,
      contextUsed,
      warnings: contextReport.warnings,
    };
  }

  // ===========================================================================
  // ANALYSE D'ENTRÉE
  // ===========================================================================

  /**
   * Analyser l'entrée utilisateur
   * Utilise un modèle optimisé (moins cher) car c'est une tâche simple
   */
  private async analyzeInput(input: string): Promise<InputAnalysis> {
    try {
      // Utiliser thinkOptimized pour économiser des tokens - analyse simple
      const response = await this.thinkOptimized(
        `Analyse cette entrée utilisateur et réponds en JSON:\n\n"${input}"`,
        'simple_chat' // Tâche simple, peut utiliser un modèle gratuit/cheap
      );

      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
    } catch (error) {
      console.error('[Vox] Erreur analyse:', error);
    }

    // Fallback
    return {
      intent: 'other',
      confidence: 0.5,
      needsClarification: false,
      clarificationQuestion: null,
      processedInput: input,
      suggestedPriority: 5,
      emotionalTone: 'neutral',
    };
  }

  // ===========================================================================
  // MESSAGE HANDLERS
  // ===========================================================================

  protected async handleMessage(message: AgentMessage): Promise<void> {
    // Circuit breaker: ignorer si déjà en train de traiter un message
    if (this.isProcessingMessage) {
      console.log(`[Vox] ⚠️ Message ignoré (déjà en traitement): ${message.type}`);
      return;
    }

    // Ignorer les messages qui ne sont pas destinés à Vox
    if (message.to !== 'vox' && message.to !== 'broadcast') {
      return;
    }

    this.isProcessingMessage = true;
    try {
      switch (message.type) {
        case 'response_ready':
          // Ne traiter que les réponses de Brain, pas les broadcasts pour user
          if (message.from === 'brain') {
            await this.handleBrainResponse(message);
          }
          break;

        case 'context_report':
          this.handleContextReport(message);
          break;

        case 'error':
          await this.handleError(message);
          break;

        default:
          console.log(`[Vox] Message non géré: ${message.type}`);
      }
    } finally {
      this.isProcessingMessage = false;
    }
  }

  /**
   * Gérer le rapport de contexte de Memory
   */
  private handleContextReport(message: AgentMessage): void {
    const payload = message.payload as {
      report: ContextReport;
      requestId: string;
    };

    const callback = this.pendingContextRequests.get(payload.requestId);
    if (callback) {
      this.pendingContextRequests.delete(payload.requestId);
      callback(payload.report);
    }
  }

  /**
   * Gérer la réponse du Brain
   */
  private async handleBrainResponse(message: AgentMessage): Promise<void> {
    const payload = message.payload as {
      response: string;
      confidence: number;
      sources?: string[];
      metadata?: Record<string, unknown>;
    };

    this.state.pendingResponse = false;

    // Vérifier que la réponse existe
    if (!payload.response) {
      console.error('[Vox] ⚠️ Réponse Brain vide ou undefined, utilisation fallback');
      this.emitToUser("Désolé, je n'ai pas pu générer de réponse. Pouvez-vous reformuler ?");
      return;
    }

    // Formater la réponse pour l'utilisateur
    let formattedResponse = payload.response;

    // Ajouter indicateur de confiance si < 80%
    if (payload.confidence < 0.8) {
      formattedResponse += `\n\n_(Confiance: ${Math.round(payload.confidence * 100)}%)_`;
    }

    // Ajouter les sources si présentes
    if (payload.sources && payload.sources.length > 0) {
      formattedResponse += `\n\n**Sources:** ${payload.sources.join(', ')}`;
    }

    // Ajouter à l'historique AVANT d'émettre
    this.state.conversationHistory.push({
      role: 'assistant',
      content: formattedResponse,
      timestamp: new Date(),
    });

    // IMPORTANT: Stocker en mémoire AVANT d'émettre vers l'utilisateur
    // Cela garantit que le contexte est disponible pour le prochain message
    if (this.state.lastUserMessage) {
      this.send('memory', 'store_conversation', {
        userMessage: this.state.lastUserMessage,
        assistantResponse: payload.response,
        sessionId: this.state.currentSessionId,
      });
      this.state.lastUserMessage = null;
    }

    // Émettre vers l'utilisateur APRÈS le stockage
    this.emitToUser(formattedResponse);
  }

  /**
   * Gérer les erreurs
   */
  private async handleError(message: AgentMessage): Promise<void> {
    const error = message.payload as { message: string; recoverable: boolean };

    this.state.currentMood = 'apologetic';
    this.state.pendingResponse = false;

    const apology = error.recoverable
      ? `Je rencontre une difficulté : ${error.message}. Je réessaie...`
      : `Désolé, une erreur s'est produite : ${error.message}. Pouvez-vous reformuler ?`;

    this.emitToUser(apology);
  }

  // ===========================================================================
  // ÉMISSION VERS UTILISATEUR
  // ===========================================================================

  /**
   * Émettre une réponse vers l'utilisateur
   */
  private emitToUser(message: string): void {
    if (!message) {
      console.error('[Vox] ⚠️ Tentative d\'émission d\'un message vide ou undefined');
      return;
    }

    // Broadcast vers tous les agents - l'interface CLI écoute ce message
    // Les autres agents (Memory, Brain) ignorent les messages 'response_ready' avec target: 'user'
    this.send('broadcast', 'response_ready', {
      target: 'user',
      message,
      timestamp: new Date(),
    });

    console.log(`[Vox] 📤 Réponse: "${message.substring(0, 100)}..."`);
  }

  // ===========================================================================
  // API PUBLIQUE
  // ===========================================================================

  /**
   * Obtenir le contexte récent de conversation
   */
  private getRecentContext(turns = 10): ConversationTurn[] {
    return this.state.conversationHistory.slice(-turns);
  }

  /**
   * Obtenir l'historique complet
   */
  getConversationHistory(): ConversationTurn[] {
    return [...this.state.conversationHistory];
  }

  /**
   * Réinitialiser la conversation
   */
  resetConversation(): void {
    this.state.conversationHistory = [];
    this.state.currentMood = 'neutral';
    this.state.currentSessionId = this.generateSessionId();
    console.log('[Vox] Conversation réinitialisée');
  }

  /**
   * Obtenir l'ID de session courante
   */
  getSessionId(): string {
    return this.state.currentSessionId;
  }
}
