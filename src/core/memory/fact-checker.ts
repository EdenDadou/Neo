/**
 * Fact Checker - Vérification des faits et détection de contradictions
 *
 * Ce module permet de:
 * 1. Vérifier si une réponse contredit des faits stockés
 * 2. Détecter les corrections de l'utilisateur
 * 3. Calculer la confiance basée sur les mémoires
 */

import Anthropic from '@anthropic-ai/sdk';
import { MemoryEntry } from '../types';

export interface FactCheckResult {
  isConsistent: boolean;
  contradictions: Array<{
    claim: string;
    storedFact: string;
    memoryId: string;
    severity: 'minor' | 'major' | 'critical';
  }>;
  supportedBy: string[]; // IDs des mémoires qui supportent la réponse
  confidence: number;
  warnings: string[];
}

export interface CorrectionDetectionResult {
  isCorrection: boolean;
  confidence: number;
  details: {
    originalError: string;
    correction: string;
    feedback: string;
    correctionType: 'factual' | 'preference' | 'misunderstanding' | 'clarification';
  } | null;
  triggerWords: string[];
}

const CORRECTION_TRIGGERS = [
  'non', 'pas vraiment', 'en fait', 'c\'est pas', 'ce n\'est pas',
  'tu te trompes', 'erreur', 'faux', 'incorrect', 'plutôt',
  'je voulais dire', 'je parlais de', 'pas ça', 'mauvais',
  'corrige', 'rectifie', 'au contraire', 'absolument pas'
];

export class FactChecker {
  private client: Anthropic;

  constructor() {
    this.client = new Anthropic();
  }

  /**
   * Détecter si un message utilisateur est une correction
   */
  async detectCorrection(
    userMessage: string,
    previousAssistantResponse: string
  ): Promise<CorrectionDetectionResult> {
    const lowerMessage = userMessage.toLowerCase();

    // Détection rapide par mots-clés
    const foundTriggers = CORRECTION_TRIGGERS.filter(t => lowerMessage.includes(t));
    const hasQuickTrigger = foundTriggers.length > 0;

    // Si pas de trigger évident, confiance basse
    if (!hasQuickTrigger && userMessage.length > 100) {
      return {
        isCorrection: false,
        confidence: 0.9,
        details: null,
        triggerWords: [],
      };
    }

    // Analyse approfondie avec Claude
    try {
      const response = await this.client.messages.create({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 500,
        temperature: 0,
        system: `Tu es un détecteur de corrections. Analyse si l'utilisateur corrige ou contredit ce que l'assistant a dit.

IMPORTANT: Sois précis. Une question n'est PAS une correction. Un complément d'info n'est PAS une correction.
Une CORRECTION c'est quand l'utilisateur dit explicitement que l'assistant s'est trompé.

Réponds UNIQUEMENT en JSON valide:
{
  "isCorrection": boolean,
  "confidence": 0.0-1.0,
  "correctionType": "factual|preference|misunderstanding|clarification|none",
  "originalError": "ce que l'assistant a dit de faux" ou null,
  "correction": "la bonne information" ou null,
  "feedback": "explication courte de l'erreur" ou null
}`,
        messages: [{
          role: 'user',
          content: `RÉPONSE PRÉCÉDENTE DE L'ASSISTANT:
"${previousAssistantResponse.substring(0, 500)}"

MESSAGE DE L'UTILISATEUR:
"${userMessage}"

Analyse: est-ce une correction?`
        }]
      });

      const content = response.content[0];
      if (content.type === 'text') {
        const jsonMatch = content.text.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);

          return {
            isCorrection: parsed.isCorrection === true,
            confidence: parsed.confidence || 0.8,
            details: parsed.isCorrection ? {
              originalError: parsed.originalError || '',
              correction: parsed.correction || '',
              feedback: parsed.feedback || '',
              correctionType: parsed.correctionType || 'factual',
            } : null,
            triggerWords: foundTriggers,
          };
        }
      }
    } catch (error) {
      console.error('[FactChecker] Erreur détection correction:', error);
    }

    // Fallback basé sur les triggers
    return {
      isCorrection: hasQuickTrigger,
      confidence: hasQuickTrigger ? 0.6 : 0.3,
      details: hasQuickTrigger ? {
        originalError: 'Non déterminé (fallback)',
        correction: userMessage,
        feedback: 'Correction détectée par mots-clés',
        correctionType: 'factual',
      } : null,
      triggerWords: foundTriggers,
    };
  }

  /**
   * Vérifier si une réponse contredit des faits stockés
   */
  async checkFactConsistency(
    proposedResponse: string,
    relevantMemories: MemoryEntry[]
  ): Promise<FactCheckResult> {
    // Si pas de mémoires pertinentes, pas de contradiction possible
    if (relevantMemories.length === 0) {
      return {
        isConsistent: true,
        contradictions: [],
        supportedBy: [],
        confidence: 0.5, // Confiance basse car pas de vérification possible
        warnings: ['Aucune mémoire pertinente pour vérifier'],
      };
    }

    // Extraire les faits des mémoires
    const factsContext = relevantMemories
      .filter(m => m.type === 'fact' || m.type === 'correction' || m.type === 'preference')
      .slice(0, 15)
      .map(m => `[${m.id.substring(0, 8)}] ${m.type}: ${m.content}`)
      .join('\n');

    if (!factsContext) {
      return {
        isConsistent: true,
        contradictions: [],
        supportedBy: [],
        confidence: 0.6,
        warnings: ['Pas de faits vérifiables dans les mémoires'],
      };
    }

    try {
      const response = await this.client.messages.create({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 800,
        temperature: 0,
        system: `Tu es un vérificateur de cohérence. Compare la réponse proposée aux faits stockés.

RÈGLES:
- Une contradiction = la réponse dit le CONTRAIRE d'un fait stocké
- Un fait non mentionné n'est PAS une contradiction
- Une généralisation n'est PAS une contradiction
- Seuls les VRAIS conflits sont des contradictions

Réponds en JSON:
{
  "isConsistent": boolean,
  "contradictions": [
    {
      "claim": "ce que dit la réponse",
      "storedFact": "ce qui est en mémoire",
      "memoryId": "ID de la mémoire",
      "severity": "minor|major|critical"
    }
  ],
  "supportingMemories": ["id1", "id2"],
  "confidence": 0.0-1.0,
  "warnings": ["avertissement si pertinent"]
}`,
        messages: [{
          role: 'user',
          content: `FAITS STOCKÉS EN MÉMOIRE:
${factsContext}

RÉPONSE PROPOSÉE:
"${proposedResponse.substring(0, 1000)}"

Vérifie la cohérence.`
        }]
      });

      const content = response.content[0];
      if (content.type === 'text') {
        const jsonMatch = content.text.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);

          return {
            isConsistent: parsed.isConsistent !== false,
            contradictions: (parsed.contradictions || []).map((c: {
              claim?: string;
              storedFact?: string;
              memoryId?: string;
              severity?: string;
            }) => ({
              claim: c.claim || '',
              storedFact: c.storedFact || '',
              memoryId: c.memoryId || '',
              severity: c.severity || 'minor',
            })),
            supportedBy: parsed.supportingMemories || [],
            confidence: parsed.confidence || 0.7,
            warnings: parsed.warnings || [],
          };
        }
      }
    } catch (error) {
      console.error('[FactChecker] Erreur vérification faits:', error);
    }

    // Fallback
    return {
      isConsistent: true,
      contradictions: [],
      supportedBy: [],
      confidence: 0.5,
      warnings: ['Vérification impossible (erreur)'],
    };
  }

  /**
   * Calculer un score de confiance basé sur les mémoires
   */
  calculateConfidenceFromMemories(
    relevantMemories: MemoryEntry[],
    factCheckResult: FactCheckResult
  ): number {
    let confidence = 0.5; // Base

    // Bonus si des mémoires supportent la réponse
    const supportCount = factCheckResult.supportedBy.length;
    confidence += Math.min(supportCount * 0.1, 0.3);

    // Malus pour contradictions
    for (const contradiction of factCheckResult.contradictions) {
      switch (contradiction.severity) {
        case 'critical':
          confidence -= 0.3;
          break;
        case 'major':
          confidence -= 0.2;
          break;
        case 'minor':
          confidence -= 0.1;
          break;
      }
    }

    // Bonus si les mémoires pertinentes sont de haute confiance
    const avgMemoryConfidence = relevantMemories.length > 0
      ? relevantMemories.reduce((sum, m) => sum + (m.metadata.confidence || 0.5), 0) / relevantMemories.length
      : 0.5;
    confidence += (avgMemoryConfidence - 0.5) * 0.2;

    // Bonus si les mémoires sont récentes et fréquemment accédées
    const avgAccessCount = relevantMemories.length > 0
      ? relevantMemories.reduce((sum, m) => sum + m.accessCount, 0) / relevantMemories.length
      : 0;
    confidence += Math.min(avgAccessCount * 0.02, 0.1);

    // Clamp entre 0.1 et 0.99
    return Math.max(0.1, Math.min(0.99, confidence));
  }

  /**
   * Résoudre une contradiction en choisissant la meilleure source
   */
  async resolveContradiction(
    _proposedClaim: string,
    contradictingMemory: MemoryEntry
  ): Promise<{
    keepProposed: boolean;
    reason: string;
    suggestedAction: 'update_memory' | 'reject_response' | 'flag_for_review';
  }> {
    // Si la mémoire est une correction utilisateur, elle a priorité
    if (contradictingMemory.type === 'correction') {
      return {
        keepProposed: false,
        reason: 'La mémoire contradictoire est une correction utilisateur (priorité)',
        suggestedAction: 'reject_response',
      };
    }

    // Si la mémoire est très ancienne et peu accédée, la nouvelle info peut être plus fiable
    const ageInDays = (Date.now() - contradictingMemory.createdAt.getTime()) / (1000 * 60 * 60 * 24);
    if (ageInDays > 180 && contradictingMemory.accessCount < 5) {
      return {
        keepProposed: true,
        reason: 'La mémoire contradictoire est ancienne et peu utilisée',
        suggestedAction: 'update_memory',
      };
    }

    // Par défaut, garder la mémoire existante (conservateur)
    return {
      keepProposed: false,
      reason: 'Conservation de la mémoire existante par défaut',
      suggestedAction: 'reject_response',
    };
  }
}
