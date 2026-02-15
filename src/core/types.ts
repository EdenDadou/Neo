/**
 * Types centraux du système multi-agents
 */

// ============================================================================
// MESSAGES INTER-AGENTS
// ============================================================================

export type AgentRole = 'vox' | 'memory' | 'brain' | 'worker';

export interface AgentMessage {
  id: string;
  from: AgentRole;
  to: AgentRole | 'broadcast';
  type: MessageType;
  payload: unknown;
  timestamp: Date;
  correlationId?: string; // Pour tracer les conversations
}

export type MessageType =
  | 'user_input'           // Vox → Brain : entrée utilisateur enrichie
  | 'context_request'      // Brain → Memory : demande de contexte
  | 'context_response'     // Memory → Brain : contexte enrichi
  | 'context_report_request' // Vox → Memory : demande de rapport proactif
  | 'context_report'       // Memory → Vox : rapport de contexte proactif
  | 'store_conversation'   // Vox → Memory : stocker un échange complet
  | 'fact_check_request'   // Brain → Memory : vérifier cohérence d'une réponse
  | 'fact_check_response'  // Memory → Brain : résultat de la vérification
  | 'task_spawn'           // Brain → Worker : créer un agent spécialisé
  | 'task_result'          // Worker → Brain : résultat d'une tâche
  | 'memory_store'         // Any → Memory : stocker une information
  | 'memory_query'         // Any → Memory : rechercher une information
  | 'response_ready'       // Brain → Vox : réponse prête pour l'utilisateur
  | 'learning_update'      // Memory → Brain : mise à jour d'apprentissage
  | 'skill_detected'       // Memory → Brain : nouvelle compétence détectée
  | 'error';               // Any : erreur

// ============================================================================
// MÉMOIRE
// ============================================================================

export interface MemoryEntry {
  id: string;
  type: MemoryType;
  content: string;
  embedding?: number[];
  metadata: MemoryMetadata;
  createdAt: Date;
  lastAccessedAt: Date;
  accessCount: number;
  importance: number; // 0-1, détermine la rétention long terme
}

export type MemoryType =
  | 'fact'              // Fait établi
  | 'preference'        // Préférence utilisateur
  | 'skill'             // Compétence apprise
  | 'conversation'      // Extrait de conversation
  | 'task_result'       // Résultat d'une tâche passée
  | 'correction'        // Correction utilisateur (pour learning loop)
  | 'system';           // Information système

export interface MemoryMetadata {
  source: string;
  confidence: number;   // 0-1
  tags: string[];
  relatedIds: string[];
  expiresAt?: Date;     // Pour mémoire court terme
}

// ============================================================================
// TÂCHES
// ============================================================================

export interface Task {
  id: string;
  title: string;
  description: string;
  status: TaskStatus;
  priority: number;     // 1-10
  createdAt: Date;
  updatedAt: Date;
  attempts: TaskAttempt[];
  dependencies: string[];
  requiredSkills: string[];
  assignedAgent?: string;
}

export type TaskStatus = 'pending' | 'in_progress' | 'blocked' | 'completed' | 'failed';

export interface TaskAttempt {
  attemptNumber: number;
  startedAt: Date;
  endedAt?: Date;
  result?: unknown;
  error?: string;
  learnings: string[];  // Ce qu'on a appris de cet essai
}

// ============================================================================
// SKILLS
// ============================================================================

export interface Skill {
  id: string;
  name: string;
  description: string;
  triggers: string[];   // Mots-clés qui activent ce skill
  handler: string;      // Chemin vers le module
  dependencies: string[];
  learnedAt: Date;
  successRate: number;  // 0-1
  usageCount: number;
}

// ============================================================================
// CONTEXTE (injecté par Memory vers Brain)
// ============================================================================

export interface EnrichedContext {
  userInput: string;
  relevantMemories: MemoryEntry[];
  activeTasks: Task[];
  availableSkills: Skill[];
  recentConversation: ConversationTurn[];
  userProfile: UserProfile;
  suggestedApproach?: string; // Basé sur les essais précédents
}

export interface ConversationTurn {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface UserProfile {
  id: string;
  preferences: Record<string, unknown>;
  communicationStyle: string;
  expertise: string[];
  timezone?: string;
}

// ============================================================================
// LEARNING LOOP
// ============================================================================

export interface LearningEntry {
  id: string;
  type: LearningType;
  context: string;
  originalResponse: string;
  correction?: string;
  feedback: string;
  createdAt: Date;
  applied: boolean;
}

export type LearningType =
  | 'user_correction'   // L'utilisateur a corrigé une erreur
  | 'task_failure'      // Une tâche a échoué
  | 'skill_improvement' // Amélioration d'un skill
  | 'pattern_detected'; // Pattern détecté dans les interactions

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface AgentConfig {
  name: string;
  role: AgentRole;
  model: string;
  maxTokens: number;
  temperature: number;
  systemPrompt: string;
}

export interface SystemConfig {
  agents: {
    vox: AgentConfig;
    memory: AgentConfig;
    brain: AgentConfig;
  };
  memory: {
    vectorDbUrl: string;
    postgresUrl: string;
    redisUrl: string;
    embeddingModel: string;
  };
  retention: {
    shortTermDays: number;
    mediumTermDays: number;
    longTermYears: number;
  };
}
