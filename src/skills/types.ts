/**
 * Types pour le système de Skills dynamiques
 *
 * Les skills sont des actions réutilisables que Neo peut:
 * - Créer automatiquement en détectant des patterns
 * - Exécuter dans un environnement sandboxé
 * - Améliorer au fil du temps via feedback
 */

// ===========================================================================
// CAPABILITIES - Ce qu'un skill peut faire
// ===========================================================================

/**
 * Types de capabilities qu'un skill peut demander
 */
export type CapabilityType =
  | 'web_fetch'      // Requêtes HTTP (domaines limités)
  | 'memory_read'    // Lire la mémoire de Neo
  | 'memory_write'   // Écrire dans la mémoire
  | 'browser'        // Contrôle navigateur (Playwright)
  | 'file_read'      // Lire fichiers (paths limités)
  | 'llm_call';      // Appeler un LLM

/**
 * Configuration d'une capability
 */
export interface CapabilityConfig {
  type: CapabilityType;
  options?: {
    // Pour web_fetch
    allowedDomains?: string[];
    maxRequestsPerMinute?: number;

    // Pour browser
    allowedUrls?: string[];
    timeout?: number;

    // Pour file_read
    allowedPaths?: string[];
    maxFileSizeKB?: number;

    // Pour memory_read/write
    allowedTypes?: string[];
    maxItems?: number;

    // Pour llm_call
    maxTokens?: number;
    allowedModels?: string[];
  };
}

/**
 * Ensemble de capabilities accordées à un skill
 */
export interface CapabilityGrant {
  skillId: string;
  capabilities: CapabilityConfig[];
  grantedAt: Date;
  expiresAt?: Date;
  revokedAt?: Date;
}

// ===========================================================================
// SKILL DEFINITION - Structure d'un skill
// ===========================================================================

/**
 * Schéma JSON pour validation des inputs/outputs
 */
export interface JSONSchema {
  type: 'object' | 'array' | 'string' | 'number' | 'boolean';
  properties?: Record<string, JSONSchema>;
  items?: JSONSchema;
  required?: string[];
  description?: string;
}

/**
 * Définition complète d'un skill
 */
export interface SkillDefinition {
  id: string;
  name: string;
  version: string;
  description: string;

  // Activation
  triggers: string[];                    // Mots-clés pour activer le skill
  requiredCapabilities: CapabilityType[]; // Capabilities nécessaires

  // Code
  code: string;                          // Code JavaScript/TypeScript du skill
  inputSchema?: JSONSchema;              // Validation des inputs
  outputSchema?: JSONSchema;             // Validation des outputs

  // Métadonnées
  createdAt: Date;
  updatedAt: Date;
  createdBy: 'user' | 'neo' | 'system';

  // Statistiques
  successRate: number;                   // 0-1
  usageCount: number;
  lastUsedAt?: Date;
  lastError?: string;

  // État
  isEnabled: boolean;
  isBuiltin: boolean;                    // Skill fourni par défaut
}

/**
 * Version simplifiée pour création
 */
export interface SkillCreateInput {
  name: string;
  description: string;
  triggers: string[];
  requiredCapabilities: CapabilityType[];
  code: string;
  inputSchema?: JSONSchema;
  outputSchema?: JSONSchema;
  createdBy?: 'user' | 'neo' | 'system';
}

/**
 * Mise à jour partielle d'un skill
 */
export interface SkillUpdateInput {
  name?: string;
  description?: string;
  triggers?: string[];
  code?: string;
  isEnabled?: boolean;
}

// ===========================================================================
// EXECUTION - Exécution d'un skill
// ===========================================================================

/**
 * Contexte d'exécution d'un skill
 */
export interface SkillExecutionContext {
  skillId: string;
  executionId: string;
  capabilities: CapabilityGrant;
  timeout: number;           // ms
  memoryLimit: number;       // bytes
  startedAt: Date;
}

/**
 * Input pour exécuter un skill
 */
export interface SkillExecutionInput {
  skillId: string;
  input: Record<string, unknown>;
  options?: {
    timeout?: number;
    priority?: 'low' | 'normal' | 'high';
  };
}

/**
 * Résultat d'exécution d'un skill
 */
export interface SkillExecutionResult {
  success: boolean;
  executionId: string;
  skillId: string;
  skillName: string;

  // Résultat
  output?: unknown;
  error?: {
    message: string;
    code: string;
    stack?: string;
  };

  // Métriques
  executionTimeMs: number;
  tokensUsed?: number;
  capabilitiesUsed: CapabilityType[];

  // Pour learning
  shouldLearn: boolean;
  learningNotes?: string;
}

// ===========================================================================
// WORKER COMMUNICATION
// ===========================================================================

/**
 * Message envoyé au worker
 */
export interface WorkerMessage {
  type: 'EXECUTE_SKILL' | 'CANCEL' | 'PING';
  executionId: string;
  payload?: {
    code: string;
    input: Record<string, unknown>;
    capabilities: SerializedCapabilities;
    timeout: number;
  };
}

/**
 * Message reçu du worker
 */
export interface WorkerResponse {
  type: 'RESULT' | 'ERROR' | 'LOG' | 'PONG' | 'CAPABILITY_REQUEST';
  executionId: string;
  payload: {
    output?: unknown;
    error?: {
      message: string;
      code: string;
      stack?: string;
    };
    log?: {
      level: 'info' | 'warn' | 'error';
      message: string;
    };
    capabilityRequest?: {
      type: CapabilityType;
      method: string;
      args: unknown[];
    };
  };
}

/**
 * Capabilities sérialisées pour le worker
 * (le worker ne peut pas accéder aux objets du main thread)
 */
export interface SerializedCapabilities {
  allowed: CapabilityType[];
  config: Record<CapabilityType, CapabilityConfig['options']>;
}

// ===========================================================================
// SKILL MANAGER
// ===========================================================================

/**
 * Options de recherche de skills
 */
export interface SkillSearchOptions {
  query?: string;
  capabilities?: CapabilityType[];
  createdBy?: 'user' | 'neo' | 'system';
  isEnabled?: boolean;
  limit?: number;
  offset?: number;
}

/**
 * Statistiques du système de skills
 */
export interface SkillStats {
  totalSkills: number;
  enabledSkills: number;
  builtinSkills: number;
  userCreatedSkills: number;
  neoCreatedSkills: number;
  totalExecutions: number;
  averageSuccessRate: number;
  mostUsedSkills: Array<{
    id: string;
    name: string;
    usageCount: number;
  }>;
}

// ===========================================================================
// EVENTS
// ===========================================================================

/**
 * Événements émis par le système de skills
 */
export type SkillEvent =
  | { type: 'skill_created'; skill: SkillDefinition }
  | { type: 'skill_updated'; skill: SkillDefinition; changes: string[] }
  | { type: 'skill_deleted'; skillId: string }
  | { type: 'skill_enabled'; skillId: string }
  | { type: 'skill_disabled'; skillId: string }
  | { type: 'execution_started'; executionId: string; skillId: string }
  | { type: 'execution_completed'; result: SkillExecutionResult }
  | { type: 'execution_failed'; executionId: string; error: string }
  | { type: 'capability_violation'; skillId: string; capability: CapabilityType; details: string };

// ===========================================================================
// CONSTANTS
// ===========================================================================

export const SKILL_DEFAULTS = {
  TIMEOUT_MS: 30000,           // 30 secondes
  MEMORY_LIMIT_MB: 256,        // 256 MB
  MAX_WORKERS: 5,              // Pool de 5 workers max
  MIN_WORKERS: 0,              // Lazy loading: workers créés à la demande
  WORKER_IDLE_TIMEOUT_MS: 60000, // Arrêter worker inactif après 1 min
} as const;

export const CAPABILITY_LIMITS = {
  web_fetch: {
    maxRequestsPerMinute: 30,
    timeoutMs: 10000,
  },
  browser: {
    maxPagesPerSkill: 3,
    timeoutMs: 30000,
  },
  memory_read: {
    maxItemsPerQuery: 100,
  },
  memory_write: {
    maxItemsPerExecution: 10,
  },
  llm_call: {
    maxTokensPerCall: 4096,
    maxCallsPerExecution: 5,
  },
  file_read: {
    maxFileSizeKB: 1024,
  },
} as const;
