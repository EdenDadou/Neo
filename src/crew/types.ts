/**
 * Types pour l'intégration CrewAI
 *
 * Architecture Hybrid: Python CrewAI subprocess communique via JSON IPC
 */

// ===========================================================================
// AGENT DEFINITIONS
// ===========================================================================

/**
 * Configuration d'un agent dans le Crew
 */
export interface CrewAgentConfig {
  name: string;
  role: string;
  goal: string;
  backstory: string;

  // Configuration LLM
  llm: LLMConfig;

  // Outils disponibles pour cet agent
  tools?: CrewToolConfig[];

  // Options comportementales
  verbose?: boolean;
  allowDelegation?: boolean;
  maxIterations?: number;
  memoryEnabled?: boolean;
}

/**
 * Configuration LLM pour un agent
 * Priorise les modèles économiques (Ollama) par défaut
 */
export interface LLMConfig {
  // Provider: ollama (local, gratuit), anthropic, openai
  provider: 'ollama' | 'anthropic' | 'openai';

  // Modèle à utiliser
  model: string;

  // URL de base (pour Ollama)
  baseUrl?: string;

  // Paramètres
  temperature?: number;
  maxTokens?: number;

  // API key (pour providers cloud)
  apiKey?: string;
}

/**
 * Presets LLM pour faciliter la configuration
 */
export const LLM_PRESETS = {
  // Modèles locaux Ollama (GRATUITS)
  OLLAMA_FAST: {
    provider: 'ollama' as const,
    model: 'llama3.2:3b',
    baseUrl: 'http://localhost:11434',
    temperature: 0.7,
  },
  OLLAMA_BALANCED: {
    provider: 'ollama' as const,
    model: 'llama3.2:8b',
    baseUrl: 'http://localhost:11434',
    temperature: 0.7,
  },
  OLLAMA_SMART: {
    provider: 'ollama' as const,
    model: 'mixtral:8x7b',
    baseUrl: 'http://localhost:11434',
    temperature: 0.5,
  },
  OLLAMA_CODE: {
    provider: 'ollama' as const,
    model: 'codellama:13b',
    baseUrl: 'http://localhost:11434',
    temperature: 0.3,
  },

  // Modèles cloud (PAYANTS - utiliser en dernier recours)
  CLAUDE_HAIKU: {
    provider: 'anthropic' as const,
    model: 'claude-3-haiku-20240307',
    temperature: 0.7,
  },
  CLAUDE_SONNET: {
    provider: 'anthropic' as const,
    model: 'claude-3-5-sonnet-20241022',
    temperature: 0.7,
  },
} as const;

// ===========================================================================
// TASK DEFINITIONS
// ===========================================================================

/**
 * Configuration d'une tâche pour le Crew
 */
export interface CrewTaskConfig {
  id: string;
  description: string;
  expectedOutput: string;

  // Agent assigné (par nom)
  agent: string;

  // Contexte de tâches précédentes (par ID)
  context?: string[];

  // Données d'entrée
  input?: Record<string, unknown>;

  // Options
  asyncExecution?: boolean;
  humanInput?: boolean;
}

// ===========================================================================
// CREW DEFINITIONS
// ===========================================================================

/**
 * Type de processus d'exécution
 */
export type CrewProcessType = 'sequential' | 'hierarchical';

/**
 * Configuration complète d'un Crew
 */
export interface CrewConfig {
  name: string;
  description?: string;

  // Agents du crew
  agents: CrewAgentConfig[];

  // Tâches à exécuter
  tasks: CrewTaskConfig[];

  // Mode d'exécution
  process: CrewProcessType;

  // Options
  verbose?: boolean;
  memory?: boolean;
  maxRpm?: number;  // Rate limit

  // Pour process hierarchical
  managerLlm?: LLMConfig;
}

// ===========================================================================
// EXECUTION
// ===========================================================================

/**
 * Résultat d'exécution d'une tâche
 */
export interface TaskResult {
  taskId: string;
  success: boolean;
  output?: string;
  error?: string;
  tokensUsed?: number;
  executionTimeMs?: number;
}

/**
 * Résultat complet d'exécution d'un Crew
 */
export interface CrewExecutionResult {
  success: boolean;
  crewName: string;

  // Résultat final
  finalOutput?: string;

  // Résultats par tâche
  taskResults: TaskResult[];

  // Métriques
  totalTokensUsed: number;
  totalExecutionTimeMs: number;

  // Erreur globale si échec
  error?: string;
}

// ===========================================================================
// IPC COMMUNICATION
// ===========================================================================

/**
 * Message envoyé au subprocess Python
 */
export interface CrewIPCRequest {
  type: 'EXECUTE_CREW' | 'CHECK_HEALTH' | 'LIST_MODELS' | 'STOP';
  requestId: string;
  payload?: {
    crew?: CrewConfig;
    timeout?: number;
  };
}

/**
 * Réponse du subprocess Python
 */
export interface CrewIPCResponse {
  type: 'RESULT' | 'ERROR' | 'LOG' | 'PROGRESS' | 'HEALTH';
  requestId: string;
  payload: {
    result?: CrewExecutionResult;
    error?: string;
    log?: {
      level: 'info' | 'warn' | 'error' | 'debug';
      message: string;
      agent?: string;
      task?: string;
    };
    progress?: {
      currentTask: string;
      completedTasks: number;
      totalTasks: number;
    };
    health?: {
      status: 'ok' | 'error';
      ollamaConnected: boolean;
      availableModels: string[];
    };
  };
}

// ===========================================================================
// TOOL DEFINITIONS
// ===========================================================================

/**
 * Configuration d'un outil pour les agents
 */
export interface CrewToolConfig {
  name: string;
  description: string;

  // Type d'outil
  type: 'web_search' | 'web_scrape' | 'file_read' | 'code_execute' | 'custom';

  // Configuration spécifique
  config?: Record<string, unknown>;
}

// ===========================================================================
// PRESET CREWS
// ===========================================================================

/**
 * Crews prédéfinis pour des tâches communes
 */
export interface PresetCrew {
  id: string;
  name: string;
  description: string;
  requiredInputs: string[];
  buildCrew: (inputs: Record<string, unknown>) => CrewConfig;
}
