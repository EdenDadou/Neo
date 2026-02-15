/**
 * Skills Module - Système de Skills Dynamiques Auto-Évolutif
 *
 * Ce module permet à Neo de:
 * - Créer des skills exécutables dynamiquement
 * - Les exécuter dans un environnement sandboxé (Worker Threads)
 * - Apprendre et améliorer ses skills au fil du temps
 * - Garantir la sécurité via le système de capabilities (OCAP)
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────┐
 * │                     SkillManager                            │
 * │  - CRUD skills                                              │
 * │  - Recherche par triggers                                   │
 * │  - Gestion du cycle de vie                                  │
 * └─────────────────────┬───────────────────────────────────────┘
 *                       │
 * ┌─────────────────────▼───────────────────────────────────────┐
 * │                  WorkerExecutor                             │
 * │  - Pool de Worker Threads                                   │
 * │  - Timeout et memory limits                                 │
 * │  - Communication via messages                               │
 * └─────────────────────┬───────────────────────────────────────┘
 *                       │
 * ┌─────────────────────▼───────────────────────────────────────┐
 * │               CapabilityManager                             │
 * │  - Accorde/révoque capabilities                             │
 * │  - Rate limiting                                            │
 * │  - Audit trail                                              │
 * └─────────────────────────────────────────────────────────────┘
 */

// ===========================================================================
// TYPES
// ===========================================================================

export type {
  // Types de capabilities
  CapabilityType,
  CapabilityConfig,
  CapabilityGrant,

  // Types de skills
  SkillDefinition,
  SkillCreateInput,
  SkillUpdateInput,
  JSONSchema,

  // Types d'exécution
  SkillExecutionContext,
  SkillExecutionInput,
  SkillExecutionResult,

  // Types de communication worker
  WorkerMessage,
  WorkerResponse,
  SerializedCapabilities,

  // Options et stats
  SkillSearchOptions,
  SkillStats,

  // Events
  SkillEvent,
} from './types';

// Constants (valeurs, pas types)
export {
  SKILL_DEFAULTS,
  CAPABILITY_LIMITS,
} from './types';

// ===========================================================================
// SKILL MANAGER
// ===========================================================================

export { SkillManager, getSkillManager } from './skill-manager';

// ===========================================================================
// CAPABILITIES
// ===========================================================================

export {
  CapabilityManager,
  getCapabilityManager,
} from './capabilities';

// Interfaces pour connexion avec le Core
export type {
  MemoryInterface,
  BrowserInterface,
  LLMInterface,
} from './capabilities';

// ===========================================================================
// EXECUTOR
// ===========================================================================

export { WorkerExecutor } from './executor/worker-executor';

// ===========================================================================
// HELPER: Créer et configurer le système de skills complet
// ===========================================================================

import { SkillManager, getSkillManager } from './skill-manager';
import type { SkillPersistence } from './skill-manager';
import { CapabilityManager, getCapabilityManager } from './capabilities';
import type { MemoryInterface, LLMInterface } from './capabilities';
import type { PersistenceLayer } from '../core/memory/persistence';
import type { SkillDefinition } from './types';

// Re-export SkillPersistence interface
export type { SkillPersistence };

/**
 * Configuration pour initialiser le système de skills
 */
export interface SkillSystemConfig {
  persistence: PersistenceLayer;
  memory?: MemoryInterface;
  llm?: LLMInterface;
}

/**
 * Adaptateur pour connecter PersistenceLayer au système de skills
 * Convertit les méthodes de PersistenceLayer vers l'interface SkillPersistence
 */
function createPersistenceAdapter(persistence: PersistenceLayer): SkillPersistence {
  return {
    async loadSkills(): Promise<SkillDefinition[]> {
      const rows = persistence.getSkillsV2({ enabledOnly: false });
      return rows.map(row => ({
        id: row.id,
        name: row.name,
        version: row.version,
        description: row.description,
        triggers: JSON.parse(row.triggers),
        requiredCapabilities: JSON.parse(row.required_capabilities),
        code: row.code,
        inputSchema: row.input_schema ? JSON.parse(row.input_schema) : undefined,
        outputSchema: row.output_schema ? JSON.parse(row.output_schema) : undefined,
        createdAt: new Date(row.created_at),
        updatedAt: new Date(row.updated_at),
        createdBy: row.created_by,
        successRate: row.success_rate,
        usageCount: row.usage_count,
        lastUsedAt: row.last_used_at ? new Date(row.last_used_at) : undefined,
        lastError: row.last_error || undefined,
        isEnabled: row.is_enabled,
        isBuiltin: row.is_builtin,
      }));
    },

    async saveSkills(skills: SkillDefinition[]): Promise<void> {
      for (const skill of skills) {
        persistence.saveSkillV2({
          id: skill.id,
          name: skill.name,
          version: skill.version,
          description: skill.description,
          triggers: JSON.stringify(skill.triggers),
          required_capabilities: JSON.stringify(skill.requiredCapabilities),
          code: skill.code,
          input_schema: skill.inputSchema ? JSON.stringify(skill.inputSchema) : null,
          output_schema: skill.outputSchema ? JSON.stringify(skill.outputSchema) : null,
          created_at: skill.createdAt.toISOString(),
          updated_at: skill.updatedAt.toISOString(),
          created_by: skill.createdBy,
          success_rate: skill.successRate,
          usage_count: skill.usageCount,
          is_enabled: skill.isEnabled,
          is_builtin: skill.isBuiltin,
          last_used_at: skill.lastUsedAt?.toISOString() || null,
          last_error: skill.lastError || null,
        });
      }
    },

    async saveSkill(skill: SkillDefinition): Promise<void> {
      persistence.saveSkillV2({
        id: skill.id,
        name: skill.name,
        version: skill.version,
        description: skill.description,
        triggers: JSON.stringify(skill.triggers),
        required_capabilities: JSON.stringify(skill.requiredCapabilities),
        code: skill.code,
        input_schema: skill.inputSchema ? JSON.stringify(skill.inputSchema) : null,
        output_schema: skill.outputSchema ? JSON.stringify(skill.outputSchema) : null,
        created_at: skill.createdAt.toISOString(),
        updated_at: skill.updatedAt.toISOString(),
        created_by: skill.createdBy,
        success_rate: skill.successRate,
        usage_count: skill.usageCount,
        is_enabled: skill.isEnabled,
        is_builtin: skill.isBuiltin,
        last_used_at: skill.lastUsedAt?.toISOString() || null,
        last_error: skill.lastError || null,
      });
    },

    async deleteSkill(id: string): Promise<void> {
      persistence.deleteSkillV2(id);
    },
  };
}

/**
 * Initialiser le système de skills complet
 *
 * Usage:
 * ```typescript
 * const { skillManager, capabilityManager } = initializeSkillSystem({
 *   persistence: memoryAgent.persistence,
 *   memory: memoryInterface,
 *   llm: llmInterface,
 * });
 *
 * await skillManager.start();
 * ```
 */
export function initializeSkillSystem(config: SkillSystemConfig): {
  skillManager: SkillManager;
  capabilityManager: CapabilityManager;
} {
  // Initialiser le capability manager
  const capabilityManager = getCapabilityManager();

  // Configurer les interfaces
  capabilityManager.configure({
    memory: config.memory,
    llm: config.llm,
  });

  // Initialiser le skill manager
  const skillManager = getSkillManager();

  // Configurer la persistence via l'adaptateur
  if (config.persistence) {
    const adapter = createPersistenceAdapter(config.persistence);
    skillManager.setPersistence(adapter);
  }

  console.log('[Skills] ✅ Système de skills initialisé');

  return {
    skillManager,
    capabilityManager,
  };
}

// ===========================================================================
// HELPER: Créer un skill depuis un pattern détecté
// ===========================================================================

import type { SkillCreateInput, CapabilityType } from './types';

/**
 * Template pour créer un skill de web scraping
 */
export function createWebScraperSkillTemplate(
  name: string,
  description: string,
  targetUrl: string,
  selectors: Record<string, string>
): SkillCreateInput {
  const selectorsCode = Object.entries(selectors)
    .map(([key, selector]) => `    "${key}": "${selector}"`)
    .join(',\n');

  return {
    name,
    description,
    triggers: [name.toLowerCase(), 'scraper', 'extraire', targetUrl],
    requiredCapabilities: ['web_fetch'] as CapabilityType[],
    code: `
// Skill: ${name}
// Auto-généré pour scraper: ${targetUrl}

const selectors = {
${selectorsCode}
};

const response = await webFetch(input.url || "${targetUrl}");
const html = response.text;

// Extraction basique (à améliorer avec un parser HTML côté main thread)
const results = {};

for (const [key, selector] of Object.entries(selectors)) {
  // Extraction simplifiée - le vrai parsing se fait côté capability
  results[key] = "Extraction via selector: " + selector;
}

return {
  url: input.url || "${targetUrl}",
  extracted: results,
  timestamp: new Date().toISOString(),
};
`,
    createdBy: 'neo',
  };
}

/**
 * Template pour créer un skill de recherche mémoire
 */
export function createMemorySearchSkillTemplate(
  name: string,
  description: string,
  defaultQuery: string,
  memoryTypes?: string[]
): SkillCreateInput {
  return {
    name,
    description,
    triggers: [name.toLowerCase(), 'chercher', 'trouver', 'rechercher'],
    requiredCapabilities: ['memory_read'] as CapabilityType[],
    code: `
// Skill: ${name}
// Recherche dans la mémoire de Neo

const query = input.query || "${defaultQuery}";
const limit = input.limit || 10;

const results = await memory.search(query, limit);

// Filtrer par type si spécifié
const filteredResults = results${memoryTypes ? `.filter(r => ${JSON.stringify(memoryTypes)}.includes(r.type))` : ''};

return {
  query,
  count: filteredResults.length,
  results: filteredResults,
  timestamp: new Date().toISOString(),
};
`,
    createdBy: 'neo',
  };
}

/**
 * Template pour créer un skill d'appel LLM
 */
export function createLLMSkillTemplate(
  name: string,
  description: string,
  systemPrompt: string
): SkillCreateInput {
  return {
    name,
    description,
    triggers: [name.toLowerCase(), 'analyser', 'générer', 'reformuler'],
    requiredCapabilities: ['llm_call'] as CapabilityType[],
    code: `
// Skill: ${name}
// Utilise un LLM pour traiter l'input

const systemPrompt = \`${systemPrompt}\`;

const userPrompt = typeof input === 'string' ? input : JSON.stringify(input);

const response = await llm.complete(
  \`\${systemPrompt}\\n\\nInput: \${userPrompt}\`,
  { maxTokens: 1024 }
);

return {
  input: userPrompt,
  output: response,
  timestamp: new Date().toISOString(),
};
`,
    createdBy: 'neo',
  };
}
