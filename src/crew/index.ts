/**
 * CrewAI Integration Module
 *
 * Ce module permet à Neo de déléguer des tâches complexes à des équipes d'agents.
 * Il utilise un subprocess Python pour exécuter CrewAI avec Ollama comme LLM par défaut.
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────┐
 * │                     TypeScript (Neo)                        │
 * │  ┌─────────────────────────────────────────────────────┐   │
 * │  │                  CrewManager                         │   │
 * │  │  - Gestion du subprocess Python                     │   │
 * │  │  - Communication IPC (JSON via stdin/stdout)        │   │
 * │  │  - Sélection automatique du LLM le plus économique  │   │
 * │  └──────────────────────┬──────────────────────────────┘   │
 * └─────────────────────────┼───────────────────────────────────┘
 *                           │ JSON IPC
 * ┌─────────────────────────▼───────────────────────────────────┐
 * │                     Python (CrewAI)                         │
 * │  ┌─────────────────────────────────────────────────────┐   │
 * │  │               crew_runner.py                         │   │
 * │  │  - Exécute les crews d'agents                       │   │
 * │  │  - Connexion Ollama/Claude/OpenAI                   │   │
 * │  │  - Retourne résultats + métriques                   │   │
 * │  └─────────────────────────────────────────────────────┘   │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Usage:
 * ```typescript
 * import { getCrewManager, PRESET_CREWS } from './crew';
 *
 * const crewManager = getCrewManager();
 * await crewManager.start();
 *
 * // Utiliser un preset
 * const crew = PRESET_CREWS.research.buildCrew({
 *   topic: 'AI trends 2025',
 *   depth: 'deep',
 * });
 *
 * const result = await crewManager.executeCrew(crew);
 * console.log(result.finalOutput);
 * ```
 */

// ===========================================================================
// TYPES
// ===========================================================================

export type {
  // Agent & Task types
  CrewAgentConfig,
  CrewTaskConfig,
  CrewConfig,
  CrewProcessType,

  // LLM types
  LLMConfig,

  // Execution types
  TaskResult,
  CrewExecutionResult,

  // IPC types
  CrewIPCRequest,
  CrewIPCResponse,

  // Tool types
  CrewToolConfig,

  // Preset types
  PresetCrew,
} from './types';

// Constants (valeurs, pas types)
export { LLM_PRESETS } from './types';

// ===========================================================================
// CREW MANAGER
// ===========================================================================

export { CrewManager, getCrewManager } from './crew-manager';

// ===========================================================================
// PRESETS
// ===========================================================================

export {
  PRESET_CREWS,
  getPresetCrew,
  listPresetCrews,

  // Individual presets
  researchCrew,
  codeReviewCrew,
  contentCrew,
  dataAnalysisCrew,
} from './presets';

// ===========================================================================
// HELPER: Quick execution
// ===========================================================================

import { getCrewManager } from './crew-manager';
import { getPresetCrew } from './presets';
import type { CrewConfig, CrewExecutionResult } from './types';

/**
 * Exécute rapidement un preset de crew
 *
 * @example
 * const result = await executePresetCrew('research', {
 *   topic: 'AI in healthcare',
 *   depth: 'deep',
 * });
 */
export async function executePresetCrew(
  presetId: string,
  inputs: Record<string, unknown>
): Promise<CrewExecutionResult> {
  const preset = getPresetCrew(presetId);
  if (!preset) {
    throw new Error(`Unknown preset: ${presetId}`);
  }

  // Validate required inputs
  for (const required of preset.requiredInputs) {
    if (!(required in inputs)) {
      throw new Error(`Missing required input: ${required}`);
    }
  }

  const crewManager = getCrewManager();

  // Start if not running
  if (!crewManager.isRunning()) {
    await crewManager.start();
  }

  const crew = preset.buildCrew(inputs);
  return crewManager.executeCrew(crew);
}

/**
 * Exécute un crew custom
 *
 * @example
 * const result = await executeCrew({
 *   name: 'My Crew',
 *   agents: [...],
 *   tasks: [...],
 *   process: 'sequential',
 * });
 */
export async function executeCrew(config: CrewConfig): Promise<CrewExecutionResult> {
  const crewManager = getCrewManager();

  // Start if not running
  if (!crewManager.isRunning()) {
    await crewManager.start();
  }

  return crewManager.executeCrew(config);
}
