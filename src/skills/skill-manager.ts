/**
 * SkillManager - Gestionnaire central des skills dynamiques
 *
 * Responsabilités:
 * - CRUD des skills (create, read, update, delete)
 * - Recherche de skills par triggers/query
 * - Délégation de l'exécution au WorkerExecutor
 * - Tracking des statistiques et learning
 */

import { EventEmitter } from 'events';
import { randomUUID } from 'crypto';
import {
  SkillDefinition,
  SkillCreateInput,
  SkillUpdateInput,
  SkillSearchOptions,
  SkillExecutionInput,
  SkillExecutionResult,
  SkillStats,
  SkillEvent,
  CapabilityType,
  SKILL_DEFAULTS,
} from './types';
import { WorkerExecutor } from './executor/worker-executor';
import { CapabilityManager } from './capabilities';

// ===========================================================================
// SKILL MANAGER
// ===========================================================================

export class SkillManager extends EventEmitter {
  private skills: Map<string, SkillDefinition> = new Map();
  private executor: WorkerExecutor;
  private capabilityManager: CapabilityManager;
  private persistence: SkillPersistence | null = null;

  constructor(options?: { persistence?: SkillPersistence }) {
    super();
    this.executor = new WorkerExecutor(this);
    this.capabilityManager = new CapabilityManager();

    if (options?.persistence) {
      this.persistence = options.persistence;
    }

    console.log('[SkillManager] Initialisé');
  }

  /**
   * Configurer la persistence (peut être appelé après construction)
   */
  setPersistence(persistence: SkillPersistence): void {
    this.persistence = persistence;
    console.log('[SkillManager] Persistence configurée');
  }

  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================

  /**
   * Démarrer le manager et charger les skills
   */
  async start(): Promise<void> {
    // Charger les skills depuis la persistence
    if (this.persistence) {
      const savedSkills = await this.persistence.loadSkills();
      for (const skill of savedSkills) {
        this.skills.set(skill.id, skill);
      }
      console.log(`[SkillManager] ${savedSkills.length} skills chargés`);
    }

    // Charger les skills built-in
    await this.loadBuiltinSkills();

    // Démarrer le pool de workers
    await this.executor.start();

    console.log('[SkillManager] Démarré');
  }

  /**
   * Arrêter le manager
   */
  async stop(): Promise<void> {
    // Sauvegarder les skills
    if (this.persistence) {
      const skills = Array.from(this.skills.values()).filter(s => !s.isBuiltin);
      await this.persistence.saveSkills(skills);
    }

    // Arrêter le pool de workers
    await this.executor.stop();

    console.log('[SkillManager] Arrêté');
  }

  // ===========================================================================
  // CRUD
  // ===========================================================================

  /**
   * Créer un nouveau skill
   */
  async createSkill(input: SkillCreateInput): Promise<SkillDefinition> {
    // Valider le code
    const validation = this.validateSkillCode(input.code);
    if (!validation.valid) {
      throw new Error(`Code invalide: ${validation.error}`);
    }

    // Vérifier que le nom est unique
    const existing = Array.from(this.skills.values()).find(
      s => s.name.toLowerCase() === input.name.toLowerCase()
    );
    if (existing) {
      throw new Error(`Un skill avec ce nom existe déjà: ${input.name}`);
    }

    const now = new Date();
    const skill: SkillDefinition = {
      id: randomUUID(),
      name: input.name,
      version: '1.0.0',
      description: input.description,
      triggers: input.triggers,
      requiredCapabilities: input.requiredCapabilities,
      code: input.code,
      inputSchema: input.inputSchema,
      outputSchema: input.outputSchema,
      createdAt: now,
      updatedAt: now,
      createdBy: input.createdBy || 'user',
      successRate: 1.0,
      usageCount: 0,
      isEnabled: true,
      isBuiltin: false,
    };

    this.skills.set(skill.id, skill);

    // Persister
    if (this.persistence) {
      await this.persistence.saveSkill(skill);
    }

    // Émettre événement
    this.emitEvent({ type: 'skill_created', skill });

    console.log(`[SkillManager] Skill créé: ${skill.name} (${skill.id})`);
    return skill;
  }

  /**
   * Mettre à jour un skill
   */
  async updateSkill(id: string, input: SkillUpdateInput): Promise<SkillDefinition> {
    const skill = this.skills.get(id);
    if (!skill) {
      throw new Error(`Skill non trouvé: ${id}`);
    }

    if (skill.isBuiltin) {
      throw new Error('Les skills built-in ne peuvent pas être modifiés');
    }

    const changes: string[] = [];

    if (input.name !== undefined && input.name !== skill.name) {
      skill.name = input.name;
      changes.push('name');
    }

    if (input.description !== undefined && input.description !== skill.description) {
      skill.description = input.description;
      changes.push('description');
    }

    if (input.triggers !== undefined) {
      skill.triggers = input.triggers;
      changes.push('triggers');
    }

    if (input.code !== undefined && input.code !== skill.code) {
      const validation = this.validateSkillCode(input.code);
      if (!validation.valid) {
        throw new Error(`Code invalide: ${validation.error}`);
      }
      skill.code = input.code;
      skill.version = this.incrementVersion(skill.version);
      changes.push('code');
    }

    if (input.isEnabled !== undefined && input.isEnabled !== skill.isEnabled) {
      skill.isEnabled = input.isEnabled;
      changes.push('isEnabled');
    }

    skill.updatedAt = new Date();

    // Persister
    if (this.persistence) {
      await this.persistence.saveSkill(skill);
    }

    // Émettre événement
    this.emitEvent({ type: 'skill_updated', skill, changes });

    console.log(`[SkillManager] Skill mis à jour: ${skill.name} (${changes.join(', ')})`);
    return skill;
  }

  /**
   * Supprimer un skill
   */
  async deleteSkill(id: string): Promise<void> {
    const skill = this.skills.get(id);
    if (!skill) {
      throw new Error(`Skill non trouvé: ${id}`);
    }

    if (skill.isBuiltin) {
      throw new Error('Les skills built-in ne peuvent pas être supprimés');
    }

    this.skills.delete(id);

    // Persister
    if (this.persistence) {
      await this.persistence.deleteSkill(id);
    }

    // Révoquer les capabilities
    this.capabilityManager.revokeAll(id);

    // Émettre événement
    this.emitEvent({ type: 'skill_deleted', skillId: id });

    console.log(`[SkillManager] Skill supprimé: ${skill.name}`);
  }

  /**
   * Obtenir un skill par ID
   */
  getSkill(id: string): SkillDefinition | undefined {
    return this.skills.get(id);
  }

  /**
   * Activer un skill
   */
  async enableSkill(id: string): Promise<void> {
    await this.updateSkill(id, { isEnabled: true });
    this.emitEvent({ type: 'skill_enabled', skillId: id });
  }

  /**
   * Désactiver un skill
   */
  async disableSkill(id: string): Promise<void> {
    await this.updateSkill(id, { isEnabled: false });
    this.emitEvent({ type: 'skill_disabled', skillId: id });
  }

  // ===========================================================================
  // SEARCH
  // ===========================================================================

  /**
   * Rechercher des skills
   */
  searchSkills(options: SkillSearchOptions = {}): SkillDefinition[] {
    let results = Array.from(this.skills.values());

    // Filtrer par enabled
    if (options.isEnabled !== undefined) {
      results = results.filter(s => s.isEnabled === options.isEnabled);
    }

    // Filtrer par créateur
    if (options.createdBy) {
      results = results.filter(s => s.createdBy === options.createdBy);
    }

    // Filtrer par capabilities requises
    if (options.capabilities && options.capabilities.length > 0) {
      results = results.filter(s =>
        options.capabilities!.some(cap => s.requiredCapabilities.includes(cap))
      );
    }

    // Recherche par query (nom, description, triggers)
    if (options.query) {
      const query = options.query.toLowerCase();
      results = results.filter(s =>
        s.name.toLowerCase().includes(query) ||
        s.description.toLowerCase().includes(query) ||
        s.triggers.some(t => t.toLowerCase().includes(query))
      );
    }

    // Trier par usage
    results.sort((a, b) => b.usageCount - a.usageCount);

    // Pagination
    if (options.offset) {
      results = results.slice(options.offset);
    }
    if (options.limit) {
      results = results.slice(0, options.limit);
    }

    return results;
  }

  /**
   * Trouver les skills pertinents pour une requête utilisateur
   */
  findRelevantSkills(userInput: string, limit = 5): SkillDefinition[] {
    const inputLower = userInput.toLowerCase();
    const words = inputLower.split(/\s+/);

    const scored = Array.from(this.skills.values())
      .filter(s => s.isEnabled)
      .map(skill => {
        let score = 0;

        // Score basé sur les triggers
        for (const trigger of skill.triggers) {
          const triggerLower = trigger.toLowerCase();
          if (inputLower.includes(triggerLower)) {
            score += 10; // Match exact
          } else if (words.some(w => triggerLower.includes(w))) {
            score += 5; // Match partiel
          }
        }

        // Bonus pour les skills souvent utilisés avec succès
        score += skill.successRate * skill.usageCount * 0.1;

        return { skill, score };
      })
      .filter(s => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    return scored.map(s => s.skill);
  }

  /**
   * Vérifier si un skill existe pour cette tâche
   */
  hasSkillFor(userInput: string): boolean {
    return this.findRelevantSkills(userInput, 1).length > 0;
  }

  // ===========================================================================
  // EXECUTION
  // ===========================================================================

  /**
   * Exécuter un skill
   */
  async executeSkill(input: SkillExecutionInput): Promise<SkillExecutionResult> {
    const skill = this.skills.get(input.skillId);
    if (!skill) {
      throw new Error(`Skill non trouvé: ${input.skillId}`);
    }

    if (!skill.isEnabled) {
      throw new Error(`Skill désactivé: ${skill.name}`);
    }

    const executionId = randomUUID();
    const timeout = input.options?.timeout || SKILL_DEFAULTS.TIMEOUT_MS;

    console.log(`[SkillManager] Exécution: ${skill.name} (${executionId})`);

    // Émettre début
    this.emitEvent({
      type: 'execution_started',
      executionId,
      skillId: skill.id,
    });

    // Accorder les capabilities
    const grant = this.capabilityManager.grant(skill.id, skill.requiredCapabilities);

    const startTime = Date.now();
    let result: SkillExecutionResult;

    try {
      // Exécuter dans le worker
      const output = await this.executor.execute({
        executionId,
        skill,
        input: input.input,
        capabilities: grant,
        timeout,
      });

      result = {
        success: true,
        executionId,
        skillId: skill.id,
        skillName: skill.name,
        output,
        executionTimeMs: Date.now() - startTime,
        capabilitiesUsed: skill.requiredCapabilities,
        shouldLearn: false,
      };

      // Mettre à jour les stats
      skill.usageCount++;
      skill.lastUsedAt = new Date();
      skill.successRate = this.updateSuccessRate(skill.successRate, true);
      skill.lastError = undefined;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);

      result = {
        success: false,
        executionId,
        skillId: skill.id,
        skillName: skill.name,
        error: {
          message: errorMessage,
          code: 'EXECUTION_ERROR',
          stack: error instanceof Error ? error.stack : undefined,
        },
        executionTimeMs: Date.now() - startTime,
        capabilitiesUsed: skill.requiredCapabilities,
        shouldLearn: true,
        learningNotes: `Échec d'exécution: ${errorMessage}`,
      };

      // Mettre à jour les stats
      skill.usageCount++;
      skill.lastUsedAt = new Date();
      skill.successRate = this.updateSuccessRate(skill.successRate, false);
      skill.lastError = errorMessage;

      this.emitEvent({
        type: 'execution_failed',
        executionId,
        error: errorMessage,
      });
    }

    // Révoquer les capabilities
    this.capabilityManager.revoke(skill.id);

    // Persister les stats mises à jour
    if (this.persistence) {
      await this.persistence.saveSkill(skill);
    }

    // Émettre fin
    this.emitEvent({
      type: 'execution_completed',
      result,
    });

    return result;
  }

  /**
   * Exécuter le skill le plus pertinent pour une requête
   */
  async executeRelevantSkill(
    userInput: string,
    input: Record<string, unknown>
  ): Promise<SkillExecutionResult | null> {
    const relevantSkills = this.findRelevantSkills(userInput, 1);

    if (relevantSkills.length === 0) {
      return null;
    }

    return this.executeSkill({
      skillId: relevantSkills[0].id,
      input,
    });
  }

  // ===========================================================================
  // CAPABILITIES PROXY
  // ===========================================================================

  /**
   * Appelé par le worker quand il a besoin d'une capability
   * (le worker ne peut pas accéder directement aux APIs)
   */
  async handleCapabilityRequest(
    skillId: string,
    capabilityType: CapabilityType,
    method: string,
    args: unknown[]
  ): Promise<unknown> {
    // Vérifier que le skill a cette capability
    const grant = this.capabilityManager.getGrant(skillId);
    if (!grant) {
      throw new Error(`Aucune capability accordée au skill: ${skillId}`);
    }

    const hasCapability = grant.capabilities.some(c => c.type === capabilityType);
    if (!hasCapability) {
      this.emitEvent({
        type: 'capability_violation',
        skillId,
        capability: capabilityType,
        details: `Tentative d'accès non autorisé à ${capabilityType}.${method}`,
      });
      throw new Error(`Capability non autorisée: ${capabilityType}`);
    }

    // Déléguer à l'implémentation de la capability
    return this.capabilityManager.execute(skillId, capabilityType, method, args);
  }

  // ===========================================================================
  // STATS
  // ===========================================================================

  /**
   * Obtenir les statistiques
   */
  getStats(): SkillStats {
    const skills = Array.from(this.skills.values());

    const enabledSkills = skills.filter(s => s.isEnabled);
    const builtinSkills = skills.filter(s => s.isBuiltin);
    const userCreatedSkills = skills.filter(s => s.createdBy === 'user');
    const neoCreatedSkills = skills.filter(s => s.createdBy === 'neo');

    const totalExecutions = skills.reduce((sum, s) => sum + s.usageCount, 0);
    const averageSuccessRate = skills.length > 0
      ? skills.reduce((sum, s) => sum + s.successRate, 0) / skills.length
      : 0;

    const mostUsedSkills = skills
      .filter(s => s.usageCount > 0)
      .sort((a, b) => b.usageCount - a.usageCount)
      .slice(0, 5)
      .map(s => ({ id: s.id, name: s.name, usageCount: s.usageCount }));

    return {
      totalSkills: skills.length,
      enabledSkills: enabledSkills.length,
      builtinSkills: builtinSkills.length,
      userCreatedSkills: userCreatedSkills.length,
      neoCreatedSkills: neoCreatedSkills.length,
      totalExecutions,
      averageSuccessRate,
      mostUsedSkills,
    };
  }

  // ===========================================================================
  // HELPERS
  // ===========================================================================

  private validateSkillCode(code: string): { valid: boolean; error?: string } {
    // Vérifications de sécurité basiques
    const forbidden = [
      'require(',
      'import(',
      'eval(',
      'Function(',
      'process.',
      '__dirname',
      '__filename',
      'global.',
      'globalThis.',
    ];

    for (const pattern of forbidden) {
      if (code.includes(pattern)) {
        return {
          valid: false,
          error: `Pattern interdit détecté: ${pattern}`,
        };
      }
    }

    // Vérifier la syntaxe
    try {
      // Tenter de parser le code comme une fonction
      new Function(code);
    } catch (e) {
      return {
        valid: false,
        error: `Erreur de syntaxe: ${e instanceof Error ? e.message : String(e)}`,
      };
    }

    return { valid: true };
  }

  private incrementVersion(version: string): string {
    const parts = version.split('.').map(Number);
    parts[2] = (parts[2] || 0) + 1;
    return parts.join('.');
  }

  private updateSuccessRate(current: number, success: boolean): number {
    // Moyenne mobile exponentielle
    const alpha = 0.1;
    return current * (1 - alpha) + (success ? 1 : 0) * alpha;
  }

  private emitEvent(event: SkillEvent): void {
    this.emit(event.type, event);
    this.emit('event', event);
  }

  private async loadBuiltinSkills(): Promise<void> {
    // Les skills built-in seront chargés depuis src/skills/builtin/
    // Pour l'instant, on ne charge rien
    console.log('[SkillManager] Skills built-in chargés');
  }
}

// ===========================================================================
// PERSISTENCE INTERFACE
// ===========================================================================

/**
 * Interface pour la persistence des skills
 * (sera implémentée par MemoryAgent/PersistenceLayer)
 */
export interface SkillPersistence {
  loadSkills(): Promise<SkillDefinition[]>;
  saveSkills(skills: SkillDefinition[]): Promise<void>;
  saveSkill(skill: SkillDefinition): Promise<void>;
  deleteSkill(id: string): Promise<void>;
}

// ===========================================================================
// SINGLETON
// ===========================================================================

let skillManagerInstance: SkillManager | null = null;

export function getSkillManager(): SkillManager {
  if (!skillManagerInstance) {
    skillManagerInstance = new SkillManager();
  }
  return skillManagerInstance;
}

export function initSkillManager(options?: { persistence?: SkillPersistence }): SkillManager {
  skillManagerInstance = new SkillManager(options);
  return skillManagerInstance;
}
