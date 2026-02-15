/**
 * CapabilityManager - Gestion des capabilities (Object Capabilities / OCAP)
 *
 * Principes OCAP appliqués:
 * - Les skills ne reçoivent QUE les capabilities dont ils ont besoin
 * - Pas d'accès global par défaut (least privilege)
 * - Capabilities révocables à tout moment
 * - Rate limiting par capability
 * - Audit trail de chaque usage
 */

import { EventEmitter } from 'events';
import type {
  CapabilityType,
  CapabilityConfig,
  CapabilityGrant,
} from './types';
import { CAPABILITY_LIMITS } from './types';

// ===========================================================================
// RATE LIMITER
// ===========================================================================

interface RateLimitBucket {
  count: number;
  resetAt: number;
}

class RateLimiter {
  private buckets: Map<string, RateLimitBucket> = new Map();

  /**
   * Vérifier si une action est autorisée
   */
  check(key: string, limit: number, windowMs: number): boolean {
    const now = Date.now();
    const bucket = this.buckets.get(key);

    if (!bucket || now >= bucket.resetAt) {
      // Nouveau bucket
      this.buckets.set(key, {
        count: 1,
        resetAt: now + windowMs,
      });
      return true;
    }

    if (bucket.count >= limit) {
      return false;
    }

    bucket.count++;
    return true;
  }

  /**
   * Nettoyer les buckets expirés
   */
  cleanup(): void {
    const now = Date.now();
    for (const [key, bucket] of this.buckets) {
      if (now >= bucket.resetAt) {
        this.buckets.delete(key);
      }
    }
  }
}

// ===========================================================================
// CAPABILITY IMPLEMENTATIONS
// ===========================================================================

/**
 * Interface pour les implémentations de capabilities
 */
interface CapabilityImplementation {
  execute(method: string, args: unknown[], config?: Record<string, unknown>): Promise<unknown>;
  cleanup?(): Promise<void>;
}

/**
 * Implémentation de web_fetch
 */
class WebFetchCapability implements CapabilityImplementation {
  private rateLimiter = new RateLimiter();

  async execute(
    method: string,
    args: unknown[],
    config?: Record<string, unknown>
  ): Promise<unknown> {
    if (method !== 'fetch') {
      throw new Error(`Méthode non supportée: ${method}`);
    }

    const [url, options] = args as [string, RequestInit?];

    // Valider l'URL
    const parsedUrl = new URL(url);
    const allowedDomains = (config?.allowedDomains as string[]) || [];

    if (allowedDomains.length > 0) {
      const domain = parsedUrl.hostname;
      const isAllowed = allowedDomains.some(d =>
        domain === d || domain.endsWith(`.${d}`)
      );
      if (!isAllowed) {
        throw new Error(`Domaine non autorisé: ${domain}`);
      }
    }

    // Rate limit
    const maxReq = (config?.maxRequestsPerMinute as number) ||
      CAPABILITY_LIMITS.web_fetch.maxRequestsPerMinute;

    if (!this.rateLimiter.check(`web_fetch`, maxReq, 60000)) {
      throw new Error('Rate limit atteint pour web_fetch');
    }

    // Effectuer la requête
    const response = await fetch(url, {
      ...options,
      signal: AbortSignal.timeout(CAPABILITY_LIMITS.web_fetch.timeoutMs),
    });

    // Retourner une version sérialisable
    return {
      ok: response.ok,
      status: response.status,
      statusText: response.statusText,
      headers: Object.fromEntries(response.headers.entries()),
      text: await response.text(),
    };
  }
}

/**
 * Implémentation de memory_read/memory_write
 * (Sera connecté au MemoryAgent)
 */
class MemoryCapability implements CapabilityImplementation {
  private memoryInterface: MemoryInterface | null = null;

  setMemoryInterface(mem: MemoryInterface): void {
    this.memoryInterface = mem;
  }

  async execute(
    method: string,
    args: unknown[],
    config?: Record<string, unknown>
  ): Promise<unknown> {
    if (!this.memoryInterface) {
      throw new Error('Memory interface non configurée');
    }

    const maxItems = (config?.maxItems as number) || CAPABILITY_LIMITS.memory_read.maxItemsPerQuery;

    switch (method) {
      case 'search': {
        const [query, limit] = args as [string, number?];
        const actualLimit = Math.min(limit || 10, maxItems);
        return this.memoryInterface.search(query, actualLimit);
      }

      case 'get': {
        const [id] = args as [string];
        return this.memoryInterface.get(id);
      }

      case 'store': {
        const [type, content, metadata] = args as [string, string, Record<string, unknown>?];
        return this.memoryInterface.store(type, content, metadata);
      }

      default:
        throw new Error(`Méthode non supportée: ${method}`);
    }
  }
}

/**
 * Implémentation de browser (Playwright)
 * (Sera connecté à un BrowserPool)
 */
class BrowserCapability implements CapabilityImplementation {
  private browserInterface: BrowserInterface | null = null;
  private activeSessions: Map<string, unknown> = new Map();

  setBrowserInterface(browser: BrowserInterface): void {
    this.browserInterface = browser;
  }

  async execute(
    method: string,
    args: unknown[],
    config?: Record<string, unknown>
  ): Promise<unknown> {
    if (!this.browserInterface) {
      throw new Error('Browser interface non configurée');
    }

    const allowedUrls = (config?.allowedUrls as string[]) || [];
    const timeout = (config?.timeout as number) || CAPABILITY_LIMITS.browser.timeoutMs;

    switch (method) {
      case 'navigate': {
        const [url] = args as [string];

        // Valider l'URL
        if (allowedUrls.length > 0) {
          // Valider que c'est une URL valide
          new URL(url);
          const isAllowed = allowedUrls.some(pattern => {
            if (pattern.includes('*')) {
              const regex = new RegExp(pattern.replace(/\*/g, '.*'));
              return regex.test(url);
            }
            return url.startsWith(pattern);
          });
          if (!isAllowed) {
            throw new Error(`URL non autorisée: ${url}`);
          }
        }

        return this.browserInterface.navigate(url, timeout);
      }

      case 'click': {
        const [selector] = args as [string];
        return this.browserInterface.click(selector, timeout);
      }

      case 'type': {
        const [selector, text] = args as [string, string];
        return this.browserInterface.type(selector, text, timeout);
      }

      case 'screenshot': {
        return this.browserInterface.screenshot();
      }

      case 'getContent': {
        return this.browserInterface.getContent();
      }

      case 'waitFor': {
        const [selector, waitTimeout] = args as [string, number?];
        return this.browserInterface.waitFor(selector, waitTimeout || timeout);
      }

      default:
        throw new Error(`Méthode non supportée: ${method}`);
    }
  }

  async cleanup(): Promise<void> {
    // Fermer toutes les sessions
    this.activeSessions.clear();
  }
}

/**
 * Implémentation de file_read
 */
class FileReadCapability implements CapabilityImplementation {
  private fs: typeof import('fs/promises') | null = null;

  async execute(
    method: string,
    args: unknown[],
    config?: Record<string, unknown>
  ): Promise<unknown> {
    // Lazy load fs
    if (!this.fs) {
      this.fs = await import('fs/promises');
    }

    const allowedPaths = (config?.allowedPaths as string[]) || [];
    const maxSize = (config?.maxFileSizeKB as number) || CAPABILITY_LIMITS.file_read.maxFileSizeKB;

    switch (method) {
      case 'read': {
        const [filePath] = args as [string];

        // Valider le chemin
        if (allowedPaths.length > 0) {
          const isAllowed = allowedPaths.some(p => filePath.startsWith(p));
          if (!isAllowed) {
            throw new Error(`Chemin non autorisé: ${filePath}`);
          }
        }

        // Vérifier la taille
        const stats = await this.fs.stat(filePath);
        if (stats.size > maxSize * 1024) {
          throw new Error(`Fichier trop volumineux: ${stats.size} bytes (max: ${maxSize}KB)`);
        }

        return this.fs.readFile(filePath, 'utf-8');
      }

      case 'exists': {
        const [filePath] = args as [string];

        // Valider le chemin
        if (allowedPaths.length > 0) {
          const isAllowed = allowedPaths.some(p => filePath.startsWith(p));
          if (!isAllowed) {
            throw new Error(`Chemin non autorisé: ${filePath}`);
          }
        }

        try {
          await this.fs.access(filePath);
          return true;
        } catch {
          return false;
        }
      }

      default:
        throw new Error(`Méthode non supportée: ${method}`);
    }
  }
}

/**
 * Implémentation de llm_call
 * (Sera connecté au ModelRouter)
 */
class LLMCapability implements CapabilityImplementation {
  private llmInterface: LLMInterface | null = null;
  private callCount: Map<string, number> = new Map();

  setLLMInterface(llm: LLMInterface): void {
    this.llmInterface = llm;
  }

  async execute(
    method: string,
    args: unknown[],
    config?: Record<string, unknown>
  ): Promise<unknown> {
    if (!this.llmInterface) {
      throw new Error('LLM interface non configurée');
    }

    const maxTokens = (config?.maxTokens as number) || CAPABILITY_LIMITS.llm_call.maxTokensPerCall;
    const maxCalls = CAPABILITY_LIMITS.llm_call.maxCallsPerExecution;

    switch (method) {
      case 'complete': {
        // Vérifier le nombre d'appels
        const currentCalls = this.callCount.get('total') || 0;
        if (currentCalls >= maxCalls) {
          throw new Error(`Limite d'appels LLM atteinte: ${maxCalls}`);
        }
        this.callCount.set('total', currentCalls + 1);

        const [prompt, options] = args as [string, { model?: string; maxTokens?: number }?];

        return this.llmInterface.complete(prompt, {
          model: options?.model,
          maxTokens: Math.min(options?.maxTokens || maxTokens, maxTokens),
        });
      }

      default:
        throw new Error(`Méthode non supportée: ${method}`);
    }
  }

  async cleanup(): Promise<void> {
    this.callCount.clear();
  }
}

// ===========================================================================
// CAPABILITY MANAGER
// ===========================================================================

export class CapabilityManager extends EventEmitter {
  private grants: Map<string, CapabilityGrant> = new Map();
  private implementations: Map<CapabilityType, CapabilityImplementation> = new Map();
  private auditLog: CapabilityAuditEntry[] = [];

  constructor() {
    super();

    // Initialiser les implémentations
    this.implementations.set('web_fetch', new WebFetchCapability());
    this.implementations.set('memory_read', new MemoryCapability());
    this.implementations.set('memory_write', new MemoryCapability());
    this.implementations.set('browser', new BrowserCapability());
    this.implementations.set('file_read', new FileReadCapability());
    this.implementations.set('llm_call', new LLMCapability());
  }

  /**
   * Configurer les interfaces externes
   */
  configure(options: {
    memory?: MemoryInterface;
    browser?: BrowserInterface;
    llm?: LLMInterface;
  }): void {
    if (options.memory) {
      (this.implementations.get('memory_read') as MemoryCapability).setMemoryInterface(options.memory);
      (this.implementations.get('memory_write') as MemoryCapability).setMemoryInterface(options.memory);
    }
    if (options.browser) {
      (this.implementations.get('browser') as BrowserCapability).setBrowserInterface(options.browser);
    }
    if (options.llm) {
      (this.implementations.get('llm_call') as LLMCapability).setLLMInterface(options.llm);
    }
  }

  /**
   * Accorder des capabilities à un skill
   */
  grant(skillId: string, capabilities: CapabilityType[]): CapabilityGrant {
    const grant: CapabilityGrant = {
      skillId,
      capabilities: capabilities.map(type => ({
        type,
        options: this.getDefaultOptions(type),
      })),
      grantedAt: new Date(),
    };

    this.grants.set(skillId, grant);

    this.audit(skillId, 'grant', capabilities.join(', '));

    return grant;
  }

  /**
   * Révoquer les capabilities d'un skill
   */
  revoke(skillId: string): void {
    const grant = this.grants.get(skillId);
    if (grant) {
      grant.revokedAt = new Date();
      this.grants.delete(skillId);
      this.audit(skillId, 'revoke', 'all');
    }
  }

  /**
   * Révoquer toutes les capabilities d'un skill
   */
  revokeAll(skillId: string): void {
    this.revoke(skillId);
  }

  /**
   * Obtenir le grant d'un skill
   */
  getGrant(skillId: string): CapabilityGrant | undefined {
    return this.grants.get(skillId);
  }

  /**
   * Exécuter une capability
   */
  async execute(
    skillId: string,
    capabilityType: CapabilityType,
    method: string,
    args: unknown[]
  ): Promise<unknown> {
    // Vérifier le grant
    const grant = this.grants.get(skillId);
    if (!grant) {
      this.audit(skillId, 'denied', `${capabilityType}.${method} - no grant`);
      throw new Error(`Aucun grant pour le skill: ${skillId}`);
    }

    // Vérifier que la capability est accordée
    const capConfig = grant.capabilities.find(c => c.type === capabilityType);
    if (!capConfig) {
      this.audit(skillId, 'denied', `${capabilityType}.${method} - not granted`);
      throw new Error(`Capability non accordée: ${capabilityType}`);
    }

    // Obtenir l'implémentation
    const impl = this.implementations.get(capabilityType);
    if (!impl) {
      throw new Error(`Implémentation non trouvée: ${capabilityType}`);
    }

    // Exécuter
    try {
      const result = await impl.execute(method, args, capConfig.options);
      this.audit(skillId, 'execute', `${capabilityType}.${method}`, true);
      return result;
    } catch (error) {
      this.audit(skillId, 'execute', `${capabilityType}.${method}`, false, error instanceof Error ? error.message : String(error));
      throw error;
    }
  }

  /**
   * Obtenir les options par défaut pour une capability
   */
  private getDefaultOptions(type: CapabilityType): CapabilityConfig['options'] {
    switch (type) {
      case 'web_fetch':
        return {
          allowedDomains: [], // Tout autorisé par défaut (peut être restreint)
          maxRequestsPerMinute: CAPABILITY_LIMITS.web_fetch.maxRequestsPerMinute,
        };
      case 'browser':
        return {
          allowedUrls: [],
          timeout: CAPABILITY_LIMITS.browser.timeoutMs,
        };
      case 'memory_read':
        return {
          maxItems: CAPABILITY_LIMITS.memory_read.maxItemsPerQuery,
        };
      case 'memory_write':
        return {
          maxItems: CAPABILITY_LIMITS.memory_write.maxItemsPerExecution,
        };
      case 'file_read':
        return {
          allowedPaths: [],
          maxFileSizeKB: CAPABILITY_LIMITS.file_read.maxFileSizeKB,
        };
      case 'llm_call':
        return {
          maxTokens: CAPABILITY_LIMITS.llm_call.maxTokensPerCall,
        };
      default:
        return {};
    }
  }

  /**
   * Logger une entrée d'audit
   */
  private audit(
    skillId: string,
    action: string,
    details: string,
    success?: boolean,
    error?: string
  ): void {
    const entry: CapabilityAuditEntry = {
      timestamp: new Date(),
      skillId,
      action,
      details,
      success,
      error,
    };

    this.auditLog.push(entry);

    // Garder les 1000 dernières entrées
    if (this.auditLog.length > 1000) {
      this.auditLog.shift();
    }

    // Émettre pour monitoring
    this.emit('audit', entry);
  }

  /**
   * Obtenir le log d'audit
   */
  getAuditLog(skillId?: string, limit = 100): CapabilityAuditEntry[] {
    let entries = this.auditLog;

    if (skillId) {
      entries = entries.filter(e => e.skillId === skillId);
    }

    return entries.slice(-limit);
  }
}

// ===========================================================================
// INTERFACES
// ===========================================================================

interface CapabilityAuditEntry {
  timestamp: Date;
  skillId: string;
  action: string;
  details: string;
  success?: boolean;
  error?: string;
}

export interface MemoryInterface {
  search(query: string, limit: number): Promise<unknown[]>;
  get(id: string): Promise<unknown | null>;
  store(type: string, content: string, metadata?: Record<string, unknown>): Promise<string>;
}

export interface BrowserInterface {
  navigate(url: string, timeout: number): Promise<void>;
  click(selector: string, timeout: number): Promise<void>;
  type(selector: string, text: string, timeout: number): Promise<void>;
  screenshot(): Promise<Buffer>;
  getContent(): Promise<string>;
  waitFor(selector: string, timeout: number): Promise<void>;
}

export interface LLMInterface {
  complete(prompt: string, options: { model?: string; maxTokens?: number }): Promise<string>;
}

// ===========================================================================
// SINGLETON
// ===========================================================================

let capabilityManagerInstance: CapabilityManager | null = null;

export function getCapabilityManager(): CapabilityManager {
  if (!capabilityManagerInstance) {
    capabilityManagerInstance = new CapabilityManager();
  }
  return capabilityManagerInstance;
}
