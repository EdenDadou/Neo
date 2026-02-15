/**
 * Skill Worker - Exécution isolée des skills
 *
 * Ce worker reçoit du code à exécuter et des capabilities sérialisées.
 * Il ne peut PAS:
 * - Importer des modules (require/import bloqués)
 * - Accéder au système de fichiers directement
 * - Accéder au réseau directement
 * - Accéder aux variables globales de Node.js
 *
 * Il PEUT:
 * - Exécuter du JavaScript pur
 * - Demander des capabilities au main thread via messages
 * - Utiliser les APIs fournies (capabilities)
 */

import { parentPort } from 'worker_threads';

if (!parentPort) {
  throw new Error('Ce fichier doit être exécuté comme worker thread');
}

// ===========================================================================
// TYPES
// ===========================================================================

interface CapabilityRequest {
  type: string;
  method: string;
  args: unknown[];
  requestId: number;
}

interface WorkerMessage {
  type: string;
  executionId: string;
  payload: {
    code?: string;
    input?: unknown;
    capabilities?: {
      allowed: string[];
      config: Record<string, unknown>;
    };
    timeout?: number;
    requestId?: number;
    result?: unknown;
    error?: { message: string };
  };
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}

interface CapabilityProxies {
  webFetch?: (url: string, options?: unknown) => Promise<unknown>;
  memory?: {
    search?: (query: string, limit?: number) => Promise<unknown>;
    get?: (id: string) => Promise<unknown>;
    store?: (type: string, content: string, metadata?: unknown) => Promise<unknown>;
  };
  browser?: {
    navigate: (url: string) => Promise<unknown>;
    click: (selector: string) => Promise<unknown>;
    type: (selector: string, text: string) => Promise<unknown>;
    screenshot: () => Promise<unknown>;
    getContent: () => Promise<unknown>;
    waitFor: (selector: string, timeout?: number) => Promise<unknown>;
  };
  file?: {
    read: (path: string) => Promise<unknown>;
    exists: (path: string) => Promise<unknown>;
  };
  llm?: {
    complete: (prompt: string, options?: unknown) => Promise<unknown>;
  };
  log: {
    info: (message: string) => void;
    warn: (message: string) => void;
    error: (message: string) => void;
  };
}

// ===========================================================================
// SECURITY: Supprimer les accès dangereux
// ===========================================================================

// Bloquer les variables globales dangereuses
// @ts-expect-error - Suppression intentionnelle
delete globalThis.process;
// @ts-expect-error - Suppression intentionnelle
delete globalThis.Buffer;

// ===========================================================================
// CAPABILITY PROXIES
// ===========================================================================

const port = parentPort;

/**
 * Crée des proxies pour les capabilities
 * Chaque appel est envoyé au main thread et attend une réponse
 */
function createCapabilityProxies(
  executionId: string,
  allowedCapabilities: string[],
  _config: Record<string, unknown>
): CapabilityProxies {
  const proxies: Partial<CapabilityProxies> = {};
  const pendingRequests = new Map<number, PendingRequest>();
  let requestId = 0;

  // Fonction pour appeler une capability
  async function callCapability(type: string, method: string, args: unknown[]): Promise<unknown> {
    if (!allowedCapabilities.includes(type)) {
      throw new Error(`Capability non autorisée: ${type}`);
    }

    const reqId = ++requestId;

    return new Promise((resolve, reject) => {
      pendingRequests.set(reqId, { resolve, reject });

      port.postMessage({
        type: 'CAPABILITY_REQUEST',
        executionId,
        payload: {
          capabilityRequest: {
            type,
            method,
            args,
            requestId: reqId,
          },
        },
      });

      // Timeout pour la capability
      setTimeout(() => {
        if (pendingRequests.has(reqId)) {
          pendingRequests.delete(reqId);
          reject(new Error(`Timeout capability: ${type}.${method}`));
        }
      }, 30000);
    });
  }

  // Gérer les réponses de capabilities
  port.on('message', (msg: WorkerMessage) => {
    if (msg.type === 'CAPABILITY_RESPONSE' && msg.executionId === executionId) {
      const reqId = msg.payload.requestId;
      if (reqId === undefined) return;

      const pending = pendingRequests.get(reqId);
      if (pending) {
        pendingRequests.delete(reqId);
        if (msg.payload.error) {
          pending.reject(new Error(msg.payload.error.message));
        } else {
          pending.resolve(msg.payload.result);
        }
      }
    }
  });

  // Créer les proxies selon les capabilities autorisées

  if (allowedCapabilities.includes('web_fetch')) {
    proxies.webFetch = async (url: string, options?: unknown) => {
      return callCapability('web_fetch', 'fetch', [url, options]);
    };
  }

  if (allowedCapabilities.includes('memory_read')) {
    proxies.memory = proxies.memory || {};
    proxies.memory.search = async (query: string, limit?: number) => {
      return callCapability('memory_read', 'search', [query, limit]);
    };
    proxies.memory.get = async (id: string) => {
      return callCapability('memory_read', 'get', [id]);
    };
  }

  if (allowedCapabilities.includes('memory_write')) {
    proxies.memory = proxies.memory || {};
    proxies.memory.store = async (type: string, content: string, metadata?: unknown) => {
      return callCapability('memory_write', 'store', [type, content, metadata]);
    };
  }

  if (allowedCapabilities.includes('browser')) {
    proxies.browser = {
      navigate: async (url: string) => {
        return callCapability('browser', 'navigate', [url]);
      },
      click: async (selector: string) => {
        return callCapability('browser', 'click', [selector]);
      },
      type: async (selector: string, text: string) => {
        return callCapability('browser', 'type', [selector, text]);
      },
      screenshot: async () => {
        return callCapability('browser', 'screenshot', []);
      },
      getContent: async () => {
        return callCapability('browser', 'getContent', []);
      },
      waitFor: async (selector: string, timeout?: number) => {
        return callCapability('browser', 'waitFor', [selector, timeout]);
      },
    };
  }

  if (allowedCapabilities.includes('file_read')) {
    proxies.file = {
      read: async (path: string) => {
        return callCapability('file_read', 'read', [path]);
      },
      exists: async (path: string) => {
        return callCapability('file_read', 'exists', [path]);
      },
    };
  }

  if (allowedCapabilities.includes('llm_call')) {
    proxies.llm = {
      complete: async (prompt: string, options?: unknown) => {
        return callCapability('llm_call', 'complete', [prompt, options]);
      },
    };
  }

  // Utilitaires toujours disponibles
  proxies.log = {
    info: (message: string) => {
      port.postMessage({
        type: 'LOG',
        executionId,
        payload: { log: { level: 'info', message: String(message) } },
      });
    },
    warn: (message: string) => {
      port.postMessage({
        type: 'LOG',
        executionId,
        payload: { log: { level: 'warn', message: String(message) } },
      });
    },
    error: (message: string) => {
      port.postMessage({
        type: 'LOG',
        executionId,
        payload: { log: { level: 'error', message: String(message) } },
      });
    },
  };

  return proxies as CapabilityProxies;
}

// ===========================================================================
// SKILL EXECUTION
// ===========================================================================

/**
 * Exécuter le code du skill
 */
async function executeSkill(
  executionId: string,
  code: string,
  input: unknown,
  capabilities: { allowed: string[]; config: Record<string, unknown> }
): Promise<unknown> {
  // Créer les proxies de capabilities
  const proxies = createCapabilityProxies(
    executionId,
    capabilities.allowed,
    capabilities.config
  );

  // Wrapper le code dans une fonction async
  // Le code reçoit: input (les données) et les capabilities (proxies)
  const wrappedCode = `
    return (async function(input, capabilities) {
      const { webFetch, memory, browser, file, llm, log } = capabilities;
      ${code}
    })(input, capabilities);
  `;

  // Créer et exécuter la fonction
  const fn = new Function('input', 'capabilities', wrappedCode);
  const result = await fn(input, proxies);

  return result;
}

// ===========================================================================
// MESSAGE HANDLER
// ===========================================================================

port.on('message', async (message: WorkerMessage) => {
  if (message.type === 'EXECUTE_SKILL') {
    const { executionId, payload } = message;
    const { code, input, capabilities, timeout } = payload;

    if (!code || !capabilities || !timeout) {
      port.postMessage({
        type: 'ERROR',
        executionId,
        payload: {
          error: {
            message: 'Missing required fields: code, capabilities, or timeout',
            code: 'INVALID_MESSAGE',
          },
        },
      });
      return;
    }

    try {
      // Exécuter avec timeout
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => {
          reject(new Error(`Timeout après ${timeout}ms`));
        }, timeout);
      });

      const result = await Promise.race([
        executeSkill(executionId, code, input, capabilities),
        timeoutPromise,
      ]);

      // Envoyer le résultat
      port.postMessage({
        type: 'RESULT',
        executionId,
        payload: { output: result },
      });
    } catch (error) {
      // Envoyer l'erreur
      const err = error as Error;
      port.postMessage({
        type: 'ERROR',
        executionId,
        payload: {
          error: {
            message: err.message || String(error),
            code: (err as Error & { code?: string }).code || 'EXECUTION_ERROR',
            stack: err.stack,
          },
        },
      });
    }
  } else if (message.type === 'PING') {
    port.postMessage({
      type: 'PONG',
      executionId: message.executionId,
      payload: {},
    });
  } else if (message.type === 'CANCEL') {
    // On ne peut pas vraiment annuler, mais on peut ignorer le résultat
    // Le timeout côté main thread s'en chargera
  }
});

// Signaler que le worker est prêt
port.postMessage({
  type: 'READY',
  executionId: 'init',
  payload: {},
});
