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

const { parentPort } = require('worker_threads');

// ===========================================================================
// SECURITY: Supprimer les accès dangereux
// ===========================================================================

// Bloquer require
const originalRequire = require;
global.require = function blockedRequire() {
  throw new Error('require() est bloqué dans les skills');
};

// Bloquer les variables globales dangereuses
delete global.process;
delete global.Buffer;
delete global.__dirname;
delete global.__filename;

// ===========================================================================
// CAPABILITY PROXIES
// ===========================================================================

/**
 * Crée des proxies pour les capabilities
 * Chaque appel est envoyé au main thread et attend une réponse
 */
function createCapabilityProxies(executionId, allowedCapabilities, config) {
  const proxies = {};
  const pendingRequests = new Map();
  let requestId = 0;

  // Fonction pour appeler une capability
  async function callCapability(type, method, args) {
    if (!allowedCapabilities.includes(type)) {
      throw new Error(`Capability non autorisée: ${type}`);
    }

    const reqId = ++requestId;

    return new Promise((resolve, reject) => {
      pendingRequests.set(reqId, { resolve, reject });

      parentPort.postMessage({
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
  parentPort.on('message', (msg) => {
    if (msg.type === 'CAPABILITY_RESPONSE' && msg.executionId === executionId) {
      const reqId = msg.payload.requestId;
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
    proxies.webFetch = async (url, options) => {
      return callCapability('web_fetch', 'fetch', [url, options]);
    };
  }

  if (allowedCapabilities.includes('memory_read')) {
    proxies.memory = proxies.memory || {};
    proxies.memory.search = async (query, limit) => {
      return callCapability('memory_read', 'search', [query, limit]);
    };
    proxies.memory.get = async (id) => {
      return callCapability('memory_read', 'get', [id]);
    };
  }

  if (allowedCapabilities.includes('memory_write')) {
    proxies.memory = proxies.memory || {};
    proxies.memory.store = async (type, content, metadata) => {
      return callCapability('memory_write', 'store', [type, content, metadata]);
    };
  }

  if (allowedCapabilities.includes('browser')) {
    proxies.browser = {
      navigate: async (url) => {
        return callCapability('browser', 'navigate', [url]);
      },
      click: async (selector) => {
        return callCapability('browser', 'click', [selector]);
      },
      type: async (selector, text) => {
        return callCapability('browser', 'type', [selector, text]);
      },
      screenshot: async () => {
        return callCapability('browser', 'screenshot', []);
      },
      getContent: async () => {
        return callCapability('browser', 'getContent', []);
      },
      waitFor: async (selector, timeout) => {
        return callCapability('browser', 'waitFor', [selector, timeout]);
      },
    };
  }

  if (allowedCapabilities.includes('file_read')) {
    proxies.file = {
      read: async (path) => {
        return callCapability('file_read', 'read', [path]);
      },
      exists: async (path) => {
        return callCapability('file_read', 'exists', [path]);
      },
    };
  }

  if (allowedCapabilities.includes('llm_call')) {
    proxies.llm = {
      complete: async (prompt, options) => {
        return callCapability('llm_call', 'complete', [prompt, options]);
      },
    };
  }

  // Utilitaires toujours disponibles
  proxies.log = {
    info: (message) => {
      parentPort.postMessage({
        type: 'LOG',
        executionId,
        payload: { log: { level: 'info', message: String(message) } },
      });
    },
    warn: (message) => {
      parentPort.postMessage({
        type: 'LOG',
        executionId,
        payload: { log: { level: 'warn', message: String(message) } },
      });
    },
    error: (message) => {
      parentPort.postMessage({
        type: 'LOG',
        executionId,
        payload: { log: { level: 'error', message: String(message) } },
      });
    },
  };

  return proxies;
}

// ===========================================================================
// SKILL EXECUTION
// ===========================================================================

/**
 * Exécuter le code du skill
 */
async function executeSkill(executionId, code, input, capabilities) {
  // Créer les proxies de capabilities
  const proxies = createCapabilityProxies(
    executionId,
    capabilities.allowed,
    capabilities.config
  );

  try {
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
  } catch (error) {
    throw error;
  }
}

// ===========================================================================
// MESSAGE HANDLER
// ===========================================================================

parentPort.on('message', async (message) => {
  if (message.type === 'EXECUTE_SKILL') {
    const { executionId, payload } = message;
    const { code, input, capabilities, timeout } = payload;

    try {
      // Exécuter avec timeout
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => {
          reject(new Error(`Timeout après ${timeout}ms`));
        }, timeout);
      });

      const result = await Promise.race([
        executeSkill(executionId, code, input, capabilities),
        timeoutPromise,
      ]);

      // Envoyer le résultat
      parentPort.postMessage({
        type: 'RESULT',
        executionId,
        payload: { output: result },
      });
    } catch (error) {
      // Envoyer l'erreur
      parentPort.postMessage({
        type: 'ERROR',
        executionId,
        payload: {
          error: {
            message: error.message || String(error),
            code: error.code || 'EXECUTION_ERROR',
            stack: error.stack,
          },
        },
      });
    }
  } else if (message.type === 'PING') {
    parentPort.postMessage({
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
parentPort.postMessage({
  type: 'READY',
  executionId: 'init',
  payload: {},
});
