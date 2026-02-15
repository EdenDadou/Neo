/**
 * Logger centralisé pour Neo
 *
 * Par défaut, affiche uniquement:
 * - Les réponses finales (encadrées)
 * - Les erreurs critiques
 *
 * Mode verbose (DEBUG=true): affiche tout
 */

export type LogLevel = 'debug' | 'info' | 'agent' | 'warn' | 'error';

// Patterns à toujours afficher
const ALWAYS_SHOW_PATTERNS = [
  /^┌─/,               // Début de réponse encadrée
  /^│/,                // Contenu de réponse encadrée
  /^└─/,               // Fin de réponse encadrée
  /^✅ Serveur/,       // Démarrage serveur
  /^🧠 NEO/,           // Bannière
  /^───/,              // Séparateurs bannière
  /^💡/,               // Tips
  /WebSocket:/,        // Info connexion
  /^Vous >/,           // Input utilisateur (CLI)
];

// Patterns à masquer en mode non-verbose (tout le reste par défaut)
const HIDE_PATTERNS = [
  /^\[/,               // Tout ce qui commence par [xxx]
  /^╔|^║|^╚/,          // Bordures bannière
  /Fallback/i,         // Messages de fallback
  /OAuth/i,            // Messages OAuth
  /tokens?.*\$/i,      // Messages de tokens/coût
  /Modèle optimisé/i,  // Sélection de modèle
  /warmup/i,           // Warmup
  /embedding/i,        // Embeddings
];

// Sauvegarde des fonctions originales
const originalConsole = {
  log: console.log.bind(console),
  warn: console.warn.bind(console),
  error: console.error.bind(console),
};

let isVerbose = false;

/**
 * Vérifie si un message doit être affiché
 */
function shouldShow(args: unknown[]): boolean {
  if (isVerbose) return true;

  const message = args.map(a => String(a)).join(' ');

  // Toujours afficher certains patterns
  if (ALWAYS_SHOW_PATTERNS.some(p => p.test(message))) {
    return true;
  }

  // Masquer les patterns système
  if (HIDE_PATTERNS.some(p => p.test(message))) {
    return false;
  }

  // Par défaut, masquer
  return false;
}

/**
 * Initialise le logger et configure le filtrage de console
 */
export function initLogger(): void {
  isVerbose = process.env.DEBUG === 'true' || process.env.VERBOSE === 'true';

  if (!isVerbose) {
    // Remplacer console.log pour filtrer les messages système
    console.log = (...args: unknown[]) => {
      if (shouldShow(args)) {
        originalConsole.log(...args);
      }
    };

    // console.warn - masquer en mode non-verbose
    console.warn = (...args: unknown[]) => {
      // Masquer tous les warnings sauf erreurs critiques
      const message = args.map(a => String(a)).join(' ');
      if (message.includes('EADDRINUSE') || message.includes('FATAL')) {
        originalConsole.warn(...args);
      }
    };

    // console.error - afficher uniquement les erreurs critiques
    console.error = (...args: unknown[]) => {
      const message = args.map(a => String(a)).join(' ');
      // Masquer les erreurs OAuth/fallback non critiques
      if (message.includes('OAuth') ||
          message.includes('fallback') ||
          message.includes('Fallback') ||
          message.includes('warmup')) {
        return;
      }
      originalConsole.error(...args);
    };
  }
}

// ============================================================================
// Logger class pour usage explicite dans le code
// ============================================================================

class Logger {
  /**
   * Log de debug - uniquement en mode verbose
   */
  debug(source: string, message: string, ...args: unknown[]): void {
    if (isVerbose) {
      originalConsole.log(`[${source}] ${message}`, ...args);
    }
  }

  /**
   * Réponse d'un agent - toujours visible avec formatage encadré
   */
  response(source: string, message: string): void {
    originalConsole.log('');
    originalConsole.log('┌─────────────────────────────────────────────────────────────');
    originalConsole.log(`│ ${this.getAgentIcon(source)} ${this.getAgentName(source)}:`);
    originalConsole.log('├─────────────────────────────────────────────────────────────');
    message.split('\n').forEach(line => {
      originalConsole.log(`│ ${line}`);
    });
    originalConsole.log('└─────────────────────────────────────────────────────────────');
    originalConsole.log('');
  }

  /**
   * Message entrant de l'utilisateur
   */
  userInput(message: string): void {
    originalConsole.log(`\nVous > ${message}`);
  }

  /**
   * Avertissement - uniquement en mode verbose
   */
  warn(source: string, message: string, ...args: unknown[]): void {
    if (isVerbose) {
      originalConsole.warn(`[${source}] ⚠️ ${message}`, ...args);
    }
  }

  /**
   * Erreur critique - toujours visible
   */
  error(source: string, message: string, error?: unknown): void {
    originalConsole.error(`❌ [${source}] ${message}`);
    if (error && isVerbose) {
      originalConsole.error(error);
    }
  }

  /**
   * Agent message - pour les messages importants des agents
   */
  agent(source: string, message: string): void {
    this.response(source, message);
  }

  private getAgentIcon(source: string): string {
    switch (source.toLowerCase()) {
      case 'vox': return '🎙️';
      case 'brain': return '🧠';
      case 'memory': return '💾';
      case 'neo': return '🧠';
      default: return '🤖';
    }
  }

  private getAgentName(source: string): string {
    switch (source.toLowerCase()) {
      case 'vox': return 'Vox';
      case 'brain': return 'Brain';
      case 'memory': return 'Memory';
      case 'neo': return 'Neo';
      default: return source;
    }
  }
}

// Singleton
export const logger = new Logger();
