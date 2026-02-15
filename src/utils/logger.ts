/**
 * Logger centralisé pour Neo
 *
 * Par défaut, affiche uniquement:
 * - Les réponses des agents (Vox, Brain, Memory)
 * - Les erreurs
 *
 * Mode verbose (DEBUG=true): affiche tout
 */

export type LogLevel = 'debug' | 'info' | 'agent' | 'warn' | 'error';

// Patterns à toujours afficher (réponses importantes)
const ALWAYS_SHOW_PATTERNS = [
  /^\[Vox\] 📤/,       // Réponses Vox
  /^🎙️ Vox:/,          // Réponses formatées
  /^🧠 Brain:/,        // Réponses Brain
  /^💾 Memory:/,       // Réponses Memory
  /^👤 User:/,         // Input utilisateur
  /^✅ Serveur/,       // Démarrage serveur
  /^🧠 NEO/,           // Bannière
  /^───/,              // Séparateurs
  /^💡/,               // Tips
  /WebSocket:/,        // Info connexion
];

// Patterns à masquer en mode non-verbose
const HIDE_PATTERNS = [
  /^\[Env\]/,
  /^\[Auth\]/,
  /^\[Persistence\]/,
  /^\[Embeddings\]/,
  /^\[ModelRouter\]/,
  /^\[SkillManager\]/,
  /^\[Skills\]/,
  /^\[WorkerExecutor\]/,
  /^\[Core\]/,
  /^\[Memory\]/,         // Tous les logs Memory système
  /^\[Brain\].*init/i,
  /^\[Brain\].*démarr/i,
  /^\[Brain\].*modèle/i,
  /^\[Brain\].*Agent/,
  /^\[Brain\].*Recherche/i,
  /^\[Vox\].*Agent/,
  /^\[Vox\].*📋/,        // Demande contexte
  /^\[Vox\].*✍️/,        // Réécriture
  /^\[Vox\].*✅/,        // Prompt enrichi
  /^\[Vox\].*📥/,        // Entrée utilisateur (doublon)
  /^\[WebSearch\]/,
  /^\[Gateway\]/,
  /^\[UserStore\]/,
  /^\[TokenManager\]/,
  /^\[FactChecker\]/,
  /^╔|^║|^╚/,          // Bordures bannière
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

  return true;
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

    // console.warn - garder les warnings importants seulement
    console.warn = (...args: unknown[]) => {
      const message = args.map(a => String(a)).join(' ');
      // Masquer les warnings de configuration
      if (message.includes('ANTHROPIC_API_KEY') ||
          message.includes('No valid') ||
          message.includes('warmup')) {
        return;
      }
      originalConsole.warn(...args);
    };
  }

  // console.error reste toujours actif
}

// ============================================================================
// Logger class pour usage explicite
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
   * Réponse d'un agent vers l'utilisateur - toujours visible avec formatage
   */
  response(source: string, message: string): void {
    const prefix = this.getAgentPrefix(source);
    originalConsole.log(`\n${prefix} ${message}\n`);
  }

  /**
   * Message entrant de l'utilisateur - toujours visible
   */
  userInput(message: string): void {
    originalConsole.log(`\n👤 User: ${message}`);
  }

  /**
   * Avertissement - toujours visible
   */
  warn(source: string, message: string, ...args: unknown[]): void {
    originalConsole.warn(`[${source}] ⚠️ ${message}`, ...args);
  }

  /**
   * Erreur - toujours visible
   */
  error(source: string, message: string, error?: unknown): void {
    originalConsole.error(`[${source}] ❌ ${message}`);
    if (error && isVerbose) {
      originalConsole.error(error);
    }
  }

  /**
   * Agent message - toujours visible
   */
  agent(source: string, message: string, ...args: unknown[]): void {
    const prefix = this.getAgentPrefix(source);
    originalConsole.log(`${prefix} ${message}`, ...args);
  }

  private getAgentPrefix(source: string): string {
    switch (source.toLowerCase()) {
      case 'vox':
        return '🎙️ Vox:';
      case 'brain':
        return '🧠 Brain:';
      case 'memory':
        return '💾 Memory:';
      default:
        return `[${source}]`;
    }
  }
}

// Singleton
export const logger = new Logger();
