/**
 * Environment configuration loader
 * Loads .env file into process.env
 */

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Load environment variables from .env file
 */
export function config(): void {
  // Find .env file (look in project root)
  const projectRoot = resolve(__dirname, '..', '..');
  const envPath = resolve(projectRoot, '.env');

  if (!existsSync(envPath)) {
    console.warn('[Env] No .env file found at', envPath);
    return;
  }

  try {
    const envContent = readFileSync(envPath, 'utf-8');
    const lines = envContent.split('\n');

    for (const line of lines) {
      // Skip empty lines and comments
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) {
        continue;
      }

      // Parse KEY=VALUE
      const equalsIndex = trimmed.indexOf('=');
      if (equalsIndex === -1) continue;

      const key = trimmed.substring(0, equalsIndex).trim();
      let value = trimmed.substring(equalsIndex + 1).trim();

      // Remove quotes if present
      if ((value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }

      // Only set if not already defined (allow system env to override)
      if (process.env[key] === undefined) {
        process.env[key] = value;
      }
    }

    console.log('[Env] Configuration loaded from .env');

    // Validate critical variables
    if (!process.env.ANTHROPIC_API_KEY ||
        process.env.ANTHROPIC_API_KEY === 'test-key-for-structure-check' ||
        process.env.ANTHROPIC_API_KEY === 'your_api_key_here') {
      console.warn('[Env] ⚠️ ANTHROPIC_API_KEY not configured!');
      console.warn('[Env] Run ./neo config to configure Neo');
    }

    // Log permissions
    const allowWrite = process.env.ALLOW_FILE_WRITE === 'true';
    const allowShell = process.env.ALLOW_SHELL_EXEC === 'true';
    if (allowWrite || allowShell) {
      console.log(`[Env] Permissions: write=${allowWrite}, shell=${allowShell}`);
    }
  } catch (error) {
    console.error('[Env] Error loading .env:', error);
  }
}

/**
 * Check if Neo has permission to write files
 */
export function canWriteFiles(): boolean {
  return process.env.ALLOW_FILE_WRITE === 'true';
}

/**
 * Check if Neo has permission to execute shell commands
 */
export function canExecuteShell(): boolean {
  return process.env.ALLOW_SHELL_EXEC === 'true';
}

/**
 * Check if Anthropic API key is configured
 */
export function hasAnthropicKey(): boolean {
  const key = process.env.ANTHROPIC_API_KEY;
  return !!(key && key !== 'test-key-for-structure-check' && key !== 'your_api_key_here' && key.startsWith('sk-ant-'));
}
