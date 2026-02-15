/**
 * Environment configuration loader
 * Loads .env file into process.env
 */

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Simple debug flag (logger not initialized yet)
const isDebug = () => process.env.DEBUG === 'true' || process.env.VERBOSE === 'true';

/**
 * Load environment variables from .env file
 */
export function config(): void {
  // Find .env file (look in project root)
  const projectRoot = resolve(__dirname, '..', '..');
  const envPath = resolve(projectRoot, '.env');

  if (!existsSync(envPath)) {
    if (isDebug()) console.warn('[Env] No .env file found at', envPath);
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

    // Validate critical variables (only warn if not configured)
    if (!process.env.ANTHROPIC_API_KEY ||
        process.env.ANTHROPIC_API_KEY === 'test-key-for-structure-check' ||
        process.env.ANTHROPIC_API_KEY === 'your_api_key_here') {
      console.warn('⚠️  ANTHROPIC_API_KEY non configurée - ./neo config');
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

/**
 * Check if the key is an OAuth token (from Claude Pro/Max subscription)
 * OAuth tokens start with sk-ant-oat
 */
export function isOAuthToken(): boolean {
  const key = process.env.ANTHROPIC_API_KEY;
  return !!(key && key.startsWith('sk-ant-oat'));
}

/**
 * Check if the key is a standard API key (from console.anthropic.com)
 * Standard API keys start with sk-ant-api
 */
export function isStandardApiKey(): boolean {
  const key = process.env.ANTHROPIC_API_KEY;
  return !!(key && key.startsWith('sk-ant-api'));
}

/**
 * Get the authentication type for Anthropic
 */
export function getAnthropicAuthType(): 'oauth' | 'api_key' | 'none' {
  if (isOAuthToken()) {
    return 'oauth';
  }
  if (isStandardApiKey()) {
    return 'api_key';
  }
  if (hasAnthropicKey()) {
    return 'api_key'; // Assume API key for other sk-ant- prefixes
  }
  return 'none';
}
