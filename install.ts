#!/usr/bin/env npx tsx
import 'node:process';
/**
 * Neo Installation Wizard
 *
 * Wizard d'installation brandé pour Neo.
 *
 * Usage:
 *   npx tsx install.ts
 *   npm run setup
 *
 * Ce wizard vous guide à travers:
 * - Configuration des clés API (Anthropic, Groq, etc.)
 * - Installation des modèles locaux (Ollama)
 * - Sécurisation VPS (si applicable)
 * - Configuration du dashboard
 * - Lancement de Neo
 */

import { runWizard } from './src/cli/wizard';

runWizard().catch((error) => {
  console.error('\x1b[31mInstallation failed:\x1b[0m', error);
  process.exit(1);
});
