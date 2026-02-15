/**
 * Serveur principal avec Gateway
 *
 * Point d'entrée pour le mode serveur (API + WebSocket)
 */

// Load environment variables first
import { config as loadEnv } from './utils/env';
loadEnv();

import { Core } from './core';
import { Gateway } from './interfaces/gateway';
import { initLogger, logger } from './utils/logger';

async function main() {
  // Initialiser le logger (mode verbose si DEBUG=true)
  initLogger();

  // Bannière toujours visible
  console.log('');
  console.log('🧠 NEO - Assistant IA Multi-Agents');
  console.log('───────────────────────────────────');

  // Créer le Core (debug mode active les logs système)
  const core = new Core({ debug: process.env.DEBUG === 'true' });

  // Créer la Gateway
  const gateway = new Gateway(core);

  // Démarrer
  await core.start();
  await gateway.start();

  const port = process.env.PORT || 3001;
  console.log(`\n✅ Serveur prêt sur http://localhost:${port}`);
  console.log('   WebSocket: ws://localhost:' + port);
  console.log('\n💡 Utilisez DEBUG=true pour voir tous les logs système\n');

  // Gestion de l'arrêt
  const shutdown = async () => {
    logger.debug('Server', 'Arrêt en cours...');
    await gateway.stop();
    await core.stop();
    process.exit(0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main().catch(console.error);
