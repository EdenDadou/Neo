/**
 * Serveur principal avec Gateway
 *
 * Point d'entrée pour le mode serveur (API + WebSocket)
 */

import { Core } from './core';
import { Gateway } from './interfaces/gateway';

async function main() {
  console.log('');
  console.log('╔═══════════════════════════════════════════════════════════╗');
  console.log('║                                                           ║');
  console.log('║     🧠 NEO - Assistant IA Multi-Agents                    ║');
  console.log('║                                                           ║');
  console.log('║     Core Agents:                                          ║');
  console.log('║       • VOX    - Interface utilisateur                    ║');
  console.log('║       • MEMORY - Mémoire 10 ans + Learning                ║');
  console.log('║       • BRAIN  - Orchestrateur intelligent                ║');
  console.log('║                                                           ║');
  console.log('║     Capacités:                                            ║');
  console.log('║       • Recherche web native                              ║');
  console.log('║       • Modèles gratuits/pas chers                        ║');
  console.log('║       • Personnalité persistante                          ║');
  console.log('║                                                           ║');
  console.log('╚═══════════════════════════════════════════════════════════╝');
  console.log('');

  // Créer le Core
  const core = new Core({ debug: process.env.DEBUG === 'true' });

  // Créer la Gateway
  const gateway = new Gateway(core);

  // Démarrer
  await core.start();
  await gateway.start();

  console.log('');
  console.log('📡 API Endpoints:');
  console.log('   POST /api/auth/register  - Créer un compte');
  console.log('   POST /api/auth/login     - Se connecter');
  console.log('   POST /api/chat           - Envoyer un message');
  console.log('   POST /api/memory         - Stocker une mémoire');
  console.log('   GET  /api/memory/search  - Rechercher en mémoire');
  console.log('   GET  /api/stats          - Statistiques');
  console.log('   POST /api/correct        - Enregistrer une correction');
  console.log('');
  console.log('🔌 WebSocket: ws://localhost:3001?token=YOUR_JWT');
  console.log('');

  // Gestion de l'arrêt
  const shutdown = async () => {
    console.log('\n\nArrêt en cours...');
    await gateway.stop();
    await core.stop();
    process.exit(0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main().catch(console.error);
