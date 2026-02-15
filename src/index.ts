/**
 * Point d'entrée principal de l'application
 */

// Load environment variables first
import { config as loadEnv } from './utils/env';
loadEnv();

import { Core } from './core';
import * as readline from 'readline';

async function main() {
  console.log('');
  console.log('╔═══════════════════════════════════════════════════════════╗');
  console.log('║                                                           ║');
  console.log('║     🧠 NEO - Assistant IA Multi-Agents                    ║');
  console.log('║                                                           ║');
  console.log('║     Agents:                                               ║');
  console.log('║       • VOX    - Interface utilisateur                    ║');
  console.log('║       • MEMORY - Mémoire 10 ans + Learning                ║');
  console.log('║       • BRAIN  - Orchestrateur intelligent                ║');
  console.log('║                                                           ║');
  console.log('╚═══════════════════════════════════════════════════════════╝');
  console.log('');

  // Créer et démarrer le système
  const core = new Core({ debug: process.env.DEBUG === 'true' });

  // Écouter les réponses
  core.on('response', (message: string) => {
    console.log('');
    console.log('┌─────────────────────────────────────────────────────────────');
    console.log('│ 🧠 Neo:');
    console.log('├─────────────────────────────────────────────────────────────');
    message.split('\n').forEach((line) => {
      console.log(`│ ${line}`);
    });
    console.log('└─────────────────────────────────────────────────────────────');
    console.log('');
    rl.prompt();
  });

  await core.start();

  // Interface readline pour le chat
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  rl.setPrompt('Vous > ');

  console.log('');
  console.log('Tapez votre message (ou "quit" pour quitter, "stats" pour les métriques)');
  console.log('');

  rl.prompt();

  rl.on('line', async (input) => {
    const trimmed = input.trim();

    if (!trimmed) {
      rl.prompt();
      return;
    }

    // Commandes spéciales
    if (trimmed.toLowerCase() === 'quit' || trimmed.toLowerCase() === 'exit') {
      console.log('\nArrêt du système...');
      await core.stop();
      rl.close();
      process.exit(0);
    }

    if (trimmed.toLowerCase() === 'stats') {
      const metrics = core.getMetrics();
      console.log('\n📊 Métriques:');
      console.log(JSON.stringify(metrics, null, 2));
      rl.prompt();
      return;
    }

    if (trimmed.toLowerCase() === 'reset') {
      core.resetConversation();
      console.log('\n🔄 Conversation réinitialisée');
      rl.prompt();
      return;
    }

    if (trimmed.startsWith('/remember ')) {
      const content = trimmed.replace('/remember ', '');
      await core.remember('fact', content, ['user_fact']);
      console.log('\n💾 Mémorisé!');
      rl.prompt();
      return;
    }

    if (trimmed.startsWith('/recall ')) {
      const query = trimmed.replace('/recall ', '');
      const results = await core.recall(query);
      console.log('\n🔍 Résultats:');
      console.log(JSON.stringify(results, null, 2));
      rl.prompt();
      return;
    }

    // Envoyer au système
    try {
      await core.chat(trimmed);
      // La réponse arrivera via l'événement 'response'
    } catch (error) {
      console.error('\n❌ Erreur:', error);
      rl.prompt();
    }
  });

  rl.on('close', async () => {
    await core.stop();
    process.exit(0);
  });

  // Gestion des signaux
  process.on('SIGINT', async () => {
    console.log('\n\nInterruption reçue, arrêt...');
    await core.stop();
    process.exit(0);
  });
}

main().catch(console.error);
