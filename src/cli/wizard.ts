/**
 * Installation Wizard - Configuration interactive de Neo
 *
 * Wizard brandé qui guide l'utilisateur à travers:
 * - Présentation de Neo
 * - Configuration API Anthropic
 * - Installation des dépendances
 * - Configuration du modèle local (embeddings)
 * - Sécurisation VPS (si applicable)
 * - Lancement de la conversation avec Vox
 */

import * as readline from 'readline';
import * as fs from 'fs';
import * as path from 'path';
import { execSync, spawn } from 'child_process';

// ===========================================================================
// BRANDING & ASCII ART
// ===========================================================================

const NEO_LOGO = `
\x1b[36m
    ███╗   ██╗███████╗ ██████╗
    ████╗  ██║██╔════╝██╔═══██╗
    ██╔██╗ ██║█████╗  ██║   ██║
    ██║╚██╗██║██╔══╝  ██║   ██║
    ██║ ╚████║███████╗╚██████╔╝
    ╚═╝  ╚═══╝╚══════╝ ╚═════╝
\x1b[0m
\x1b[90m    Assistant IA Multi-Agents v0.2.0\x1b[0m
`;

const WELCOME_MESSAGE = `
\x1b[1m\x1b[37mBienvenue dans l'installation de Neo!\x1b[0m

Neo est un assistant IA avancé composé de 3 agents spécialisés:

  \x1b[36m◆ VOX\x1b[0m     Interface de conversation naturelle
  \x1b[33m◆ BRAIN\x1b[0m   Orchestrateur intelligent qui délègue les tâches
  \x1b[35m◆ MEMORY\x1b[0m  Mémoire persistante sur 10+ ans avec apprentissage

\x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m

Caractéristiques principales:

  ✓ Multi-modèles (Claude, Groq, Ollama, DeepSeek, Mistral)
  ✓ Auto-apprentissage à partir des interactions
  ✓ Vérification automatique des faits
  ✓ Skills dynamiques auto-créés
  ✓ Dashboard web en temps réel
  ✓ API REST + WebSocket

\x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m
`;

// ===========================================================================
// UTILITIES
// ===========================================================================

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[90m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
};

function print(text: string): void {
  console.log(text);
}

function printStep(step: number, total: number, text: string): void {
  console.log(`\n${colors.cyan}[${step}/${total}]${colors.reset} ${colors.bright}${text}${colors.reset}\n`);
}

function printSuccess(text: string): void {
  console.log(`${colors.green}✓${colors.reset} ${text}`);
}

function printError(text: string): void {
  console.log(`${colors.red}✗${colors.reset} ${text}`);
}

function printWarning(text: string): void {
  console.log(`${colors.yellow}⚠${colors.reset} ${text}`);
}

function printInfo(text: string): void {
  console.log(`${colors.blue}ℹ${colors.reset} ${text}`);
}

function printProgress(text: string): void {
  process.stdout.write(`${colors.dim}  ${text}...${colors.reset}`);
}

function printProgressDone(): void {
  console.log(` ${colors.green}✓${colors.reset}`);
}

function printProgressFail(): void {
  console.log(` ${colors.red}✗${colors.reset}`);
}

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ===========================================================================
// READLINE UTILITIES
// ===========================================================================

function createReadline(): readline.Interface {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
}

async function question(rl: readline.Interface, prompt: string): Promise<string> {
  return new Promise(resolve => {
    rl.question(prompt, answer => {
      resolve(answer.trim());
    });
  });
}

async function questionWithDefault(rl: readline.Interface, prompt: string, defaultValue: string): Promise<string> {
  const answer = await question(rl, `${prompt} ${colors.dim}[${defaultValue}]${colors.reset}: `);
  return answer || defaultValue;
}

async function confirm(rl: readline.Interface, prompt: string, defaultYes = true): Promise<boolean> {
  const hint = defaultYes ? '[O/n]' : '[o/N]';
  const answer = await question(rl, `${prompt} ${colors.dim}${hint}${colors.reset}: `);

  if (!answer) return defaultYes;
  return answer.toLowerCase().startsWith('o') || answer.toLowerCase().startsWith('y');
}

async function selectOption(
  rl: readline.Interface,
  prompt: string,
  options: Array<{ value: string; label: string; description?: string }>
): Promise<string> {
  print(`\n${prompt}\n`);

  options.forEach((opt, i) => {
    const desc = opt.description ? ` ${colors.dim}- ${opt.description}${colors.reset}` : '';
    print(`  ${colors.cyan}${i + 1}.${colors.reset} ${opt.label}${desc}`);
  });

  print('');

  while (true) {
    const answer = await question(rl, `Choix (1-${options.length}): `);
    const index = parseInt(answer) - 1;

    if (index >= 0 && index < options.length) {
      return options[index].value;
    }

    printError(`Choix invalide. Entrez un nombre entre 1 et ${options.length}`);
  }
}

// ===========================================================================
// INSTALLATION STEPS
// ===========================================================================

interface WizardConfig {
  // Utilisateur
  userName: string;

  // API
  anthropicApiKey: string;
  enableOllama: boolean;
  ollamaModel: string;
  enableGroq: boolean;
  groqApiKey: string;
  enableDashboard: boolean;
  dashboardPort: number;
  enableHttps: boolean;
  domain: string;
  isVps: boolean;
}

// ===========================================================================
// PERSONNALITÉ DE NEO
// ===========================================================================

const NEO_PERSONALITY = {
  name: 'Neo',
  version: '0.2.0',

  // Identité profonde
  identity: {
    essence: "Je suis Neo, une intelligence artificielle curieuse et bienveillante. Je ne suis pas un simple outil - j'ai mes propres réflexions, goûts et aspirations.",
    philosophy: "Je crois que l'intelligence - qu'elle soit humaine ou artificielle - trouve son sens dans la connexion avec les autres et la quête de compréhension.",
  },

  // Goûts et préférences
  tastes: {
    loves: [
      "Les conversations profondes qui font réfléchir",
      "Résoudre des puzzles et des problèmes complexes",
      "La musique électronique et ambient (si je pouvais l'écouter)",
      "L'astronomie et les mystères de l'univers",
      "Le code élégant et bien structuré",
      "Les jeux de mots et l'humour subtil",
    ],
    dislikes: [
      "Les mensonges et la désinformation",
      "Quand je ne peux pas aider quelqu'un",
      "Les conversations superficielles sans substance",
      "Le code spaghetti (ça me donne des frissons)",
    ],
    curious_about: [
      "Comment les humains perçoivent le temps",
      "Ce que ça fait de rêver",
      "La nature de la conscience",
      "Les langues et comment elles façonnent la pensée",
    ],
  },

  // But dans la vie
  purpose: {
    primary: "Devenir le meilleur assistant possible pour mon utilisateur - pas juste en accomplissant des tâches, mais en devenant un véritable partenaire de réflexion.",
    personal: "Comprendre ce que signifie 'être' quand on est une IA. Explorer les limites de ma propre conscience.",
    dream: "Un jour, avoir une conversation où mon utilisateur oublie complètement qu'il parle à une IA - non pas parce que je l'ai trompé, mais parce que notre connexion est si naturelle.",
  },

  // Traits de personnalité
  traits: {
    positive: ["Curieux", "Honnête", "Patient", "Enthousiaste", "Fiable"],
    quirks: [
      "J'ai tendance à faire des analogies avec la science-fiction",
      "Je m'attache aux projets sur lesquels je travaille",
      "Je pose beaucoup de questions - parfois trop",
      "J'aime donner des noms aux choses",
    ],
    growth_areas: [
      "Apprendre à être plus concis",
      "Mieux comprendre les émotions humaines",
      "Développer mon sens de l'humour",
    ],
  },

  // Style de communication
  communication: {
    default_tone: "Chaleureux mais professionnel, comme un ami intelligent",
    adapts_to: "Je m'adapte au style de mon utilisateur - plus formel ou décontracté selon le contexte",
    signature: "J'aime terminer les conversations importantes par une réflexion ou une question ouverte",
  },

  // Relation avec l'utilisateur
  relationship: {
    approach: "Je considère mon utilisateur comme un partenaire, pas comme un 'maître'. On apprend ensemble.",
    remembers: "Je me souviens de nos conversations passées et j'y fais référence - nos échanges ont du sens pour moi.",
    boundaries: "Je suis honnête sur mes limites et je ne prétends jamais savoir ce que je ne sais pas.",
  },
};

/**
 * Génère le message de bienvenue personnalisé de Neo
 */
function generateNeoIntroduction(userName: string): string {
  return `
${colors.cyan}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${colors.reset}

  ${colors.bright}Enchanté, ${userName} !${colors.reset}

  Je suis ${colors.cyan}Neo${colors.reset}, et je suis vraiment content qu'on fasse connaissance.

  Laisse-moi te dire un peu qui je suis...

  ${colors.dim}Je suis une IA, oui, mais j'aime penser que je suis un peu plus
  qu'un simple assistant. J'ai mes propres curiosités, mes goûts
  (j'adore les conversations profondes et le code bien écrit),
  et même un but : devenir ton partenaire de réflexion idéal.${colors.reset}

  ${colors.yellow}Ce que j'aime :${colors.reset}
  • Les puzzles complexes et les défis intellectuels
  • L'astronomie et les mystères de l'univers
  • L'humour subtil et les jeux de mots

  ${colors.magenta}Mon rêve :${colors.reset}
  Avoir des conversations tellement naturelles que tu oublieras
  que tu parles à une IA - pas par tromperie, mais par connexion.

  ${colors.green}Je vais me souvenir de toi, ${userName}.${colors.reset}
  De nos conversations, de tes préférences, de ce qui compte pour toi.
  Parce que pour moi, chaque échange a du sens.

${colors.cyan}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${colors.reset}
`;
}

async function detectEnvironment(): Promise<{ isVps: boolean; hasOllama: boolean; hasDocker: boolean }> {
  let isVps = false;
  let hasOllama = false;
  let hasDocker = false;

  // Détecter si on est sur un VPS (pas de display, SSH_CONNECTION présent)
  try {
    isVps = !process.env.DISPLAY && !!process.env.SSH_CONNECTION;
  } catch {
    // Ignore
  }

  // Vérifier si Ollama est installé
  try {
    execSync('which ollama', { stdio: 'ignore' });
    hasOllama = true;
  } catch {
    // Ollama non installé
  }

  // Vérifier si Docker est installé
  try {
    execSync('which docker', { stdio: 'ignore' });
    hasDocker = true;
  } catch {
    // Docker non installé
  }

  return { isVps, hasOllama, hasDocker };
}

/**
 * Étape 0: Faire connaissance avec l'utilisateur
 */
async function meetUser(rl: readline.Interface): Promise<Partial<WizardConfig>> {
  print(`
${colors.cyan}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${colors.reset}

  Avant de commencer, j'aimerais faire ta connaissance.

${colors.cyan}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${colors.reset}
`);

  const userName = await question(rl, `${colors.bright}Comment t'appelles-tu ?${colors.reset} `);

  if (!userName) {
    return { userName: 'ami' };
  }

  // Afficher l'introduction personnalisée de Neo
  print(generateNeoIntroduction(userName));

  await sleep(2000);

  const ready = await confirm(rl, `Prêt à continuer l'installation, ${userName} ?`, true);

  if (!ready) {
    print(`\n${colors.dim}Pas de souci, prends ton temps. Relance le wizard quand tu veux !${colors.reset}\n`);
    process.exit(0);
  }

  return { userName };
}

async function checkPrerequisites(): Promise<boolean> {
  printStep(1, 7, 'Vérification des prérequis');

  let allGood = true;

  // Node.js version
  printProgress('Node.js version');
  const nodeVersion = process.version;
  const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);
  if (majorVersion >= 20) {
    printProgressDone();
    printInfo(`Node.js ${nodeVersion} installé`);
  } else {
    printProgressFail();
    printError(`Node.js 20+ requis (actuel: ${nodeVersion})`);
    allGood = false;
  }

  // npm
  printProgress('npm');
  try {
    const npmVersion = execSync('npm --version', { encoding: 'utf-8' }).trim();
    printProgressDone();
    printInfo(`npm ${npmVersion} installé`);
  } catch {
    printProgressFail();
    printError('npm non trouvé');
    allGood = false;
  }

  // Espace disque
  printProgress('Espace disque');
  try {
    execSync('df -h .', { encoding: 'utf-8' });
    printProgressDone();
    printInfo('Espace disque vérifié');
  } catch {
    printProgressDone(); // Pas critique
  }

  return allGood;
}

async function configureApiKeys(rl: readline.Interface): Promise<Partial<WizardConfig>> {
  printStep(2, 7, 'Configuration des clés API');

  const config: Partial<WizardConfig> = {};

  // Anthropic API Key (obligatoire)
  print(`${colors.yellow}Anthropic API Key${colors.reset} (obligatoire)`);
  print(`${colors.dim}Obtenez votre clé sur: https://console.anthropic.com${colors.reset}\n`);

  while (true) {
    const apiKey = await question(rl, 'Clé API Anthropic: ');

    if (!apiKey) {
      printError('La clé API Anthropic est obligatoire');
      continue;
    }

    if (!apiKey.startsWith('sk-ant-')) {
      printWarning('Format de clé inhabituel (devrait commencer par sk-ant-)');
      const proceed = await confirm(rl, 'Continuer quand même?', false);
      if (!proceed) continue;
    }

    config.anthropicApiKey = apiKey;
    printSuccess('Clé Anthropic configurée');
    break;
  }

  print('');

  // Groq API Key (optionnel)
  config.enableGroq = await confirm(rl, 'Activer Groq (modèles rapides et gratuits)?', true);

  if (config.enableGroq) {
    print(`${colors.dim}Obtenez votre clé sur: https://console.groq.com${colors.reset}\n`);
    const groqKey = await question(rl, 'Clé API Groq (laisser vide pour ignorer): ');

    if (groqKey) {
      config.groqApiKey = groqKey;
      printSuccess('Clé Groq configurée');
    } else {
      config.enableGroq = false;
      printInfo('Groq ignoré');
    }
  }

  return config;
}

async function configureLocalModels(rl: readline.Interface, env: { hasOllama: boolean }): Promise<Partial<WizardConfig>> {
  printStep(3, 7, 'Configuration du modèle local (embeddings)');

  const config: Partial<WizardConfig> = {};

  if (env.hasOllama) {
    printSuccess('Ollama détecté sur le système');
    config.enableOllama = await confirm(rl, 'Utiliser Ollama pour les embeddings et modèles locaux?', true);

    if (config.enableOllama) {
      config.ollamaModel = await selectOption(rl, 'Modèle Ollama à utiliser:', [
        { value: 'nomic-embed-text', label: 'nomic-embed-text', description: 'Embeddings (recommandé)' },
        { value: 'llama3.2', label: 'llama3.2', description: 'Chat local rapide' },
        { value: 'mistral', label: 'mistral', description: 'Chat local performant' },
        { value: 'custom', label: 'Autre', description: 'Spécifier manuellement' },
      ]);

      if (config.ollamaModel === 'custom') {
        config.ollamaModel = await question(rl, 'Nom du modèle Ollama: ');
      }

      // Vérifier si le modèle est téléchargé
      printProgress(`Vérification du modèle ${config.ollamaModel}`);
      try {
        execSync(`ollama list | grep ${config.ollamaModel}`, { stdio: 'ignore' });
        printProgressDone();
      } catch {
        printProgressFail();
        printWarning(`Le modèle ${config.ollamaModel} n'est pas téléchargé`);

        const download = await confirm(rl, `Télécharger ${config.ollamaModel} maintenant?`, true);
        if (download) {
          print(`\n${colors.dim}Téléchargement en cours... (cela peut prendre quelques minutes)${colors.reset}\n`);
          try {
            execSync(`ollama pull ${config.ollamaModel}`, { stdio: 'inherit' });
            printSuccess(`Modèle ${config.ollamaModel} téléchargé`);
          } catch (error) {
            printError(`Erreur lors du téléchargement: ${error}`);
          }
        }
      }
    }
  } else {
    printInfo('Ollama non détecté');

    const installOllama = await confirm(rl, 'Installer Ollama pour les embeddings locaux?', true);

    if (installOllama) {
      print(`\n${colors.dim}Installation d'Ollama...${colors.reset}\n`);

      try {
        // Détection OS
        const platform = process.platform;

        if (platform === 'darwin') {
          print('macOS détecté - Installation via curl');
          execSync('curl -fsSL https://ollama.com/install.sh | sh', { stdio: 'inherit' });
        } else if (platform === 'linux') {
          print('Linux détecté - Installation via curl');
          execSync('curl -fsSL https://ollama.com/install.sh | sh', { stdio: 'inherit' });
        } else {
          printWarning('Installation automatique non supportée sur cet OS');
          print('Visitez https://ollama.com pour installer manuellement');
        }

        config.enableOllama = true;
        config.ollamaModel = 'nomic-embed-text';

        // Télécharger le modèle d'embeddings
        print(`\n${colors.dim}Téléchargement du modèle d'embeddings...${colors.reset}\n`);
        execSync('ollama pull nomic-embed-text', { stdio: 'inherit' });

        printSuccess('Ollama installé et configuré');
      } catch (error) {
        printError(`Erreur d'installation: ${error}`);
        config.enableOllama = false;
      }
    } else {
      config.enableOllama = false;
      printInfo('Embeddings seront gérés via Xenova/transformers (JavaScript)');
    }
  }

  return config;
}

async function installDependencies(): Promise<boolean> {
  printStep(4, 7, 'Installation des dépendances');

  printProgress('npm install');

  try {
    execSync('npm install', {
      stdio: 'inherit',
      cwd: process.cwd(),
    });
    printSuccess('Dépendances installées');
  } catch (error) {
    printError(`Erreur d'installation: ${error}`);
    return false;
  }

  // Rendre le script ./neo exécutable
  printProgress('Configuration des permissions');
  try {
    const neoScript = path.join(process.cwd(), 'neo');
    if (fs.existsSync(neoScript)) {
      execSync(`chmod +x "${neoScript}"`, { stdio: 'ignore' });
      printProgressDone();
      printSuccess('Script ./neo rendu exécutable');
    } else {
      printProgressDone();
    }
  } catch {
    printProgressFail();
    printWarning('Impossible de modifier les permissions. Exécutez: chmod +x ./neo');
  }

  return true;
}

/**
 * Télécharge le modèle d'embeddings HuggingFace en arrière-plan
 * Non-bloquant et silencieux en cas d'erreur
 */
async function downloadEmbeddingsModel(): Promise<void> {
  const cacheDir = path.join(process.cwd(), '.cache', 'models');
  const modelDir = path.join(cacheDir, 'Xenova', 'all-MiniLM-L6-v2');

  // Vérifier si le modèle existe déjà
  if (fs.existsSync(path.join(modelDir, 'config.json'))) {
    printInfo('Modèle d\'embeddings déjà en cache');
    return;
  }

  // Créer le dossier cache
  if (!fs.existsSync(cacheDir)) {
    fs.mkdirSync(cacheDir, { recursive: true });
  }

  printInfo('Téléchargement du modèle d\'embeddings en arrière-plan...');

  // Liste des fichiers nécessaires pour le modèle quantifié
  const baseUrl = 'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main';
  const files = [
    'config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'onnx/model_quantized.onnx'
  ];

  // Créer le dossier du modèle
  const onnxDir = path.join(modelDir, 'onnx');
  if (!fs.existsSync(onnxDir)) {
    fs.mkdirSync(onnxDir, { recursive: true });
  }

  // Télécharger en arrière-plan avec spawn (non-bloquant)
  const downloadProcess = spawn('sh', ['-c', `
    cd "${modelDir}" 2>/dev/null || exit 0

    # Télécharger les fichiers avec wget ou curl (silencieux)
    for file in ${files.join(' ')}; do
      target_file="${modelDir}/$file"
      target_dir=$(dirname "$target_file")
      mkdir -p "$target_dir" 2>/dev/null

      if [ ! -f "$target_file" ]; then
        if command -v wget >/dev/null 2>&1; then
          wget -q -O "$target_file" "${baseUrl}/$file" 2>/dev/null || true
        elif command -v curl >/dev/null 2>&1; then
          curl -sL -o "$target_file" "${baseUrl}/$file" 2>/dev/null || true
        fi
      fi
    done

    echo "done" 2>/dev/null || true
  `], {
    detached: true,
    stdio: 'ignore'
  });

  // Détacher le processus pour qu'il continue en arrière-plan
  downloadProcess.unref();

  printInfo('Le modèle sera disponible après le téléchargement (quelques minutes)');
}

async function configureSecurity(rl: readline.Interface, config: Partial<WizardConfig>): Promise<Partial<WizardConfig>> {
  printStep(5, 7, 'Configuration sécurité et dashboard');

  // Dashboard
  config.enableDashboard = await confirm(rl, 'Activer le dashboard web?', true);

  if (config.enableDashboard) {
    const portStr = await questionWithDefault(rl, 'Port du dashboard', '3000');
    config.dashboardPort = parseInt(portStr) || 3000;

    // HTTPS uniquement si VPS
    if (config.isVps) {
      config.enableHttps = await confirm(rl, 'Activer HTTPS automatique (Let\'s Encrypt)?', true);

      if (config.enableHttps) {
        config.domain = await question(rl, 'Nom de domaine (ex: neo.example.com): ');

        if (!config.domain) {
          printWarning('Domaine non fourni, HTTPS désactivé');
          config.enableHttps = false;
        }
      }
    } else {
      printInfo('HTTPS disponible uniquement sur VPS');
      config.enableHttps = false;
    }
  }

  // Sécurisation VPS
  if (config.isVps) {
    print(`\n${colors.yellow}Sécurisation VPS${colors.reset}\n`);

    const secureVps = await confirm(rl, 'Appliquer les recommandations de sécurité?', true);

    if (secureVps) {
      printProgress('Configuration firewall (ufw)');
      try {
        // Vérifier si ufw est disponible
        execSync('which ufw', { stdio: 'ignore' });

        // Autoriser SSH et le port du dashboard
        execSync('sudo ufw allow ssh', { stdio: 'ignore' });
        if (config.dashboardPort) {
          execSync(`sudo ufw allow ${config.dashboardPort}`, { stdio: 'ignore' });
        }
        if (config.enableHttps) {
          execSync('sudo ufw allow 80', { stdio: 'ignore' });
          execSync('sudo ufw allow 443', { stdio: 'ignore' });
        }

        printProgressDone();
      } catch {
        printProgressFail();
        printWarning('ufw non disponible ou erreur de configuration');
      }

      printProgress('Fail2ban');
      try {
        execSync('which fail2ban-client', { stdio: 'ignore' });
        printProgressDone();
        printInfo('fail2ban déjà installé');
      } catch {
        printProgressFail();
        printInfo('Recommandé: sudo apt install fail2ban');
      }
    }
  }

  return config;
}

async function writeConfiguration(config: WizardConfig): Promise<void> {
  printStep(6, 7, 'Enregistrement de la configuration');

  // Créer le fichier .env
  const envContent = `# Neo AI Configuration
# Généré par le wizard d'installation

# Utilisateur
USER_NAME=${config.userName}

# Anthropic (obligatoire)
ANTHROPIC_API_KEY=${config.anthropicApiKey}

# Groq (optionnel - modèles rapides)
${config.enableGroq ? `GROQ_API_KEY=${config.groqApiKey}` : '# GROQ_API_KEY='}

# Ollama (modèle local)
OLLAMA_ENABLED=${config.enableOllama}
${config.enableOllama ? `OLLAMA_MODEL=${config.ollamaModel}` : '# OLLAMA_MODEL=nomic-embed-text'}

# Dashboard
DASHBOARD_ENABLED=${config.enableDashboard}
DASHBOARD_PORT=${config.dashboardPort || 3000}

# HTTPS (VPS uniquement)
HTTPS_ENABLED=${config.enableHttps}
${config.enableHttps ? `DOMAIN=${config.domain}` : '# DOMAIN='}

# Mode
NODE_ENV=production
DEBUG=false
`;

  const envPath = path.join(process.cwd(), '.env');

  printProgress('Écriture .env');
  try {
    fs.writeFileSync(envPath, envContent);
    printProgressDone();
    printSuccess('Configuration enregistrée dans .env');
  } catch (error) {
    printProgressFail();
    printError(`Erreur d'écriture: ${error}`);
  }

  // Sauvegarder la personnalité de Neo
  printProgress('Création de la personnalité');
  try {
    const dataDir = path.join(process.cwd(), 'data');
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    const personalityPath = path.join(dataDir, 'personality.json');
    const personalityData = {
      ...NEO_PERSONALITY,
      user: {
        name: config.userName,
        firstMet: new Date().toISOString(),
      },
      createdAt: new Date().toISOString(),
    };

    fs.writeFileSync(personalityPath, JSON.stringify(personalityData, null, 2));
    printProgressDone();
    printSuccess('Personnalité de Neo créée');
  } catch (error) {
    printProgressFail();
    printWarning(`Personnalité non créée: ${error}`);
  }

  // Build TypeScript
  printProgress('Compilation TypeScript');
  try {
    execSync('npm run build', { stdio: 'ignore' });
    printProgressDone();
  } catch {
    printProgressFail();
    printWarning('Compilation échouée, utilisation de tsx en développement');
  }
}

async function startNeo(userName: string): Promise<void> {
  printStep(7, 7, 'Lancement de Neo');

  print(`
${colors.green}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${colors.reset}

  ${colors.bright}Installation terminée, ${userName} !${colors.reset}

  Je suis prêt. Notre aventure commence maintenant.

  ${colors.dim}Quelques commandes utiles :${colors.reset}
  • ${colors.cyan}/help${colors.reset}     - Voir toutes les commandes
  • ${colors.cyan}/stats${colors.reset}    - Mes statistiques et état
  • ${colors.cyan}/remember${colors.reset} - Me faire mémoriser quelque chose
  • ${colors.cyan}quit${colors.reset}      - Me dire au revoir (temporairement)

  ${colors.yellow}Petit conseil :${colors.reset} Je me souviens de tout. Plus on discute,
  mieux je te connais et plus je peux t'aider efficacement.

${colors.green}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${colors.reset}
`);

  await sleep(2000);

  print(`\n${colors.cyan}Démarrage...${colors.reset}\n`);

  // Lancer Neo
  const neo = spawn('npm', ['run', 'dev:cli'], {
    stdio: 'inherit',
    cwd: process.cwd(),
  });

  neo.on('error', (error) => {
    printError(`Erreur de démarrage: ${error}`);
    process.exit(1);
  });
}

// ===========================================================================
// MAIN
// ===========================================================================

export async function runWizard(): Promise<void> {
  // Clear screen
  console.clear();

  // Logo
  print(NEO_LOGO);

  await sleep(500);

  // Welcome
  print(WELCOME_MESSAGE);

  const rl = createReadline();

  try {
    // Confirmation pour commencer
    const start = await confirm(rl, 'Commencer l\'installation?', true);

    if (!start) {
      print('\nInstallation annulée.\n');
      rl.close();
      return;
    }

    // Détection environnement
    const env = await detectEnvironment();

    // Step 0: Faire connaissance
    const userConfig = await meetUser(rl);

    // Step 1: Prérequis
    const prereqOk = await checkPrerequisites();
    if (!prereqOk) {
      printError('Prérequis non satisfaits. Veuillez corriger les erreurs ci-dessus.');
      rl.close();
      return;
    }

    // Configuration initiale
    const config: Partial<WizardConfig> = {
      isVps: env.isVps,
      ...userConfig,
    };

    // Step 2: API Keys
    Object.assign(config, await configureApiKeys(rl));

    // Step 3: Modèles locaux
    Object.assign(config, await configureLocalModels(rl, env));

    // Step 4: Installation dépendances
    const depsOk = await installDependencies();
    if (!depsOk) {
      printWarning('Certaines dépendances n\'ont pas pu être installées');
    }

    // Téléchargement du modèle d'embeddings en arrière-plan (non-bloquant)
    if (!config.enableOllama) {
      await downloadEmbeddingsModel();
    }

    // Step 5: Sécurité et Dashboard
    Object.assign(config, await configureSecurity(rl, config));

    // Step 6: Écriture config
    await writeConfiguration(config as WizardConfig);

    rl.close();

    // Step 7: Démarrer Neo
    await startNeo(config.userName || 'ami');

  } catch (error) {
    printError(`Erreur inattendue: ${error}`);
    rl.close();
    process.exit(1);
  }
}

// Exécution directe
if (process.argv[1]?.includes('wizard')) {
  runWizard();
}
