/**
 * VPS Auto-Configuration Script
 *
 * Configure automatiquement un VPS pour Neo:
 * - Installation de Node.js 20+
 * - Installation d'Ollama pour embeddings locaux
 * - Configuration du firewall (ufw)
 * - Installation de fail2ban
 * - Configuration HTTPS avec Let's Encrypt
 * - Création du service systemd
 */

import { execSync, spawnSync } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';

// ===========================================================================
// TYPES
// ===========================================================================

interface VpsConfig {
  domain?: string;
  email?: string;
  port: number;
  enableHttps: boolean;
  installOllama: boolean;
  ollamaModel: string;
  enableFirewall: boolean;
  enableFail2ban: boolean;
  createSystemdService: boolean;
}

interface SetupResult {
  success: boolean;
  message: string;
  details?: string;
}

// ===========================================================================
// COLORS
// ===========================================================================

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[90m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function print(text: string): void {
  console.log(text);
}

function printStep(step: string): void {
  console.log(`\n${colors.cyan}▶${colors.reset} ${colors.bright}${step}${colors.reset}`);
}

function printSuccess(text: string): void {
  console.log(`  ${colors.green}✓${colors.reset} ${text}`);
}

function printError(text: string): void {
  console.log(`  ${colors.red}✗${colors.reset} ${text}`);
}

function printWarning(text: string): void {
  console.log(`  ${colors.yellow}⚠${colors.reset} ${text}`);
}

function printInfo(text: string): void {
  console.log(`  ${colors.dim}ℹ${colors.reset} ${text}`);
}

// ===========================================================================
// UTILITIES
// ===========================================================================

function isRoot(): boolean {
  return process.getuid?.() === 0;
}

function commandExists(cmd: string): boolean {
  try {
    execSync(`which ${cmd}`, { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

function getDistro(): string {
  try {
    const osRelease = fs.readFileSync('/etc/os-release', 'utf-8');
    const match = osRelease.match(/^ID=(.*)$/m);
    return match?.[1]?.replace(/"/g, '') || 'unknown';
  } catch {
    return 'unknown';
  }
}

function runCommand(cmd: string, options: { sudo?: boolean; silent?: boolean } = {}): SetupResult {
  const { sudo = false, silent = false } = options;
  const fullCmd = sudo && !isRoot() ? `sudo ${cmd}` : cmd;

  try {
    if (!silent) {
      printInfo(`Exécution: ${cmd}`);
    }
    execSync(fullCmd, { stdio: silent ? 'ignore' : 'inherit' });
    return { success: true, message: 'Commande exécutée avec succès' };
  } catch (error) {
    return {
      success: false,
      message: `Erreur: ${error}`,
    };
  }
}

// ===========================================================================
// SETUP FUNCTIONS
// ===========================================================================

/**
 * Installer Node.js 20+ si nécessaire
 */
export function setupNodeJs(): SetupResult {
  printStep('Installation de Node.js');

  // Vérifier version actuelle
  const currentVersion = process.version;
  const majorVersion = parseInt(currentVersion.slice(1).split('.')[0]);

  if (majorVersion >= 20) {
    printSuccess(`Node.js ${currentVersion} déjà installé`);
    return { success: true, message: 'Node.js déjà à jour' };
  }

  printWarning(`Node.js ${currentVersion} détecté, 20+ requis`);

  const distro = getDistro();

  if (distro === 'ubuntu' || distro === 'debian') {
    // Installation via NodeSource
    const result = runCommand(
      'curl -fsSL https://deb.nodesource.com/setup_20.x | bash -',
      { sudo: true }
    );

    if (!result.success) return result;

    return runCommand('apt-get install -y nodejs', { sudo: true });
  } else if (distro === 'centos' || distro === 'rhel' || distro === 'fedora') {
    const result = runCommand(
      'curl -fsSL https://rpm.nodesource.com/setup_20.x | bash -',
      { sudo: true }
    );

    if (!result.success) return result;

    return runCommand('yum install -y nodejs', { sudo: true });
  }

  return {
    success: false,
    message: `Distribution non supportée: ${distro}`,
    details: 'Installez Node.js 20+ manuellement depuis https://nodejs.org',
  };
}

/**
 * Installer Ollama pour les embeddings locaux
 */
export function setupOllama(model: string = 'nomic-embed-text'): SetupResult {
  printStep('Installation d\'Ollama');

  if (commandExists('ollama')) {
    printSuccess('Ollama déjà installé');
  } else {
    printInfo('Téléchargement et installation d\'Ollama...');

    const result = runCommand('curl -fsSL https://ollama.com/install.sh | sh');

    if (!result.success) {
      return result;
    }

    printSuccess('Ollama installé');
  }

  // Télécharger le modèle
  printInfo(`Téléchargement du modèle ${model}...`);

  const modelResult = runCommand(`ollama pull ${model}`);

  if (!modelResult.success) {
    printWarning(`Impossible de télécharger ${model}`);
    return modelResult;
  }

  printSuccess(`Modèle ${model} prêt`);

  // Démarrer le service Ollama si systemd est disponible
  if (fs.existsSync('/etc/systemd/system')) {
    runCommand('systemctl enable ollama', { sudo: true, silent: true });
    runCommand('systemctl start ollama', { sudo: true, silent: true });
    printSuccess('Service Ollama activé');
  }

  return { success: true, message: 'Ollama configuré' };
}

/**
 * Configurer le firewall (ufw)
 */
export function setupFirewall(config: { port: number; enableHttps: boolean }): SetupResult {
  printStep('Configuration du firewall (ufw)');

  if (!commandExists('ufw')) {
    printInfo('Installation de ufw...');
    const distro = getDistro();

    if (distro === 'ubuntu' || distro === 'debian') {
      runCommand('apt-get install -y ufw', { sudo: true });
    } else {
      return {
        success: false,
        message: 'ufw non disponible sur cette distribution',
      };
    }
  }

  // Règles de base
  runCommand('ufw default deny incoming', { sudo: true, silent: true });
  runCommand('ufw default allow outgoing', { sudo: true, silent: true });

  // SSH
  runCommand('ufw allow ssh', { sudo: true, silent: true });
  printSuccess('SSH autorisé');

  // Port de l'application
  runCommand(`ufw allow ${config.port}/tcp`, { sudo: true, silent: true });
  printSuccess(`Port ${config.port} autorisé`);

  // HTTPS si activé
  if (config.enableHttps) {
    runCommand('ufw allow 80/tcp', { sudo: true, silent: true });
    runCommand('ufw allow 443/tcp', { sudo: true, silent: true });
    printSuccess('Ports HTTP/HTTPS autorisés');
  }

  // Activer le firewall
  runCommand('ufw --force enable', { sudo: true, silent: true });
  printSuccess('Firewall activé');

  return { success: true, message: 'Firewall configuré' };
}

/**
 * Installer et configurer fail2ban
 */
export function setupFail2ban(): SetupResult {
  printStep('Configuration de fail2ban');

  if (!commandExists('fail2ban-client')) {
    printInfo('Installation de fail2ban...');

    const distro = getDistro();

    if (distro === 'ubuntu' || distro === 'debian') {
      runCommand('apt-get install -y fail2ban', { sudo: true });
    } else if (distro === 'centos' || distro === 'rhel' || distro === 'fedora') {
      runCommand('yum install -y fail2ban', { sudo: true });
    } else {
      return {
        success: false,
        message: 'Distribution non supportée pour fail2ban',
      };
    }
  }

  // Configuration basique
  const jailLocal = `
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = ssh
logpath = %(sshd_log)s
backend = %(sshd_backend)s
`;

  try {
    fs.writeFileSync('/etc/fail2ban/jail.local', jailLocal);
  } catch {
    // Essayer avec sudo via un fichier temporaire
    const tmpFile = '/tmp/jail.local';
    fs.writeFileSync(tmpFile, jailLocal);
    runCommand(`mv ${tmpFile} /etc/fail2ban/jail.local`, { sudo: true });
  }

  // Activer et démarrer
  runCommand('systemctl enable fail2ban', { sudo: true, silent: true });
  runCommand('systemctl restart fail2ban', { sudo: true, silent: true });

  printSuccess('fail2ban configuré et activé');

  return { success: true, message: 'fail2ban configuré' };
}

/**
 * Configurer HTTPS avec Let's Encrypt (certbot)
 */
export function setupHttps(domain: string, email: string): SetupResult {
  printStep('Configuration HTTPS (Let\'s Encrypt)');

  if (!domain) {
    return {
      success: false,
      message: 'Domaine requis pour HTTPS',
    };
  }

  // Installer certbot
  if (!commandExists('certbot')) {
    printInfo('Installation de certbot...');

    const distro = getDistro();

    if (distro === 'ubuntu' || distro === 'debian') {
      runCommand('apt-get install -y certbot', { sudo: true });
    } else {
      runCommand('snap install --classic certbot', { sudo: true });
    }
  }

  // Obtenir le certificat (mode standalone)
  printInfo(`Obtention du certificat pour ${domain}...`);

  const certbotCmd = email
    ? `certbot certonly --standalone -d ${domain} --email ${email} --agree-tos --non-interactive`
    : `certbot certonly --standalone -d ${domain} --register-unsafely-without-email --agree-tos --non-interactive`;

  const result = runCommand(certbotCmd, { sudo: true });

  if (!result.success) {
    printWarning('Échec de l\'obtention du certificat');
    printInfo('Assurez-vous que le domaine pointe vers ce serveur');
    return result;
  }

  // Créer un cron pour le renouvellement automatique
  runCommand('certbot renew --dry-run', { sudo: true, silent: true });

  printSuccess(`Certificat SSL obtenu pour ${domain}`);
  printInfo(`Certificat: /etc/letsencrypt/live/${domain}/fullchain.pem`);
  printInfo(`Clé privée: /etc/letsencrypt/live/${domain}/privkey.pem`);

  return { success: true, message: 'HTTPS configuré' };
}

/**
 * Créer un service systemd pour Neo
 */
export function createSystemdService(config: { port: number; workingDir: string }): SetupResult {
  printStep('Création du service systemd');

  const user = os.userInfo().username;
  const nodePathResult = spawnSync('which', ['node'], { encoding: 'utf-8' });
  const nodePath = nodePathResult.stdout?.trim() || '/usr/bin/node';

  const serviceContent = `[Unit]
Description=Neo AI Assistant
After=network.target

[Service]
Type=simple
User=${user}
WorkingDirectory=${config.workingDir}
ExecStart=${nodePath} dist/server.js
Restart=on-failure
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=neo-ai
Environment=NODE_ENV=production
Environment=PORT=${config.port}

[Install]
WantedBy=multi-user.target
`;

  const servicePath = '/etc/systemd/system/neo-ai.service';

  try {
    // Écrire via fichier temporaire
    const tmpFile = '/tmp/neo-ai.service';
    fs.writeFileSync(tmpFile, serviceContent);
    runCommand(`mv ${tmpFile} ${servicePath}`, { sudo: true });
    runCommand('chmod 644 ' + servicePath, { sudo: true });
  } catch (error) {
    return {
      success: false,
      message: `Erreur de création du service: ${error}`,
    };
  }

  // Recharger systemd
  runCommand('systemctl daemon-reload', { sudo: true, silent: true });
  runCommand('systemctl enable neo-ai', { sudo: true, silent: true });

  printSuccess('Service systemd créé');
  printInfo('Commandes utiles:');
  printInfo('  sudo systemctl start neo-ai');
  printInfo('  sudo systemctl stop neo-ai');
  printInfo('  sudo systemctl status neo-ai');
  printInfo('  sudo journalctl -u neo-ai -f');

  return { success: true, message: 'Service systemd créé' };
}

/**
 * Créer un reverse proxy nginx (optionnel)
 */
export function setupNginx(config: { domain: string; port: number; enableHttps: boolean }): SetupResult {
  printStep('Configuration de nginx');

  if (!commandExists('nginx')) {
    printInfo('Installation de nginx...');

    const distro = getDistro();

    if (distro === 'ubuntu' || distro === 'debian') {
      runCommand('apt-get install -y nginx', { sudo: true });
    } else if (distro === 'centos' || distro === 'rhel' || distro === 'fedora') {
      runCommand('yum install -y nginx', { sudo: true });
    }
  }

  const sslConfig = config.enableHttps
    ? `
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    ssl_certificate /etc/letsencrypt/live/${config.domain}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${config.domain}/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
`
    : `
    listen 80;
    listen [::]:80;
`;

  const nginxConfig = `
server {
    server_name ${config.domain};
    ${sslConfig}

    location / {
        proxy_pass http://127.0.0.1:${config.port};
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://127.0.0.1:${config.port};
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}

${config.enableHttps ? `
# Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name ${config.domain};
    return 301 https://$server_name$request_uri;
}
` : ''}
`;

  const configPath = `/etc/nginx/sites-available/${config.domain}`;
  const enabledPath = `/etc/nginx/sites-enabled/${config.domain}`;

  try {
    const tmpFile = `/tmp/${config.domain}.nginx`;
    fs.writeFileSync(tmpFile, nginxConfig);
    runCommand(`mv ${tmpFile} ${configPath}`, { sudo: true });

    // Activer le site
    runCommand(`ln -sf ${configPath} ${enabledPath}`, { sudo: true });

    // Supprimer le site par défaut
    runCommand('rm -f /etc/nginx/sites-enabled/default', { sudo: true, silent: true });

    // Tester la configuration
    const testResult = runCommand('nginx -t', { sudo: true, silent: true });
    if (!testResult.success) {
      printWarning('Configuration nginx invalide');
      return testResult;
    }

    // Recharger nginx
    runCommand('systemctl reload nginx', { sudo: true, silent: true });

    printSuccess('nginx configuré');
  } catch (error) {
    return {
      success: false,
      message: `Erreur de configuration nginx: ${error}`,
    };
  }

  return { success: true, message: 'nginx configuré' };
}

// ===========================================================================
// MAIN SETUP FUNCTION
// ===========================================================================

export async function runVpsSetup(config: VpsConfig): Promise<void> {
  print(`
${colors.cyan}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${colors.reset}
${colors.bright}       Configuration VPS pour Neo${colors.reset}
${colors.cyan}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${colors.reset}
`);

  const results: Array<{ step: string; result: SetupResult }> = [];

  // 1. Node.js
  results.push({
    step: 'Node.js',
    result: setupNodeJs(),
  });

  // 2. Ollama (si demandé)
  if (config.installOllama) {
    results.push({
      step: 'Ollama',
      result: setupOllama(config.ollamaModel),
    });
  }

  // 3. Firewall (si demandé)
  if (config.enableFirewall) {
    results.push({
      step: 'Firewall',
      result: setupFirewall({
        port: config.port,
        enableHttps: config.enableHttps,
      }),
    });
  }

  // 4. Fail2ban (si demandé)
  if (config.enableFail2ban) {
    results.push({
      step: 'Fail2ban',
      result: setupFail2ban(),
    });
  }

  // 5. HTTPS (si demandé avec domaine)
  if (config.enableHttps && config.domain) {
    results.push({
      step: 'HTTPS',
      result: setupHttps(config.domain, config.email || ''),
    });

    // 6. Nginx
    results.push({
      step: 'Nginx',
      result: setupNginx({
        domain: config.domain,
        port: config.port,
        enableHttps: config.enableHttps,
      }),
    });
  }

  // 7. Service systemd (si demandé)
  if (config.createSystemdService) {
    results.push({
      step: 'Systemd',
      result: createSystemdService({
        port: config.port,
        workingDir: process.cwd(),
      }),
    });
  }

  // Résumé
  print(`
${colors.cyan}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${colors.reset}
${colors.bright}       Résumé de la configuration${colors.reset}
${colors.cyan}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${colors.reset}
`);

  let allSuccess = true;

  for (const { step, result } of results) {
    if (result.success) {
      printSuccess(`${step}: ${result.message}`);
    } else {
      printError(`${step}: ${result.message}`);
      if (result.details) {
        printInfo(`  ${result.details}`);
      }
      allSuccess = false;
    }
  }

  print('');

  if (allSuccess) {
    printSuccess('Configuration VPS terminée avec succès!');

    if (config.createSystemdService) {
      print(`
${colors.dim}Pour démarrer Neo:${colors.reset}
  sudo systemctl start neo-ai

${colors.dim}Pour voir les logs:${colors.reset}
  sudo journalctl -u neo-ai -f
`);
    }

    if (config.domain && config.enableHttps) {
      print(`
${colors.dim}Neo accessible sur:${colors.reset}
  https://${config.domain}
`);
    }
  } else {
    printWarning('Certaines étapes ont échoué');
    printInfo('Vérifiez les erreurs ci-dessus et réessayez');
  }
}

