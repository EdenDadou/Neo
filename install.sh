#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  Neo Core — Installation automatique
# ═══════════════════════════════════════════════════════════
#
#  Usage (une seule commande) :
#    curl -fsSL https://raw.githubusercontent.com/EdenDadou/Neo/main/install.sh | sudo bash
#
#  Ou manuellement :
#    git clone https://github.com/EdenDadou/Neo.git /opt/neo-core
#    cd /opt/neo-core && sudo bash install.sh
#
# ═══════════════════════════════════════════════════════════

# Ne PAS utiliser set -e — on gère les erreurs manuellement
set -uo pipefail

# ─── Couleurs ─────────────────────────────────────────────
CYAN='\033[96m'
GREEN='\033[92m'
YELLOW='\033[93m'
RED='\033[91m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

INSTALL_DIR="/opt/neo-core"
VENV_DIR="${INSTALL_DIR}/.venv"
NEO_USER="neo"
LOG_FILE="/tmp/neo-install.log"

# ─── Fonctions utilitaires ────────────────────────────────

log_info()  { echo -e "  ${GREEN}✓${RESET} $1"; }
log_warn()  { echo -e "  ${YELLOW}⚠${RESET} $1"; }
log_error() { echo -e "  ${RED}✗${RESET} $1"; }
log_step()  { echo -e "\n${CYAN}${BOLD}  [$1/$2] $3${RESET}\n"; }

# Exécuter une commande avec log visible en cas d'erreur
run_or_fail() {
    local description="$1"
    shift
    echo -e "  ${DIM}⧗ ${description}...${RESET}"
    if "$@" >> "$LOG_FILE" 2>&1; then
        log_info "$description"
        return 0
    else
        log_error "$description — ÉCHEC"
        echo -e "  ${DIM}  Voir les détails: tail -50 ${LOG_FILE}${RESET}"
        return 1
    fi
}

# Exécuter une commande, continuer même en cas d'erreur
run_optional() {
    local description="$1"
    shift
    echo -e "  ${DIM}⧗ ${description}...${RESET}"
    if "$@" >> "$LOG_FILE" 2>&1; then
        log_info "$description"
    else
        log_warn "$description — ignoré (non critique)"
    fi
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "\n  ${RED}Ce script doit être lancé en root (sudo).${RESET}"
        echo -e "  ${DIM}Relancez avec : sudo bash install.sh${RESET}\n"
        exit 1
    fi
}

# ─── Banner ───────────────────────────────────────────────

clear
echo -e "${CYAN}${BOLD}"
cat << 'BANNER'

    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║             ███╗   ██╗███████╗ ██████╗            ║
    ║             ████╗  ██║██╔════╝██╔═══██╗           ║
    ║             ██╔██╗ ██║█████╗  ██║   ██║           ║
    ║             ██║╚██╗██║██╔══╝  ██║   ██║           ║
    ║             ██║ ╚████║███████╗╚██████╔╝           ║
    ║             ╚═╝  ╚═══╝╚══════╝ ╚═════╝           ║
    ║                                                   ║
    ║         Installation automatique                  ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝

BANNER
echo -e "${RESET}"

check_root

# Initialiser le fichier de log
echo "=== Neo Core Install — $(date) ===" > "$LOG_FILE"

TOTAL_STEPS=6

# ═══════════════════════════════════════════════════════════
#  Étape 1 : Dépendances système
# ═══════════════════════════════════════════════════════════

log_step 1 $TOTAL_STEPS "Installation des dépendances système"

export DEBIAN_FRONTEND=noninteractive

run_or_fail "Mise à jour de l'index des paquets" apt-get update -qq

# Paquets essentiels
run_or_fail "Installation de Python 3 + outils de base" apt-get install -y -qq \
    python3 python3-pip python3-venv python3-full python3-dev \
    git curl wget \
    build-essential libffi-dev libssl-dev pkg-config \
    sqlite3 \
    cmake

# Rust (nécessaire pour chromadb/tokenizers)
if ! command -v rustc &>/dev/null; then
    echo -e "  ${DIM}⧗ Installation de Rust (requis pour chromadb)...${RESET}"
    curl -fsSL https://sh.rustup.rs | sh -s -- -y >> "$LOG_FILE" 2>&1
    source "$HOME/.cargo/env" 2>/dev/null || true
    export PATH="$HOME/.cargo/bin:$PATH"
    if command -v rustc &>/dev/null; then
        log_info "Rust installé ($(rustc --version 2>/dev/null | head -1))"
    else
        log_warn "Rust non installé — chromadb pourrait échouer"
    fi
else
    log_info "Rust déjà installé ($(rustc --version 2>/dev/null | head -1))"
fi

# Vérifier la version Python
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MINOR -lt 10 ]]; then
    log_error "Python 3.10+ requis (version actuelle: $PYTHON_VERSION)"
    exit 1
fi

log_info "Python $PYTHON_VERSION détecté"

# Swap file si RAM < 2GB (aide à la compilation de chromadb)
TOTAL_RAM_MB=$(free -m | awk '/^Mem:/{print $2}')
if [[ $TOTAL_RAM_MB -lt 2048 ]] && [[ ! -f /swapfile ]]; then
    echo -e "  ${DIM}⧗ RAM faible (${TOTAL_RAM_MB}MB) — création d'un swap de 2GB...${RESET}"
    fallocate -l 2G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=2048 >> "$LOG_FILE" 2>&1
    chmod 600 /swapfile
    mkswap /swapfile >> "$LOG_FILE" 2>&1
    swapon /swapfile >> "$LOG_FILE" 2>&1
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    log_info "Swap 2GB activé (aide à la compilation)"
else
    log_info "RAM: ${TOTAL_RAM_MB}MB"
fi

# ═══════════════════════════════════════════════════════════
#  Étape 2 : Utilisateur système Neo
# ═══════════════════════════════════════════════════════════

log_step 2 $TOTAL_STEPS "Création de l'utilisateur système"

if id "$NEO_USER" &>/dev/null; then
    log_info "Utilisateur '$NEO_USER' existe déjà"
else
    useradd -r -m -s /bin/bash -d /home/$NEO_USER $NEO_USER 2>/dev/null || true
    log_info "Utilisateur '$NEO_USER' créé"
fi

# Donner les droits sudo sans mot de passe à Neo (autonomie complète)
echo "${NEO_USER} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/neo
chmod 440 /etc/sudoers.d/neo
log_info "Droits sudo accordés à '$NEO_USER' (NOPASSWD)"

# Ajouter le venv au PATH du user neo et du user courant
NEO_PATH_LINE='export PATH="/opt/neo-core/.venv/bin:/usr/local/bin:$PATH"'
if [[ -d /home/${NEO_USER} ]]; then
    grep -qF "/opt/neo-core/.venv/bin" /home/${NEO_USER}/.bashrc 2>/dev/null || \
        echo "$NEO_PATH_LINE" >> /home/${NEO_USER}/.bashrc
fi
# Aussi pour l'utilisateur qui lance l'install (souvent ubuntu)
REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"
if [[ "$REAL_USER" != "root" ]] && [[ -f "/home/${REAL_USER}/.bashrc" ]]; then
    grep -qF "/opt/neo-core/.venv/bin" "/home/${REAL_USER}/.bashrc" 2>/dev/null || \
        echo "$NEO_PATH_LINE" >> "/home/${REAL_USER}/.bashrc"
    log_info "PATH mis à jour pour l'utilisateur '${REAL_USER}'"
fi

# ═══════════════════════════════════════════════════════════
#  Étape 3 : Cloner le dépôt
# ═══════════════════════════════════════════════════════════

log_step 3 $TOTAL_STEPS "Téléchargement de Neo Core"

# Toujours supprimer et re-cloner pour garantir un code à jour
# (évite les problèmes de safe.directory quand le dossier appartient à un autre user)
if [[ -d "$INSTALL_DIR" ]]; then
    # Sauvegarder les données utilisateur si elles existent
    if [[ -d "${INSTALL_DIR}/data" ]]; then
        cp -r "${INSTALL_DIR}/data" /tmp/neo-data-backup 2>/dev/null || true
        log_info "Données existantes sauvegardées dans /tmp/neo-data-backup"
    fi
    if [[ -f "${INSTALL_DIR}/.env" ]]; then
        cp "${INSTALL_DIR}/.env" /tmp/neo-env-backup 2>/dev/null || true
        log_info "Configuration .env sauvegardée"
    fi
    rm -rf "$INSTALL_DIR"
fi

if run_or_fail "Clonage du dépôt dans $INSTALL_DIR" git clone --quiet https://github.com/EdenDadou/Neo.git "$INSTALL_DIR"; then
    :
else
    log_error "Impossible de cloner le dépôt. Vérifiez votre connexion internet."
    exit 1
fi

# Restaurer les données si backup existe
if [[ -d /tmp/neo-data-backup ]]; then
    cp -r /tmp/neo-data-backup "${INSTALL_DIR}/data"
    rm -rf /tmp/neo-data-backup
    log_info "Données restaurées"
fi
if [[ -f /tmp/neo-env-backup ]]; then
    cp /tmp/neo-env-backup "${INSTALL_DIR}/.env"
    rm -f /tmp/neo-env-backup
    log_info "Configuration .env restaurée"
fi

cd "$INSTALL_DIR"

# ═══════════════════════════════════════════════════════════
#  Étape 4 : Virtual environment + dépendances Python
# ═══════════════════════════════════════════════════════════

log_step 4 $TOTAL_STEPS "Installation de Neo Core"

# Créer le venv si nécessaire
if [[ ! -d "$VENV_DIR" ]]; then
    run_or_fail "Création du virtual environment" python3 -m venv "$VENV_DIR"
else
    log_info "Virtual environment existant"
fi

# Activer le venv
source "${VENV_DIR}/bin/activate"

# S'assurer que Rust est dans le PATH pour le venv aussi
export PATH="$HOME/.cargo/bin:$PATH"

# Installer pip à jour
run_or_fail "Mise à jour de pip" pip install --upgrade pip --no-cache-dir

# Purger le cache pip (évite de réutiliser des métadonnées obsolètes)
pip cache purge >> "$LOG_FILE" 2>&1 || true

# Installer Neo Core (SANS [dev] en production — moins de deps)
echo -e "  ${DIM}⧗ Installation de Neo Core + dépendances (peut prendre 2-5 min)...${RESET}"
if pip install -e "." --no-cache-dir >> "$LOG_FILE" 2>&1; then
    log_info "Neo Core + dépendances installés"
else
    log_error "Installation de Neo Core échouée"
    echo -e "  ${DIM}  Dernières lignes du log :${RESET}"
    tail -15 "$LOG_FILE" | sed 's/^/    /'
    echo
    echo -e "  ${YELLOW}Tentative d'installation sans les extras...${RESET}"
    # Fallback : installer les deps une par une
    pip install langchain langchain-anthropic rich python-dotenv fastapi uvicorn httpx psutil cryptography ddgs >> "$LOG_FILE" 2>&1 || true
    pip install -e "." --no-deps >> "$LOG_FILE" 2>&1 || true
    log_warn "Installation partielle — certains modules pourraient manquer"
fi

# Installer chromadb séparément (c'est souvent lui qui pose problème)
if python3 -c "import chromadb" 2>/dev/null; then
    log_info "chromadb déjà installé"
else
    echo -e "  ${DIM}⧗ Installation de chromadb (compilation, peut prendre 3-5 min)...${RESET}"
    if pip install chromadb >> "$LOG_FILE" 2>&1; then
        log_info "chromadb installé"
    else
        log_warn "chromadb échoué — les fonctions de mémoire vectorielle seront limitées"
        echo -e "  ${DIM}  Vous pouvez réessayer plus tard : ${VENV_DIR}/bin/pip install chromadb${RESET}"
    fi
fi

# Installer les providers optionnels (gratuits) — chacun séparément
run_optional "Installation de Groq (LLM cloud gratuit)" pip install groq
run_optional "Installation de Gemini (LLM cloud gratuit)" pip install google-generativeai
run_optional "Installation de Ollama (LLM local)" pip install ollama
run_optional "Installation du bot Telegram" pip install python-telegram-bot

# Vérifier que la commande neo fonctionne
if "${VENV_DIR}/bin/neo" version > /dev/null 2>&1; then
    NEO_VERSION=$("${VENV_DIR}/bin/neo" version 2>/dev/null || echo "unknown")
    log_info "Commande 'neo' fonctionnelle ($NEO_VERSION)"
else
    log_warn "La commande 'neo' n'est pas encore fonctionnelle — le wizard la configurera"
fi

# Créer un wrapper global pour que 'neo' soit accessible de partout
# Le wrapper exécute en tant que user neo pour éviter les problèmes de permissions
# (.env, data/ sont propriété de neo)
# --preserve-env=TERM garde le terminal fonctionnel pour les commandes interactives
cat > /usr/local/bin/neo << 'WRAPPER'
#!/usr/bin/env bash
exec sudo --preserve-env=TERM,LANG,LC_ALL -u neo /opt/neo-core/.venv/bin/neo "$@"
WRAPPER
chmod 755 /usr/local/bin/neo
log_info "Commande 'neo' ajoutée au PATH (/usr/local/bin/neo → sudo -u neo)"

# ═══════════════════════════════════════════════════════════
#  Étape 5 : Permissions + dossier data
# ═══════════════════════════════════════════════════════════

log_step 5 $TOTAL_STEPS "Configuration des permissions"

# Créer le dossier data
mkdir -p "${INSTALL_DIR}/data"

# Donner la propriété à l'utilisateur neo
chown -R ${NEO_USER}:${NEO_USER} "$INSTALL_DIR"
chmod 700 "${INSTALL_DIR}/data"

log_info "Permissions configurées (propriétaire: $NEO_USER)"

# ═══════════════════════════════════════════════════════════
#  Étape 6 : Service systemd
# ═══════════════════════════════════════════════════════════

log_step 6 $TOTAL_STEPS "Installation du service systemd"

cat > /etc/systemd/system/neo-guardian.service << 'EOF'
[Unit]
Description=Neo Core — AI Multi-Agent System
Documentation=https://github.com/EdenDadou/Neo
After=network.target
StartLimitIntervalSec=3600
StartLimitBurst=10

[Service]
Type=simple
User=neo
Group=neo
WorkingDirectory=/opt/neo-core
ExecStart=/opt/neo-core/.venv/bin/neo start --foreground
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
Environment=NEO_ENV=production
EnvironmentFile=-/opt/neo-core/.env

# Timeouts
TimeoutStartSec=60
TimeoutStopSec=30

# Security hardening (lecture seule sauf data + fichiers Neo)
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/neo-core

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=neo-core

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable neo-guardian --quiet 2>/dev/null || true
log_info "Service neo-guardian installé et activé au boot"

# ═══════════════════════════════════════════════════════════
#  Résumé + lancement du wizard
# ═══════════════════════════════════════════════════════════

echo -e "\n${CYAN}${BOLD}  ╔═══════════════════════════════════════════════╗"
echo -e "  ║       Installation système terminée !         ║"
echo -e "  ╚═══════════════════════════════════════════════╝${RESET}"
echo
echo -e "  ${BOLD}Installé dans :${RESET} ${INSTALL_DIR}"
echo -e "  ${BOLD}Utilisateur :${RESET}   ${NEO_USER}"
echo -e "  ${BOLD}Python :${RESET}        ${VENV_DIR}/bin/python3"
echo -e "  ${BOLD}Service :${RESET}       neo-guardian (systemd)"
echo -e "  ${BOLD}Log install :${RESET}   ${LOG_FILE}"
echo

echo -e "  ${CYAN}${BOLD}Dernière étape : le wizard de configuration${RESET}"
echo -e "  ${DIM}Il va vous demander votre nom et optionnellement vos clés API.${RESET}"
echo -e "  ${DIM}Sans clé API, Neo fonctionne en mode démo (réponses simulées).${RESET}"
echo

# Lancer le wizard en tant que root (le user neo n'a pas accès au TTY)
# On chown les fichiers de config après
echo -e "  ${DIM}Lancement du wizard...${RESET}\n"
cd "${INSTALL_DIR}" && source "${VENV_DIR}/bin/activate" && neo setup --auto < /dev/tty
WIZARD_EXIT=$?

# Rendre les fichiers de config accessibles à l'utilisateur neo
chown -R ${NEO_USER}:${NEO_USER} "${INSTALL_DIR}/data" 2>/dev/null || true
chown ${NEO_USER}:${NEO_USER} "${INSTALL_DIR}/.env" 2>/dev/null || true

# Si le wizard réussit, démarrer le service systemd
if [[ $WIZARD_EXIT -eq 0 ]]; then
    echo
    echo -e "  ${DIM}⧗ Démarrage du service Neo...${RESET}"

    # S'assurer que le .env est chargé par systemd
    systemctl daemon-reload 2>/dev/null || true
    systemctl start neo-guardian 2>/dev/null || true
    sleep 3

    if systemctl is-active --quiet neo-guardian; then
        log_info "Neo est en ligne ! (service systemd actif)"
    else
        log_warn "Le service n'a pas démarré automatiquement"
        echo -e "  ${DIM}Vérifiez les logs : sudo journalctl -u neo-guardian -n 30${RESET}"
    fi

    echo
    echo -e "  ${BOLD}Commandes utiles :${RESET}"
    echo -e "    ${CYAN}neo chat${RESET}                            Discuter avec Neo"
    echo -e "    ${CYAN}neo status${RESET}                          État du système"
    echo -e "    ${CYAN}neo setup${RESET}                           Relancer le wizard complet"
    echo -e "    ${CYAN}neo logs${RESET}                            Voir les logs Neo"
    echo -e "    ${CYAN}sudo journalctl -u neo-guardian -f${RESET}  Logs systemd en temps réel"
    echo -e "    ${CYAN}sudo systemctl restart neo-guardian${RESET} Redémarrer le service"
    echo
else
    log_warn "Le wizard a rencontré un problème"
    echo -e "  ${DIM}Relancez-le manuellement :${RESET}"
    echo -e "    ${CYAN}neo setup${RESET}"
fi

echo -e "\n${GREEN}${BOLD}  Installation terminée.${RESET}\n"
