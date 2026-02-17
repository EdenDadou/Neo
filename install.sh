#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  Neo Core — Installation automatique
# ═══════════════════════════════════════════════════════════
#
#  Usage (une seule commande) :
#    curl -fsSL https://raw.githubusercontent.com/EdenDadou/Neo/main/install.sh | bash
#
#  Ou manuellement :
#    git clone https://github.com/EdenDadou/Neo.git /opt/neo-core
#    cd /opt/neo-core && bash install.sh
#
# ═══════════════════════════════════════════════════════════

set -euo pipefail

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

# ─── Fonctions utilitaires ────────────────────────────────

log_info()  { echo -e "  ${GREEN}✓${RESET} $1"; }
log_warn()  { echo -e "  ${YELLOW}⚠${RESET} $1"; }
log_error() { echo -e "  ${RED}✗${RESET} $1"; }
log_step()  { echo -e "\n${CYAN}${BOLD}  [$1/$2] $3${RESET}\n"; }

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

TOTAL_STEPS=6

# ═══════════════════════════════════════════════════════════
#  Étape 1 : Dépendances système
# ═══════════════════════════════════════════════════════════

log_step 1 $TOTAL_STEPS "Installation des dépendances système"

export DEBIAN_FRONTEND=noninteractive

apt-get update -qq > /dev/null 2>&1
log_info "Index des paquets mis à jour"

apt-get install -y -qq \
    python3 python3-pip python3-venv python3-full \
    git curl wget \
    build-essential libffi-dev libssl-dev \
    sqlite3 \
    > /dev/null 2>&1

log_info "Python 3, pip, venv, git, curl, SQLite installés"

# Vérifier la version Python
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MINOR -lt 10 ]]; then
    log_error "Python 3.10+ requis (version actuelle: $PYTHON_VERSION)"
    exit 1
fi

log_info "Python $PYTHON_VERSION détecté"

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

# ═══════════════════════════════════════════════════════════
#  Étape 3 : Cloner le dépôt
# ═══════════════════════════════════════════════════════════

log_step 3 $TOTAL_STEPS "Téléchargement de Neo Core"

if [[ -d "${INSTALL_DIR}/.git" ]]; then
    log_info "Dépôt existant détecté — mise à jour..."
    cd "$INSTALL_DIR"
    git fetch --all --quiet
    git checkout main --quiet 2>/dev/null || true
    git pull --quiet 2>/dev/null || true
else
    if [[ -d "$INSTALL_DIR" ]]; then
        rm -rf "$INSTALL_DIR"
    fi
    git clone --quiet https://github.com/EdenDadou/Neo.git "$INSTALL_DIR"
    log_info "Dépôt cloné dans $INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# ═══════════════════════════════════════════════════════════
#  Étape 4 : Virtual environment + dépendances Python
# ═══════════════════════════════════════════════════════════

log_step 4 $TOTAL_STEPS "Installation de Neo Core"

# Créer le venv si nécessaire
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    log_info "Virtual environment créé"
else
    log_info "Virtual environment existant"
fi

# Activer le venv
source "${VENV_DIR}/bin/activate"

# Installer Neo + toutes les deps
pip install --upgrade pip -q 2>/dev/null
log_info "pip mis à jour"

pip install -e ".[dev]" -q 2>/dev/null
log_info "Neo Core + dépendances installés"

# Installer les providers optionnels (gratuits)
pip install groq google-generativeai ollama python-telegram-bot -q 2>/dev/null || true
log_info "Providers LLM optionnels installés (Groq, Gemini, Ollama, Telegram)"

# Vérifier que la commande neo fonctionne
if "${VENV_DIR}/bin/neo" version > /dev/null 2>&1; then
    NEO_VERSION=$("${VENV_DIR}/bin/neo" version 2>/dev/null || echo "unknown")
    log_info "Commande 'neo' fonctionnelle ($NEO_VERSION)"
else
    log_warn "La commande 'neo' n'est pas encore fonctionnelle — le wizard la configurera"
fi

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
Description=Neo Core Guardian — AI Multi-Agent System
Documentation=https://github.com/EdenDadou/Neo
After=network.target
StartLimitIntervalSec=3600
StartLimitBurst=10

[Service]
Type=simple
User=neo
Group=neo
WorkingDirectory=/opt/neo-core
ExecStart=/opt/neo-core/.venv/bin/neo guardian
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=NEO_ENV=production

# Timeouts
TimeoutStartSec=30
TimeoutStopSec=30

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/neo-core/data /opt/neo-core/.env /opt/neo-core/.venv

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=neo-guardian

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable neo-guardian --quiet 2>/dev/null
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
echo

echo -e "  ${CYAN}${BOLD}Dernière étape : le wizard de configuration${RESET}"
echo -e "  ${DIM}Il va vous demander votre nom et optionnellement vos clés API.${RESET}"
echo -e "  ${DIM}Sans clé API, Neo fonctionne en mode démo (réponses simulées).${RESET}"
echo

# Lancer le wizard en mode auto (minimal questions)
echo -e "  ${DIM}Lancement du wizard...${RESET}\n"
sudo -u $NEO_USER bash -c "cd ${INSTALL_DIR} && source ${VENV_DIR}/bin/activate && neo setup --auto"

# Si le wizard réussit, démarrer le service
if [[ $? -eq 0 ]]; then
    echo
    echo -e "  ${DIM}⧗ Démarrage de Neo...${RESET}"
    systemctl start neo-guardian 2>/dev/null || true
    sleep 2

    if systemctl is-active --quiet neo-guardian; then
        log_info "Neo est en ligne ! 🚀"
        echo
        echo -e "  ${BOLD}Commandes utiles :${RESET}"
        echo -e "    ${CYAN}sudo -u neo ${VENV_DIR}/bin/neo chat${RESET}      Discuter avec Neo"
        echo -e "    ${CYAN}sudo -u neo ${VENV_DIR}/bin/neo status${RESET}    État du système"
        echo -e "    ${CYAN}sudo journalctl -u neo-guardian -f${RESET}        Voir les logs"
        echo -e "    ${CYAN}sudo systemctl restart neo-guardian${RESET}       Redémarrer"
        echo
        echo -e "  ${DIM}Alias recommandé (ajoutez dans ~/.bashrc) :${RESET}"
        echo -e "    ${CYAN}alias neo='sudo -u neo ${VENV_DIR}/bin/neo'${RESET}"
        echo
    else
        log_warn "Le service n'a pas démarré automatiquement"
        echo -e "  ${DIM}Lancez manuellement :${RESET}"
        echo -e "    ${CYAN}sudo systemctl start neo-guardian${RESET}"
        echo -e "    ${CYAN}sudo journalctl -u neo-guardian -f${RESET}"
    fi
else
    log_warn "Le wizard a rencontré un problème"
    echo -e "  ${DIM}Relancez-le manuellement :${RESET}"
    echo -e "    ${CYAN}sudo -u neo ${VENV_DIR}/bin/neo setup --auto${RESET}"
fi

echo -e "\n${GREEN}${BOLD}  Installation terminée.${RESET}\n"
