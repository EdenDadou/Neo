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

# ─── Fonctions de préservation des données utilisateur ────

_preserve_user_data() {
    rm -rf /tmp/neo-data-backup 2>/dev/null || true
    rm -f /tmp/neo-env-backup 2>/dev/null || true

    if [[ -d "${INSTALL_DIR}/data" ]]; then
        cp -r "${INSTALL_DIR}/data" /tmp/neo-data-backup 2>/dev/null || true
        log_info "Données existantes sauvegardées"
    fi
    if [[ -f "${INSTALL_DIR}/.env" ]]; then
        cp "${INSTALL_DIR}/.env" /tmp/neo-env-backup 2>/dev/null || true
        chmod 600 /tmp/neo-env-backup 2>/dev/null || true
        log_info "Configuration .env sauvegardée"
    fi
}

_restore_user_data() {
    if [[ -d /tmp/neo-data-backup ]]; then
        cp -r /tmp/neo-data-backup "${INSTALL_DIR}/data" 2>/dev/null || true
        rm -rf /tmp/neo-data-backup 2>/dev/null || true
        log_info "Données restaurées"
    fi
    if [[ -f /tmp/neo-env-backup ]]; then
        cp /tmp/neo-env-backup "${INSTALL_DIR}/.env" 2>/dev/null || true
        chmod 600 "${INSTALL_DIR}/.env" 2>/dev/null || true
        shred -u /tmp/neo-env-backup 2>/dev/null || rm -f /tmp/neo-env-backup 2>/dev/null || true
        log_info "Configuration .env restaurée"
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

# Initialiser le fichier de log (tronquer si > 10MB)
if [[ -f "$LOG_FILE" ]] && [[ $(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0) -gt 10485760 ]]; then
    tail -1000 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
fi
echo "=== Neo Core Install — $(date) ===" >> "$LOG_FILE"

# ─── Vérification de l'espace disque ─────────────────────
check_disk_space() {
    local required_mb=3000  # 3GB minimum
    local available_mb
    available_mb=$(df -BM / | awk 'NR==2{gsub(/M/,"",$4); print $4}')

    if [[ -z "$available_mb" ]]; then
        log_warn "Impossible de vérifier l'espace disque — on continue"
        return 0
    fi

    echo -e "  ${DIM}Espace disque disponible : ${available_mb}MB${RESET}"

    if [[ $available_mb -lt $required_mb ]]; then
        log_error "Espace disque insuffisant : ${available_mb}MB disponibles, ${required_mb}MB requis"
        echo -e "  ${YELLOW}Nettoyage automatique en cours...${RESET}"

        # Nettoyer le cache apt
        apt-get clean >> "$LOG_FILE" 2>&1 || true
        apt-get autoremove -y -qq >> "$LOG_FILE" 2>&1 || true

        # Nettoyer les anciens backups Neo s'ils existent
        rm -rf /tmp/neo-data-backup 2>/dev/null || true
        rm -f /tmp/neo-env-backup 2>/dev/null || true
        rm -f /tmp/neo-install.log 2>/dev/null || true

        # Nettoyer les anciens logs
        journalctl --vacuum-size=50M >> "$LOG_FILE" 2>&1 || true

        # Nettoyer pip cache (peut être volumineux)
        pip cache purge >> "$LOG_FILE" 2>&1 || true
        rm -rf /root/.cache/pip 2>/dev/null || true
        rm -rf /home/*/.cache/pip 2>/dev/null || true

        # Nettoyer les caches HuggingFace/torch si présents
        rm -rf /root/.cache/huggingface 2>/dev/null || true
        rm -rf /root/.cache/torch 2>/dev/null || true

        # Re-vérifier
        available_mb=$(df -BM / | awk 'NR==2{gsub(/M/,"",$4); print $4}')
        if [[ $available_mb -lt 1500 ]]; then
            log_error "Toujours pas assez d'espace : ${available_mb}MB (minimum 1.5GB)"
            echo -e "  ${DIM}Conseils :${RESET}"
            echo -e "  ${DIM}  - Vérifiez avec : df -h${RESET}"
            echo -e "  ${DIM}  - Supprimez les fichiers inutiles${RESET}"
            echo -e "  ${DIM}  - Si swap prend de la place : sudo swapoff /swapfile && sudo rm /swapfile${RESET}"
            exit 1
        fi
        log_info "Espace récupéré : ${available_mb}MB disponibles"
    else
        log_info "Espace disque OK : ${available_mb}MB disponibles"
    fi
}

check_disk_space

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

# Note : Rust n'est plus nécessaire (FAISS remplace ChromaDB depuis v0.9.3)

# Vérifier la version Python
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MINOR -lt 10 ]]; then
    log_error "Python 3.10+ requis (version actuelle: $PYTHON_VERSION)"
    exit 1
fi

log_info "Python $PYTHON_VERSION détecté"

# Swap file si RAM < 2GB (aide au chargement des modèles d'embedding)
TOTAL_RAM_MB=$(free -m | awk '/^Mem:/{print $2}')
SWAP_ACTIVE=$(swapon --show --noheadings 2>/dev/null | wc -l)
if [[ $TOTAL_RAM_MB -lt 2048 ]] && [[ $SWAP_ACTIVE -eq 0 ]]; then
    # Vérifier si un swapfile existe déjà mais n'est pas actif
    if [[ -f /swapfile ]]; then
        swapon /swapfile >> "$LOG_FILE" 2>&1 && log_info "Swap existant réactivé" || {
            rm -f /swapfile
            log_warn "Ancien swapfile corrompu — recréation"
        }
    fi
    # Créer seulement si toujours pas de swap
    if [[ $(swapon --show --noheadings 2>/dev/null | wc -l) -eq 0 ]]; then
        # Utiliser 1GB si l'espace est limité (< 5GB libre), sinon 2GB
        AVAIL_MB=$(df -BM / | awk 'NR==2{gsub(/M/,"",$4); print $4}')
        if [[ $AVAIL_MB -lt 5000 ]]; then
            SWAP_SIZE="1G"
            SWAP_BYTES=$((1024*1024*1024))
        else
            SWAP_SIZE="2G"
            SWAP_BYTES=$((2*1024*1024*1024))
        fi
        echo -e "  ${DIM}⧗ RAM faible (${TOTAL_RAM_MB}MB) — création d'un swap de ${SWAP_SIZE}...${RESET}"
        fallocate -l "$SWAP_SIZE" /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_BYTES/1024/1024)) >> "$LOG_FILE" 2>&1
        chmod 600 /swapfile
        mkswap /swapfile >> "$LOG_FILE" 2>&1
        swapon /swapfile >> "$LOG_FILE" 2>&1
        # Ajouter à fstab seulement si pas déjà présent
        grep -qF '/swapfile' /etc/fstab 2>/dev/null || echo '/swapfile none swap sw 0 0' >> /etc/fstab
        log_info "Swap ${SWAP_SIZE} activé (aide à la compilation)"
    fi
elif [[ $SWAP_ACTIVE -gt 0 ]]; then
    SWAP_TOTAL=$(free -m | awk '/^Swap:/{print $2}')
    log_info "RAM: ${TOTAL_RAM_MB}MB + Swap: ${SWAP_TOTAL}MB (déjà actif)"
else
    log_info "RAM: ${TOTAL_RAM_MB}MB (suffisant)"
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

# Donner les droits sudo limités à Neo (commandes spécifiques uniquement)
cat > /etc/sudoers.d/neo << 'SUDOERS'
# Neo Core — permissions sudo restreintes
# Seules les commandes nécessaires au fonctionnement sont autorisées
neo ALL=(ALL) NOPASSWD: /usr/bin/systemctl start neo-guardian
neo ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop neo-guardian
neo ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart neo-guardian
neo ALL=(ALL) NOPASSWD: /usr/bin/systemctl status neo-guardian
neo ALL=(ALL) NOPASSWD: /usr/bin/journalctl -u neo-guardian *
neo ALL=(ALL) NOPASSWD: /opt/neo-core/.venv/bin/pip install *
neo ALL=(ALL) NOPASSWD: /opt/neo-core/.venv/bin/pip uninstall *
SUDOERS
chmod 440 /etc/sudoers.d/neo
log_info "Droits sudo restreints accordés à '$NEO_USER' (commandes spécifiques)"

# Ajouter le venv au PATH du user neo (accès direct aux binaires)
NEO_PATH_LINE='export PATH="/opt/neo-core/.venv/bin:/usr/local/bin:$PATH"'
if [[ -d /home/${NEO_USER} ]]; then
    grep -qF "/opt/neo-core/.venv/bin" /home/${NEO_USER}/.bashrc 2>/dev/null || \
        echo "$NEO_PATH_LINE" >> /home/${NEO_USER}/.bashrc
fi
# Pour l'utilisateur courant (ubuntu), NE PAS ajouter le venv au PATH
# car il faut passer par le wrapper /usr/local/bin/neo qui fait sudo -u neo.
# On ajoute seulement /usr/local/bin s'il n'est pas déjà dans le PATH.
REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"
WRAPPER_PATH_LINE='export PATH="/usr/local/bin:$PATH"'
if [[ "$REAL_USER" != "root" ]] && [[ -f "/home/${REAL_USER}/.bashrc" ]]; then
    # Nettoyer l'ancien PATH qui incluait le venv (corrige les installs précédentes)
    sed -i '\|/opt/neo-core/.venv/bin|d' "/home/${REAL_USER}/.bashrc" 2>/dev/null || true
    grep -qF "/usr/local/bin" "/home/${REAL_USER}/.bashrc" 2>/dev/null || \
        echo "$WRAPPER_PATH_LINE" >> "/home/${REAL_USER}/.bashrc"
    log_info "PATH mis à jour pour l'utilisateur '${REAL_USER}'"
fi

# ═══════════════════════════════════════════════════════════
#  Étape 3 : Code source Neo Core
# ═══════════════════════════════════════════════════════════

log_step 3 $TOTAL_STEPS "Code source Neo Core"

REPO_URL="https://github.com/EdenDadou/Neo.git"

if [[ -d "$INSTALL_DIR" ]]; then
    if [[ -d "${INSTALL_DIR}/.git" ]]; then
        # ─── Repo git existant → mise à jour ───
        log_info "Code existant détecté dans ${INSTALL_DIR}"

        # Fix safe.directory (root exécute git sur un dossier d'un autre user)
        git config --global --add safe.directory "$INSTALL_DIR" 2>/dev/null || true

        systemctl stop neo-guardian 2>/dev/null || true

        cd "$INSTALL_DIR"
        if run_or_fail "Mise à jour du code (git pull)" git pull --ff-only origin main; then
            :
        else
            log_warn "git pull échoué — tentative de synchronisation forcée"
            if git fetch origin >> "$LOG_FILE" 2>&1 && \
               git reset --hard origin/main >> "$LOG_FILE" 2>&1; then
                log_info "Code synchronisé depuis GitHub"
            else
                log_warn "Mise à jour impossible — utilisation du code existant"
                log_warn "Vérifiez votre connexion internet pour les prochaines mises à jour"
            fi
        fi
    else
        # ─── Dossier existant mais PAS un repo git (install cassée) ───
        log_warn "${INSTALL_DIR} existe mais n'est pas un dépôt Git — réinstallation"

        _preserve_user_data

        systemctl stop neo-guardian 2>/dev/null || true
        rm -rf "$INSTALL_DIR"

        if run_or_fail "Clonage du dépôt dans $INSTALL_DIR" git clone --quiet "$REPO_URL" "$INSTALL_DIR"; then
            :
        else
            log_error "Impossible de cloner le dépôt. Vérifiez votre connexion internet."
            exit 1
        fi

        _restore_user_data
    fi
else
    # ─── Aucun code présent → installation fraîche ───
    if run_or_fail "Clonage du dépôt dans $INSTALL_DIR" git clone --quiet "$REPO_URL" "$INSTALL_DIR"; then
        :
    else
        log_error "Impossible de cloner le dépôt. Vérifiez votre connexion internet."
        exit 1
    fi
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

# Vérifier FAISS + sentence-transformers (installés via pyproject.toml)
if python3 -c "import faiss" 2>/dev/null; then
    log_info "faiss-cpu déjà installé"
else
    echo -e "  ${DIM}⧗ Installation de faiss-cpu (mémoire vectorielle)...${RESET}"
    if pip install faiss-cpu --no-cache-dir >> "$LOG_FILE" 2>&1; then
        log_info "faiss-cpu installé"
    else
        log_warn "faiss-cpu échoué — les fonctions de mémoire vectorielle seront limitées"
    fi
fi

if python3 -c "from sentence_transformers import SentenceTransformer" 2>/dev/null; then
    log_info "sentence-transformers déjà installé"
else
    echo -e "  ${DIM}⧗ Installation de sentence-transformers (embeddings)...${RESET}"
    if pip install sentence-transformers --no-cache-dir >> "$LOG_FILE" 2>&1; then
        log_info "sentence-transformers installé"
    else
        log_warn "sentence-transformers échoué — embeddings limités"
    fi
fi

# Installer les providers optionnels (gratuits) — chacun séparément
run_optional "Installation de Groq (LLM cloud gratuit)" pip install groq
run_optional "Installation de Gemini (LLM cloud gratuit)" pip install google-generativeai
run_optional "Installation de Ollama (LLM local)" pip install ollama
run_optional "Installation du bot Telegram" pip install python-telegram-bot

# Pré-télécharger le modèle d'embedding (mémoire de Neo)
# Le faire ici garantit que le cache existe AVANT le wizard setup.
# Avec le cache présent, setup.py/store.py chargent en mode offline → zéro 429.
#
# Stratégie : git clone direct (PAS l'API HuggingFace Hub qui rate-limit 429).
# Le git clone télécharge les fichiers du modèle sans passer par l'API REST.
echo -e "  ${DIM}⧗ Pré-téléchargement du modèle d'embedding...${RESET}"
EMBEDDING_OK=false
NEO_HOME=$(eval echo ~${NEO_USER})
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_DIR="${NEO_HOME}/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"

# Vérifier si déjà en cache
if HOME="$NEO_HOME" python3 -c "
import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('all-MiniLM-L6-v2')
    r = m.encode(['test'])
    if r is not None and len(r) > 0:
        sys.exit(0)
except: pass
sys.exit(1)
" >> "$LOG_FILE" 2>&1; then
    EMBEDDING_OK=true
    log_info "Modèle d'embedding all-MiniLM-L6-v2 déjà en cache"
fi

if [[ "$EMBEDDING_OK" = false ]]; then
    # Méthode 1 : huggingface-cli download (plus fiable que SentenceTransformer())
    if command -v huggingface-cli &>/dev/null || "${VENV_DIR}/bin/huggingface-cli" --version &>/dev/null 2>&1; then
        echo -e "  ${DIM}  Téléchargement via huggingface-cli...${RESET}"
        HF_CLI="huggingface-cli"
        command -v huggingface-cli &>/dev/null || HF_CLI="${VENV_DIR}/bin/huggingface-cli"

        if HOME="$NEO_HOME" $HF_CLI download "${EMBEDDING_MODEL}" --quiet >> "$LOG_FILE" 2>&1; then
            EMBEDDING_OK=true
        fi
    fi
fi

if [[ "$EMBEDDING_OK" = false ]]; then
    # Méthode 2 : git clone direct (contourne l'API HuggingFace → pas de 429)
    echo -e "  ${DIM}  Téléchargement via git clone (contourne les rate-limits)...${RESET}"

    # Installer git-lfs si nécessaire (pour les fichiers binaires du modèle)
    if ! command -v git-lfs &>/dev/null; then
        apt-get install -y git-lfs >> "$LOG_FILE" 2>&1 || true
        git lfs install >> "$LOG_FILE" 2>&1 || true
    fi

    TEMP_MODEL_DIR=$(mktemp -d)
    if git clone --depth 1 "https://huggingface.co/${EMBEDDING_MODEL}" "$TEMP_MODEL_DIR" >> "$LOG_FILE" 2>&1; then
        # Placer dans le cache HuggingFace au format attendu par sentence-transformers
        mkdir -p "${MODEL_CACHE_DIR}/snapshots/local"
        cp -r "${TEMP_MODEL_DIR}"/* "${MODEL_CACHE_DIR}/snapshots/local/" 2>/dev/null || true
        # Créer le fichier refs/main pour que HF Hub trouve le snapshot
        mkdir -p "${MODEL_CACHE_DIR}/refs"
        echo "local" > "${MODEL_CACHE_DIR}/refs/main"

        # Vérifier que ça charge
        if HOME="$NEO_HOME" python3 -c "
import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('all-MiniLM-L6-v2')
    r = m.encode(['test'])
    if r is not None and len(r) > 0:
        sys.exit(0)
except: pass
sys.exit(1)
" >> "$LOG_FILE" 2>&1; then
            EMBEDDING_OK=true
        fi
    fi
    rm -rf "$TEMP_MODEL_DIR" 2>/dev/null || true
fi

if [[ "$EMBEDDING_OK" = false ]]; then
    # Méthode 3 : fallback SentenceTransformer (API HuggingFace Hub — risque 429)
    echo -e "  ${DIM}  Fallback: téléchargement via Python API...${RESET}"
    for ATTEMPT in 1 2 3; do
        if HOME="$NEO_HOME" python3 -c "
import os, sys
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('all-MiniLM-L6-v2')
    r = m.encode(['test'])
    if r is not None and len(r) > 0:
        sys.exit(0)
except: pass
sys.exit(1)
" >> "$LOG_FILE" 2>&1; then
            EMBEDDING_OK=true
            break
        else
            if [[ $ATTEMPT -lt 3 ]]; then
                WAIT=$((5 * ATTEMPT))
                echo -e "  ${YELLOW}⚠${RESET} Tentative ${ATTEMPT}/3 échouée — retry dans ${WAIT}s..."
                sleep $WAIT
            fi
        fi
    done
fi

# S'assurer que le cache HF appartient à neo
chown -R ${NEO_USER}:${NEO_USER} "${NEO_HOME}/.cache" 2>/dev/null || true

if [[ "$EMBEDDING_OK" = true ]]; then
    log_info "Modèle d'embedding all-MiniLM-L6-v2 en cache"
else
    log_warn "Modèle d'embedding non téléchargé — mémoire en mode dégradé (bag-of-words)"
    echo -e "  ${DIM}  La recherche mémoire fonctionnera par mots-clés au lieu de sémantique.${RESET}"
    echo -e "  ${DIM}  Fix: neo setup (relancera le téléchargement)${RESET}"
fi

# Nettoyer les caches pour libérer de l'espace disque
echo -e "  ${DIM}⧗ Nettoyage des caches...${RESET}"
pip cache purge >> "$LOG_FILE" 2>&1 || true
apt-get clean >> "$LOG_FILE" 2>&1 || true
rm -rf /root/.cache/pip 2>/dev/null || true
log_info "Caches nettoyés"

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
# cd to project root so that .env, data/ are accessible as neo user
cd /opt/neo-core 2>/dev/null || true
exec sudo --preserve-env=TERM,LANG,LC_ALL -u neo /opt/neo-core/.venv/bin/neo "$@"
WRAPPER
chmod 755 /usr/local/bin/neo
log_info "Commande 'neo' ajoutée au PATH (/usr/local/bin/neo → sudo -u neo)"

# ═══════════════════════════════════════════════════════════
#  Étape 5 : Permissions + dossier data
# ═══════════════════════════════════════════════════════════

log_step 5 $TOTAL_STEPS "Configuration des permissions"

# Créer les dossiers data
mkdir -p "${INSTALL_DIR}/data"
mkdir -p "${INSTALL_DIR}/data/memory"
mkdir -p "${INSTALL_DIR}/data/plugins"
mkdir -p "${INSTALL_DIR}/data/patches"
mkdir -p "${INSTALL_DIR}/data/tool_metadata"
mkdir -p "${INSTALL_DIR}/data/system_docs"

# Donner la propriété à l'utilisateur neo
chown -R ${NEO_USER}:${NEO_USER} "$INSTALL_DIR"

# Le répertoire principal doit être lisible/traversable par tous
# pour que 'ubuntu' puisse cd dans /opt/neo-core et lancer le wrapper
chmod 755 "$INSTALL_DIR"

# data/ est privé à neo (contient .env, mémoire, etc.)
chmod 700 "${INSTALL_DIR}/data"

# Le .venv doit être exécutable par neo
chmod 755 "${INSTALL_DIR}/.venv" 2>/dev/null || true

# Ajouter ubuntu au groupe neo pour accès lecture
REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"
if [[ "$REAL_USER" != "root" ]] && [[ "$REAL_USER" != "$NEO_USER" ]]; then
    usermod -aG ${NEO_USER} ${REAL_USER} 2>/dev/null || true
    log_info "Utilisateur '${REAL_USER}' ajouté au groupe '${NEO_USER}'"
fi

log_info "Permissions configurées (propriétaire: $NEO_USER, accès: 755)"

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
TimeoutStartSec=120
TimeoutStopSec=30

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/opt/neo-core/data /opt/neo-core/.env /tmp

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

# Lancer le wizard directement via le venv (PAS via le wrapper /usr/local/bin/neo
# qui fait sudo -u neo, car sudo peut échouer si le disque est plein ou le PTY indisponible)
# On chown les fichiers de config après
echo -e "  ${DIM}Lancement du wizard...${RESET}\n"
cd "${INSTALL_DIR}" && "${VENV_DIR}/bin/neo" setup < /dev/tty
WIZARD_EXIT=$?

# Rendre les fichiers de config accessibles à l'utilisateur neo
chown -R ${NEO_USER}:${NEO_USER} "${INSTALL_DIR}/data" 2>/dev/null || true
chown ${NEO_USER}:${NEO_USER} "${INSTALL_DIR}/.env" 2>/dev/null || true

# Si le wizard réussit, démarrer le service systemd
if [[ $WIZARD_EXIT -eq 0 ]]; then
    echo
    echo -e "  ${DIM}⧗ Démarrage du service Neo...${RESET}"

    # S'assurer que le .env est chargé par systemd
    # IMPORTANT : restart (pas start) pour recharger la config du vault
    # si le service tournait déjà d'une installation précédente
    systemctl daemon-reload 2>/dev/null || true
    systemctl restart neo-guardian 2>/dev/null || true
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
    echo -e "    ${CYAN}neo setup${RESET}                           Reconfigurer (clés API, providers...)"
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
