#!/bin/bash

# Neo AI - Script d'installation automatique
# Usage: curl -fsSL https://raw.githubusercontent.com/EdenDadou/Neo/main/setup.sh | bash
# Ou:    ./setup.sh

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'
DIM='\033[2m'

# Logo Neo
print_logo() {
    echo ""
    echo -e "${CYAN}"
    echo "    ███╗   ██╗███████╗ ██████╗ "
    echo "    ████╗  ██║██╔════╝██╔═══██╗"
    echo "    ██╔██╗ ██║█████╗  ██║   ██║"
    echo "    ██║╚██╗██║██╔══╝  ██║   ██║"
    echo "    ██║ ╚████║███████╗╚██████╔╝"
    echo "    ╚═╝  ╚═══╝╚══════╝ ╚═════╝ "
    echo -e "${NC}"
    echo -e "${DIM}    Assistant IA Multi-Agents v0.2.0${NC}"
    echo ""
}

print_step() {
    echo ""
    echo -e "${CYAN}[$1/$2]${NC} ${BOLD}$3${NC}"
    echo -e "${DIM}────────────────────────────────────────────────────────${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Détecter l'OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            echo "debian"
        elif [ -f /etc/redhat-release ]; then
            echo "redhat"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Installer Node.js
install_nodejs() {
    print_step 1 6 "Installation de Node.js 20"

    if command -v node &> /dev/null; then
        NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
        if [ "$NODE_VERSION" -ge 20 ]; then
            print_success "Node.js $(node -v) déjà installé"
            return 0
        else
            print_warning "Node.js $(node -v) détecté, mise à jour vers 20+"
        fi
    fi

    OS=$(detect_os)

    case $OS in
        debian)
            print_info "Installation via NodeSource (Debian/Ubuntu)..."
            curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - > /dev/null 2>&1
            sudo apt-get install -y nodejs > /dev/null 2>&1
            ;;
        redhat)
            print_info "Installation via NodeSource (RHEL/CentOS)..."
            curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash - > /dev/null 2>&1
            sudo yum install -y nodejs > /dev/null 2>&1
            ;;
        macos)
            if command -v brew &> /dev/null; then
                print_info "Installation via Homebrew..."
                brew install node@20 > /dev/null 2>&1
            else
                print_error "Homebrew requis pour macOS. Installez-le: https://brew.sh"
                exit 1
            fi
            ;;
        *)
            print_error "OS non supporté pour l'installation automatique"
            print_info "Installez Node.js 20+ manuellement: https://nodejs.org"
            exit 1
            ;;
    esac

    if command -v node &> /dev/null; then
        print_success "Node.js $(node -v) installé"
    else
        print_error "Échec de l'installation de Node.js"
        exit 1
    fi
}

# Installer les outils de build (pour better-sqlite3)
install_build_tools() {
    print_step 2 6 "Installation des outils de compilation"

    OS=$(detect_os)

    case $OS in
        debian)
            # Vérifier si build-essential est installé
            if ! dpkg -l | grep -q build-essential; then
                print_info "Installation de build-essential, python3..."
                sudo apt-get update > /dev/null 2>&1
                sudo apt-get install -y build-essential python3 > /dev/null 2>&1
                print_success "Outils de compilation installés"
            else
                print_success "Outils de compilation déjà présents"
            fi
            ;;
        redhat)
            print_info "Installation des outils de développement..."
            sudo yum groupinstall -y "Development Tools" > /dev/null 2>&1
            sudo yum install -y python3 > /dev/null 2>&1
            print_success "Outils de compilation installés"
            ;;
        macos)
            if ! xcode-select -p &> /dev/null; then
                print_info "Installation de Xcode Command Line Tools..."
                xcode-select --install 2>/dev/null || true
                print_warning "Suivez les instructions à l'écran pour installer Xcode CLI"
            else
                print_success "Xcode CLI déjà installé"
            fi
            ;;
        *)
            print_warning "Vérifiez que les outils de compilation sont installés"
            ;;
    esac
}

# Installer les dépendances npm
install_dependencies() {
    print_step 3 6 "Installation des dépendances Node.js"

    print_info "Installation des packages npm..."
    npm install 2>&1 | tail -5
    print_success "Dépendances backend installées"

    if [ -d "web" ]; then
        print_info "Installation des packages frontend..."
        cd web && npm install 2>&1 | tail -3 && cd ..
        print_success "Dépendances frontend installées"
    fi
}

# Créer les répertoires de données
create_directories() {
    print_step 4 6 "Création des répertoires"

    mkdir -p data/vectors data/logs data/sessions data/backups
    print_success "Répertoires de données créés"
}

# Configurer les clés API
configure_api() {
    print_step 5 6 "Configuration des clés API"

    # Vérifier si .env existe déjà
    if [ -f ".env" ] && grep -q "ANTHROPIC_API_KEY=sk-ant-" .env; then
        print_success "Configuration existante détectée"
        read -p "  Reconfigurer? [y/N] " RECONFIG
        RECONFIG=${RECONFIG:-N}
        if [[ ! "$RECONFIG" =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi

    echo ""
    echo -e "  ${YELLOW}Clé API Anthropic (obligatoire)${NC}"
    echo -e "  ${DIM}Obtenez-la sur: https://console.anthropic.com${NC}"
    echo ""

    while true; do
        read -p "  Clé API Anthropic: " API_KEY

        if [[ "$API_KEY" =~ ^sk-ant- ]]; then
            break
        else
            print_error "Format invalide. La clé doit commencer par sk-ant-"
        fi
    done

    # Groq (optionnel)
    echo ""
    echo -e "  ${YELLOW}Clé API Groq (optionnel - modèles gratuits rapides)${NC}"
    echo -e "  ${DIM}Obtenez-la sur: https://console.groq.com${NC}"
    echo ""
    read -p "  Clé API Groq (laisser vide pour ignorer): " GROQ_KEY

    # Générer le fichier .env
    cat > .env << EOF
# Neo AI Configuration
# Généré automatiquement par setup.sh

# Anthropic (obligatoire)
ANTHROPIC_API_KEY=$API_KEY

# Groq (optionnel)
GROQ_API_KEY=${GROQ_KEY:-}

# Serveur
PORT=3001
CORS_ORIGIN=http://localhost:5173

# Sécurité
JWT_SECRET=neo-$(date +%s)-$(openssl rand -hex 8 2>/dev/null || echo $RANDOM$RANDOM)

# Mode
NODE_ENV=production
DEBUG=false
EOF

    print_success "Fichier .env créé"
}

# Installer Ollama (optionnel)
install_ollama() {
    print_step 6 6 "Configuration d'Ollama (embeddings locaux)"

    if command -v ollama &> /dev/null; then
        print_success "Ollama déjà installé"
    else
        echo ""
        read -p "  Installer Ollama pour les embeddings locaux? [Y/n] " INSTALL_OLLAMA
        INSTALL_OLLAMA=${INSTALL_OLLAMA:-Y}

        if [[ "$INSTALL_OLLAMA" =~ ^[Yy]$ ]]; then
            print_info "Installation d'Ollama..."
            curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1

            if command -v ollama &> /dev/null; then
                print_success "Ollama installé"

                # Télécharger le modèle d'embeddings
                print_info "Téléchargement du modèle d'embeddings (nomic-embed-text)..."
                ollama pull nomic-embed-text 2>&1 | tail -3
                print_success "Modèle d'embeddings prêt"
            else
                print_warning "Échec de l'installation d'Ollama"
            fi
        else
            print_info "Ollama ignoré - embeddings via Xenova/transformers"
        fi
    fi
}

# Démarrer Neo
start_neo() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${GREEN}${BOLD}Installation terminée!${NC}"
    echo ""
    echo -e "  ${BOLD}Commandes disponibles:${NC}"
    echo ""
    echo -e "    ${CYAN}npm run dev${NC}       Serveur API (port 3001)"
    echo -e "    ${CYAN}npm run dev:cli${NC}   Mode CLI interactif"
    echo -e "    ${CYAN}npm run dev:web${NC}   Dashboard React (port 5173)"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    read -p "  Démarrer Neo maintenant? [Y/n] " START
    START=${START:-Y}

    if [[ "$START" =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${GREEN}🚀 Démarrage de Neo...${NC}"
        echo ""
        npm run dev:cli
    else
        echo ""
        echo -e "  Pour démarrer plus tard: ${CYAN}npm run dev:cli${NC}"
        echo ""
    fi
}

# Main
main() {
    print_logo

    echo -e "${BOLD}Bienvenue dans l'installation de Neo!${NC}"
    echo ""
    echo "Neo est un assistant IA avec:"
    echo -e "  ${CYAN}◆${NC} 3 agents spécialisés (Vox, Brain, Memory)"
    echo -e "  ${CYAN}◆${NC} Mémoire persistante 10+ ans"
    echo -e "  ${CYAN}◆${NC} Auto-apprentissage"
    echo -e "  ${CYAN}◆${NC} Multi-modèles (Claude, Groq, Ollama)"
    echo ""

    read -p "Commencer l'installation? [Y/n] " START_INSTALL
    START_INSTALL=${START_INSTALL:-Y}

    if [[ ! "$START_INSTALL" =~ ^[Yy]$ ]]; then
        echo "Installation annulée."
        exit 0
    fi

    install_nodejs
    install_build_tools
    install_dependencies
    create_directories
    configure_api
    install_ollama
    start_neo
}

# Exécuter
main "$@"
