#!/bin/bash

# AI Bot - Script d'installation rapide
# Usage: ./setup.sh ou bash setup.sh

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                                                               ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}     🤖 ${BOLD}AI BOT - Installation Rapide${NC}                          ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                                                               ${CYAN}║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Vérifier Node.js
echo -e "${BLUE}[1/4]${NC} Vérification de Node.js..."
if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js n'est pas installé${NC}"
    echo -e "  Installez Node.js 20+ depuis: https://nodejs.org"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 20 ]; then
    echo -e "${RED}✗ Node.js 20+ requis. Version actuelle: $(node -v)${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Node.js $(node -v)"

# Installer les dépendances
echo ""
echo -e "${BLUE}[2/4]${NC} Installation des dépendances..."
npm install --silent
echo -e "${GREEN}✓${NC} Dépendances backend installées"

if [ -d "web" ]; then
    cd web && npm install --silent && cd ..
    echo -e "${GREEN}✓${NC} Dépendances frontend installées"
fi

# Créer les répertoires
echo ""
echo -e "${BLUE}[3/4]${NC} Configuration..."
mkdir -p data/vectors data/logs data/sessions
echo -e "${GREEN}✓${NC} Répertoires créés"

# Configurer la clé API
echo ""
echo -e "${BLUE}[4/4]${NC} Configuration de l'API Claude"

if [ -f ".env" ] && grep -q "ANTHROPIC_API_KEY" .env; then
    echo -e "${GREEN}✓${NC} Clé API trouvée dans .env"
else
    echo ""
    echo -e "  ${YELLOW}Obtenez votre clé sur: https://console.anthropic.com/${NC}"
    echo ""
    read -p "  Clé API Anthropic (sk-ant-...): " API_KEY

    if [[ ! "$API_KEY" =~ ^sk-ant- ]]; then
        echo -e "${RED}✗ Format de clé invalide${NC}"
        exit 1
    fi

    cat > .env << EOF
ANTHROPIC_API_KEY=$API_KEY
PORT=3001
DEBUG=false
CORS_ORIGIN=http://localhost:5173
JWT_SECRET=aibot-$(date +%s)-$(openssl rand -hex 4)
EOF

    echo -e "${GREEN}✓${NC} Fichier .env créé"
fi

# Terminé
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}✓ Installation terminée!${NC}"
echo ""
echo -e "  ${BOLD}Pour démarrer:${NC}"
echo ""
echo -e "    ${CYAN}npm run dev${NC}      # Serveur API (port 3001)"
echo -e "    ${CYAN}npm run dev:web${NC}  # Frontend React (port 5173)"
echo -e "    ${CYAN}npm run dev:cli${NC}  # Mode CLI"
echo ""
echo -e "  ${BOLD}Ou lancez le wizard interactif:${NC}"
echo ""
echo -e "    ${CYAN}npm run setup${NC}"
echo ""

# Demander si on démarre
read -p "  Démarrer le serveur maintenant? [Y/n] " START
START=${START:-Y}

if [[ "$START" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${GREEN}🚀 Démarrage du AI Bot...${NC}"
    echo ""
    echo -e "  ${YELLOW}📡 API:${NC}      http://localhost:3001"
    echo -e "  ${YELLOW}🌐 Web UI:${NC}   Lancez 'npm run dev:web' dans un autre terminal"
    echo ""
    npm run dev
fi
