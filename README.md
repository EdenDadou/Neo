# Neo Core

**Écosystème IA Multi-Agents Autonome** — Un système d'intelligence artificielle modulaire basé sur une architecture multi-agents avec mémoire persistante, orchestration intelligente et providers LLM multiples.

## Architecture

Neo Core repose sur trois agents principaux qui collaborent :

- **Vox** — Interface utilisateur. Reçoit les messages, détecte les intentions, délègue au Brain.
- **Brain** — Orchestrateur. Planifie, raisonne, gère les workers spécialisés (researcher, coder, analyst, etc.).
- **Memory** — Mémoire persistante. Double stockage ChromaDB (sémantique) + SQLite (métadonnées structurées).

Les agents communiquent via un système de providers LLM interchangeables (Anthropic, Groq, Gemini, Ollama) avec routing intelligent et fallback automatique.

## Installation

```bash
# Cloner le repo
git clone https://github.com/EdenDadou/Neo.git
cd Neo

# Installer (mode développement)
pip install -e ".[dev]"

# Installer avec les providers optionnels
pip install -e ".[dev,providers]"

# Configurer l'environnement
cp .env.example .env
# Éditer .env avec vos clés API
```

## Configuration

Créez un fichier `.env` à la racine :

```env
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...           # Optionnel
GEMINI_API_KEY=...             # Optionnel
OLLAMA_URL=http://localhost:11434  # Optionnel
NEO_PROVIDER_MODE=economic     # economic | quality
```

Puis lancez le wizard d'installation :

```bash
neo setup
```

## Utilisation

```bash
# Chat interactif
neo chat

# Statut du système
neo status

# Lister les providers LLM actifs
neo providers
```

## Commandes CLI

| Commande | Description |
|----------|-------------|
| `neo setup` | Wizard d'installation |
| `neo chat` | Chat interactif avec Neo |
| `neo status` | Dashboard de santé du système |
| `neo providers` | Providers LLM et routing |

## Tests

```bash
# Tous les tests
pytest tests/ -v

# Avec couverture
pytest tests/ -v --cov=neo_core --cov-report=term-missing

# Un stage spécifique
pytest tests/test_stage14_providers.py -v
```

## Structure du projet

```
neo_core/
├── api/              # API REST FastAPI (routes, middleware, websocket)
├── cli/              # Interface CLI (chat, status, setup)
├── core/             # Agents principaux (vox, brain, memory_agent, workers)
├── memory/           # Stockage persistant (store, consolidation)
├── providers/        # LLM providers (anthropic, groq, gemini, ollama, registry)
├── tools/            # Outils des agents (web search, etc.)
├── config.py         # Configuration centralisée
└── oauth.py          # Authentification OAuth
tests/
├── conftest.py       # Fixtures partagées
├── test_stage*.py    # Tests par stage (1-15+)
```

## Providers LLM

Neo Core supporte plusieurs fournisseurs de modèles avec routing intelligent :

| Provider | Type | Modèles | Usage |
|----------|------|---------|-------|
| Anthropic | Cloud payant | Claude Sonnet, Haiku | Brain, workers complexes |
| Groq | Cloud gratuit | Llama 3.3 70B | Workers, fallback rapide |
| Gemini | Cloud gratuit | Gemini Pro/Flash | Workers légers |
| Ollama | Local gratuit | Deepseek, Llama, Mistral | Workers, mode économique |

Le mode `economic` privilégie les modèles locaux/gratuits. Le mode `quality` privilégie les modèles cloud performants.

## Licence

Projet privé — Eden Dadou.
