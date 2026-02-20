# Neo Core

**Écosystème IA Multi-Agents Autonome** — Un système d'intelligence artificielle modulaire basé sur une architecture multi-agents avec mémoire persistante, orchestration intelligente et providers LLM multiples.

## Architecture

Neo Core repose sur trois agents principaux qui collaborent :

- **Vox** — Routeur et gestionnaire de sessions. Reçoit les messages, gère l'historique (budget tokens), passe le message brut à Brain. Aucun appel LLM dans le pipeline principal.
- **Brain** — Orchestrateur. Comprend, décide, agit, délègue aux workers spécialisés (researcher, coder, analyst, writer, summarizer, translator). System prompt compact (~600 tokens).
- **Memory** — Mémoire persistante. Double stockage FAISS (vectoriel, 384 dim) + SQLite (métadonnées structurées).

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
| `neo start` | Lancer le daemon (API + Heartbeat + Telegram) |
| `neo stop` | Arrêter le daemon |
| `neo restart` | Redémarrer |
| `neo history` | Sessions précédentes |
| `neo health` | Vérification santé |
| `neo plugins` | Plugins chargés |
| `neo guardian` | Mode supervision (auto-restart) |
| `neo api` | Serveur REST FastAPI seul |
| `neo install-service` | Générer le service systemd |
| `neo update` | Mise à jour (git pull + restart) |
| `neo version` | Version actuelle |
| `neo logs` | Logs du daemon |

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
├── brain/                 # Orchestrateur (core.py, prompts.py, teams/, tools/, providers/)
│   ├── core.py            # Brain main class
│   ├── prompts.py         # System prompt compact + BrainDecision
│   ├── teams/             # Workers spécialisés (worker.py, factory.py, crew.py)
│   ├── tools/             # Outils intégrés + plugins + génération auto
│   └── providers/         # Multi-provider LLM (anthropic, groq, gemini, ollama)
├── vox/                   # Interface humaine
│   ├── interface.py       # Vox agent (routage, sessions, historique token-based)
│   ├── cli/               # CLI (chat TUI, setup wizard, status dashboard)
│   ├── api/               # REST API FastAPI (routes, middleware, websocket)
│   └── integrations/      # Telegram bot
├── memory/                # Mémoire persistante
│   ├── store.py           # FAISS + SQLite
│   ├── task_registry.py   # Tasks, Epics, projets récurrents
│   ├── learning.py        # Apprentissage boucle fermée
│   └── persona.py         # Identité évolutive
├── infra/                 # Infrastructure
│   ├── daemon.py          # Daemon lifecycle + systemd
│   ├── guardian.py        # Auto-restart crash recovery
│   ├── heartbeat.py       # Pouls autonome + projets récurrents
│   ├── resilience.py      # Retry, circuit breaker, health
│   └── security/          # Sanitizer + Vault chiffré
├── config.py              # Configuration centralisée
└── oauth.py               # Authentification OAuth Anthropic
tests/
├── conftest.py            # Fixtures partagées
├── test_stage*.py         # Tests par stage (1-18)
└── test_*.py              # Tests unitaires
```

## Providers LLM

Neo Core supporte plusieurs fournisseurs de modèles avec routing intelligent :

| Provider | Type | Modèles | Usage |
|----------|------|---------|-------|
| Anthropic | Cloud payant | Claude Sonnet 4.6, Haiku 4.5 | Brain, workers critiques (coder, analyst) |
| Groq | Cloud gratuit | LLaMA 3.3 70B | Workers, fallback rapide |
| Gemini | Cloud gratuit | Gemini 2.5 Flash | Workers légers |
| Ollama | Local gratuit | DeepSeek-R1:8b, Llama, Mistral | Workers secondaires, mode économique |

Le mode `economic` privilégie les modèles locaux/gratuits. Le mode `quality` privilégie les modèles cloud performants.

## Documentation

- **ARCHITECTURE.md** — Documentation technique complète (10 sections)
- **NEO_MANUAL.md** — Manuel d'installation et exploitation VPS

## Licence

Projet privé — Eden Dadou.
