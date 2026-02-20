# Neo Core — Architecture Technique

> **v0.9.7** | Écosystème IA Multi-Agents Autonome
> Dernière mise à jour : 20 février 2026

---

## Vue d'ensemble

Neo Core est un système d'IA multi-agents composé de **4 domaines** :

```
neo_core/
├── brain/           Orchestrateur — décisions, exécution, LLM, workers
├── vox/             Interface humaine — CLI, API REST, Telegram
├── memory/          Mémoire persistante — FAISS, apprentissage, persona
├── infra/           Infrastructure — daemon, guardian, résilience, sécurité
├── config.py        Configuration centrale
├── logging_config.py   Logging structuré
├── oauth.py         Gestion OAuth Anthropic
└── validation.py    Validation des inputs
```

### Les 3 agents

| Agent | Rôle | Modèle par défaut |
|-------|------|-------------------|
| **Vox** | Routeur et gestionnaire de sessions (zéro LLM dans le pipeline principal) | — |
| **Brain** | Orchestrateur : comprend, décide, agit, délègue | Claude Sonnet 4.6 |
| **Memory** | Bibliothécaire : stocke, cherche, consolide, apprend | Claude Haiku 4.5 (consolidation) |

### Flux principal — Smart Pipeline v5.0

```
Humain
  ↓ message brut (aucune reformulation)
Vox (routage + historique)
  ↓ message brut + historique conversation
Brain — UN SEUL appel Sonnet
  ├─ question simple → réponse directe
  ├─ action détectée → neo-action JSON en fin de réponse
  │   ├── search → Worker Researcher
  │   ├── delegate → Worker spécialisé
  │   ├── code → Worker Coder
  │   ├── create_project → ProjectManager (N workers)
  │   ├── create_recurring_project → Projet cyclique (heartbeat)
  │   └── crew_directive → Pilotage d'un projet en cours
  └─ réponse finale
  ↓
Memory ← stocke l'échange + apprend
Heartbeat ← consolide, auto-tune, self-patch (en arrière-plan)
```

**Point clé :** Vox ne fait **aucun appel LLM** dans le pipeline de messages. Le message utilisateur arrive brut à Brain (Sonnet), qui a l'historique complet et le contexte mémoire. Zéro intermédiaire = zéro perte d'intelligence.

---

## 1. VOX — Routeur et Gestionnaire de Sessions

### 1.1 interface.py — Agent Vox

**Classe `Vox`** (~590 lignes) — Point de contact entre humain et système.

**Pipeline `process_message(human_message)` :**
1. Validation input (Sanitizer)
2. Ajout à l'historique avec gestion par budget tokens (12K tokens, pas un compteur de messages)
3. Le message passe **brut** à Brain — plus de reformulation, plus de quick reply
4. Mode async (TUI) : génère un ACK statique, lance Brain en arrière-plan, délivre via callback
5. Mode sync (API/Telegram) : attend Brain, retourne la réponse
6. Sauvegarde dans ConversationStore + Memory

**Gestion de l'historique (token-based) :**
```python
_MAX_HISTORY_TOKENS = 12_000   # budget tokens
_SUMMARY_TRIGGER = 14_000      # seuil compression
```
Quand l'historique dépasse le budget, les 20 derniers messages sont conservés intacts et les plus anciens sont compressés en un résumé condensé.

**Méthodes encore présentes mais non utilisées dans le pipeline principal :**
- `_is_simple_message()` — détection de messages simples (dead code dans le flux principal)
- `_vox_quick_reply()` — réponse Vox sans Brain (dead code dans le flux principal)
- `format_request_async()` — reformulation LLM (dead code dans le flux principal)

**Vox utilise encore le LLM pour :**
- `generate_session_summary()` — résumé de fin de session via Haiku

**Gestion sessions** : `start_new_session()`, `resume_session()` (thread-safe via Lock)

**`bootstrap()`** : Initialise Config → Memory → Brain → Vox, retourne Vox prêt

### 1.2 api/ — REST API (FastAPI)

#### server.py — Factory
`create_app(config)` :
- Lifespan : initialise CoreRegistry au démarrage
- Middleware : CORS → APIKeyMiddleware → RateLimitMiddleware → SanitizerMiddleware

#### routes.py — Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| POST | `/chat` | Envoie un message (120s timeout) |
| GET | `/health` | Santé du système (public) |
| GET | `/status` | Statut détaillé (auth requise) |
| GET | `/sessions` | Liste les sessions (limit=20) |
| GET | `/sessions/{id}/history` | Historique d'une session |
| GET | `/persona` | Personnalité Neo + profil utilisateur |
| WS | `/ws/chat?token=` | Chat temps réel WebSocket |

#### middleware.py — Sécurité
- **APIKeyMiddleware** : header `X-Neo-Key`, comparaison timing-safe (`hmac.compare_digest`)
- **RateLimitMiddleware** : 60 req/min par IP, SQLite persistant
- **SanitizerMiddleware** : bloque les menaces medium/high sur POST/PUT/PATCH

### 1.3 cli/ — Interface Terminal

#### \_\_init\_\_.py — Dispatch des commandes
`main()` route les commandes : `setup`, `chat`, `start`, `stop`, `restart`, `status`, `history`, `health`, `providers`, `plugins`, `guardian`, `version`, `api`, `install-service`, `telegram-setup`, `update`, `logs`

#### chat.py — Boucle conversationnelle Rich
- Intégration Guardian (GracefulShutdown, StateSnapshot)
- Heartbeat en arrière-plan
- File de résultats Brain async
- Commandes slash : `/quit`, `/status`, `/health`, `/help`, `/history`, `/sessions`, `/skills`, `/tasks`, `/epics`, `/heartbeat`, `/persona`, `/profile`, `/reflect`, `/restart`

#### setup.py — Wizard d'installation (8 étapes)
#### status.py — Dashboard système (tables Rich)
#### history.py — Navigateur de sessions

### 1.4 integrations/telegram.py — Bot Telegram

- Whitelist user_ids, rate limit 20 req/min
- Sanitisation input, split réponses en chunks 4096 chars
- Partage l'instance Vox avec CLI/API

---

## 2. BRAIN — Orchestrateur

### 2.1 core.py — Cerveau principal

**Classe `Brain`** (~700 lignes) — Point d'entrée de toute requête.

**Pipeline `process(request, conversation_history)` :**
1. Valide l'input
2. Détecte les commandes slash → traitement direct
3. Appel LLM Sonnet avec :
   - System prompt compact (~600 tokens)
   - Historique conversation complet
   - Contexte mémoire enrichi (projets actifs, sessions récentes)
   - Contexte utilisateur (profil, préférences)
4. Parse la réponse pour détecter les blocs `neo-action`
5. Si action détectée → exécute l'action correspondante
6. Retourne la réponse (message humain + résultat d'action)

**Actions supportées :**
- `search {query}` → Worker Researcher
- `delegate {task, worker}` → Worker spécialisé
- `code {code}` → Worker Coder
- `create_project {name, steps}` → Décompose en Epic, exécute via ProjectManager
- `create_recurring_project {name, steps, goal, schedule}` → Projet cyclique
- `crew_directive {project, type, detail}` → Pilotage d'un projet en cours

**Injection de contexte dans le prompt :**
```python
# Projets et sessions sont injectés dans memory_context, pas dans le system prompt
projects_context = self._build_full_projects_context()
if projects_context:
    memory_context += f"\n\n=== PROJETS ACTIFS ===\n{projects_context}"
recent_sessions = self._build_recent_sessions_context()
if recent_sessions:
    memory_context += f"\n\n=== SESSIONS RÉCENTES ===\n{recent_sessions}"
```

**Workers reçoivent le contexte conversation :**
Les 5 derniers échanges (10 messages) sont transmis aux workers pour qu'ils comprennent le contexte.

**Classe `WorkerLifecycleManager`** — Thread-safe, gère le cycle de vie des Workers :
- `register(worker)` → génère un worker_id
- `unregister(worker)` → sauvegarde dans l'historique (max 50)
- `cleanup_all()` → nettoyage forcé à l'arrêt

### 2.2 prompts.py — Prompts système

- `BRAIN_SYSTEM_PROMPT` : ~18 lignes, compact, avec placeholders `{memory_context}`, `{user_context}`, `{current_date}`, `{current_time}`
- `DECOMPOSE_PROMPT` : Template JSON pour décomposition de tâches (sans limite artificielle de nombre d'étapes)
- `BrainDecision` : Dataclass de décision stratégique

### 2.3 llm.py — Infrastructure LLM

**Authentification (3 méthodes)** :
1. **OAuth Bearer** + beta header (`sk-ant-oat...`)
2. **Clé API convertie** depuis OAuth (`sk-ant-api...`)
3. **LangChain classique** (`x-api-key`)

### 2.4 execution.py — Pipeline d'exécution

**`execute_with_worker()`** : Crée un Worker, exécute avec retry (max 3 tentatives), apprend du résultat.

**`execute_as_epic()`** : Décompose en sous-tâches, exécute séquentiellement via Workers, compile les résultats.

### 2.5 self_patcher.py — Auto-correction (Level 3)

Cycle : DETECTION → GENERATION → VALIDATION → APPLICATION → MONITORING → ROLLBACK
Détecte les patterns d'erreurs (count >= 3, recent < 7j), génère des patches comportementaux.

### 2.6 auto_tuner.py — Optimisation paramétrique (Level 1)

Ajustement automatique par worker_type : température, timeout, retries, tool iterations.
Persisté dans `data/auto_tuning.json`.

### 2.7 providers/ — Routage multi-LLM

#### base.py — Abstractions
- `LLMProvider` (abstract) : `list_models()`, `chat()`, `test_model()`
- `ModelInfo` : model_id, capability (BASIC/STANDARD/ADVANCED)
- `ChatResponse` : format unifié cross-provider

#### registry.py — Catalogue et routage (~600 lignes)
**Classe `ModelRegistry`** (singleton) — Découverte, test, et sélection des modèles.

**Priorité des providers :**
1. Ollama (local, gratuit, illimité)
2. Groq (cloud gratuit)
3. Gemini (cloud gratuit)
4. Anthropic (cloud payant)

**Agents core (toujours Cloud Anthropic)** : vox, brain, memory, worker:coder, worker:analyst

#### router.py — Routage unifié
`route_chat(agent_name, messages, ...)` : essaie chaque modèle de la fallback chain, puis `_fallback_anthropic()` en dernier recours.

#### Providers spécifiques

| Provider | Modèles | Auth | Particularités |
|----------|---------|------|----------------|
| **Anthropic** | Claude Sonnet 4.6, Haiku 4.5 | API key ou OAuth Bearer | Tool use natif |
| **Groq** | LLaMA 3.3 70B, Llama 3 8B, Gemma 2 9B | Bearer token | Format OpenAI |
| **Gemini** | 2.5 Flash, 2.0 Flash-Lite | API key en URL | systemInstruction séparé |
| **Ollama** | Auto-découverts (DeepSeek-R1:8b, Llama, Mistral...) | Aucun (local) | Catalogue dynamique |

### 2.8 teams/ — Workers spécialisés

#### worker.py — Agent d'exécution
**Types** : RESEARCHER, CODER, SUMMARIZER, ANALYST, WRITER, TRANSLATOR, GENERIC

Chaque worker reçoit un system prompt spécialisé et les 5 derniers échanges de la conversation.

**Classe `Worker`** : async context manager avec lifecycle garanti
- `execute()` → `WorkerResult` (success, output, errors, tool_calls, execution_time)

#### factory.py — WorkerFactory
Classifie la tâche par score regex, crée le Worker approprié avec ses outils.

#### project_manager.py — ProjectManager
Coordonne l'exécution d'un Epic (projet multi-étapes) avec plusieurs Workers.

#### crew.py — CrewExecutor
Exécution parallèle/séquentielle de tâches au sein d'un projet.

### 2.9 tools/ — Outils des Workers

#### base_tools.py — 6 outils intégrés

| Outil | Fonction |
|-------|----------|
| `web_search_tool(query)` | Recherche DuckDuckGo (ddgs) |
| `web_fetch_tool(url)` | Fetch + nettoyage HTML |
| `file_read_tool(path)` | Lecture avec sécurité racine |
| `file_write_tool(path, content)` | Écriture avec contrôle d'accès |
| `code_execute_tool(code)` | Exécution Python sandbox |
| `memory_search_tool(query)` | Recherche sémantique via Memory |

#### plugin_loader.py — Plugins dynamiques
Charge des plugins depuis `data/plugins/` avec hot-reload, timeout 30s, namespace isolé.

#### tool_generator.py — Génération autonome (Level 4)
Détecte les besoins récurrents, génère des plugins Python, valide et déploie automatiquement.

---

## 3. MEMORY — Mémoire Persistante

### 3.1 agent.py — MemoryAgent (Bibliothécaire)

**Orchestrateur** de tous les sous-systèmes mémoire :
- `initialize()` : crée Store, ContextEngine, Consolidator, LearningEngine, TaskRegistry, PersonaEngine
- `store_memory(content, source, tags, importance)` → stocke dans FAISS + SQLite
- `get_context(query)` → recherche sémantique pour injection dans les prompts
- `on_conversation_turn(user_msg, ai_response)` : stocke, enrichit les tâches, analyse (persona), consolide (tous les 50 tours)

### 3.2 store.py — Stockage vectoriel FAISS

**Double couche :**

| Couche | Technologie | Rôle |
|--------|-------------|------|
| Vecteurs | FAISS IndexFlatIP (384 dim) | Recherche sémantique |
| Métadonnées | SQLite (memory_meta.db) | Stockage structuré |
| Embeddings | FastEmbed/ONNX all-MiniLM-L6-v2 | Vectorisation |

Optimisations : modèle d'embedding caché au niveau process, cache sémantique par requête, index SQLite, WAL journal mode.

### 3.3 context.py — Injection de contexte

`build_context(query)` → `ContextBlock` avec : relevant_memories (top 5), important_memories (>= 0.7, max 2), recent_memories (max 2). Tronqué si > max_context_tokens.

### 3.4 conversation.py — Persistance des sessions

Tables SQLite : `sessions` + `turns`. Thread-safe, export JSON.

### 3.5 consolidator.py — Consolidation mémoire

Comme le sommeil humain : cleanup (> 30 jours, importance < 0.2), merge (doublons sémantiques > 0.85), promote (entrées populaires).

### 3.6 learning.py — Apprentissage (boucle fermée)

Enregistre succès/échecs, met à jour les LearnedSkill, ErrorPattern, WorkerPerformance. Donne des conseils pour les retries.

### 3.7 persona.py — Identité et personnalité (Stage 9)

**3 Commandements immuables** : "Neo ne s'éteint jamais", "Neo n'oublie jamais", "Neo apprend tous les jours"

**Traits** (échelle 0.0–1.0, évolution EMA) : communication_style, humor_level, patience, verbosity, curiosity, empathy, formality, expressiveness

**Profil utilisateur** (auto-appris) : préférences, patterns, observations.

### 3.8 task_registry.py — Registre de tâches

**Task** : id, description, worker_type, status (pending/in_progress/done/failed), context_notes
**Epic** : id, description, task_ids, strategy, status

**Projets récurrents** (nouveau) :
```python
recurring: bool = False
schedule_cron: str = ""
schedule_interval_minutes: int = 0
cycle_count: int = 0
max_cycles: int = 0
cycle_template: list[dict]   # template de tâches pour chaque cycle
goal: str = ""
goal_reached: bool = False
accumulated_results: list[str]  # résultats chaînés entre cycles
```
Méthodes : `get_recurring_epics_due()`, `advance_recurring_cycle()`, `_next_cron_time()`

### 3.9 migrations.py — Versioning schéma SQLite

---

## 4. INFRA — Infrastructure

### 4.1 registry.py — Singleton des agents

**Classe `CoreRegistry`** — Garantit UNE seule instance Vox/Brain/Memory par process.
Singleton via `__new__()` + threading.Lock. Utilisé par CLI, API, Telegram et Daemon.

### 4.2 daemon.py — Gestion daemon

- Mode foreground (pour systemd) ou background (double fork)
- Lance en parallèle : uvicorn + Heartbeat + Telegram (optionnel)
- Signaux : SIGTERM/SIGINT → arrêt propre, SIGHUP → rechargement config
- `generate_systemd_service()` : template avec hardening sécurité

### 4.3 guardian.py — Supervision de processus

Boucle infinie de restart avec backoff exponentiel (1s → 60s max). Protection : max 10 restarts/heure. StateSnapshot pour reprise après crash.

### 4.4 heartbeat.py — Pouls autonome

| Action | Fréquence | Description |
|--------|-----------|-------------|
| Advance epics | Chaque pulse (~5 min) | Exécute la prochaine tâche des epics en cours |
| Detect stale | Chaque pulse | Alerte si tâche in_progress > 10 min |
| Recurring cycles | Chaque pulse | Vérifie et déclenche les projets récurrents dus |
| Consolidation | 10 pulses (~50 min) | Nettoyage + merge mémoire |
| Self-reflection | 24h | LLM analyse les interactions, ajuste la persona |
| Auto-tuning | 5 pulses (~25 min) | Level 1 : ajuste température, timeout, retries |
| Self-patching | 10 pulses (~50 min) | Level 3 : détecte + corrige les patterns d'erreur |
| Tool generation | 15 pulses (~75 min) | Level 4 : génère des plugins automatiquement |

### 4.5 resilience.py — Tolérance aux pannes

**RetryConfig** : max 3 retries, backoff exponentiel avec jitter.
**CircuitBreaker** : Closed → Open (>= 5 échecs, timeout 60s) → Half-Open.
**HealthMonitor** : error_rate, avg_response_time, statut (healthy/degraded).

### 4.6 security/sanitizer.py — Filtrage des inputs

| Menace | Sévérité | Action |
|--------|----------|--------|
| Prompt injection | HIGH | Bloque |
| SQL injection | HIGH | Bloque |
| XSS | MEDIUM | [FILTERED] |
| Path traversal | MEDIUM | [BLOCKED] |
| Dépassement longueur (> 10 000 chars) | LOW | Tronque |

### 4.7 security/vault.py — Chiffrement des secrets

Fernet (AES-128-CBC + HMAC-SHA256), PBKDF2 480K itérations, lié à la machine (machine-id).

---

## 5. Configuration

### config.py — Configuration centrale

**NeoConfig** agrège : LLMConfig, MemoryConfig, ResilienceConfig, SelfImprovementConfig.

**AGENT_MODELS** : mapping agent → modèle LLM par défaut
- vox, memory, worker:summarizer/writer/translator/generic → Haiku 4.5 (rapide)
- brain, worker:researcher/coder/analyst → Sonnet 4.6 (raisonnement)

**Priorité de chargement** :
1. Vault (secrets chiffrés)
2. Variables d'environnement (.env)
3. Wizard config (data/neo_config.json)
4. Defaults intégrés

### oauth.py — Gestion OAuth Anthropic

Types de token : `sk-ant-oat...` (access, ~8h), `sk-ant-ort...` (refresh), `sk-ant-api...` (permanent).
Import auto depuis Claude Code credentials. Conversion OAuth → API key.

---

## 6. Points d'entrée

| Commande | Description |
|----------|-------------|
| `neo chat` | Chat interactif TUI (Rich) |
| `neo setup` | Wizard d'installation |
| `neo start` | Lancement daemon (API + Heartbeat + Telegram) |
| `neo stop` | Arrêt daemon |
| `neo restart` | Redémarrage |
| `neo status` | Dashboard système |
| `neo providers` | Providers LLM et routing |
| `neo history` | Sessions précédentes |
| `neo health` | Vérification santé |
| `neo plugins` | Plugins chargés |
| `neo guardian` | Mode supervision (auto-restart) |
| `neo api` | Serveur REST seul |
| `neo install-service` | Génère le service systemd |
| `neo update` | git pull + restart |
| `neo version` | Version actuelle |

---

## 7. Niveaux d'autonomie

| Niveau | Nom | Description | Localisation |
|--------|-----|-------------|--------------|
| 0 | Exécution | Répond aux requêtes via LLM | brain/core.py |
| 1 | Auto-tuning | Ajuste température, timeout, retries | brain/auto_tuner.py |
| 2 | Apprentissage | Apprend des succès/échecs, conseille | memory/learning.py |
| 3 | Self-patching | Détecte et corrige les patterns d'erreur | brain/self_patcher.py |
| 4 | Tool generation | Génère des plugins automatiquement | brain/tools/tool_generator.py |
| 5 | Résilience | Circuit breaker, retry, health monitoring | infra/resilience.py |
| 9 | Persona | Évolue sa personnalité, apprend l'utilisateur | memory/persona.py |

---

## 8. Sécurité

| Couche | Module | Protection |
|--------|--------|------------|
| Input | validation.py | Longueur, type, format |
| Sanitisation | infra/security/sanitizer.py | Prompt injection, SQL, XSS, path traversal |
| Auth API | vox/api/middleware.py | API key timing-safe, rate limit 60/min |
| Auth Telegram | vox/integrations/telegram.py | Whitelist user_ids, rate limit 20/min |
| Secrets | infra/security/vault.py | Fernet AES-128, PBKDF2 480K, machine-bound |
| OAuth | oauth.py | Tokens chiffrés, refresh auto, conversion API key |
| Systemd | infra/daemon.py | ProtectSystem=strict, NoNewPrivileges |
| Tools | brain/tools/tool_generator.py | Imports interdits (os, subprocess, sys...) |
| Workers | brain/tools/base_tools.py | Racines autorisées pour fichiers |

---

## 9. Dépendances

### Core
```
langchain, langchain-anthropic, langchain-community  — LLM integration
faiss-cpu                                             — Recherche vectorielle
fastembed (ONNX)                                      — Embeddings (all-MiniLM-L6-v2)
fastapi, uvicorn                                      — API REST
httpx                                                 — Client HTTP async
rich                                                  — TUI terminal
python-dotenv                                         — Variables .env
ddgs                                                  — DuckDuckGo search
psutil                                                — Monitoring processus
cryptography                                          — Fernet/PBKDF2 pour le vault
```

### Optionnels
```
groq                    — Provider Groq (cloud gratuit)
google-generativeai     — Provider Gemini (cloud gratuit)
ollama                  — Provider Ollama (local)
python-telegram-bot     — Bot Telegram
```

---

## 10. Structure complète des fichiers

```
neo_core/
├── brain/
│   ├── core.py                # Brain main class (~700 lignes)
│   ├── prompts.py             # System prompt compact + BrainDecision
│   ├── llm.py                 # OAuth + appels LLM
│   ├── execution.py           # Pipeline worker/epic
│   ├── auto_tuner.py          # Level 1 auto-adjustment
│   ├── self_patcher.py        # Level 3 auto-correction
│   ├── providers/
│   │   ├── base.py            # Abstractions LLMProvider
│   │   ├── registry.py        # ModelRegistry singleton (~600 lignes)
│   │   ├── router.py          # route_chat unifié
│   │   ├── anthropic_provider.py
│   │   ├── groq_provider.py
│   │   ├── gemini_provider.py
│   │   ├── ollama_provider.py
│   │   ├── bootstrap.py       # Initialisation providers
│   │   └── hardware.py        # Détection CPU/RAM/GPU
│   ├── teams/
│   │   ├── worker.py          # Worker + 7 types + lifecycle
│   │   ├── factory.py         # WorkerFactory
│   │   ├── project_manager.py # Coordination multi-worker
│   │   └── crew.py            # Exécution parallèle
│   └── tools/
│       ├── base_tools.py      # 6 outils intégrés
│       ├── plugin_loader.py   # Plugins dynamiques
│       └── tool_generator.py  # Génération autonome
│
├── vox/
│   ├── interface.py           # Vox agent (routage, sessions, historique)
│   ├── cli/
│   │   ├── __init__.py        # CLI dispatcher (20+ commandes)
│   │   ├── chat.py            # TUI Rich interactif
│   │   ├── setup.py           # Wizard 8 étapes
│   │   ├── status.py          # Dashboard
│   │   └── history.py         # Navigateur sessions
│   ├── api/
│   │   ├── server.py          # FastAPI factory
│   │   ├── routes.py          # 8 endpoints
│   │   ├── middleware.py      # Auth, rate limit, sanitizer
│   │   └── schemas.py         # Pydantic models
│   └── integrations/
│       └── telegram.py        # Bot Telegram
│
├── memory/
│   ├── agent.py               # MemoryAgent orchestrateur
│   ├── store.py               # FAISS + SQLite (384-dim)
│   ├── context.py             # Injection de contexte
│   ├── consolidator.py        # Consolidation (sommeil)
│   ├── conversation.py        # Sessions SQLite
│   ├── learning.py            # Boucle fermée
│   ├── task_registry.py       # Tasks, Epics, projets récurrents
│   ├── persona.py             # Identité évolutive
│   ├── working_memory.py      # Scratchpad temps réel
│   └── migrations.py          # Versioning SQLite
│
├── infra/
│   ├── registry.py            # CoreRegistry singleton
│   ├── daemon.py              # Daemon lifecycle + systemd
│   ├── guardian.py            # Auto-restart crash recovery
│   ├── heartbeat.py           # Pouls autonome + projets récurrents
│   ├── resilience.py          # Retry, circuit breaker, health
│   └── security/
│       ├── sanitizer.py       # Input filtering
│       └── vault.py           # Fernet encrypted secrets
│
├── config.py                  # Configuration centralisée
├── logging_config.py          # Structured logging
├── oauth.py                   # OAuth Anthropic
├── validation.py              # Input validation
└── features.py                # Feature flags

data/
├── .vault.db                  # Secrets chiffrés (Fernet-AES)
├── .oauth_credentials.json    # Tokens OAuth
├── neo_config.json            # Config runtime
├── neo.log                    # Logs daemon (rotation 5MB x3)
├── working_memory.json        # Scratchpad
├── auto_tuning.json           # Paramètres auto-tuning
├── memory/
│   ├── memory.db              # SQLite mémoire
│   ├── faiss_index.bin        # Index vectoriel
│   └── faiss_id_map.json      # Mapping IDs
├── guardian/state.json        # Snapshots crash recovery
├── plugins/                   # Plugins installés + auto-générés
├── patches/                   # Patches comportementaux
└── system_docs/               # Documentation système embarquée

tests/                         # 35 fichiers de test (pytest)
```
