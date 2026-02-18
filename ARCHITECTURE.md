# Neo Core — Architecture & Documentation

> **v0.9.4** | Ecosysteme IA Multi-Agents Autonome

---

## Vue d'ensemble

Neo Core est un systeme d'IA multi-agents compose de **4 domaines** :

```
neo_core/
├── brain/           Orchestrateur — decisions, execution, LLM, workers
├── vox/             Interface humaine — CLI, API REST, Telegram
├── memory/          Memoire persistante — FAISS, apprentissage, persona
├── infra/           Infrastructure — daemon, guardian, resilience, securite
├── config.py        Configuration centrale
├── logging_config.py   Logging structure
├── oauth.py         Gestion OAuth Anthropic
└── validation.py    Validation des inputs
```

### Les 3 agents

| Agent | Role | Modele par defaut |
|-------|------|-------------------|
| **Vox** | Interface humaine : recoit, reformule, restitue | Claude Haiku (rapide) |
| **Brain** | Orchestrateur : decide, delegue, apprend | Claude Sonnet (raisonnement) |
| **Memory** | Bibliothecaire : stocke, cherche, consolide | Claude Haiku (consolidation) |

### Flux principal

```
Humain
  ↓ message
Vox (reformulation)
  ↓ requete structuree
Brain (decision)
  ├─ simple → reponse directe LLM
  ├─ moderate → delegue a 1 Worker
  └─ complexe → decompose en Epic (N Workers)
  ↓ reponse
Vox (restitution)
  ↓ message
Humain

[En parallele]
Memory ← stocke chaque echange
Memory ← apprend des succes/echecs
Heartbeat ← consolide, auto-tune, self-patch
```

---

## 1. BRAIN — Orchestrateur

### 1.1 core.py — Cerveau principal

**Classe `Brain`** (~700 lignes) — Point d'entree de toute requete complexe.

**Pipeline `process(request, conversation_history)` :**
1. Valide l'input (ValidationError)
2. Analyse la complexite (simple/moderate/complexe)
3. Prend une decision (`BrainDecision`) — AVANT de charger le contexte memoire (optim v0.9.1)
4. Charge le contexte memoire si necessaire
5. Verifie le circuit breaker (Stage 5)
6. Execute la strategie choisie
7. Apprend du resultat (LearningEngine)

**Classe `WorkerLifecycleManager`** — Thread-safe, gere le cycle de vie des Workers :
- `register(worker)` → genere un worker_id
- `unregister(worker)` → sauvegarde dans l'historique (max 50)
- `cleanup_all()` → nettoyage force a l'arret

**Decision (`BrainDecision`)** :
- `action` : "direct_response" | "delegate_worker" | "delegate_crew"
- `worker_type`, `subtasks`, `confidence`, `reasoning`
- Patches Level 3 appliques avant la decision finale

### 1.2 prompts.py — Prompts systeme

- `BRAIN_SYSTEM_PROMPT` : 104 lignes definissant le role de Brain, ses capacites ("NEO CAN DO EVERYTHING"), regles de routing
- `DECOMPOSE_PROMPT` : Template JSON pour decomposition de taches
- `BrainDecision` : Dataclass de decision strategique

### 1.3 llm.py — Infrastructure LLM

**Authentification (3 methodes)** :
1. **OAuth Bearer** + beta header (`sk-ant-oat...`)
2. **Cle API convertie** depuis OAuth (`sk-ant-api...`)
3. **LangChain classique** (`x-api-key`)

**Fonctions cles** :
- `init_llm(brain)` → detecte le type d'auth, initialise le client
- `raw_llm_call(brain, prompt)` → appel LLM via multi-provider, fallback LangChain
- `oauth_response(brain, request, context, history)` → reponse complete avec system prompt
- `llm_response(brain, request, context, history)` → reponse via LangChain
- `mock_response(request, complexity, context)` → reponse deterministe (tests)

### 1.4 execution.py — Pipeline d'execution

**`execute_with_worker(brain, request, decision, context, analysis)`** :
1. Cree un TaskRecord dans le TaskRegistry
2. Boucle de retry (max 3 tentatives) :
   - Tentative 2+ : ameliore la strategie via Memory
   - Cree un Worker via Factory
   - Execute le Worker (async context manager)
   - Apprend du resultat AVANT desinscription
3. Si echec total → marque la tache "failed"

**`execute_as_epic(brain, request, decision, context)`** :
- Cree un Epic dans le TaskRegistry
- Execute les sous-taches sequentiellement via Workers
- Compile les resultats avec indicateurs succes/echec

**`improve_strategy(brain, request, decision, errors, attempt)`** :
- Tentative 2 : consulte Memory pour un worker alternatif
- Tentative 3 : fallback worker generique + simplification

### 1.5 self_patcher.py — Auto-correction (Level 3)

**Cycle** : DETECTION → GENERATION → VALIDATION → APPLICATION → MONITORING → ROLLBACK

**Classe `SelfPatcher`** :
- `detect_patchable_patterns()` : lit les ErrorPattern du LearningEngine (count >= 3, recent < 7j)
- `generate_patch(pattern)` : cree un PatchMetadata selon le type d'erreur :
  - hallucination → baisse temperature a 0.3
  - tool_failure → routing vers worker fallback
  - timeout → augmentation timeout
  - routing_error → regle de routing alternative
- `validate_patch()` : estime l'amelioration, active si >= 50%
- `apply_patches(request, worker_type)` : retourne les overrides actifs
- `evaluate_and_rollback_all()` : desactive les patches inefficaces

**Persistance** : `data/patches/*.json`

### 1.6 auto_tuner.py — Optimisation parametrique (Level 1)

**Classe `AutoTuner`** — Ajustement automatique par worker_type :
- Temperature : baisse si hallucinations (bornes 0.1–1.0)
- Timeout : augmente si >= 40% timeouts (bornes 60–600s)
- Max retries : augmente si >= 50% erreurs transitoires (bornes 1–5)
- Max tool iterations : ajustement stable (bornes 5–25)

**Persistance** : `data/auto_tuning.json`

### 1.7 providers/ — Routage multi-LLM

#### base.py — Abstractions
- `LLMProvider` (abstract) : `list_models()`, `chat()`, `test_model()`
- `ModelInfo` : model_id, capability (BASIC/STANDARD/ADVANCED), latence, statut
- `ChatMessage` / `ChatResponse` : format unifie cross-provider

#### registry.py — Catalogue et routage (~600 lignes)
**Classe `ModelRegistry`** (singleton via `get_model_registry()`) :
- Decouverte : `discover_models()` fusionne tous les providers
- Test : `test_model(id)`, `test_all()`
- Routing : `get_best_for(agent_name)` → selectionne le meilleur modele

**Priorite des providers** :
1. Ollama (local, gratuit, illimite)
2. Groq (cloud gratuit)
3. Gemini (cloud gratuit)
4. Anthropic (cloud payant)

**Agents core (toujours Cloud)** : vox, brain, memory, worker:coder, worker:analyst

#### router.py — Routage unifie
**`route_chat(agent_name, messages, ...)`** :
1. Obtient la fallback chain du registry
2. Pour chaque modele : appelle `provider.chat()`
3. Si tout echoue → `_fallback_anthropic()` (httpx direct)

**`route_chat_raw()`** : meme chose mais retourne le format dict Anthropic (pour les Workers)

#### Providers specifiques
| Provider | Modeles | Auth | Particularites |
|----------|---------|------|----------------|
| **Anthropic** | Sonnet 4.5, Haiku 4.5 | API key ou OAuth Bearer | Tool use natif |
| **Groq** | Llama 3.3 70B, Llama 3 8B, Gemma 2 9B | Bearer token | Format OpenAI, 14 400 req/jour |
| **Gemini** | 2.5 Flash, 2.0 Flash-Lite | API key en URL | systemInstruction separe, role "model" |
| **Ollama** | Auto-decouverts (DeepSeek, Llama, Mistral...) | Aucun (local) | Timeout 120s, catalogue dynamique |

#### bootstrap.py — Initialisation
`bootstrap_providers(config)` : enregistre les providers configures → `discover_models()` → singleton

#### hardware.py — Detection materielle
`HardwareDetector.detect()` → CPU, RAM, GPU (nvidia-smi) → recommande les modeles Ollama adaptes

### 1.8 teams/ — Workers specialises

#### worker.py — Agent d'execution
**Types** : RESEARCHER, CODER, SUMMARIZER, ANALYST, WRITER, TRANSLATOR, GENERIC

Chaque type a son system prompt specialise :
- **RESEARCHER** : regles anti-hallucination, verification des sources
- **CODER** : ecriture, analyse, debug, tests
- **ANALYST** : identification de tendances, conclusions

**Classe `Worker`** : async context manager avec lifecycle garanti
- `execute()` → `WorkerResult` (success, output, errors, tool_calls, execution_time)

#### factory.py — Creation de Workers
**Classe `WorkerFactory`** :
- `classify_task(request)` → score par regex (patterns sport, code, recherche, etc.)
- `analyze_task(request)` → `TaskAnalysis` (type, subtasks, tools)
- `create_worker(analysis)` → Worker pret avec outils

### 1.9 tools/ — Outils des Workers

#### base_tools.py — Outils integres
| Outil | Fonction |
|-------|----------|
| `web_search_tool(query)` | Recherche DuckDuckGo (ddgs) |
| `web_fetch_tool(url)` | Fetch + nettoyage HTML |
| `file_read_tool(path)` | Lecture avec securite racine |
| `file_write_tool(path, content)` | Ecriture avec controle d'acces |
| `code_execute_tool(code)` | Execution Python sandbox |
| `memory_search_tool(query)` | Recherche semantique via Memory |

`ToolRegistry` : charge les outils par type de Worker
`TOOL_SCHEMAS` : schemas au format Anthropic tool_use

#### plugin_loader.py — Plugins dynamiques
Charge des plugins depuis `data/plugins/` :
```python
# Contrat plugin
PLUGIN_META = {"name": str, "description": str, "input_schema": dict, "worker_types": [str]}
def execute(**kwargs) -> str: ...
```
- `discover()`, `load_plugin()`, `execute_plugin()` (timeout 30s)
- Thread-safe, hot-reload

#### tool_generator.py — Generation autonome (Level 4)
**Cycle** : DETECT → GENERATE → VALIDATE → DEPLOY → MONITOR → PRUNE
- Detecte les patterns recurrents (erreurs tool_not_found >= seuil)
- Genere un plugin Python (`data/plugins/auto_*.py`)
- Valide : syntaxe, securite (imports interdits : os, subprocess, sys...), dry-run
- Deploie via hot-reload du PluginLoader
- Monitore : usage_count, success_rate
- Elague : deprecie si inutilise > 7j ou success_rate < 30%

---

## 2. VOX — Interface Humaine

### 2.1 interface.py — Agent Vox

**Classe `Vox`** (~590 lignes) — Point de contact entre humain et systeme.

**Pipeline `process_message(human_message)`** :
1. Validation input (Sanitizer)
2. Ajoute a l'historique (borne a 20 messages)
3. Detection message simple (salutations, statut, conversation legere)
4. **Simple** → `_vox_quick_reply()` (sans Brain, instantane)
5. **Complexe** → reformulation LLM → Brain
   - Mode async (callback) : genere un ack, lance Brain en arriere-plan
   - Mode sync (API/Telegram) : attend Brain, retourne reponse
6. Sauvegarde dans ConversationStore + Memory

**LLM dedie** : Haiku pour reformulation/restitution via `_vox_llm_call()` → route_chat multi-provider

**Gestion sessions** : `start_new_session()`, `resume_session()` (thread-safe via Lock)

**`bootstrap()`** : Initialise Config → Memory → Brain → Vox, retourne Vox pret

### 2.2 api/ — REST API (FastAPI)

#### server.py — Factory
`create_app(config)` :
- Lifespan : initialise CoreRegistry au demarrage
- Middleware : CORS → APIKeyMiddleware → RateLimitMiddleware → SanitizerMiddleware
- Routes depuis routes.py

#### routes.py — Endpoints
| Methode | Route | Description |
|---------|-------|-------------|
| POST | `/chat` | Envoie un message (120s timeout) |
| GET | `/health` | Sante du systeme (public) |
| GET | `/status` | Statut detaille (auth requise) |
| GET | `/sessions` | Liste les sessions (limit=20) |
| GET | `/sessions/{id}/history` | Historique d'une session |
| GET | `/persona` | Personnalite Neo + profil utilisateur |
| WS | `/ws/chat?token=` | Chat temps reel WebSocket |

#### middleware.py — Securite
- **APIKeyMiddleware** : header `X-Neo-Key`, comparaison timing-safe (`hmac.compare_digest`)
- **RateLimitMiddleware** : 60 req/min par IP, SQLite persistant
- **SanitizerMiddleware** : bloque les menaces medium/high sur POST/PUT/PATCH

#### schemas.py — Modeles Pydantic
ChatRequest, ChatResponse, StatusResponse, SessionInfo, HistoryTurn, ErrorResponse

### 2.3 cli/ — Interface Terminal

#### \_\_init\_\_.py — Dispatch des commandes
`main()` route les commandes : `setup`, `chat`, `start`, `stop`, `restart`, `status`, `history`, `version`

#### chat.py — Boucle conversationnelle Rich
- Integration Guardian (GracefulShutdown, StateSnapshot)
- Heartbeat en arriere-plan (30 min)
- File de resultats Brain async
- Commandes slash : `/quit`, `/status`, `/health`, `/help`, `/history`, `/sessions`, `/skills`, `/tasks`, `/epics`, `/heartbeat`, `/persona`, `/profile`, `/reflect`, `/restart`

#### setup.py — Wizard d'installation
8 etapes (mode interactif) ou 5 etapes (mode auto) :
1. Verification Python 3.10+
2. Setup venv
3. Installation dependances
4. Configuration auth (auto-import Claude Code ou saisie manuelle)
5. Test connexion Anthropic
6. Detection hardware + providers optionnels (Ollama, Groq, Gemini)
7. Sauvegarde config (`data/neo_config.json` + `.env`)
8. "Make it live" (daemon + systemd + Telegram)

#### status.py — Dashboard systeme
Tables Rich : config, agents, workers, lifecycle, health monitor, providers LLM

#### history.py — Navigateur de sessions
Table Rich : session_id, user_name, message_count, dates

### 2.4 integrations/telegram.py — Bot Telegram

**Classe `TelegramBot`** :
- Whitelist user_ids (securite)
- Rate limit 20 req/min par user (SQLite)
- Sanitisation input (Sanitizer)
- Partage l'instance Vox avec CLI/API
- Commandes : `/start`, `/help`, `/status`, `/whoami`
- Split reponses en chunks 4096 chars (limite Telegram)

---

## 3. MEMORY — Memoire Persistante

### 3.1 agent.py — MemoryAgent (Bibliothecaire)

**Orchestrateur** de tous les sous-systemes memoire :
- `initialize()` : cree Store, ContextEngine, Consolidator, LearningEngine, TaskRegistry, PersonaEngine
- `store_memory(content, source, tags, importance)` → stocke dans FAISS + SQLite
- `get_context(query)` → recherche semantique pour injection dans les prompts
- `on_conversation_turn(user_msg, ai_response)` :
  - Stocke l'echange
  - Enrichit les taches actives
  - Analyse la conversation (persona)
  - Declenche la consolidation (tous les 50 tours)
- Charge la documentation systeme depuis `data/system_docs/` (hash cache)

### 3.2 store.py — Stockage vectoriel FAISS

**Classe `MemoryStore`** — Double couche :

| Couche | Technologie | Role |
|--------|-------------|------|
| Vecteurs | FAISS IndexFlatIP (384 dim) | Recherche semantique |
| Metadata | SQLite (memory_meta.db) | Stockage structure |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Vectorisation |

**Methodes** :
- `store(content, source, tags, importance)` → embedding + FAISS + SQLite (atomique)
- `search_semantic(query, n_results, min_importance)` → recherche cosinus top-k
- `search_by_source()`, `search_by_tags()` → filtres exacts
- `delete(record_id)` → marque supprime, rebuild necessaire
- `rebuild_index()` → elimine les entrees supprimees

**Optimisations** :
- Modele d'embedding cache au niveau process (global)
- Cache semantique par requete (vide entre chaque tour)
- Index SQLite sur source, importance, timestamp
- WAL journal mode

### 3.3 context.py — Injection de contexte

**Classe `ContextEngine`** :
- `build_context(query)` → `ContextBlock` avec :
  - `relevant_memories` : top 5 recherche semantique
  - `important_memories` : importance >= 0.7 (max 2)
  - `recent_memories` : derniers (max 2)
- Tronque si > max_context_tokens
- `store_conversation_turn()` : estime l'importance (mots-cles identite → 0.9, preferences → 0.8, taches → 0.7)

### 3.4 conversation.py — Persistance des sessions

**Classe `ConversationStore`** (SQLite) :
- Tables : `sessions` (session_id, user_name, dates) + `turns` (role, content, turn_number)
- `start_session()`, `append_turn()` (thread-safe), `get_history()`, `get_sessions()`
- `export_to_json()` pour sauvegarde

### 3.5 consolidator.py — Consolidation memoire

**Classe `MemoryConsolidator`** — Comme le sommeil humain :
- `cleanup(max_age_days=30, min_importance=0.2)` → supprime les vieux/peu importants
- `merge_similar(threshold=0.85)` → fusionne les doublons semantiques
- `promote_important(boost=0.1)` → augmente l'importance des entrees populaires
- `full_consolidation()` → cleanup + merge + promote (thread-safe)

### 3.6 learning.py — Apprentissage (boucle fermee)

**Classe `LearningEngine`** :
- `record_result(request, worker_type, success, time, errors)` :
  - Met a jour les `LearnedSkill` (nom, succes, temps moyen, meilleure approche)
  - Enregistre les `ErrorPattern` (type, count, exemples, regle d'evitement)
  - Met a jour les `WorkerPerformance` (taux succes/echec, temps moyen)
- `get_advice(request, proposed_worker)` → `LearningAdvice` :
  - Workers a eviter (erreurs recurrentes)
  - Ajustement de confiance
  - Warnings et skills pertinents

### 3.7 persona.py — Identite et personnalite (Stage 9)

**3 Commandements immuables** (FrozenInstanceError si modification) :
1. "Neo ne s'eteint jamais"
2. "Neo n'oublie jamais"
3. "Neo apprend tous les jours"

**Traits de personnalite** (echelle 0.0–1.0, evolution EMA) :
communication_style, humor_level, patience, verbosity, curiosity, empathy, formality, expressiveness

**Profil utilisateur** (appris automatiquement) :
- Preferences : langue, longueur reponse, niveau technique, ton
- Patterns : heures de pointe, longueur moyenne, sujets, langues
- Observations : satisfaction, frustration, comportements (max 200)

**PersonaEngine** :
- `analyze_conversation()` : detecte langue, ton, sujets, adapte les traits
- `perform_self_reflection()` : LLM analyse les interactions recentes (toutes les 24h)
- Injection dans le prompt Vox → affecte le style de communication

### 3.8 task_registry.py — Registre de taches

**Task** : id, description, worker_type, status (pending/in_progress/done/failed), context_notes
**Epic** : id, description, task_ids, strategy, status

CRUD complet + persistance dans MemoryStore (source="task_registry:*")

### 3.9 migrations.py — Versioning schema SQLite

3 migrations appliquees sequentiellement : creation table versions, index timestamp, cleanup

---

## 4. INFRA — Infrastructure

### 4.1 registry.py — Singleton des agents

**Classe `CoreRegistry`** — Garantit UNE seule instance Vox/Brain/Memory par process :
- Singleton via `__new__()` + threading.Lock
- `_bootstrap()` : cree les 3 agents + initialise les providers
- `get_vox()`, `get_brain()`, `get_memory()` : lazy init, thread-safe
- Reinitialisation post-fork (`os.register_at_fork`)

Utilise par CLI, API, Telegram et Daemon pour partager les memes agents.

### 4.2 daemon.py — Gestion daemon

**`start(foreground, host, port)`** :
- Mode foreground : bloquant (pour systemd)
- Mode background : double fork → detache TTY
- Lance en parallele : uvicorn + Heartbeat + Telegram (optionnel)

**Signaux** :
- SIGTERM/SIGINT → arret propre
- SIGHUP → `config.reload()` (rechargement a chaud)

**`generate_systemd_service()`** : template de service systemd avec hardening securite

### 4.3 guardian.py — Supervision de processus

**Commandement #1** : "Neo ne s'eteint jamais"

**Classe `Guardian`** :
- Boucle infinie : lance `neo chat` → attend exit → decide restart
- Exit code 0 → arret normal
- Exit code 42 → restart immediat (`/restart`)
- Autre → crash, backoff exponentiel (1s → 60s max)
- Protection : max 10 restarts/heure
- Reset backoff si process stable > 5 min

**StateSnapshot** : sauvegarde l'etat avant crash (session_id, turn_count, taches actives) pour reprise

**GracefulShutdown** : intercepte SIGTERM/SIGINT, execute les callbacks de nettoyage

### 4.4 heartbeat.py — Pouls autonome

**Classe `HeartbeatManager`** — asyncio.Task qui pulse toutes les 30 min :

| Action | Frequence | Description |
|--------|-----------|-------------|
| Advance epics | Chaque pulse | Execute la prochaine tache des epics en cours |
| Detect stale | Chaque pulse | Alerte si tache in_progress > 10 min |
| Consolidation | 10 pulses (~5h) | Nettoyage + merge memoire |
| Self-reflection | 24h | LLM analyse les interactions, ajuste la persona |
| Auto-tuning | 5 pulses (~2.5h) | Level 1 : ajuste temperature, timeout, retries |
| Self-patching | 10 pulses (~5h) | Level 3 : detecte + corrige les patterns d'erreur |
| Tool generation | 15 pulses (~7.5h) | Level 4 : genere des plugins automatiquement |

### 4.5 resilience.py — Tolerance aux pannes

**RetryConfig** : max 3 retries, backoff exponentiel avec jitter (evite le thundering herd)

**CircuitBreaker** (3 etats) :
- Closed → normal, autorise les appels
- Open → bloque (>= 5 echecs consecutifs, timeout 60s)
- Half-Open → autorise 1 appel test

**HealthMonitor** : error_rate, avg_response_time, statut (healthy/ok_with_errors/warning/degraded)

### 4.6 security/sanitizer.py — Filtrage des inputs

**Classe `Sanitizer`** — Detecte et filtre :

| Menace | Severite | Action |
|--------|----------|--------|
| Prompt injection ("ignore previous instructions", "you are now a"...) | HIGH | Bloque |
| SQL injection (`' OR 1=1`, `; DROP TABLE`) | HIGH | Bloque |
| XSS (`<script>`, `javascript:`, `onerror=`) | MEDIUM | [FILTERED] |
| Path traversal (`../`, `/etc/passwd`) | MEDIUM | [BLOCKED] |
| Depassement longueur (> 10 000 chars) | LOW | Tronque |

### 4.7 security/vault.py — Chiffrement des secrets

**Classe `KeyVault`** :
- **Chiffrement** : Fernet (AES-128-CBC + HMAC-SHA256)
- **Derivation** : PBKDF2 480 000 iterations
- **Materiau** : machine-id + master password (optionnel)
- **Liaison machine** : le vault ne peut pas etre transfere entre machines
- **Stockage** : SQLite (`.vault.db`) + sel persistant (`.vault.salt`)
- **Permissions** : 0o600 sur tous les fichiers vault

---

## 5. Modules racine

### config.py — Configuration centrale

**NeoConfig** agrege :
- `LLMConfig` : provider, model, api_key, temperature, max_tokens
- `MemoryConfig` : storage_path, vector_db (faiss), max_context_tokens
- `ResilienceConfig` : retries, timeouts, circuit breaker
- `SelfImprovementConfig` : Level 1 (auto-tuning), Level 3 (self-patching), Level 4 (tool-generation)

**AGENT_MODELS** : mapping agent → modele LLM par defaut
- vox, memory, worker:summarizer/writer/translator/generic → Haiku (rapide)
- brain, worker:researcher/coder/analyst → Sonnet (raisonnement)

**`get_agent_model(agent_name)`** : consulte ModelRegistry, fallback sur AGENT_MODELS

### oauth.py — Gestion OAuth Anthropic

**Types de token** :
- `sk-ant-oat...` : Access token (temporaire, ~8h)
- `sk-ant-ort...` : Refresh token (long-terme)
- `sk-ant-api...` : Cle API permanente (apres conversion)

**Fonctions cles** :
- `get_best_auth()` : priorite cle API > token valide > refresh+convert
- `import_claude_code_credentials()` : import auto depuis `~/.claude/.credentials.json`
- `convert_oauth_to_api_key()` : convertit OAuth en cle permanente
- Stockage dans KeyVault (chiffre)

### validation.py — Validation des inputs

- `validate_message(msg)` : longueur [1, 10 000], type str, strip
- `validate_task_description(desc)` : max 5 000 chars
- `validate_session_id(id)` : regex `^[a-zA-Z0-9_-]+$`, max 100 chars

### logging_config.py — Logging structure

- Console (stderr, format lisible) + Fichier (rotation 5MB x 3, format JSON)
- Niveaux par module : resilience → WARNING, reste → INFO
- Deps externes (faiss, langchain, httpx) → WARNING

---

## 6. Points d'entree

| Commande | Chemin | Description |
|----------|--------|-------------|
| `./neo chat` | bash → `python -m neo_core.vox.cli chat` | Chat interactif |
| `./neo setup` | bash → `python -m neo_core.vox.cli setup` | Wizard d'installation |
| `./neo start` | bash → daemon.start() | Lancement daemon (API + Heartbeat + Telegram) |
| `./neo stop` | bash → daemon.stop() | Arret daemon |
| `./neo status` | bash → status.run_status() | Dashboard systeme |
| `pip install -e .` | pyproject.toml → `neo_core.vox.cli:main` | Installation console_scripts |
| `install.sh` | Script bash | Deploiement VPS (systemd, user neo, swap, venv) |

---

## 7. Niveaux d'autonomie

| Niveau | Nom | Description | Localisation |
|--------|-----|-------------|--------------|
| 0 | Execution | Repond aux requetes via LLM | brain/core.py |
| 1 | Auto-tuning | Ajuste temperature, timeout, retries | brain/auto_tuner.py |
| 2 | Apprentissage | Apprend des succes/echecs, conseille | memory/learning.py |
| 3 | Self-patching | Detecte et corrige les patterns d'erreur | brain/self_patcher.py |
| 4 | Tool generation | Genere des plugins automatiquement | brain/tools/tool_generator.py |
| 5 | Resilience | Circuit breaker, retry, health monitoring | infra/resilience.py |
| 9 | Persona | Evolue sa personnalite, apprend l'utilisateur | memory/persona.py |

---

## 8. Securite

| Couche | Module | Protection |
|--------|--------|------------|
| Input | validation.py | Longueur, type, format |
| Sanitisation | infra/security/sanitizer.py | Prompt injection, SQL, XSS, path traversal |
| Auth API | vox/api/middleware.py | API key timing-safe, rate limit 60/min |
| Auth Telegram | vox/integrations/telegram.py | Whitelist user_ids, rate limit 20/min |
| Secrets | infra/security/vault.py | Fernet AES-128, PBKDF2 480K, machine-bound |
| OAuth | oauth.py | Tokens chiffres, refresh auto, conversion API key |
| Systemd | infra/daemon.py | ProtectSystem=strict, NoNewPrivileges |
| Tools | brain/tools/tool_generator.py | Imports interdits (os, subprocess, sys...) |
| Workers | brain/tools/base_tools.py | Racines autorisees pour fichiers |

---

## 9. Dependances

### Core
```
langchain, langchain-anthropic, langchain-community  — LLM integration
faiss-cpu                                             — Recherche vectorielle
sentence-transformers                                 — Embeddings (all-MiniLM-L6-v2)
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
