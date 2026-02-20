# Neo Core — Manuel d'Installation & Exploitation

> Document basé sur les faits récoltés le 20 février 2026 sur le VPS de production.

---

## 1. Environnement de Production

| Paramètre | Valeur |
|-----------|--------|
| **OS** | Ubuntu 25.04 (Plucky) |
| **Python** | 3.13.3 |
| **Répertoire** | `/opt/neo-core` |
| **Virtualenv** | `/opt/neo-core/.venv` |
| **Utilisateur système** | `neo` (propriétaire de tous les fichiers) |
| **Utilisateur SSH** | `ubuntu` (nécessite `sudo -u neo` pour les commandes Neo) |
| **Service systemd** | `neo-guardian.service` |
| **Version** | 0.9.7, Stage 6 |
| **Entry point** | `neo = "neo_core.vox.cli:main"` (défini dans `pyproject.toml`) |

---

## 2. Structure des Fichiers

```
/opt/neo-core/
├── .env                      # Clés API (permission 600, owner: neo)
├── .env.example              # Template des variables requises
├── .venv/                    # Virtualenv Python 3.13
│   └── bin/
│       ├── python3           # Interpréteur
│       └── neo               # CLI Neo (installé via pip)
├── neo_core/                 # Code source
│   ├── brain/                # Intelligence (core.py, prompts.py, teams/)
│   ├── vox/                  # Interface (interface.py, cli/, api/, integrations/)
│   ├── memory/               # Mémoire (agent.py, conversation.py, task_registry.py)
│   ├── infra/                # Infrastructure (daemon.py, heartbeat.py)
│   └── config.py             # Configuration et modèles
├── data/                     # Données persistantes (permission 700, owner: neo)
│   ├── .vault.db             # Secrets chiffrés
│   ├── .oauth_credentials.json
│   ├── neo_config.json       # Configuration runtime
│   ├── neo.log               # Logs du daemon
│   ├── working_memory.json   # Mémoire de travail
│   ├── memory/               # SQLite mémoire long-terme
│   ├── guardian/              # Snapshots Guardian
│   ├── plugins/              # Plugins installés
│   ├── patches/              # Auto-patches
│   └── system_docs/          # Documentation système
└── pyproject.toml            # Définition du package Python
```

---

## 3. Permissions — CRITIQUE

Neo tourne sous l'utilisateur système `neo`. Le user SSH `ubuntu` n'a PAS les permissions sur `data/` et `.env`.

**Règle d'or : toute commande Neo doit être lancée en tant que `neo`.**

```bash
# ❌ INCORRECT (lancé en tant que ubuntu)
neo start
neo chat
neo status

# ✅ CORRECT — via systemd (recommandé)
sudo systemctl start neo-guardian
sudo systemctl stop neo-guardian
sudo systemctl restart neo-guardian
sudo systemctl status neo-guardian

# ✅ CORRECT — commande manuelle en tant que neo
sudo -u neo bash -c 'source /opt/neo-core/.venv/bin/activate && neo start'
sudo -u neo bash -c 'source /opt/neo-core/.venv/bin/activate && neo chat'
sudo -u neo bash -c 'source /opt/neo-core/.venv/bin/activate && neo status'

# ✅ Consulter les logs
sudo -u neo tail -f /opt/neo-core/data/neo.log
```

---

## 4. Démarrage / Arrêt / Redémarrage

### Via systemd (recommandé)

```bash
sudo systemctl restart neo-guardian   # Redémarrer
sudo systemctl start neo-guardian     # Démarrer
sudo systemctl stop neo-guardian      # Arrêter
sudo systemctl status neo-guardian    # Vérifier le statut
journalctl -u neo-guardian -f         # Logs systemd en temps réel
```

### Via CLI (mode interactif)

```bash
# Mode daemon (background)
sudo -u neo bash -c 'source /opt/neo-core/.venv/bin/activate && neo start'
sudo -u neo bash -c 'source /opt/neo-core/.venv/bin/activate && neo stop'

# Mode chat direct (foreground, Ctrl+C pour quitter)
sudo -u neo bash -c 'source /opt/neo-core/.venv/bin/activate && neo chat'
```

---

## 5. Architecture — 3 Agents + Workers

```
┌──────────────────────────────────────────────────────┐
│                    NEO CORE v0.9.7                    │
├──────────┬──────────┬──────────┬─────────────────────┤
│   Vox    │  Brain   │  Memory  │     Heartbeat       │
│ (routeur)│ (Sonnet) │ (Haiku)  │  (pulse 5 min)      │
│          │          │          │                      │
│ • Gère   │ • Smart  │ • 296    │ • Avance les projets │
│   l'hist.│   Pipeline│  souven.│ • Détecte les stales │
│ • ACK    │ • Actions│ • SQLite │ • Projets récurrents │
│   statiq.│ • Projets│ • Résumés│ • Auto-consolidation │
│ • Multi- │ • Workers│   session│ • Self-patching      │
│   canal  │          │          │                      │
└──────────┴──────────┴──────────┴─────────────────────┘
         │                  │
         ▼                  ▼
┌─────────────────────────────────────────────────────┐
│              Workers Éphémères (à la demande)        │
├────────────┬─────────────┬─────────────┬────────────┤
│ researcher │   coder     │  analyst    │  writer    │
│ (Ollama    │ (Sonnet)    │ (Sonnet)    │ (Ollama    │
│  DeepSeek) │             │             │  DeepSeek) │
├────────────┼─────────────┼─────────────┼────────────┤
│ summarizer │ translator  │   generic   │            │
│ (Ollama)   │ (Ollama)    │  (Ollama)   │            │
└────────────┴─────────────┴─────────────┴────────────┘
```

---

## 6. Providers LLM Configurés

| Provider | Type | Modèles | Agents |
|----------|------|---------|--------|
| **Anthropic** | Cloud payant | claude-sonnet-4-6 | Vox, Brain, Memory, worker:coder, worker:analyst |
| **Ollama** | Local (VPS) | deepseek-r1:8b | worker:researcher, writer, summarizer, translator, generic |
| **Gemini** | Cloud gratuit | 2 modèles | Disponible (1/2) |

### Routing actuel

Les agents core (Vox, Brain, Memory) tournent sur **Anthropic Sonnet**.
Les workers secondaires (researcher, writer, etc.) tournent sur **Ollama DeepSeek-R1:8b** (local, gratuit).
Les workers critiques (coder, analyst) tournent sur **Anthropic Sonnet**.

---

## 7. Interfaces

| Interface | Accès | Commande |
|-----------|-------|----------|
| **CLI (TUI)** | Terminal SSH | `neo chat` |
| **API REST** | `http://localhost:8000` | Lancé par le daemon |
| **Telegram** | Bot `@NeoCore_bot` | Intégré au daemon |

Le daemon lance simultanément l'API REST + le Heartbeat + le bot Telegram.

---

## 8. Configuration

### Fichier `.env`

```bash
# Clé API Anthropic (obligatoire pour Brain/Vox/Memory)
ANTHROPIC_API_KEY=sk-ant-...

# Telegram (optionnel)
TELEGRAM_BOT_TOKEN=8211990364:AAF...

# Ollama (optionnel, par défaut localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434
```

### Fichier `data/neo_config.json`

Configuration runtime : nom du core, utilisateur, stage, modèles, etc.
Modifié via les commandes slash (`/persona`, `/profile`, etc.)

---

## 9. Commandes Slash (dans le chat)

| Commande | Description |
|----------|-------------|
| `/help` | Liste des commandes |
| `/status` | Statut complet du système |
| `/tasks` | Liste des tâches actives |
| `/project` | Gestion des projets (créer, voir, supprimer) |
| `/heartbeat` | Statut du heartbeat |
| `/persona` | Personnalité de Neo |
| `/profile` | Profil utilisateur |
| `/sessions` | Sessions précédentes |
| `/skills` | Compétences apprises |
| `/reflect` | Auto-réflexion |

---

## 10. Diagnostic & Troubleshooting

### Le daemon ne démarre pas

```bash
# 1. Vérifier les logs
sudo -u neo tail -50 /opt/neo-core/data/neo.log

# 2. Tester les imports Python
sudo -u neo bash -c 'source /opt/neo-core/.venv/bin/activate && python3 -c "from neo_core.vox.interface import Vox; print(\"OK\")"'

# 3. Vérifier les permissions
ls -la /opt/neo-core/data/
ls -la /opt/neo-core/.env

# 4. Vérifier systemd
sudo systemctl status neo-guardian
journalctl -u neo-guardian --no-pager -n 50
```

### Erreurs de permissions

```bash
# Tout remettre à l'utilisateur neo
sudo chown -R neo:neo /opt/neo-core/data/
sudo chmod 700 /opt/neo-core/data/
sudo chmod 600 /opt/neo-core/.env
```

### Ollama ne répond pas

```bash
# Vérifier que Ollama tourne
systemctl status ollama
curl http://localhost:11434/api/tags

# Relancer
sudo systemctl restart ollama
```

### HuggingFace 429 (Too Many Requests)

Erreur vue dans les logs :
```
Could not download model from HuggingFace: 429 Too Many Requests
```
C'est le modèle d'embedding fastembed. Non bloquant — Neo fonctionne avec le modèle en cache. Si ça persiste, redémarrer résout généralement le problème.

---

## 11. Mise à Jour du Code

Après modification des fichiers source (via Claude, git, ou manuellement) :

```bash
# 1. Redémarrer le daemon pour prendre en compte les changements
sudo systemctl restart neo-guardian

# 2. Vérifier que ça tourne
sudo systemctl status neo-guardian

# 3. Vérifier les logs pour erreurs d'import
sudo -u neo tail -20 /opt/neo-core/data/neo.log
```

**Pas besoin de `pip install` ni de rebuild.** Le code Python est lu directement depuis `/opt/neo-core/neo_core/`. Un simple restart suffit.

Si des dépendances pip ont été ajoutées :
```bash
sudo -u neo bash -c 'source /opt/neo-core/.venv/bin/activate && pip install nouveau-package'
```

---

## 12. Pipeline de Traitement (Smart Pipeline v5.0)

```
User tape un message
        │
        ▼
   [Vox reçoit]
   • Stocke dans l'historique (budget 12K tokens)
   • Passe le message BRUT à Brain (zéro reformulation)
        │
        ▼
   [Brain — UN SEUL appel Sonnet]
   • System prompt compact (~600 tokens)
   • Historique conversation complet
   • Mémoire + projets + sessions dans le contexte
   • Comprend + décide + répond en une passe
        │
        ├─── Réponse directe → retour immédiat
        │
        └─── Action détectée (```neo-action JSON```)
             │
             ├── search → Worker Researcher
             ├── delegate → Worker spécialisé
             ├── code → Worker Coder
             ├── create_project → ProjectManager (N workers parallèles)
             ├── create_recurring_project → Projet cyclique (heartbeat relance)
             └── crew_directive → Pilotage d'un projet en cours
```

---

## 13. Projets Récurrents (Nouveau)

Neo peut maintenant gérer des projets qui se relancent automatiquement.

**Exemple :** "Gère une bankroll de 300€ en paris ATP tennis, objectif doubler"

```
Cycle 1 (immédiat) : Recherche matchs → Analyse cotes → Stratégie Kelly → Simulation
Cycle 2 (J+1)      : Idem, avec résultats du cycle 1 dans le contexte
Cycle 3 (J+2)      : Adapte la stratégie selon les résultats cumulés
...
Cycle N             : Objectif atteint (600€) → projet terminé
```

Le Heartbeat vérifie toutes les 5 minutes si un cycle est dû et le déclenche automatiquement.

---

## 14. Dépendances Installées (extrait)

| Package | Version | Usage |
|---------|---------|-------|
| anthropic | 0.82.0 | API Claude |
| fastapi | 0.129.0 | API REST |
| httpx | 0.28.1 | Client HTTP |
| faiss-cpu | 1.13.2 | Recherche vectorielle |
| fastembed | 0.7.4 | Embeddings |
| ddgs | 9.10.0 | DuckDuckGo search |
| cryptography | 44.0.3 | Vault chiffré |
| groq | 1.0.0 | Provider Groq |
| google-generativeai | 0.8.6 | Provider Gemini |

---

*Dernière mise à jour : 20 février 2026 — basé sur le VPS de production `vps-a1b25d3a`*
