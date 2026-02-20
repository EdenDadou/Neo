# Neo Core — Architecture Technique Complète

> v0.9.7 — Dernière mise à jour : 20 février 2026

## Commandement Fondamental

NEO PEUT TOUT FAIRE. Il n'y a aucune limitation technique.
Neo tourne sur un VPS complet avec accès réseau, fichiers, bases de données,
exécution de code, et tous les packages Python disponibles.

## Vue d'ensemble

Neo Core est un écosystème IA multi-agents composé de 3 agents principaux (Vox, Brain, Memory) qui communiquent entre eux pour traiter les requêtes utilisateur.

## Architecture des Agents

### Vox (Routeur et Gestionnaire de Sessions)
- Rôle : Point de contact avec l'utilisateur, routage des messages
- Pipeline : le message utilisateur passe BRUT à Brain (aucune reformulation, aucun appel LLM dans le flux principal)
- Gestion de l'historique par budget tokens (12K tokens max, compression automatique des anciens messages)
- Mode async : génère un ACK statique, lance Brain en arrière-plan, délivre le résultat via callback
- Mode sync : attend Brain, retourne la réponse (API, Telegram)
- Sauvegarde chaque échange dans ConversationStore + Memory
- Seul usage LLM résiduel : résumé de fin de session via Haiku

### Brain (Orchestrateur — UN SEUL appel Sonnet)
- Modèle : Claude Sonnet 4.6 (raisonnement avancé)
- Rôle : Comprend, décide, agit, délègue
- System prompt compact (~600 tokens, ~18 lignes) avec placeholders dynamiques
- Reçoit l'historique conversation complet + contexte mémoire enrichi (projets actifs, sessions récentes, profil utilisateur)
- Capacités :
  - Réponse directe pour les questions simples
  - Délégation à un Worker spécialisé via blocs neo-action JSON
  - Création de projets multi-étapes (Epic)
  - Création de projets récurrents/cycliques (heartbeat-driven)
  - Pilotage de projets en cours (crew_directive)
- Actions supportées : search, delegate, code, create_project, create_recurring_project, crew_directive

### Memory (Mémoire Persistante)
- Modèle : Claude Haiku 4.5 (consolidation uniquement)
- Rôle : Stockage et récupération de connaissances
- Composants :
  - MemoryStore : double couche FAISS (vectoriel, 384 dim) + SQLite (métadonnées)
  - ContextEngine : sélection intelligente du contexte pertinent (top 5 sémantique + importance >= 0.7)
  - LearningEngine : apprentissage des succès et échecs (boucle fermée)
  - TaskRegistry : gestion des tâches, epics, et projets récurrents
  - PersonaEngine : identité évolutive et profil utilisateur (EMA)
  - ConversationStore : historique des sessions de conversation (SQLite)
  - Consolidator : nettoyage, merge, promotion (comme le sommeil)
- Modèle d'embedding : all-MiniLM-L6-v2 (FastEmbed/ONNX, 384 dimensions)

## Workers Spécialisés (créés dynamiquement par Brain)

Les Workers sont des agents éphémères créés par la WorkerFactory selon le besoin. Chaque worker reçoit les 5 derniers échanges de la conversation pour comprendre le contexte.

| Worker | Modèle | Rôle | Outils disponibles |
|--------|--------|------|-------------------|
| Researcher | Sonnet/Ollama | Recherche web, investigation | web_search, web_fetch |
| Coder | Sonnet | Écriture/debug de code | code_execute, file_read, file_write |
| Analyst | Sonnet | Analyse de données, tendances | code_execute, web_search |
| Writer | Haiku/Ollama | Rédaction (articles, emails, rapports) | file_write |
| Summarizer | Haiku/Ollama | Synthèse et résumé | — |
| Translator | Haiku/Ollama | Traduction multilingue | — |
| Generic | Haiku/Ollama | Tâches polyvalentes | tous les outils de base |

Le routing des workers dépend du mode (economic/quality) et de la disponibilité des providers.

## Outils Intégrés (Built-in Tools)

- **web_search** : Recherche sur internet via DuckDuckGo (actualités, météo, prix, scores)
- **web_fetch** : Récupération et lecture du contenu de pages web
- **code_execute** : Exécution de code Python dans un sandbox sécurisé
- **file_read** : Lecture de fichiers (sécurité racine)
- **file_write** : Écriture de fichiers (contrôle d'accès)
- **memory_search** : Recherche dans la mémoire sémantique

## Système de Tâches, Epics et Projets Récurrents

### Tâches
- Unité de travail unitaire avec description, worker_type, statut
- Statuts : pending → in_progress → done/failed

### Epics (Projets Multi-étapes)
- Groupe de tâches coordonnées pour un objectif complexe
- Décomposition automatique par Brain (sans limite artificielle de nombre d'étapes)
- Exécution séquentielle des sous-tâches

### Projets Récurrents (Cycliques)
- Se relancent automatiquement selon un schedule (quotidien, toutes les N heures)
- Chaque cycle reçoit les résultats accumulés des cycles précédents
- Objectif configurable : le projet se termine quand l'objectif est atteint
- Le Heartbeat vérifie toutes les 5 minutes si un cycle est dû

## Self-Improvement (Auto-amélioration)

### Level 1 — Auto-Tuning
Ajuste automatiquement température, retries, timeout par worker_type.

### Level 2 — Plugin System
Charge dynamiquement des plugins Python depuis data/plugins/ (hot-reload, timeout 30s).

### Level 3 — Self-Patching
Détecte les patterns d'erreurs récurrents (>= 3 occurrences), génère des patches comportementaux, rollback si inefficace.

### Level 4 — Autonomous Tool Creation
Détecte les besoins d'outils non couverts, génère automatiquement des plugins, valide et déploie.

## Providers LLM Supportés

- **Anthropic** : Claude Sonnet 4.6, Haiku 4.5 — provider principal
- **Groq** : LLaMA 3.3 70B — gratuit, rapide
- **Gemini** : Google Gemini 2.5 Flash — alternatif
- **Ollama** : Modèles locaux (DeepSeek-R1:8b, Llama, Mistral) — gratuit, privé

Routing : local gratuit > cloud gratuit > cloud payant (agents core toujours Anthropic)

## Interfaces

- **CLI** : Terminal interactif Rich avec commandes slash (/status, /tasks, /project, /heartbeat, /persona, /profile, /reflect, etc.)
- **API REST** : FastAPI sur port 8000 (endpoints /chat, /health, /status, /sessions, /persona, /ws/chat)
- **Telegram** : Bot connecté via token, même pipeline Vox → Brain

## Résilience

- Retry avec backoff exponentiel (max 3 tentatives, jitter)
- Circuit breaker (seuil 5 erreurs consécutives, timeout 60s)
- Health monitoring (taux d'erreur, temps de réponse)
- Guardian mode : auto-restart en cas de crash (backoff 1s → 60s, max 10/heure)
- Heartbeat : maintenance périodique (consolidation, auto-tuning, self-patching, projets récurrents)

## Environnement d'Exécution (VPS Ubuntu complet)

- **Accès réseau COMPLET** : requests, urllib, httpx, socket, aiohttp
- **Web scraping** : BeautifulSoup, lxml + web_search/web_fetch intégrés
- **Base de données** : SQLite natif + mémoire vectorielle FAISS persistante
- **Packages Python** : pip install fonctionne — pandas, numpy, scikit-learn installables
- **Fichiers** : lecture/écriture complète, pathlib, glob
- **Scheduler** : heartbeat + projets récurrents automatiques
- **Dashboard** : FastAPI + uvicorn intégrés

## Flux de Traitement d'une Requête — Smart Pipeline v5.0

1. Utilisateur → Vox : message reçu
2. Vox : stocke dans l'historique (budget 12K tokens), passe le message BRUT à Brain
3. Brain : appel Sonnet unique avec system prompt compact + historique + contexte mémoire
4. Brain : comprend + décide + répond en une passe
5. Si action détectée (neo-action JSON) : exécute l'action (worker, projet, etc.)
6. Brain → Vox : réponse finale
7. Vox → Utilisateur : réponse directe (aucune modification)
8. Memory : stocke l'échange + apprentissage
