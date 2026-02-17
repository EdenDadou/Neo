# Neo Core — Architecture Technique Complète

## Vue d'ensemble

Neo Core est un écosystème IA multi-agents composé de 3 agents principaux (Vox, Brain, Memory) qui communiquent entre eux pour traiter les requêtes utilisateur.

## Architecture des Agents

### Vox (Interface Humaine)
- Modèle : Claude Haiku (rapide, léger)
- Rôle : Point de contact avec l'utilisateur
- Capacités :
  - Reformule les requêtes avant transmission à Brain
  - Répond seul aux messages simples (salutations, conversation)
  - Génère des accusés de réception pendant que Brain travaille
  - Informe l'utilisateur des capacités du système
  - Mode asynchrone : continue la conversation pendant que Brain réfléchit

### Brain (Cortex Exécutif / Orchestrateur)
- Modèle : Claude Sonnet (puissant, raisonnement avancé)
- Rôle : Décideur stratégique, orchestrateur de Workers
- Capacités :
  - Analyse la complexité des requêtes (simple/modéré/complexe)
  - Prend des décisions : réponse directe, délégation à un Worker, création d'Epic
  - Crée et coordonne des Workers spécialisés via la Factory
  - Consulte Memory pour enrichir ses décisions avec l'apprentissage passé
  - Gère les retries intelligents (jusqu'à 3 tentatives avec changement de stratégie)
  - Applique des patches comportementaux (self-patching Level 3)

### Memory (Hippocampe / Système de Mémoire)
- Modèle : Claude Haiku (économique)
- Rôle : Stockage et récupération de connaissances
- Composants :
  - MemoryStore : double couche ChromaDB (vectoriel) + SQLite (métadonnées)
  - ContextEngine : sélection intelligente du contexte pertinent
  - LearningEngine : apprentissage des succès et échecs
  - TaskRegistry : gestion des tâches et epics
  - PersonaEngine : identité évolutive et profil utilisateur
  - ConversationStore : historique des sessions de conversation

## Workers Spécialisés (créés dynamiquement par Brain)

Les Workers sont des agents éphémères créés par la WorkerFactory selon le besoin :

| Worker | Modèle | Rôle | Outils disponibles |
|--------|--------|------|-------------------|
| Researcher | Sonnet | Recherche web, investigation | web_search, web_fetch |
| Coder | Sonnet | Écriture/debug de code | code_execute, file_read, file_write |
| Analyst | Sonnet | Analyse de données, tendances | code_execute, web_search |
| Writer | Haiku | Rédaction (articles, emails, rapports) | file_write |
| Summarizer | Haiku | Synthèse et résumé | - |
| Translator | Haiku | Traduction multilingue | - |
| Generic | Haiku | Tâches polyvalentes | tous les outils de base |

## Outils Intégrés (Built-in Tools)

- **web_search** : Recherche sur internet via DuckDuckGo (actualités, météo, prix, scores)
- **web_fetch** : Récupération et lecture du contenu de pages web
- **code_execute** : Exécution de code Python dans un sandbox sécurisé
- **file_read** : Lecture de fichiers
- **file_write** : Écriture de fichiers
- **memory_search** : Recherche dans la mémoire sémantique

## Système de Tâches et Epics

### Tâches
- Unité de travail unitaire avec description, worker_type, statut
- Statuts : pending → in_progress → done/failed
- Suivies dans le TaskRegistry en mémoire persistante

### Epics (Projets Multi-étapes)
- Groupe de tâches coordonnées pour un objectif complexe
- Décomposition automatique par Brain
- Exécution séquentielle des sous-tâches
- Suivi global de progression

## Self-Improvement (Auto-amélioration)

### Level 1 — Auto-Tuning
- Ajuste automatiquement température, retries, timeout
- Basé sur les métriques de performance du LearningEngine
- Persisté dans data/auto_tuning.json

### Level 2 — Plugin System
- Charge dynamiquement des plugins Python depuis data/plugins/
- Timeout 30s, namespace isolé, hot-reload
- Format standard avec PLUGIN_META et execute()

### Level 3 — Self-Patching
- Détecte les patterns d'erreurs récurrents (≥3 occurrences)
- Génère des patches comportementaux (JSON)
- Types : prompt_override, routing_rule, config_override
- Validation par amélioration historique ≥50%
- Rollback automatique si inefficace

### Level 4 — Autonomous Tool Creation
- Détecte les besoins d'outils récurrents non couverts
- Génère automatiquement des plugins Python
- Validation de sécurité (imports dangereux bloqués)
- Pruning des outils inutilisés après 7 jours

## Providers LLM Supportés

- **Anthropic** : Claude (Haiku, Sonnet, Opus) — provider principal
- **Groq** : LLaMA, Mixtral — gratuit, rapide
- **Gemini** : Google AI — alternatif
- **Ollama** : Modèles locaux — gratuit, privé

Routing économique : local gratuit > cloud gratuit > cloud payant

## Interfaces

- **CLI** : Terminal interactif Rich avec commandes /status, /health, /tasks, etc.
- **API REST** : FastAPI sur port 8000 (endpoints /chat, /health, /tasks, etc.)
- **Telegram** : Bot connecté via token, même pipeline Vox → Brain

## Résilience

- Retry avec backoff exponentiel (max 3 tentatives)
- Circuit breaker (seuil 5 erreurs consécutives)
- Health monitoring (taux d'erreur, temps de réponse)
- Guardian mode : auto-restart en cas de crash
- Heartbeat : maintenance périodique (consolidation, auto-tuning, self-patching)

## Flux de Traitement d'une Requête

1. Utilisateur → Vox : message reçu
2. Vox : détecte si simple (répond seul) ou complexe (transmet à Brain)
3. Vox → Brain : requête reformulée
4. Brain : analyse complexité + consulte Memory pour contexte
5. Brain : décide (réponse directe / Worker / Epic)
6. Si Worker : Factory crée un Worker spécialisé → exécute → apprend du résultat
7. Si Epic : décompose en sous-tâches → exécute séquentiellement
8. Brain → Vox : réponse finale
9. Vox → Utilisateur : réponse naturelle
10. Memory : stocke l'échange + apprentissage
