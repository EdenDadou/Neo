# Plan : Système Multi-Provider LLM (Local + Cloud Gratuit)

## Objectif

Permettre à Neo Core d'utiliser des LLMs locaux (Ollama) et cloud gratuits (Groq, Gemini) en plus d'Anthropic, avec détection automatique du hardware et test de chaque modèle avant mise en production.

---

## Architecture proposée

### Nouveau fichier : `neo_core/providers/`

```
neo_core/providers/
├── __init__.py
├── base.py          # LLMProvider (classe abstraite)
├── anthropic.py     # AnthropicProvider (existant, refactoré)
├── ollama.py        # OllamaProvider (local)
├── groq.py          # GroqProvider (cloud gratuit)
├── gemini.py        # GeminiProvider (cloud gratuit)
├── registry.py      # ModelRegistry (catalogue + test + routing)
└── hardware.py      # HardwareDetector (RAM, GPU, CPU)
```

### Providers retenus

| Provider | Type | Gratuit | Modèles clés | Limite free tier |
|----------|------|---------|--------------|------------------|
| **Ollama** | Local | 100% | Llama 3.1 8B, Qwen 3, Gemma 2, Mistral 7B | Illimité (limité par hardware) |
| **Groq** | Cloud | Oui | Llama 3.3 70B, Llama 3 8B | 14 400 req/jour |
| **Gemini** | Cloud | Oui | Gemini 2.5 Flash, Flash-Lite | 250 req/jour (Flash) |
| **Anthropic** | Cloud | Payant | Claude Sonnet, Haiku | Selon plan |

### Pourquoi ces 3 (+ Anthropic) ?

1. **Ollama** — Gratuit, illimité, privé, pas de rate limit. Parfait pour les tâches répétitives (Vox, Memory, Workers légers). La question est le hardware du VPS.

2. **Groq** — 14 400 req/jour gratuit, inférence ultra-rapide (LPU). Llama 3.3 70B rivalise avec Sonnet pour les Workers complexes. API compatible OpenAI.

3. **Gemini** — Gratuit avec contexte 1M tokens. Idéal pour l'analyse de gros documents. Mais rate limit réduit (250/jour pour Flash).

OpenRouter et Together AI écartés : OpenRouter a 50 req/jour sans crédits (trop peu), Together AI n'a qu'un trial $25 (pas pérenne).

---

## Implémentation détaillée

### Étape 1 : HardwareDetector (`hardware.py`)

Détecte automatiquement les capacités du VPS :

```python
@dataclass
class HardwareProfile:
    total_ram_gb: float          # psutil.virtual_memory()
    available_ram_gb: float
    cpu_cores: int               # psutil.cpu_count(logical=False)
    cpu_threads: int
    gpu_name: str | None         # nvidia-smi
    gpu_vram_gb: float | None
    platform: str                # Linux, Darwin, Windows

    @property
    def can_run_3b(self) -> bool:   # >= 4GB RAM
    @property
    def can_run_7b(self) -> bool:   # >= 8GB RAM ou >= 6GB VRAM
    @property
    def can_run_13b(self) -> bool:  # >= 16GB RAM ou >= 10GB VRAM

    def max_model_size(self) -> str:  # "3b" | "7b" | "13b" | "none"
    def recommend_ollama_models(self) -> list[str]
```

Détection GPU via `nvidia-smi` (subprocess) — pas besoin d'installer PyTorch.
Détection RAM via `psutil` (déjà une dep standard).

### Étape 2 : LLMProvider (base abstraite)

```python
class LLMProvider(ABC):
    """Interface commune pour tous les providers."""
    name: str              # "ollama", "groq", "gemini", "anthropic"
    provider_type: str     # "local" | "cloud"

    @abstractmethod
    async def chat(self, messages, model, **kwargs) -> str

    @abstractmethod
    async def test_connection(self, model: str) -> TestResult

    @abstractmethod
    def list_available_models(self) -> list[ModelInfo]

    @abstractmethod
    def is_configured(self) -> bool
```

### Étape 3 : ModelInfo et ModelRegistry

```python
@dataclass
class ModelInfo:
    model_id: str           # "ollama:llama3.1:8b", "groq:llama-3.3-70b"
    provider: str           # "ollama", "groq", "gemini", "anthropic"
    display_name: str       # "Llama 3.1 8B (Local)"
    capability: str         # "basic" | "standard" | "advanced"
    context_window: int     # tokens
    is_free: bool
    is_local: bool
    status: str             # "untested" | "available" | "failed" | "rate_limited"
    last_test: datetime | None
    avg_latency_ms: float | None

class ModelRegistry:
    """Catalogue centralisé de tous les modèles disponibles."""

    def register_provider(self, provider: LLMProvider)
    def discover_models(self) -> list[ModelInfo]    # Auto-discovery
    def test_model(self, model_id: str) -> TestResult  # Test obligatoire
    def test_all(self) -> dict[str, TestResult]
    def get_available(self, capability: str = None) -> list[ModelInfo]
    def get_best_for(self, task: str) -> ModelInfo  # Routing intelligent
```

### Étape 4 : Routing intelligent dans Brain

Brain ne choisit plus un modèle hardcodé — il demande au `ModelRegistry` le meilleur modèle disponible :

```python
# AVANT (hardcodé)
AGENT_MODELS = {
    "brain": AgentModelConfig(model="claude-sonnet-4-5-20250929", ...),
    "vox": AgentModelConfig(model="claude-haiku-4-5-20251001", ...),
}

# APRÈS (dynamique)
# config.py charge le registry, qui sélectionne le meilleur modèle testé
def get_agent_model(agent_name: str) -> AgentModelConfig:
    registry = get_model_registry()

    # Chaque agent a un niveau de capability requis
    requirements = {
        "vox": "basic",          # Reformulation → modèle léger/rapide
        "brain": "advanced",     # Orchestration → meilleur modèle dispo
        "memory": "basic",       # Consolidation → léger suffit
        "worker:researcher": "standard",
        "worker:coder": "advanced",
        ...
    }

    capability = requirements.get(agent_name, "standard")
    best = registry.get_best_for(capability)
    return AgentModelConfig(model=best.model_id, ...)
```

**Stratégie de sélection :**

| Capability | Priorité 1 (préféré) | Priorité 2 | Priorité 3 (fallback) |
|-----------|----------------------|------------|----------------------|
| **basic** | Ollama 3-7B (gratuit, rapide) | Groq Llama 8B | Gemini Flash-Lite |
| **standard** | Ollama 7-13B ou Groq 70B | Gemini Flash | Anthropic Haiku |
| **advanced** | Anthropic Sonnet | Groq Llama 70B | Gemini Pro |

### Étape 5 : Wizard setup (nouvelle étape)

Le wizard actuel a 5 étapes. On ajoute une étape 4 bis "Modèles LLM" entre l'auth Anthropic et la finalisation :

```
Étape 4 : Connexion Anthropic (existant)
Étape 5 : Configuration des modèles LLM  ← NOUVEAU
  5a. Détection hardware automatique
  5b. Proposition de providers selon capabilities
  5c. Configuration Ollama (si hardware ok)
  5d. Configuration Groq (clé gratuite)
  5e. Configuration Gemini (clé gratuite)
  5f. Test de tous les modèles configurés
  5g. Affichage du routing final
Étape 6 : Finalisation
```

**Exemple de flux wizard :**

```
  [5/7] Configuration des modèles LLM
  ─────────────────────────────────────

  ⧗ Détection du hardware...
  ✓ RAM: 8 GB | CPU: 4 cores | GPU: Aucun

  Modèles locaux (Ollama) :
  ✓ Votre VPS peut faire tourner des modèles jusqu'à 3B
    Recommandé : Llama 3.2 3B, Qwen 2.5 3B

  ▸ Installer Ollama et les modèles recommandés ? [O/n]: o
  ⧗ Installation d'Ollama...
  ✓ Ollama installé
  ⧗ Téléchargement de llama3.2:3b...
  ✓ llama3.2:3b prêt

  Modèles cloud gratuits :
  ▸ Configurer Groq (gratuit, 14 400 req/jour) ? [O/n]: o
  ▸ Clé Groq (console.groq.com) : gsk_xxx...
  ✓ Groq configuré

  ▸ Configurer Gemini (gratuit, 250 req/jour) ? [O/n]: n

  ⧗ Test de tous les modèles...
  ✓ ollama:llama3.2:3b → 850ms (basic)
  ✓ groq:llama-3.3-70b → 120ms (advanced)
  ✓ anthropic:claude-sonnet → 1200ms (advanced)

  Routing final :
    Vox      → ollama:llama3.2:3b (local, gratuit)
    Brain    → groq:llama-3.3-70b (cloud, gratuit)
    Memory   → ollama:llama3.2:3b (local, gratuit)
    Workers  → groq:llama-3.3-70b / anthropic:sonnet
```

### Étape 6 : Persistance dans la config

Le `neo_config.json` est enrichi :

```json
{
  "core_name": "Neo",
  "user_name": "Eden",
  "version": "0.6",
  "stage": 6,
  "providers": {
    "ollama": {
      "enabled": true,
      "base_url": "http://localhost:11434",
      "models": ["llama3.2:3b"]
    },
    "groq": {
      "enabled": true,
      "models": ["llama-3.3-70b-versatile", "llama3-8b-8192"]
    },
    "gemini": {
      "enabled": false
    },
    "anthropic": {
      "enabled": true,
      "models": ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"]
    }
  },
  "model_routing": {
    "vox": "ollama:llama3.2:3b",
    "brain": "groq:llama-3.3-70b-versatile",
    "memory": "ollama:llama3.2:3b",
    "worker:researcher": "groq:llama-3.3-70b-versatile",
    "worker:coder": "anthropic:claude-sonnet-4-5-20250929",
    "worker:generic": "ollama:llama3.2:3b"
  },
  "tested_models": {
    "ollama:llama3.2:3b": {"status": "available", "latency_ms": 850, "tested_at": "..."},
    "groq:llama-3.3-70b-versatile": {"status": "available", "latency_ms": 120, "tested_at": "..."}
  }
}
```

### Étape 7 : Adaptation de l'API call dans Worker et Brain

Actuellement, Worker._real_execute() et Brain font des appels httpx directs à l'API Anthropic. Il faudra router vers le bon provider :

```python
# Worker._real_execute() — AVANT
async with httpx.AsyncClient() as client:
    response = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers, json=payload
    )

# APRÈS — le provider gère l'appel
provider = registry.get_provider(self._model_config.provider)
response = await provider.chat(
    messages=messages,
    model=self._model_config.model,
    system=prompt,
    tools=tool_schemas,
    max_tokens=self._model_config.max_tokens,
    temperature=self._model_config.temperature,
)
```

Chaque provider traduit en son format :
- **Anthropic** : POST /v1/messages (format existant)
- **Ollama** : POST /api/chat (format natif) ou /v1/chat/completions (OpenAI compat)
- **Groq** : POST /openai/v1/chat/completions (OpenAI compat)
- **Gemini** : via SDK google-generativeai

**Point clé** : Groq et Ollama supportent le format OpenAI, donc on peut factoriser 70% du code d'appel.

---

## Fichiers impactés

| Fichier | Modification |
|---------|-------------|
| `neo_core/providers/` (NOUVEAU) | 7 nouveaux fichiers |
| `neo_core/config.py` | Charger le routing depuis neo_config.json, `get_agent_model()` dynamique |
| `neo_core/cli/setup.py` | Nouvelle étape wizard (hardware + providers + tests) |
| `neo_core/teams/worker.py` | `_real_execute()` utilise le provider au lieu de httpx direct |
| `neo_core/core/brain.py` | `_oauth_response()` / `_llm_response()` passent par le provider |
| `neo_core/core/vox.py` | `_vox_llm_call()` passe par le provider |
| `neo_core/core/memory_agent.py` | `_memory_llm_call()` passe par le provider |
| `neo_core/cli/status.py` | Afficher les providers actifs + modèles disponibles |
| `setup.py` (deps) | Ajouter `psutil`, `ollama` (optionnel), `groq` (optionnel), `google-generativeai` (optionnel) |
| `tests/` | Tests pour providers, registry, hardware, routing |

---

## Ordre d'implémentation

1. **`hardware.py`** — Détection VPS (indépendant)
2. **`base.py`** — Interface LLMProvider
3. **`anthropic.py`** — Refactor de l'existant en provider
4. **`ollama.py`** — Provider local
5. **`groq.py`** — Provider cloud gratuit
6. **`gemini.py`** — Provider cloud gratuit
7. **`registry.py`** — Catalogue + tests + routing
8. **`config.py`** — Routing dynamique
9. **`setup.py`** — Wizard enrichi
10. **Brain/Worker/Vox/Memory** — Utiliser les providers
11. **Tests** — Valider tout le pipeline

---

## Questions de design

### Tool Use (function calling)

Seuls certains providers supportent le tool_use natif :
- **Anthropic** : ✅ natif
- **Groq** : ✅ via OpenAI function calling
- **Ollama** : ⚠️ supporté depuis récemment, mais instable sur les petits modèles
- **Gemini** : ✅ via SDK

**Décision** : Pour les Workers qui utilisent des outils (researcher, coder), on privilégie les providers avec tool_use fiable. Les modèles locaux sans tool_use sont réservés à Vox et Memory (pas de tools).
