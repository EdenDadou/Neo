"""
Neo Core — HardwareDetector : Détection des capacités du VPS
=============================================================
Détecte RAM, CPU, GPU pour déterminer quels modèles locaux
peuvent tourner sur la machine.

Utilisé par le wizard setup et le ModelRegistry.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUInfo:
    """Informations sur un GPU détecté."""
    name: str
    vram_mb: int
    vram_gb: float
    driver_version: str = ""

    def __str__(self) -> str:
        return f"{self.name} ({self.vram_gb:.1f} GB VRAM)"


@dataclass
class HardwareProfile:
    """
    Profil matériel complet du système.

    Détecté automatiquement par HardwareDetector.detect().
    Utilisé pour recommander les modèles Ollama adaptés.
    """
    total_ram_gb: float
    available_ram_gb: float
    cpu_cores: int          # Cœurs physiques
    cpu_threads: int        # Threads logiques
    cpu_name: str = ""
    platform_name: str = ""
    gpu: Optional[GPUInfo] = None

    # ── Capacités modèles ──

    @property
    def has_gpu(self) -> bool:
        return self.gpu is not None and self.gpu.vram_gb > 0

    @property
    def effective_memory_gb(self) -> float:
        """
        Mémoire effective pour les modèles LLM.

        GPU VRAM est prioritaire (inférence 10-30x plus rapide).
        Sans GPU, on utilise la RAM (CPU inference, plus lent).
        """
        if self.has_gpu:
            return self.gpu.vram_gb
        return self.total_ram_gb

    @property
    def can_run_1_5b(self) -> bool:
        """Peut tourner des modèles 1-1.5B (DeepSeek-R1 1.5B, Qwen 0.5-1.5B)."""
        return self.effective_memory_gb >= 2

    @property
    def can_run_3b(self) -> bool:
        """Peut tourner des modèles 3B (Llama 3.2 3B, Phi-3 mini)."""
        return self.effective_memory_gb >= 4

    @property
    def can_run_7b(self) -> bool:
        """Peut tourner des modèles 7-8B (Llama 3.1 8B, Mistral 7B, DeepSeek-R1 8B)."""
        return self.effective_memory_gb >= 6

    @property
    def can_run_14b(self) -> bool:
        """Peut tourner des modèles 14B (DeepSeek-R1 14B, Phi-4 14B)."""
        return self.effective_memory_gb >= 10

    @property
    def can_run_32b(self) -> bool:
        """Peut tourner des modèles 30-32B (DeepSeek-R1 32B)."""
        return self.effective_memory_gb >= 20

    @property
    def can_run_70b(self) -> bool:
        """Peut tourner des modèles 70B (Llama 3.1 70B, DeepSeek-R1 70B)."""
        return self.effective_memory_gb >= 40

    def max_model_size(self) -> str:
        """Retourne la taille max de modèle supportée."""
        if self.can_run_70b:
            return "70b"
        if self.can_run_32b:
            return "32b"
        if self.can_run_14b:
            return "14b"
        if self.can_run_7b:
            return "7b"
        if self.can_run_3b:
            return "3b"
        if self.can_run_1_5b:
            return "1.5b"
        return "none"

    def recommend_ollama_models(self) -> list[dict]:
        """
        Recommande les modèles Ollama adaptés au hardware.

        Retourne une liste ordonnée par priorité (meilleur en premier).
        Chaque entrée : {"model": str, "size": str, "role": str, "priority": int}
        """
        models = []

        if self.can_run_7b:
            # ── 8GB+ : modèles performants ──
            models.append({
                "model": "deepseek-r1:8b",
                "size": "8B",
                "role": "Raisonnement, code, analyse",
                "priority": 1,
            })
            models.append({
                "model": "llama3.1:8b",
                "size": "8B",
                "role": "Général, conversation, résumé",
                "priority": 2,
            })

        if self.can_run_14b:
            # ── 16GB+ : insérer les modèles 14B en tête ──
            models.insert(0, {
                "model": "deepseek-r1:14b",
                "size": "14B",
                "role": "Raisonnement avancé, code complexe",
                "priority": 0,
            })

        if self.can_run_3b and not self.can_run_7b:
            # ── 4-8GB : modèles légers ──
            models.append({
                "model": "llama3.2:3b",
                "size": "3B",
                "role": "Général léger, reformulation",
                "priority": 1,
            })
            models.append({
                "model": "deepseek-r1:1.5b",
                "size": "1.5B",
                "role": "Raisonnement basique",
                "priority": 2,
            })

        if self.can_run_1_5b and not self.can_run_3b:
            # ── 2-4GB : ultra-léger uniquement ──
            models.append({
                "model": "deepseek-r1:1.5b",
                "size": "1.5B",
                "role": "Raisonnement basique",
                "priority": 1,
            })
            models.append({
                "model": "qwen2.5:0.5b",
                "size": "0.5B",
                "role": "Tâches très simples",
                "priority": 2,
            })

        return models

    def summary(self) -> str:
        """Résumé lisible du profil hardware."""
        parts = [
            f"RAM: {self.total_ram_gb:.1f} GB",
            f"CPU: {self.cpu_cores} cores / {self.cpu_threads} threads",
        ]
        if self.has_gpu:
            parts.append(f"GPU: {self.gpu}")
        else:
            parts.append("GPU: Aucun")

        max_size = self.max_model_size()
        if max_size == "none":
            parts.append("Modèles locaux: Non recommandé")
        else:
            parts.append(f"Modèles locaux: Jusqu'à {max_size}")

        return " | ".join(parts)


class HardwareDetector:
    """
    Détecte les capacités matérielles du système.

    Méthodes de détection :
    - RAM/CPU : via psutil (fiable, cross-platform)
    - GPU : via nvidia-smi (subprocess, pas de dep lourde)

    Usage :
        profile = HardwareDetector.detect()
        print(profile.summary())
        models = profile.recommend_ollama_models()
    """

    @staticmethod
    def detect() -> HardwareProfile:
        """Détecte le profil hardware complet."""
        ram_total, ram_avail = HardwareDetector._detect_ram()
        cpu_cores, cpu_threads, cpu_name = HardwareDetector._detect_cpu()
        gpu = HardwareDetector._detect_gpu()

        return HardwareProfile(
            total_ram_gb=ram_total,
            available_ram_gb=ram_avail,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            cpu_name=cpu_name,
            platform_name=platform.system(),
            gpu=gpu,
        )

    @staticmethod
    def _detect_ram() -> tuple[float, float]:
        """Détecte la RAM totale et disponible en GB."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return (
                mem.total / (1024 ** 3),
                mem.available / (1024 ** 3),
            )
        except ImportError:
            # Fallback Linux : /proc/meminfo
            try:
                with open("/proc/meminfo") as f:
                    lines = f.readlines()
                total = available = 0
                for line in lines:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) / (1024 * 1024)  # kB → GB
                    elif line.startswith("MemAvailable:"):
                        available = int(line.split()[1]) / (1024 * 1024)
                return (total, available)
            except Exception:
                return (0.0, 0.0)

    @staticmethod
    def _detect_cpu() -> tuple[int, int, str]:
        """Détecte les cores, threads, et nom du CPU."""
        try:
            import psutil
            cores = psutil.cpu_count(logical=False) or 1
            threads = psutil.cpu_count(logical=True) or 1
        except ImportError:
            # Fallback
            import os
            threads = os.cpu_count() or 1
            cores = threads  # Approximation

        cpu_name = platform.processor() or "Unknown"
        return (cores, threads, cpu_name)

    @staticmethod
    def _detect_gpu() -> Optional[GPUInfo]:
        """Détecte le GPU NVIDIA via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,nounits,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return None

            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]

            if len(parts) >= 2:
                vram_mb = int(float(parts[1]))
                return GPUInfo(
                    name=parts[0],
                    vram_mb=vram_mb,
                    vram_gb=vram_mb / 1024,
                    driver_version=parts[2] if len(parts) > 2 else "",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        return None

    @staticmethod
    def is_ollama_installed() -> bool:
        """Vérifie si Ollama est installé sur le système."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def is_ollama_running() -> bool:
        """Vérifie si le serveur Ollama tourne."""
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    @staticmethod
    def get_ollama_models() -> list[str]:
        """Liste les modèles Ollama déjà téléchargés."""
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []
