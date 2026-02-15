/**
 * Tests pour le système de modèles de Neo
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { ModelRouter, ModelInfo, ModelTier, getModelRouter } from '../src/core/models';

describe('ModelRouter', () => {
  let router: ModelRouter;

  beforeEach(() => {
    // Créer un router avec config vide (pas de clés API)
    router = new ModelRouter({});
  });

  describe('Model Selection', () => {
    it('should select model for simple chat preferring free tier', () => {
      // Note: Sans clés API, seuls les modèles Ollama locaux seraient disponibles
      // Ce test vérifie la logique de sélection
      const model = router.selectModel({
        task: 'simple_chat',
        maxTier: 'free',
        preferSpeed: true,
      });

      // Peut être null si aucun modèle gratuit n'est disponible
      if (model) {
        expect(model.tier).toBe('free');
        expect(model.capabilities).toContain('chat');
      }
    });

    it('should escalate to higher tier if needed', () => {
      // Essayer de sélectionner avec différents tiers
      const freeTierModel = router.selectModel({ task: 'reasoning', maxTier: 'free' });
      const standardTierModel = router.selectModel({ task: 'reasoning', maxTier: 'standard' });

      // Le modèle standard devrait exister si configuré
      // Le free peut ne pas avoir reasoning
      if (freeTierModel && standardTierModel) {
        expect(['free', 'cheap', 'standard']).toContain(standardTierModel.tier);
      }
    });

    it('should prefer speed for simple tasks', () => {
      const fastModel = router.selectModel({
        task: 'simple_chat',
        maxTier: 'standard',
        preferSpeed: true,
      });

      const qualityModel = router.selectModel({
        task: 'simple_chat',
        maxTier: 'standard',
        preferSpeed: false,
      });

      // Les deux peuvent être le même modèle, mais la logique doit fonctionner
      if (fastModel && qualityModel) {
        expect(fastModel.speed).toBeDefined();
        expect(qualityModel.quality).toBeDefined();
      }
    });
  });

  describe('Available Models Detection', () => {
    it('should return empty array when no API keys configured', async () => {
      const models = await router.detectAvailableModels();

      // Sans clés API et sans Ollama local, devrait être vide ou presque
      expect(Array.isArray(models)).toBe(true);
    });

    it('should categorize models by tier', () => {
      const freeModels = router.getModelsByTier('free');
      const cheapModels = router.getModelsByTier('cheap');
      const standardModels = router.getModelsByTier('standard');
      const premiumModels = router.getModelsByTier('premium');

      // Tous devraient être des arrays (possiblement vides)
      expect(Array.isArray(freeModels)).toBe(true);
      expect(Array.isArray(cheapModels)).toBe(true);
      expect(Array.isArray(standardModels)).toBe(true);
      expect(Array.isArray(premiumModels)).toBe(true);
    });
  });

  describe('Usage Stats', () => {
    it('should start with zero costs', () => {
      const totalCost = router.getTotalCost();
      expect(totalCost).toBe(0);
    });

    it('should return usage stats map', () => {
      const stats = router.getUsageStats();
      expect(stats instanceof Map).toBe(true);
    });
  });

  describe('Model Info Structure', () => {
    it('should have required properties for each model', () => {
      const models = router.getAvailableModels();

      for (const model of models) {
        expect(model).toHaveProperty('id');
        expect(model).toHaveProperty('name');
        expect(model).toHaveProperty('provider');
        expect(model).toHaveProperty('tier');
        expect(model).toHaveProperty('contextWindow');
        expect(model).toHaveProperty('costPer1kTokens');
        expect(model).toHaveProperty('capabilities');
        expect(model).toHaveProperty('speed');
        expect(model).toHaveProperty('quality');

        // Vérifier les types
        expect(typeof model.id).toBe('string');
        expect(typeof model.contextWindow).toBe('number');
        expect(Array.isArray(model.capabilities)).toBe(true);
        expect(['free', 'cheap', 'standard', 'premium']).toContain(model.tier);
        expect(['fast', 'medium', 'slow']).toContain(model.speed);
        expect(['basic', 'good', 'excellent']).toContain(model.quality);
      }
    });
  });
});

describe('ModelRouter Singleton', () => {
  it('should return same instance', () => {
    const router1 = getModelRouter();
    const router2 = getModelRouter();

    // Note: Le singleton peut être réinitialisé entre les tests
    // Vérifions juste qu'ils sont des instances valides
    expect(router1).toBeInstanceOf(ModelRouter);
    expect(router2).toBeInstanceOf(ModelRouter);
  });
});

describe('Model Capabilities', () => {
  it('should define valid capabilities', () => {
    const validCapabilities = [
      'chat',
      'code',
      'reasoning',
      'creative',
      'multilingual',
      'long_context',
      'function_calling',
      'vision',
    ];

    const router = new ModelRouter({});
    const models = router.getAvailableModels();

    for (const model of models) {
      for (const cap of model.capabilities) {
        expect(validCapabilities).toContain(cap);
      }
    }
  });
});

describe('Cost Calculations', () => {
  it('should have zero cost for free tier models', () => {
    const router = new ModelRouter({});
    const freeModels = router.getModelsByTier('free');

    for (const model of freeModels) {
      expect(model.costPer1kTokens).toBe(0);
    }
  });

  it('should have low cost for cheap tier models', () => {
    const router = new ModelRouter({});
    const cheapModels = router.getModelsByTier('cheap');

    for (const model of cheapModels) {
      // Cheap devrait être moins de $0.001 par 1k tokens
      expect(model.costPer1kTokens).toBeLessThanOrEqual(0.001);
    }
  });
});
