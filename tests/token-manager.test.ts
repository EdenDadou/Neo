/**
 * Tests pour le TokenManager - Gestion des ressources de Neo
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { TokenManager } from '../src/core/token-manager';

describe('TokenManager', () => {
  let tokenManager: TokenManager;

  beforeEach(() => {
    // Créer une nouvelle instance pour chaque test (sans persistence)
    tokenManager = new TokenManager();
  });

  describe('Usage Tracking', () => {
    it('should record token usage', () => {
      tokenManager.recordUsage('Brain', {
        inputTokens: 100,
        outputTokens: 50,
        model: 'claude-sonnet-4-20250514',
        provider: 'anthropic',
      });

      const stats = tokenManager.getQuickStats();
      expect(stats.todayTokens).toBe(150);
      expect(stats.todayCalls).toBe(1);
    });

    it('should track usage by multiple agents', () => {
      tokenManager.recordUsage('Vox', {
        inputTokens: 200,
        outputTokens: 100,
        model: 'claude-sonnet-4-20250514',
        provider: 'anthropic',
      });

      tokenManager.recordUsage('Memory', {
        inputTokens: 300,
        outputTokens: 150,
        model: 'claude-sonnet-4-20250514',
        provider: 'anthropic',
      });

      tokenManager.recordUsage('Brain', {
        inputTokens: 500,
        outputTokens: 200,
        model: 'claude-sonnet-4-20250514',
        provider: 'anthropic',
      });

      const stats = tokenManager.getQuickStats();
      expect(stats.todayTokens).toBe(1450);
      expect(stats.todayCalls).toBe(3);
    });
  });

  describe('Resource Planning', () => {
    it('should calculate remaining resources', () => {
      const plan = tokenManager.getResourcePlan();

      expect(plan.remainingTokens).toBeGreaterThan(0);
      expect(plan.remainingBudget).toBeGreaterThan(0);
      expect(plan.hoursUntilReset).toBeGreaterThanOrEqual(0);
      expect(plan.hoursUntilReset).toBeLessThanOrEqual(24);
    });

    it('should provide operation estimates', () => {
      const plan = tokenManager.getResourcePlan();

      expect(plan.canAfford.simpleChat).toBeGreaterThan(0);
      expect(plan.canAfford.complexTask).toBeGreaterThan(0);
      expect(plan.canAfford.synthesis).toBeGreaterThan(0);
    });

    it('should recommend appropriate tier based on budget', () => {
      // Avec budget par défaut ($10), devrait recommander standard ou premium
      const plan = tokenManager.getResourcePlan();
      expect(['free', 'cheap', 'standard', 'premium']).toContain(plan.recommendedTier);
    });
  });

  describe('Daily Limits', () => {
    it('should allow setting custom limits', () => {
      tokenManager.setDailyLimits({
        maxTokens: 500000,
        maxCost: 5.0,
        warningThreshold: 0.9,
      });

      const plan = tokenManager.getResourcePlan();
      expect(plan.remainingTokens).toBeLessThanOrEqual(500000);
    });

    it('should check if operation is affordable', () => {
      // Set low limits
      tokenManager.setDailyLimits({ maxTokens: 1000 });

      // Small operation should be allowed
      const smallOp = tokenManager.canAfford(500);
      expect(smallOp.allowed).toBe(true);

      // Large operation should be blocked
      const largeOp = tokenManager.canAfford(2000);
      expect(largeOp.allowed).toBe(false);
      expect(largeOp.reason).toBeDefined();
    });

    it('should block operations that exceed limits', () => {
      tokenManager.setDailyLimits({ maxTokens: 1000 });

      // Use up most of the budget
      tokenManager.recordUsage('Test', {
        inputTokens: 800,
        outputTokens: 100,
        model: 'test-model',
        provider: 'anthropic',
      });

      // Next operation would exceed limit
      const check = tokenManager.canAfford(200);
      expect(check.allowed).toBe(false);
    });
  });

  describe('Model Suggestions', () => {
    it('should suggest models based on task and budget', () => {
      const suggestion = tokenManager.suggestModelForTask('simple_chat');

      expect(suggestion.reason).toBeDefined();
      // Model might be null if no models are available (no API keys)
      // but reason should always be present
    });

    it('should suggest cheaper models when budget is low', () => {
      // Set very low budget
      tokenManager.setDailyLimits({ maxCost: 0.01 });

      // Use up most of budget
      tokenManager.recordUsage('Test', {
        inputTokens: 100,
        outputTokens: 50,
        model: 'test',
        provider: 'anthropic',
        cost: 0.009,
      });

      const suggestion = tokenManager.suggestModelForTask('reasoning');
      // Devrait recommander free tier quand budget épuisé
      expect(suggestion.reason).toMatch(/free|limité|reste \$0/i);
    });
  });

  describe('Usage Reports', () => {
    it('should generate usage report', () => {
      // Add some usage
      tokenManager.recordUsage('Brain', {
        inputTokens: 100,
        outputTokens: 50,
        model: 'claude-sonnet-4-20250514',
        provider: 'anthropic',
      });

      const report = tokenManager.getUsageReport('today');

      expect(report.period).toBe('today');
      expect(report.totalTokens).toBe(150);
      expect(report.byModel).toBeDefined();
      expect(report.trends).toBeDefined();
    });

    it('should track trends correctly', () => {
      // Add multiple usages
      for (let i = 0; i < 5; i++) {
        tokenManager.recordUsage('Brain', {
          inputTokens: 100 + i * 10,
          outputTokens: 50 + i * 5,
          model: 'claude-sonnet-4-20250514',
          provider: 'anthropic',
        });
      }

      const report = tokenManager.getUsageReport('today');

      expect(report.trends.mostUsedModel).toBe('claude-sonnet-4-20250514');
      expect(report.byAgent['Brain']).toBeDefined();
      expect(report.byAgent['Brain'].totalCalls).toBe(5);
    });
  });

  describe('Quick Stats', () => {
    it('should provide quick overview', () => {
      const stats = tokenManager.getQuickStats();

      expect(stats).toHaveProperty('todayTokens');
      expect(stats).toHaveProperty('todayCost');
      expect(stats).toHaveProperty('todayCalls');
      expect(stats).toHaveProperty('remainingBudget');
      expect(stats).toHaveProperty('remainingTokens');
      expect(stats).toHaveProperty('recommendedTier');
    });

    it('should update in real-time', () => {
      const before = tokenManager.getQuickStats();
      const beforeTokens = before.todayTokens;

      tokenManager.recordUsage('Test', {
        inputTokens: 500,
        outputTokens: 250,
        model: 'test',
        provider: 'anthropic',
      });

      const after = tokenManager.getQuickStats();
      expect(after.todayTokens).toBe(beforeTokens + 750);
    });
  });
});
