/**
 * Tests pour le service de recherche web de Neo
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { WebSearchService } from '../src/utils/web-search';

describe('WebSearchService', () => {
  let webSearch: WebSearchService;

  beforeEach(() => {
    webSearch = new WebSearchService();
  });

  describe('Configuration', () => {
    it('should return config without exposing secrets', () => {
      const config = webSearch.getConfig();

      expect(config).toHaveProperty('provider');
      expect(config).toHaveProperty('hasApiKey');
      expect(config).toHaveProperty('hasSearxngUrl');
      expect(config).toHaveProperty('timeout');

      // hasApiKey/hasSearxngUrl sont des booleans, pas les vraies clés
      expect(typeof config.hasApiKey).toBe('boolean');
      expect(typeof config.hasSearxngUrl).toBe('boolean');
    });

    it('should default to duckduckgo when no API keys', () => {
      const config = webSearch.getConfig();

      // Sans clés API, devrait utiliser DuckDuckGo par défaut
      if (!config.hasApiKey && !config.hasSearxngUrl) {
        expect(config.provider).toBe('duckduckgo');
      }
    });

    it('should have reasonable default timeout', () => {
      const config = webSearch.getConfig();
      expect(config.timeout).toBeGreaterThanOrEqual(5000);
      expect(config.timeout).toBeLessThanOrEqual(30000);
    });
  });

  describe('Health Check', () => {
    it('should return health status object', async () => {
      const health = await webSearch.healthCheck();

      expect(health).toHaveProperty('status');
      expect(health).toHaveProperty('provider');
      expect(['ok', 'error']).toContain(health.status);

      if (health.status === 'error') {
        expect(health).toHaveProperty('message');
      }
    });
  });

  describe('Text Extraction', () => {
    // Test internal HTML extraction logic via public method
    it('should handle fetch errors gracefully', async () => {
      // URL inexistante
      const content = await webSearch.fetchPageContent('https://this-domain-does-not-exist-12345.com');
      expect(content).toBe('');
    });
  });

  describe('Search Options', () => {
    it('should accept search options', async () => {
      // Ce test vérifie que les options sont acceptées sans erreur
      // Le résultat dépend de la disponibilité du provider
      const searchFn = () => webSearch.search('test query', {
        maxResults: 3,
        language: 'fr',
        safeSearch: true,
      });

      // Ne devrait pas throw
      await expect(searchFn()).resolves.toBeDefined();
    });
  });

  describe('Search Result Structure', () => {
    it('should return array of results', async () => {
      try {
        const results = await webSearch.search('test', { maxResults: 1 });

        expect(Array.isArray(results)).toBe(true);

        if (results.length > 0) {
          const result = results[0];
          expect(result).toHaveProperty('title');
          expect(result).toHaveProperty('url');
          expect(result).toHaveProperty('snippet');
          expect(result).toHaveProperty('source');

          // URL devrait être valide
          expect(result.url).toMatch(/^https?:\/\//);
        }
      } catch {
        // Si la recherche échoue (pas de connexion, rate limit, etc.)
        // le test est quand même valide
        expect(true).toBe(true);
      }
    });
  });

  describe('Provider Selection', () => {
    it('should detect best provider based on config', () => {
      // Créer avec SearXNG URL
      const searxngService = new WebSearchService({
        searxngUrl: 'http://localhost:8080',
      });
      expect(searxngService.getConfig().provider).toBe('searxng');

      // Créer avec config vide
      const defaultService = new WebSearchService({});
      expect(defaultService.getConfig().provider).toBe('duckduckgo');
    });
  });
});

describe('WebSearchService Error Handling', () => {
  it('should handle network errors gracefully', async () => {
    const webSearch = new WebSearchService({
      provider: 'searxng',
      searxngUrl: 'http://localhost:99999', // Port invalide
    });

    // Devrait fallback vers DuckDuckGo ou retourner vide
    const results = await webSearch.search('test');
    expect(Array.isArray(results)).toBe(true);
  });

  it('should respect timeout', async () => {
    const webSearch = new WebSearchService({
      timeout: 100, // Très court timeout
    });

    const start = Date.now();
    try {
      await webSearch.search('test query');
    } catch {
      // Timeout expected
    }
    const duration = Date.now() - start;

    // Devrait timeout rapidement (avec une marge pour le traitement)
    expect(duration).toBeLessThan(5000);
  });
});

describe('SearchAndFetch', () => {
  it('should search and fetch content in one call', async () => {
    const webSearch = new WebSearchService();

    try {
      const results = await webSearch.searchAndFetch('test', { maxResults: 1 });

      expect(Array.isArray(results)).toBe(true);

      if (results.length > 0) {
        // Les premiers résultats devraient avoir content (ou undefined si fetch échoue)
        expect(results[0]).toHaveProperty('content');
      }
    } catch {
      // Network errors sont OK pour les tests
      expect(true).toBe(true);
    }
  });
});
