/**
 * Tests pour le système de mémoire de Neo
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { PersistenceLayer } from '../src/core/memory/persistence';
import { EmbeddingsService } from '../src/core/memory/embeddings';
import * as fs from 'fs';
import * as path from 'path';

// Utiliser un dossier temporaire pour les tests
const TEST_DATA_DIR = './data-test';

describe('PersistenceLayer', () => {
  let persistence: PersistenceLayer;

  beforeEach(() => {
    // Nettoyer le dossier de test
    if (fs.existsSync(TEST_DATA_DIR)) {
      fs.rmSync(TEST_DATA_DIR, { recursive: true });
    }
    persistence = new PersistenceLayer(TEST_DATA_DIR);
  });

  afterEach(() => {
    persistence.close();
    // Nettoyer après les tests
    if (fs.existsSync(TEST_DATA_DIR)) {
      fs.rmSync(TEST_DATA_DIR, { recursive: true });
    }
  });

  describe('Memories CRUD', () => {
    it('should save and retrieve a memory', () => {
      const memory = {
        id: 'test-1',
        type: 'fact' as const,
        content: 'Test memory content',
        metadata: {
          source: 'test',
          confidence: 0.9,
          tags: ['test'],
          relatedIds: [],
        },
        createdAt: new Date(),
        lastAccessedAt: new Date(),
        accessCount: 0,
        importance: 0.8,
      };

      persistence.saveMemory(memory);
      const retrieved = persistence.getMemory('test-1');

      expect(retrieved).not.toBeNull();
      expect(retrieved?.content).toBe('Test memory content');
      expect(retrieved?.type).toBe('fact');
      expect(retrieved?.importance).toBe(0.8);
    });

    it('should search memories by type', () => {
      // Sauvegarder plusieurs mémoires
      for (let i = 0; i < 5; i++) {
        persistence.saveMemory({
          id: `fact-${i}`,
          type: 'fact',
          content: `Fact ${i}`,
          metadata: { source: 'test', confidence: 0.9, tags: [], relatedIds: [] },
          createdAt: new Date(),
          lastAccessedAt: new Date(),
          accessCount: 0,
          importance: 0.5 + i * 0.1,
        });
      }

      for (let i = 0; i < 3; i++) {
        persistence.saveMemory({
          id: `pref-${i}`,
          type: 'preference',
          content: `Preference ${i}`,
          metadata: { source: 'test', confidence: 0.9, tags: [], relatedIds: [] },
          createdAt: new Date(),
          lastAccessedAt: new Date(),
          accessCount: 0,
          importance: 0.7,
        });
      }

      const facts = persistence.searchMemories({ type: 'fact' });
      const prefs = persistence.searchMemories({ type: 'preference' });

      expect(facts.length).toBe(5);
      expect(prefs.length).toBe(3);
    });

    it('should update memory access count', () => {
      persistence.saveMemory({
        id: 'access-test',
        type: 'fact',
        content: 'Access test',
        metadata: { source: 'test', confidence: 1, tags: [], relatedIds: [] },
        createdAt: new Date(),
        lastAccessedAt: new Date(),
        accessCount: 0,
        importance: 0.5,
      });

      persistence.updateMemoryAccess('access-test');
      persistence.updateMemoryAccess('access-test');

      const memory = persistence.getMemory('access-test');
      expect(memory?.accessCount).toBe(2);
    });

    it('should delete memories', () => {
      persistence.saveMemory({
        id: 'to-delete',
        type: 'fact',
        content: 'Will be deleted',
        metadata: { source: 'test', confidence: 1, tags: [], relatedIds: [] },
        createdAt: new Date(),
        lastAccessedAt: new Date(),
        accessCount: 0,
        importance: 0.5,
      });

      expect(persistence.getMemory('to-delete')).not.toBeNull();

      const deleted = persistence.deleteMemory('to-delete');
      expect(deleted).toBe(true);
      expect(persistence.getMemory('to-delete')).toBeNull();
    });

    it('should update memory content', () => {
      persistence.saveMemory({
        id: 'to-update',
        type: 'fact',
        content: 'Original content',
        metadata: { source: 'test', confidence: 0.5, tags: ['old'], relatedIds: [] },
        createdAt: new Date(),
        lastAccessedAt: new Date(),
        accessCount: 0,
        importance: 0.5,
      });

      persistence.updateMemory('to-update', {
        content: 'Updated content',
        importance: 0.9,
        tags: ['new', 'updated'],
      });

      const memory = persistence.getMemory('to-update');
      expect(memory?.content).toBe('Updated content');
      expect(memory?.importance).toBe(0.9);
      expect(memory?.metadata.tags).toContain('updated');
    });
  });

  describe('BM25 Search (FTS5)', () => {
    beforeEach(() => {
      // Ajouter des données de test
      const testData = [
        { id: '1', content: 'Le chat mange la souris', type: 'fact' },
        { id: '2', content: 'Le chien court dans le jardin', type: 'fact' },
        { id: '3', content: 'La souris est petite et grise', type: 'fact' },
        { id: '4', content: 'Le jardin est vert en été', type: 'fact' },
        { id: '5', content: 'Python est un langage de programmation', type: 'fact' },
      ];

      for (const item of testData) {
        persistence.saveMemory({
          id: item.id,
          type: item.type as 'fact',
          content: item.content,
          metadata: { source: 'test', confidence: 1, tags: [], relatedIds: [] },
          createdAt: new Date(),
          lastAccessedAt: new Date(),
          accessCount: 0,
          importance: 0.5,
        });
      }
    });

    it('should find memories by keyword', () => {
      const results = persistence.searchBM25('souris', 10);

      expect(results.length).toBeGreaterThan(0);
      expect(results.some(r => r.id === '1' || r.id === '3')).toBe(true);
    });

    it('should find memories by multiple keywords', () => {
      const results = persistence.searchBM25('jardin vert', 10);

      expect(results.length).toBeGreaterThan(0);
      expect(results.some(r => r.id === '4')).toBe(true);
    });

    it('should return empty for non-matching query', () => {
      const results = persistence.searchBM25('xyznonexistent', 10);
      expect(results.length).toBe(0);
    });
  });

  describe('Personality', () => {
    it('should save and retrieve personality traits', () => {
      persistence.setPersonality('name', 'Neo');
      persistence.setPersonality('tone', 'friendly');

      expect(persistence.getPersonality('name')).toBe('Neo');
      expect(persistence.getPersonality('tone')).toBe('friendly');
    });

    it('should get all personality traits', () => {
      persistence.setPersonality('name', 'Neo');
      persistence.setPersonality('language', 'fr');

      const all = persistence.getAllPersonality();

      expect(all.name).toBe('Neo');
      expect(all.language).toBe('fr');
    });

    it('should update existing personality trait', () => {
      persistence.setPersonality('mood', 'happy');
      persistence.setPersonality('mood', 'neutral');

      expect(persistence.getPersonality('mood')).toBe('neutral');
    });
  });

  describe('Context Snapshots', () => {
    it('should save and retrieve context snapshots', () => {
      const snapshotId = persistence.saveContextSnapshot({
        sessionId: 'session-1',
        conversationSummary: 'Discussion about AI',
        keyFacts: ['fact1', 'fact2'],
        importantMemories: ['mem-1', 'mem-2'],
        userIntent: 'Learn about AI',
        tokenCount: 1500,
      });

      expect(snapshotId).toBeDefined();

      const snapshots = persistence.getContextSnapshots('session-1', 5);
      expect(snapshots.length).toBe(1);
      expect(snapshots[0].conversationSummary).toBe('Discussion about AI');
      expect(snapshots[0].keyFacts).toContain('fact1');
    });

    it('should get last context snapshot', () => {
      persistence.saveContextSnapshot({
        sessionId: 'session-2',
        conversationSummary: 'First conversation',
        keyFacts: [],
        importantMemories: [],
        tokenCount: 100,
      });

      persistence.saveContextSnapshot({
        sessionId: 'session-2',
        conversationSummary: 'Second conversation',
        keyFacts: ['latest'],
        importantMemories: [],
        tokenCount: 200,
      });

      const last = persistence.getLastContextSnapshot('session-2');
      expect(last?.conversationSummary).toBe('Second conversation');
      expect(last?.keyFacts).toContain('latest');
    });
  });

  describe('Stats', () => {
    it('should return correct stats', () => {
      // Ajouter des données
      for (let i = 0; i < 10; i++) {
        persistence.saveMemory({
          id: `stat-${i}`,
          type: i < 6 ? 'fact' : 'preference',
          content: `Content ${i}`,
          metadata: { source: 'test', confidence: 1, tags: [], relatedIds: [] },
          createdAt: new Date(),
          lastAccessedAt: new Date(),
          accessCount: 0,
          importance: 0.5,
        });
      }

      const stats = persistence.getStats();

      expect(stats.totalMemories).toBe(10);
      expect(stats.memoriesByType.fact).toBe(6);
      expect(stats.memoriesByType.preference).toBe(4);
    });
  });

  describe('Feedback (Règle 4: Neo s\'améliore)', () => {
    it('should save and retrieve feedback', () => {
      const feedbackId = persistence.saveFeedback({
        responseId: 'resp-1',
        sessionId: 'session-1',
        rating: 'positive',
        userMessage: 'Comment faire X?',
        assistantResponse: 'Voici comment faire X...',
        userComment: 'Super réponse!',
      });

      expect(feedbackId).toBeDefined();

      const unprocessed = persistence.getUnprocessedFeedback();
      expect(unprocessed.length).toBe(1);
      expect(unprocessed[0].rating).toBe('positive');
      expect(unprocessed[0].userComment).toBe('Super réponse!');
    });

    it('should mark feedback as processed', () => {
      const feedbackId = persistence.saveFeedback({
        responseId: 'resp-2',
        sessionId: 'session-1',
        rating: 'negative',
        userMessage: 'Question?',
        assistantResponse: 'Mauvaise réponse',
      });

      persistence.markFeedbackProcessed(feedbackId);

      const unprocessed = persistence.getUnprocessedFeedback();
      expect(unprocessed.find(f => f.id === feedbackId)).toBeUndefined();
    });

    it('should calculate feedback stats correctly', () => {
      // Ajouter plusieurs feedbacks
      persistence.saveFeedback({
        responseId: 'resp-3',
        sessionId: 'session-1',
        rating: 'positive',
        userMessage: 'Q1',
        assistantResponse: 'R1',
      });
      persistence.saveFeedback({
        responseId: 'resp-4',
        sessionId: 'session-1',
        rating: 'positive',
        userMessage: 'Q2',
        assistantResponse: 'R2',
      });
      persistence.saveFeedback({
        responseId: 'resp-5',
        sessionId: 'session-1',
        rating: 'negative',
        userMessage: 'Q3',
        assistantResponse: 'R3',
      });

      const stats = persistence.getFeedbackStats();

      expect(stats.total).toBe(3);
      expect(stats.positive).toBe(2);
      expect(stats.negative).toBe(1);
      expect(stats.satisfactionRate).toBeCloseTo(0.67, 1);
    });
  });
});

describe('EmbeddingsService', () => {
  let embeddings: EmbeddingsService;

  beforeEach(() => {
    embeddings = new EmbeddingsService();
  });

  describe('Cosine Similarity', () => {
    it('should calculate correct similarity for identical vectors', () => {
      const vec = [1, 0, 0, 1];
      const similarity = embeddings.cosineSimilarity(vec, vec);
      expect(similarity).toBeCloseTo(1.0, 5);
    });

    it('should calculate correct similarity for orthogonal vectors', () => {
      const vec1 = [1, 0, 0, 0];
      const vec2 = [0, 1, 0, 0];
      const similarity = embeddings.cosineSimilarity(vec1, vec2);
      expect(similarity).toBeCloseTo(0.0, 5);
    });

    it('should calculate correct similarity for opposite vectors', () => {
      const vec1 = [1, 0, 0, 0];
      const vec2 = [-1, 0, 0, 0];
      const similarity = embeddings.cosineSimilarity(vec1, vec2);
      expect(similarity).toBeCloseTo(-1.0, 5);
    });

    it('should handle different dimension vectors gracefully', () => {
      const vec1 = [1, 0, 0];
      const vec2 = [1, 0, 0, 0];
      const similarity = embeddings.cosineSimilarity(vec1, vec2);
      expect(similarity).toBe(0); // Should return 0 for mismatched dimensions
    });
  });

  describe('Find Most Similar', () => {
    it('should find the most similar vectors', () => {
      const query = [1, 0, 0, 0];
      const candidates = [
        { id: 'a', embedding: [0.9, 0.1, 0, 0] },   // Most similar
        { id: 'b', embedding: [0, 1, 0, 0] },       // Orthogonal
        { id: 'c', embedding: [0.5, 0.5, 0, 0] },   // Partially similar
      ];

      const results = embeddings.findMostSimilar(query, candidates, 3);

      expect(results.length).toBe(3);
      expect(results[0].id).toBe('a'); // Most similar should be first
    });
  });

  describe('Hybrid Search', () => {
    it('should combine vector and BM25 scores', () => {
      const queryEmbedding = [1, 0, 0, 0];
      const candidates = [
        { id: 'a', embedding: [0.9, 0.1, 0, 0] },  // High vector similarity
        { id: 'b', embedding: [0.1, 0.9, 0, 0] },  // Low vector similarity
      ];
      const bm25Results = [
        { id: 'b', score: 0.9 },  // High BM25 score
        { id: 'a', score: 0.1 },  // Low BM25 score
      ];

      const results = embeddings.hybridSearch(
        queryEmbedding,
        candidates,
        bm25Results,
        { vectorWeight: 0.7, bm25Weight: 0.3 }
      );

      expect(results.length).toBe(2);
      // Les deux devraient avoir des scores raisonnables car:
      // - 'a' a un bon score vector mais mauvais BM25
      // - 'b' a un mauvais score vector mais bon BM25
      expect(results[0].vectorScore).toBeGreaterThan(0);
      expect(results[0].bm25Score).toBeGreaterThanOrEqual(0);
    });

    it('should respect minScore threshold', () => {
      const queryEmbedding = [1, 0, 0, 0];
      const candidates = [
        { id: 'a', embedding: [0.1, 0.1, 0.1, 0.1] },  // Low similarity
      ];
      const bm25Results: Array<{ id: string; score: number }> = [];

      const results = embeddings.hybridSearch(
        queryEmbedding,
        candidates,
        bm25Results,
        { minScore: 0.5 }  // High threshold
      );

      // Should filter out low scores
      expect(results.length).toBe(0);
    });
  });

  describe('Cache', () => {
    it('should track cache stats', () => {
      const stats = embeddings.getCacheStats();

      expect(stats).toHaveProperty('size');
      expect(stats).toHaveProperty('memoryMB');
      expect(stats).toHaveProperty('modelLoaded');
    });

    it('should clear cache', () => {
      embeddings.clearCache();
      const stats = embeddings.getCacheStats();
      expect(stats.size).toBe(0);
    });
  });
});
