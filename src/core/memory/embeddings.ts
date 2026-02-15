/**
 * Embeddings Service - Génération et recherche vectorielle
 *
 * Utilise le modèle local all-MiniLM-L6-v2 via @xenova/transformers
 * - 100% gratuit et local (pas d'API)
 * - 384 dimensions
 * - Rapide (~50ms par embedding)
 * - Qualité sémantique excellente
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
import { pipeline, env } from '@xenova/transformers';
import { existsSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configure cache directory for models
const CACHE_DIR = resolve(__dirname, '../../../.cache/models');
if (!existsSync(CACHE_DIR)) {
  mkdirSync(CACHE_DIR, { recursive: true });
}
env.cacheDir = CACHE_DIR;
env.allowLocalModels = true;
env.localModelPath = CACHE_DIR;
// Désactiver le téléchargement si le modèle n'est pas en cache local
env.allowRemoteModels = false;

export interface EmbeddingResult {
  text: string;
  embedding: number[];
  model: string;
}

export class EmbeddingsService {
  private extractor: any = null;
  private cache: Map<string, number[]> = new Map();
  private initPromise: Promise<void> | null = null;
  private isInitializing = false;
  private initFailed = false; // Track if init failed to use fallback

  readonly embeddingDimension = 384; // Dimension de all-MiniLM-L6-v2
  readonly modelName = 'Xenova/all-MiniLM-L6-v2';

  constructor() {
    // Initialisation lazy au premier appel
  }

  /**
   * Initialiser le modèle d'embedding (téléchargement au premier lancement)
   */
  private async initialize(): Promise<void> {
    if (this.extractor) return;

    if (this.initPromise) {
      await this.initPromise;
      return;
    }

    if (this.isInitializing) {
      // Attendre que l'initialisation en cours se termine
      while (this.isInitializing) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      return;
    }

    this.isInitializing = true;

    this.initPromise = (async () => {
      console.log('[Embeddings] 🚀 Chargement du modèle local...');
      console.log(`[Embeddings] Modèle: ${this.modelName}`);

      try {
        // Créer le pipeline d'extraction de features
        this.extractor = await pipeline('feature-extraction', this.modelName, {
          quantized: true, // Utiliser le modèle quantifié (plus petit, plus rapide)
        });

        console.log('[Embeddings] ✅ Modèle chargé avec succès');
        console.log(`[Embeddings] Dimension: ${this.embeddingDimension}`);
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        if (errorMsg.includes('429')) {
          console.warn('[Embeddings] ⚠️ Rate limit HuggingFace (429) - Fallback local activé');
          console.warn('[Embeddings] 💡 Pour télécharger le modèle manuellement:');
          console.warn(`[Embeddings]    git lfs clone https://huggingface.co/Xenova/all-MiniLM-L6-v2 ${CACHE_DIR}/Xenova/all-MiniLM-L6-v2`);
        } else {
          console.error('[Embeddings] ❌ Erreur chargement modèle:', error);
        }
        // Mark as failed so embed() uses fallback instead of retrying
        this.initFailed = true;
        // Don't throw - let embed() use the fallback
      } finally {
        this.isInitializing = false;
      }
    })();

    await this.initPromise;
  }

  /**
   * Générer un embedding pour un texte
   */
  async embed(text: string): Promise<number[]> {
    // Vérifier le cache d'abord
    const cacheKey = this.hashText(text);
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    // If init previously failed, use fallback directly
    if (this.initFailed) {
      return this.localFallbackEmbed(text);
    }

    // S'assurer que le modèle est chargé
    await this.initialize();

    // If extractor failed to load, use fallback
    if (!this.extractor) {
      return this.localFallbackEmbed(text);
    }

    try {
      // Tronquer le texte si trop long (max ~512 tokens)
      const truncatedText = text.length > 2000 ? text.substring(0, 2000) : text;

      // Générer l'embedding
      const output = await this.extractor(truncatedText, {
        pooling: 'mean',      // Mean pooling sur tous les tokens
        normalize: true,       // Normalisation L2
      }) as any;

      // Extraire le vecteur
      const embedding = Array.from(output.data) as number[];

      // Vérifier la dimension
      if (embedding.length !== this.embeddingDimension) {
        console.warn(`[Embeddings] Dimension inattendue: ${embedding.length} (attendu: ${this.embeddingDimension})`);
      }

      // Mettre en cache
      this.cache.set(cacheKey, embedding);

      return embedding;
    } catch (error) {
      console.error('[Embeddings] Erreur génération:', error);
      // Fallback vers embedding local simple
      return this.localFallbackEmbed(text);
    }
  }

  /**
   * Générer des embeddings pour plusieurs textes (batch)
   */
  async embedBatch(texts: string[]): Promise<number[][]> {
    // If init failed, process each text with fallback via embed()
    if (!this.initFailed) {
      await this.initialize();
    }

    const results: number[][] = [];

    // Traiter par batch pour éviter les problèmes de mémoire
    const batchSize = 32;

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);

      for (const text of batch) {
        const embedding = await this.embed(text);
        results.push(embedding);
      }

      // Log de progression pour les gros batches
      if (texts.length > batchSize) {
        console.log(`[Embeddings] Batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(texts.length / batchSize)}`);
      }
    }

    return results;
  }

  /**
   * Calculer la similarité cosinus entre deux vecteurs
   */
  cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      // Si dimensions différentes, retourner 0
      console.warn(`[Embeddings] Dimensions différentes: ${a.length} vs ${b.length}`);
      return 0;
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  }

  /**
   * Rechercher les N vecteurs les plus similaires
   */
  findMostSimilar(
    queryEmbedding: number[],
    candidates: Array<{ id: string; embedding: number[] }>,
    topK: number = 10
  ): Array<{ id: string; similarity: number }> {
    const scored = candidates.map(candidate => ({
      id: candidate.id,
      similarity: this.cosineSimilarity(queryEmbedding, candidate.embedding)
    }));

    return scored
      .filter(s => !isNaN(s.similarity))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }

  /**
   * Recherche sémantique avec seuil de similarité
   */
  semanticSearch(
    queryEmbedding: number[],
    candidates: Array<{ id: string; embedding: number[] }>,
    options: {
      topK?: number;
      minSimilarity?: number;
    } = {}
  ): Array<{ id: string; similarity: number }> {
    const { topK = 10, minSimilarity = 0.3 } = options;

    return this.findMostSimilar(queryEmbedding, candidates, topK * 2)
      .filter(result => result.similarity >= minSimilarity)
      .slice(0, topK);
  }

  /**
   * Fallback : embedding local simple basé sur le hashing
   * Utilisé si le modèle ne charge pas
   */
  private localFallbackEmbed(text: string): number[] {
    // console.warn('[Embeddings] Utilisation du fallback local');

    const vector = new Array(this.embeddingDimension).fill(0);
    const words = text.toLowerCase().split(/\s+/);

    // TF simple avec hashing
    const wordCounts = new Map<string, number>();
    words.forEach(word => {
      if (word.length > 2) {
        wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
      }
    });

    wordCounts.forEach((count, word) => {
      const positions = this.getHashPositions(word, 5);
      const weight = Math.log(1 + count);
      positions.forEach(pos => {
        vector[pos] += weight;
      });
    });

    // Ajouter des n-grammes
    for (let i = 0; i < words.length - 1; i++) {
      const bigram = `${words[i]}_${words[i + 1]}`;
      const pos = this.simpleHash(bigram) % this.embeddingDimension;
      vector[pos] += 0.5;
    }

    return this.normalize(vector);
  }

  /**
   * Hash un texte vers une clé de cache courte
   */
  private hashText(text: string): string {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(36);
  }

  /**
   * Obtenir plusieurs positions de hash pour un mot
   */
  private getHashPositions(word: string, count: number): number[] {
    const positions: number[] = [];
    for (let i = 0; i < count; i++) {
      positions.push(this.simpleHash(`${word}_${i}`) % this.embeddingDimension);
    }
    return positions;
  }

  /**
   * Hash simple pour un string
   */
  private simpleHash(str: string): number {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) + hash) + str.charCodeAt(i);
    }
    return Math.abs(hash);
  }

  /**
   * Normaliser un vecteur (L2)
   */
  private normalize(vector: number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    if (norm === 0) return vector;
    return vector.map(val => val / norm);
  }

  /**
   * Recherche hybride: combine vector similarity (70%) + BM25 scores (30%)
   * Comme OpenClaw mais implémenté localement
   */
  hybridSearch(
    queryEmbedding: number[],
    candidates: Array<{ id: string; embedding: number[] }>,
    bm25Results: Array<{ id: string; score: number }>,
    options: {
      topK?: number;
      vectorWeight?: number;  // Default 0.7
      bm25Weight?: number;    // Default 0.3
      minScore?: number;
    } = {}
  ): Array<{ id: string; score: number; vectorScore: number; bm25Score: number }> {
    const {
      topK = 10,
      vectorWeight = 0.7,
      bm25Weight = 0.3,
      minScore = 0.2
    } = options;

    // 1. Calculer les scores vectoriels
    const vectorScores = new Map<string, number>();
    for (const candidate of candidates) {
      const similarity = this.cosineSimilarity(queryEmbedding, candidate.embedding);
      if (similarity > 0) {
        vectorScores.set(candidate.id, similarity);
      }
    }

    // 2. Créer une map des scores BM25
    const bm25Scores = new Map<string, number>();
    for (const result of bm25Results) {
      bm25Scores.set(result.id, result.score);
    }

    // 3. Fusionner tous les IDs uniques
    const allIds = new Set([
      ...vectorScores.keys(),
      ...bm25Scores.keys()
    ]);

    // 4. Calculer le score hybride pour chaque ID
    const hybridResults: Array<{
      id: string;
      score: number;
      vectorScore: number;
      bm25Score: number;
    }> = [];

    for (const id of allIds) {
      const vScore = vectorScores.get(id) || 0;
      const bScore = bm25Scores.get(id) || 0;

      // Score hybride pondéré
      const hybridScore = (vScore * vectorWeight) + (bScore * bm25Weight);

      if (hybridScore >= minScore) {
        hybridResults.push({
          id,
          score: hybridScore,
          vectorScore: vScore,
          bm25Score: bScore
        });
      }
    }

    // 5. Trier par score hybride décroissant
    hybridResults.sort((a, b) => b.score - a.score);

    return hybridResults.slice(0, topK);
  }

  /**
   * Vider le cache
   */
  clearCache(): void {
    this.cache.clear();
    console.log('[Embeddings] Cache vidé');
  }

  /**
   * Statistiques du cache
   */
  getCacheStats(): { size: number; memoryMB: number; modelLoaded: boolean } {
    let totalSize = 0;
    this.cache.forEach(embedding => {
      totalSize += embedding.length * 4; // 4 bytes per float
    });

    return {
      size: this.cache.size,
      memoryMB: totalSize / (1024 * 1024),
      modelLoaded: this.extractor !== null
    };
  }

  /**
   * Préchauffer le modèle (utile au démarrage)
   */
  async warmup(): Promise<void> {
    console.log('[Embeddings] 🔥 Préchauffage du modèle...');
    try {
      await this.embed('Test de préchauffage du modèle d\'embedding');
      if (this.initFailed) {
        console.log('[Embeddings] ⚠️ Modèle non chargé, fallback local actif');
      } else {
        console.log('[Embeddings] ✅ Modèle prêt');
      }
    } catch (error) {
      console.error('[Embeddings] ⚠️ Warmup échoué, fallback actif:', error);
      this.initFailed = true;
    }
  }
}
