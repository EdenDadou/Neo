/**
 * Persistence Layer - SQLite pour stockage local durable
 *
 * Conçu pour durer 10+ ans avec :
 * - Schéma versioned et migrations
 * - Compression des anciennes données
 * - Index optimisés pour la recherche
 */

import Database from 'better-sqlite3';
import { MemoryEntry, Task, Skill, LearningEntry, MemoryType } from '../types';
import { randomUUID } from 'crypto';
import * as path from 'path';
import * as fs from 'fs';

const SCHEMA_VERSION = 4;

export class PersistenceLayer {
  private db: Database.Database;
  private dbPath: string;

  constructor(dataDir: string = './data') {
    // Créer le répertoire si nécessaire
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    this.dbPath = path.join(dataDir, 'memory.db');
    this.db = new Database(this.dbPath);

    // Optimisations SQLite pour performance
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');
    this.db.pragma('cache_size = -64000'); // 64MB cache
    this.db.pragma('temp_store = MEMORY');

    this.initializeSchema();
    console.log(`[Persistence] 💾 Base de données initialisée: ${this.dbPath}`);
  }

  /**
   * Initialiser le schéma de la base de données
   */
  private initializeSchema(): void {
    // Table de version
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
      );
    `);

    const currentVersion = this.db.prepare(
      'SELECT MAX(version) as v FROM schema_version'
    ).get() as { v: number | null };

    if (!currentVersion?.v || currentVersion.v < SCHEMA_VERSION) {
      this.applyMigrations(currentVersion?.v || 0);
    }
  }

  /**
   * Appliquer les migrations
   */
  private applyMigrations(fromVersion: number): void {
    if (fromVersion < 1) {
      console.log('[Persistence] Applying migration v1...');

      this.db.exec(`
        -- Mémoires principales
        CREATE TABLE IF NOT EXISTS memories (
          id TEXT PRIMARY KEY,
          type TEXT NOT NULL,
          content TEXT NOT NULL,
          embedding BLOB,
          source TEXT NOT NULL,
          confidence REAL NOT NULL DEFAULT 1.0,
          tags TEXT NOT NULL DEFAULT '[]',
          related_ids TEXT NOT NULL DEFAULT '[]',
          importance REAL NOT NULL DEFAULT 0.5,
          created_at TEXT NOT NULL,
          last_accessed_at TEXT NOT NULL,
          access_count INTEGER NOT NULL DEFAULT 0,
          expires_at TEXT,
          is_archived INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
        CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
        CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(is_archived);

        -- Tâches
        CREATE TABLE IF NOT EXISTS tasks (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          description TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending',
          priority INTEGER NOT NULL DEFAULT 5,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          attempts TEXT NOT NULL DEFAULT '[]',
          dependencies TEXT NOT NULL DEFAULT '[]',
          required_skills TEXT NOT NULL DEFAULT '[]',
          assigned_agent TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority DESC);

        -- Skills
        CREATE TABLE IF NOT EXISTS skills (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL UNIQUE,
          description TEXT NOT NULL,
          triggers TEXT NOT NULL DEFAULT '[]',
          handler TEXT NOT NULL,
          dependencies TEXT NOT NULL DEFAULT '[]',
          learned_at TEXT NOT NULL,
          success_rate REAL NOT NULL DEFAULT 1.0,
          usage_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);

        -- Learnings (pour la learning loop)
        CREATE TABLE IF NOT EXISTS learnings (
          id TEXT PRIMARY KEY,
          type TEXT NOT NULL,
          context TEXT NOT NULL,
          original_response TEXT NOT NULL,
          correction TEXT,
          feedback TEXT NOT NULL,
          created_at TEXT NOT NULL,
          applied INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_learnings_type ON learnings(type);
        CREATE INDEX IF NOT EXISTS idx_learnings_applied ON learnings(applied);

        -- Synthèses (résumés périodiques)
        CREATE TABLE IF NOT EXISTS syntheses (
          id TEXT PRIMARY KEY,
          period_start TEXT NOT NULL,
          period_end TEXT NOT NULL,
          summary TEXT NOT NULL,
          key_facts TEXT NOT NULL DEFAULT '[]',
          memory_ids TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_syntheses_period ON syntheses(period_start, period_end);

        -- Conversations (pour contexte)
        CREATE TABLE IF NOT EXISTS conversations (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          embedding BLOB,
          timestamp TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_time ON conversations(timestamp DESC);
      `);

      this.db.prepare(
        'INSERT INTO schema_version (version, applied_at) VALUES (?, ?)'
      ).run(1, new Date().toISOString());
    }

    if (fromVersion < 2) {
      console.log('[Persistence] Applying migration v2 (FTS5 for hybrid search)...');

      this.db.exec(`
        -- Table FTS5 pour recherche BM25 (full-text search)
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
          id,
          content,
          tags,
          tokenize='porter unicode61'
        );

        -- Trigger pour synchroniser FTS avec memories
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
          INSERT INTO memories_fts(id, content, tags)
          VALUES (NEW.id, NEW.content, NEW.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
          DELETE FROM memories_fts WHERE id = OLD.id;
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
          DELETE FROM memories_fts WHERE id = OLD.id;
          INSERT INTO memories_fts(id, content, tags)
          VALUES (NEW.id, NEW.content, NEW.tags);
        END;

        -- Table pour le pre-compaction flush (sauvegarde contexte)
        CREATE TABLE IF NOT EXISTS context_snapshots (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          conversation_summary TEXT NOT NULL,
          key_facts TEXT NOT NULL DEFAULT '[]',
          important_memories TEXT NOT NULL DEFAULT '[]',
          user_intent TEXT,
          created_at TEXT NOT NULL,
          token_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_session ON context_snapshots(session_id);
        CREATE INDEX IF NOT EXISTS idx_snapshots_time ON context_snapshots(created_at DESC);

        -- Table pour la personnalité persistante
        CREATE TABLE IF NOT EXISTS personality (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
      `);

      // Peupler la table FTS avec les mémoires existantes
      const existingMemories = this.db.prepare('SELECT id, content, tags FROM memories').all() as Array<{ id: string; content: string; tags: string }>;
      const insertFts = this.db.prepare('INSERT OR IGNORE INTO memories_fts(id, content, tags) VALUES (?, ?, ?)');

      for (const mem of existingMemories) {
        insertFts.run(mem.id, mem.content, mem.tags);
      }

      console.log(`[Persistence] FTS5 peuplé avec ${existingMemories.length} mémoires`);

      this.db.prepare(
        'INSERT INTO schema_version (version, applied_at) VALUES (?, ?)'
      ).run(2, new Date().toISOString());
    }

    if (fromVersion < 3) {
      console.log('[Persistence] Applying migration v3 (Feedback system for self-improvement)...');

      this.db.exec(`
        -- Table pour le feedback utilisateur (Règle 4: Neo s'améliore tout seul)
        CREATE TABLE IF NOT EXISTS feedback (
          id TEXT PRIMARY KEY,
          response_id TEXT NOT NULL,
          session_id TEXT NOT NULL,
          rating TEXT NOT NULL CHECK(rating IN ('positive', 'negative', 'neutral')),
          user_message TEXT NOT NULL,
          assistant_response TEXT NOT NULL,
          user_comment TEXT,
          created_at TEXT NOT NULL,
          processed INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
        CREATE INDEX IF NOT EXISTS idx_feedback_processed ON feedback(processed);
      `);

      this.db.prepare(
        'INSERT INTO schema_version (version, applied_at) VALUES (?, ?)'
      ).run(3, new Date().toISOString());

      console.log('[Persistence] Migration v3 appliquée - système de feedback activé');
    }

    if (fromVersion < 4) {
      console.log('[Persistence] Applying migration v4 (Skills v2 + Long-term memory)...');

      this.db.exec(`
        -- Table skills_v2 pour les skills dynamiques exécutables
        CREATE TABLE IF NOT EXISTS skills_v2 (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          version TEXT NOT NULL DEFAULT '1.0.0',
          description TEXT NOT NULL,
          triggers TEXT NOT NULL DEFAULT '[]',
          required_capabilities TEXT NOT NULL DEFAULT '[]',
          code TEXT NOT NULL,
          input_schema TEXT,
          output_schema TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          created_by TEXT NOT NULL DEFAULT 'system',
          success_rate REAL NOT NULL DEFAULT 1.0,
          usage_count INTEGER NOT NULL DEFAULT 0,
          is_enabled INTEGER NOT NULL DEFAULT 1,
          is_builtin INTEGER NOT NULL DEFAULT 0,
          last_used_at TEXT,
          last_error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_skills_v2_name ON skills_v2(name);
        CREATE INDEX IF NOT EXISTS idx_skills_v2_enabled ON skills_v2(is_enabled);
        CREATE INDEX IF NOT EXISTS idx_skills_v2_created_by ON skills_v2(created_by);

        -- Historique des changements de personnalité (pour traçabilité 10+ ans)
        CREATE TABLE IF NOT EXISTS personality_history (
          id TEXT PRIMARY KEY,
          trait TEXT NOT NULL,
          old_value TEXT,
          new_value TEXT NOT NULL,
          changed_at TEXT NOT NULL,
          reason TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_personality_history_trait ON personality_history(trait);
        CREATE INDEX IF NOT EXISTS idx_personality_history_time ON personality_history(changed_at DESC);

        -- Index pour recherche temporelle rapide (mémoire long-terme)
        CREATE INDEX IF NOT EXISTS idx_memories_year_month ON memories(
          substr(created_at, 1, 7)
        );

        -- Index pour les mémoires importantes (jamais archiver)
        CREATE INDEX IF NOT EXISTS idx_memories_critical ON memories(importance, type)
        WHERE importance >= 0.8 OR type IN ('correction', 'preference');

        -- Table pour les exécutions de skills (audit trail)
        CREATE TABLE IF NOT EXISTS skill_executions (
          id TEXT PRIMARY KEY,
          skill_id TEXT NOT NULL,
          skill_name TEXT NOT NULL,
          input TEXT NOT NULL,
          output TEXT,
          success INTEGER NOT NULL,
          error TEXT,
          execution_time_ms INTEGER NOT NULL,
          capabilities_used TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL,
          FOREIGN KEY (skill_id) REFERENCES skills_v2(id)
        );

        CREATE INDEX IF NOT EXISTS idx_skill_executions_skill ON skill_executions(skill_id);
        CREATE INDEX IF NOT EXISTS idx_skill_executions_time ON skill_executions(created_at DESC);
      `);

      this.db.prepare(
        'INSERT INTO schema_version (version, applied_at) VALUES (?, ?)'
      ).run(4, new Date().toISOString());

      console.log('[Persistence] Migration v4 appliquée - Skills v2 + mémoire long-terme');
    }
  }

  // ===========================================================================
  // MEMORIES CRUD
  // ===========================================================================

  saveMemory(memory: MemoryEntry): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO memories
      (id, type, content, embedding, source, confidence, tags, related_ids,
       importance, created_at, last_accessed_at, access_count, expires_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      memory.id,
      memory.type,
      memory.content,
      memory.embedding ? Buffer.from(new Float32Array(memory.embedding).buffer) : null,
      memory.metadata.source,
      memory.metadata.confidence,
      JSON.stringify(memory.metadata.tags),
      JSON.stringify(memory.metadata.relatedIds),
      memory.importance,
      memory.createdAt.toISOString(),
      memory.lastAccessedAt.toISOString(),
      memory.accessCount,
      memory.metadata.expiresAt?.toISOString() || null
    );
  }

  getMemory(id: string): MemoryEntry | null {
    const row = this.db.prepare('SELECT * FROM memories WHERE id = ?').get(id) as MemoryRow | undefined;
    return row ? this.rowToMemory(row) : null;
  }

  searchMemories(options: {
    type?: MemoryType;
    tags?: string[];
    minImportance?: number;
    limit?: number;
    includeArchived?: boolean;
  }): MemoryEntry[] {
    let query = 'SELECT * FROM memories WHERE 1=1';
    const params: unknown[] = [];

    if (!options.includeArchived) {
      query += ' AND is_archived = 0';
    }

    if (options.type) {
      query += ' AND type = ?';
      params.push(options.type);
    }

    if (options.minImportance !== undefined) {
      query += ' AND importance >= ?';
      params.push(options.minImportance);
    }

    query += ' ORDER BY importance DESC, last_accessed_at DESC';

    if (options.limit) {
      query += ' LIMIT ?';
      params.push(options.limit);
    }

    const rows = this.db.prepare(query).all(...params) as MemoryRow[];
    let memories = rows.map(row => this.rowToMemory(row));

    // Filtrer par tags en mémoire (plus flexible)
    if (options.tags && options.tags.length > 0) {
      memories = memories.filter(m =>
        options.tags!.some(tag => m.metadata.tags.includes(tag))
      );
    }

    return memories;
  }

  updateMemoryAccess(id: string): void {
    this.db.prepare(`
      UPDATE memories
      SET last_accessed_at = ?, access_count = access_count + 1
      WHERE id = ?
    `).run(new Date().toISOString(), id);
  }

  archiveOldMemories(olderThanDays: number, minImportance: number = 0.3): number {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - olderThanDays);

    const result = this.db.prepare(`
      UPDATE memories
      SET is_archived = 1
      WHERE last_accessed_at < ? AND importance < ? AND is_archived = 0
    `).run(cutoffDate.toISOString(), minImportance);

    return result.changes;
  }

  /**
   * Supprimer une mémoire par son ID
   */
  deleteMemory(id: string): boolean {
    const result = this.db.prepare('DELETE FROM memories WHERE id = ?').run(id);
    return result.changes > 0;
  }

  /**
   * Supprimer plusieurs mémoires par leurs IDs
   */
  deleteMemories(ids: string[]): number {
    if (ids.length === 0) return 0;

    const placeholders = ids.map(() => '?').join(',');
    const result = this.db.prepare(`DELETE FROM memories WHERE id IN (${placeholders})`).run(...ids);
    return result.changes;
  }

  /**
   * Mettre à jour une mémoire existante (pour la fusion)
   */
  updateMemory(id: string, updates: {
    content?: string;
    importance?: number;
    confidence?: number;
    tags?: string[];
    relatedIds?: string[];
    embedding?: number[];
  }): boolean {
    const setClauses: string[] = [];
    const values: unknown[] = [];

    if (updates.content !== undefined) {
      setClauses.push('content = ?');
      values.push(updates.content);
    }
    if (updates.importance !== undefined) {
      setClauses.push('importance = ?');
      values.push(updates.importance);
    }
    if (updates.confidence !== undefined) {
      setClauses.push('confidence = ?');
      values.push(updates.confidence);
    }
    if (updates.tags !== undefined) {
      setClauses.push('tags = ?');
      values.push(JSON.stringify(updates.tags));
    }
    if (updates.relatedIds !== undefined) {
      setClauses.push('related_ids = ?');
      values.push(JSON.stringify(updates.relatedIds));
    }
    if (updates.embedding !== undefined) {
      setClauses.push('embedding = ?');
      values.push(Buffer.from(new Float32Array(updates.embedding).buffer));
    }

    if (setClauses.length === 0) return false;

    values.push(id);
    const result = this.db.prepare(
      `UPDATE memories SET ${setClauses.join(', ')} WHERE id = ?`
    ).run(...values);

    return result.changes > 0;
  }

  // ===========================================================================
  // TASKS CRUD
  // ===========================================================================

  saveTask(task: Task): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO tasks
      (id, title, description, status, priority, created_at, updated_at,
       attempts, dependencies, required_skills, assigned_agent)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      task.id,
      task.title,
      task.description,
      task.status,
      task.priority,
      task.createdAt.toISOString(),
      task.updatedAt.toISOString(),
      JSON.stringify(task.attempts),
      JSON.stringify(task.dependencies),
      JSON.stringify(task.requiredSkills),
      task.assignedAgent || null
    );
  }

  getTasks(status?: string): Task[] {
    let query = 'SELECT * FROM tasks';
    const params: unknown[] = [];

    if (status) {
      query += ' WHERE status = ?';
      params.push(status);
    }

    query += ' ORDER BY priority DESC, created_at DESC';

    const rows = this.db.prepare(query).all(...params) as TaskRow[];
    return rows.map(row => this.rowToTask(row));
  }

  // ===========================================================================
  // SKILLS CRUD
  // ===========================================================================

  saveSkill(skill: Skill): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO skills
      (id, name, description, triggers, handler, dependencies, learned_at, success_rate, usage_count)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      skill.id,
      skill.name,
      skill.description,
      JSON.stringify(skill.triggers),
      skill.handler,
      JSON.stringify(skill.dependencies),
      skill.learnedAt.toISOString(),
      skill.successRate,
      skill.usageCount
    );
  }

  getSkills(): Skill[] {
    const rows = this.db.prepare('SELECT * FROM skills ORDER BY usage_count DESC').all() as SkillRow[];
    return rows.map(row => ({
      id: row.id,
      name: row.name,
      description: row.description,
      triggers: JSON.parse(row.triggers),
      handler: row.handler,
      dependencies: JSON.parse(row.dependencies),
      learnedAt: new Date(row.learned_at),
      successRate: row.success_rate,
      usageCount: row.usage_count
    }));
  }

  // ===========================================================================
  // LEARNINGS CRUD
  // ===========================================================================

  saveLearning(learning: LearningEntry): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO learnings
      (id, type, context, original_response, correction, feedback, created_at, applied)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      learning.id,
      learning.type,
      learning.context,
      learning.originalResponse,
      learning.correction || null,
      learning.feedback,
      learning.createdAt.toISOString(),
      learning.applied ? 1 : 0
    );
  }

  getLearnings(options?: { type?: string; unappliedOnly?: boolean }): LearningEntry[] {
    let query = 'SELECT * FROM learnings WHERE 1=1';
    const params: unknown[] = [];

    if (options?.type) {
      query += ' AND type = ?';
      params.push(options.type);
    }

    if (options?.unappliedOnly) {
      query += ' AND applied = 0';
    }

    query += ' ORDER BY created_at DESC';

    const rows = this.db.prepare(query).all(...params) as LearningRow[];
    return rows.map(row => ({
      id: row.id,
      type: row.type as LearningEntry['type'],
      context: row.context,
      originalResponse: row.original_response,
      correction: row.correction || undefined,
      feedback: row.feedback,
      createdAt: new Date(row.created_at),
      applied: row.applied === 1
    }));
  }

  /**
   * Marquer un learning comme appliqué
   */
  markLearningApplied(learningId: string): void {
    this.db.prepare('UPDATE learnings SET applied = 1 WHERE id = ?').run(learningId);
  }

  // ===========================================================================
  // SYNTHESES
  // ===========================================================================

  saveSynthesis(synthesis: {
    periodStart: Date;
    periodEnd: Date;
    summary: string;
    keyFacts: string[];
    memoryIds: string[];
  }): string {
    const id = randomUUID();

    this.db.prepare(`
      INSERT INTO syntheses (id, period_start, period_end, summary, key_facts, memory_ids, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).run(
      id,
      synthesis.periodStart.toISOString(),
      synthesis.periodEnd.toISOString(),
      synthesis.summary,
      JSON.stringify(synthesis.keyFacts),
      JSON.stringify(synthesis.memoryIds),
      new Date().toISOString()
    );

    return id;
  }

  getRecentSyntheses(limit: number = 10): Array<{
    id: string;
    periodStart: Date;
    periodEnd: Date;
    summary: string;
    keyFacts: string[];
  }> {
    const rows = this.db.prepare(`
      SELECT * FROM syntheses ORDER BY period_end DESC LIMIT ?
    `).all(limit) as SynthesisRow[];

    return rows.map(row => ({
      id: row.id,
      periodStart: new Date(row.period_start),
      periodEnd: new Date(row.period_end),
      summary: row.summary,
      keyFacts: JSON.parse(row.key_facts)
    }));
  }

  // ===========================================================================
  // CONVERSATIONS
  // ===========================================================================

  saveConversationTurn(sessionId: string, role: string, content: string, embedding?: number[]): string {
    const id = randomUUID();

    this.db.prepare(`
      INSERT INTO conversations (id, session_id, role, content, embedding, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(
      id,
      sessionId,
      role,
      content,
      embedding ? Buffer.from(new Float32Array(embedding).buffer) : null,
      new Date().toISOString()
    );

    return id;
  }

  getConversationHistory(sessionId: string, limit: number = 50): Array<{
    role: string;
    content: string;
    timestamp: Date;
  }> {
    const rows = this.db.prepare(`
      SELECT role, content, timestamp FROM conversations
      WHERE session_id = ?
      ORDER BY timestamp DESC
      LIMIT ?
    `).all(sessionId, limit) as Array<{ role: string; content: string; timestamp: string }>;

    return rows.reverse().map(row => ({
      role: row.role,
      content: row.content,
      timestamp: new Date(row.timestamp)
    }));
  }

  // ===========================================================================
  // BM25 / FTS5 SEARCH
  // ===========================================================================

  /**
   * Recherche BM25 via FTS5 (recherche par mots-clés)
   * Retourne les IDs triés par score BM25
   */
  searchBM25(query: string, limit: number = 20): Array<{ id: string; score: number }> {
    try {
      // Nettoyer la requête pour FTS5
      const cleanQuery = query
        .replace(/[^\w\s\u00C0-\u017F]/g, ' ')  // Garder lettres, chiffres, accents
        .split(/\s+/)
        .filter(w => w.length > 1)
        .map(w => `"${w}"*`)  // Préfixe matching
        .join(' OR ');

      if (!cleanQuery) return [];

      const rows = this.db.prepare(`
        SELECT id, bm25(memories_fts, 1.2, 0.75) as score
        FROM memories_fts
        WHERE memories_fts MATCH ?
        ORDER BY score
        LIMIT ?
      `).all(cleanQuery, limit) as Array<{ id: string; score: number }>;

      // BM25 retourne des scores négatifs (plus négatif = meilleur)
      // On inverse pour avoir des scores positifs normalisés
      const maxScore = Math.abs(Math.min(...rows.map(r => r.score), -0.001));
      return rows.map(r => ({
        id: r.id,
        score: Math.abs(r.score) / maxScore  // Normaliser entre 0 et 1
      }));
    } catch (error) {
      console.error('[Persistence] BM25 search error:', error);
      return [];
    }
  }

  /**
   * Récupérer toutes les mémoires avec embeddings pour recherche vectorielle
   */
  getMemoriesWithEmbeddings(options?: {
    type?: MemoryType;
    limit?: number;
    includeArchived?: boolean;
  }): Array<{ id: string; embedding: number[]; content: string; type: string; importance: number }> {
    let query = 'SELECT id, embedding, content, type, importance FROM memories WHERE embedding IS NOT NULL';
    const params: unknown[] = [];

    if (!options?.includeArchived) {
      query += ' AND is_archived = 0';
    }

    if (options?.type) {
      query += ' AND type = ?';
      params.push(options.type);
    }

    query += ' ORDER BY importance DESC';

    if (options?.limit) {
      query += ' LIMIT ?';
      params.push(options.limit);
    }

    const rows = this.db.prepare(query).all(...params) as Array<{
      id: string;
      embedding: Buffer;
      content: string;
      type: string;
      importance: number;
    }>;

    return rows.map(row => ({
      id: row.id,
      embedding: Array.from(new Float32Array(row.embedding.buffer)),
      content: row.content,
      type: row.type,
      importance: row.importance
    }));
  }

  // ===========================================================================
  // CONTEXT SNAPSHOTS (Pre-compaction flush)
  // ===========================================================================

  /**
   * Sauvegarder un snapshot du contexte avant compaction
   */
  saveContextSnapshot(snapshot: {
    sessionId: string;
    conversationSummary: string;
    keyFacts: string[];
    importantMemories: string[];
    userIntent?: string;
    tokenCount: number;
  }): string {
    const id = randomUUID();

    this.db.prepare(`
      INSERT INTO context_snapshots
      (id, session_id, conversation_summary, key_facts, important_memories, user_intent, created_at, token_count)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      id,
      snapshot.sessionId,
      snapshot.conversationSummary,
      JSON.stringify(snapshot.keyFacts),
      JSON.stringify(snapshot.importantMemories),
      snapshot.userIntent || null,
      new Date().toISOString(),
      snapshot.tokenCount
    );

    console.log(`[Persistence] 💾 Context snapshot saved (${snapshot.tokenCount} tokens)`);
    return id;
  }

  /**
   * Récupérer les derniers snapshots d'une session
   */
  getContextSnapshots(sessionId: string, limit: number = 5): Array<{
    id: string;
    conversationSummary: string;
    keyFacts: string[];
    importantMemories: string[];
    userIntent?: string;
    createdAt: Date;
    tokenCount: number;
  }> {
    const rows = this.db.prepare(`
      SELECT * FROM context_snapshots
      WHERE session_id = ?
      ORDER BY created_at DESC, rowid DESC
      LIMIT ?
    `).all(sessionId, limit) as Array<{
      id: string;
      session_id: string;
      conversation_summary: string;
      key_facts: string;
      important_memories: string;
      user_intent: string | null;
      created_at: string;
      token_count: number;
    }>;

    return rows.map(row => ({
      id: row.id,
      conversationSummary: row.conversation_summary,
      keyFacts: JSON.parse(row.key_facts),
      importantMemories: JSON.parse(row.important_memories),
      userIntent: row.user_intent || undefined,
      createdAt: new Date(row.created_at),
      tokenCount: row.token_count
    }));
  }

  /**
   * Récupérer le dernier snapshot (pour restauration après compaction)
   */
  getLastContextSnapshot(sessionId: string): {
    conversationSummary: string;
    keyFacts: string[];
    importantMemories: string[];
    userIntent?: string;
  } | null {
    const snapshots = this.getContextSnapshots(sessionId, 1);
    return snapshots.length > 0 ? snapshots[0] : null;
  }

  // ===========================================================================
  // PERSONALITY (Persistante)
  // ===========================================================================

  /**
   * Sauvegarder/mettre à jour une valeur de personnalité
   */
  setPersonality(key: string, value: string): void {
    this.db.prepare(`
      INSERT OR REPLACE INTO personality (key, value, updated_at)
      VALUES (?, ?, ?)
    `).run(key, value, new Date().toISOString());
  }

  /**
   * Récupérer une valeur de personnalité
   */
  getPersonality(key: string): string | null {
    const row = this.db.prepare('SELECT value FROM personality WHERE key = ?').get(key) as { value: string } | undefined;
    return row?.value || null;
  }

  /**
   * Récupérer toutes les valeurs de personnalité
   */
  getAllPersonality(): Record<string, string> {
    const rows = this.db.prepare('SELECT key, value FROM personality').all() as Array<{ key: string; value: string }>;
    const result: Record<string, string> = {};
    rows.forEach(row => {
      result[row.key] = row.value;
    });
    return result;
  }

  /**
   * Supprimer une valeur de personnalité
   */
  deletePersonality(key: string): boolean {
    const result = this.db.prepare('DELETE FROM personality WHERE key = ?').run(key);
    return result.changes > 0;
  }

  // ===========================================================================
  // FEEDBACK (Règle 4: Neo s'améliore tout seul)
  // ===========================================================================

  /**
   * Enregistrer un feedback utilisateur
   */
  saveFeedback(feedback: {
    responseId: string;
    sessionId: string;
    rating: 'positive' | 'negative' | 'neutral';
    userMessage: string;
    assistantResponse: string;
    userComment?: string;
  }): string {
    const id = randomUUID();

    this.db.prepare(`
      INSERT INTO feedback
      (id, response_id, session_id, rating, user_message, assistant_response, user_comment, created_at, processed)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
    `).run(
      id,
      feedback.responseId,
      feedback.sessionId,
      feedback.rating,
      feedback.userMessage,
      feedback.assistantResponse,
      feedback.userComment || null,
      new Date().toISOString()
    );

    console.log(`[Persistence] 📝 Feedback enregistré: ${feedback.rating}`);
    return id;
  }

  /**
   * Récupérer les feedbacks non traités pour l'apprentissage
   */
  getUnprocessedFeedback(limit: number = 50): Array<{
    id: string;
    responseId: string;
    sessionId: string;
    rating: 'positive' | 'negative' | 'neutral';
    userMessage: string;
    assistantResponse: string;
    userComment?: string;
    createdAt: Date;
  }> {
    const rows = this.db.prepare(`
      SELECT * FROM feedback
      WHERE processed = 0
      ORDER BY created_at ASC
      LIMIT ?
    `).all(limit) as Array<{
      id: string;
      response_id: string;
      session_id: string;
      rating: 'positive' | 'negative' | 'neutral';
      user_message: string;
      assistant_response: string;
      user_comment: string | null;
      created_at: string;
    }>;

    return rows.map(row => ({
      id: row.id,
      responseId: row.response_id,
      sessionId: row.session_id,
      rating: row.rating,
      userMessage: row.user_message,
      assistantResponse: row.assistant_response,
      userComment: row.user_comment || undefined,
      createdAt: new Date(row.created_at)
    }));
  }

  /**
   * Marquer un feedback comme traité
   */
  markFeedbackProcessed(feedbackId: string): void {
    this.db.prepare('UPDATE feedback SET processed = 1 WHERE id = ?').run(feedbackId);
  }

  /**
   * Statistiques de feedback
   */
  getFeedbackStats(): {
    total: number;
    positive: number;
    negative: number;
    neutral: number;
    satisfactionRate: number;
    unprocessed: number;
  } {
    const stats = this.db.prepare(`
      SELECT
        COUNT(*) as total,
        SUM(CASE WHEN rating = 'positive' THEN 1 ELSE 0 END) as positive,
        SUM(CASE WHEN rating = 'negative' THEN 1 ELSE 0 END) as negative,
        SUM(CASE WHEN rating = 'neutral' THEN 1 ELSE 0 END) as neutral,
        SUM(CASE WHEN processed = 0 THEN 1 ELSE 0 END) as unprocessed
      FROM feedback
    `).get() as {
      total: number;
      positive: number;
      negative: number;
      neutral: number;
      unprocessed: number;
    };

    const satisfactionRate = stats.total > 0
      ? (stats.positive / (stats.positive + stats.negative)) || 0
      : 0;

    return {
      ...stats,
      satisfactionRate: Math.round(satisfactionRate * 100) / 100
    };
  }

  // ===========================================================================
  // SKILLS V2 CRUD
  // ===========================================================================

  /**
   * Sauvegarder un skill v2
   */
  saveSkillV2(skill: SkillV2Row): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO skills_v2
      (id, name, version, description, triggers, required_capabilities, code,
       input_schema, output_schema, created_at, updated_at, created_by,
       success_rate, usage_count, is_enabled, is_builtin, last_used_at, last_error)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      skill.id,
      skill.name,
      skill.version,
      skill.description,
      skill.triggers,
      skill.required_capabilities,
      skill.code,
      skill.input_schema || null,
      skill.output_schema || null,
      skill.created_at,
      skill.updated_at,
      skill.created_by,
      skill.success_rate,
      skill.usage_count,
      skill.is_enabled ? 1 : 0,
      skill.is_builtin ? 1 : 0,
      skill.last_used_at || null,
      skill.last_error || null
    );
  }

  /**
   * Récupérer tous les skills v2
   */
  getSkillsV2(options?: { enabledOnly?: boolean; createdBy?: string }): SkillV2Row[] {
    let query = 'SELECT * FROM skills_v2 WHERE 1=1';
    const params: unknown[] = [];

    if (options?.enabledOnly) {
      query += ' AND is_enabled = 1';
    }

    if (options?.createdBy) {
      query += ' AND created_by = ?';
      params.push(options.createdBy);
    }

    query += ' ORDER BY usage_count DESC, created_at DESC';

    return this.db.prepare(query).all(...params) as SkillV2Row[];
  }

  /**
   * Récupérer un skill v2 par ID
   */
  getSkillV2(id: string): SkillV2Row | null {
    return this.db.prepare('SELECT * FROM skills_v2 WHERE id = ?').get(id) as SkillV2Row | null;
  }

  /**
   * Supprimer un skill v2
   */
  deleteSkillV2(id: string): boolean {
    const result = this.db.prepare('DELETE FROM skills_v2 WHERE id = ? AND is_builtin = 0').run(id);
    return result.changes > 0;
  }

  /**
   * Logger une exécution de skill
   */
  logSkillExecution(execution: {
    id: string;
    skillId: string;
    skillName: string;
    input: string;
    output?: string;
    success: boolean;
    error?: string;
    executionTimeMs: number;
    capabilitiesUsed: string[];
  }): void {
    this.db.prepare(`
      INSERT INTO skill_executions
      (id, skill_id, skill_name, input, output, success, error, execution_time_ms, capabilities_used, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      execution.id,
      execution.skillId,
      execution.skillName,
      execution.input,
      execution.output || null,
      execution.success ? 1 : 0,
      execution.error || null,
      execution.executionTimeMs,
      JSON.stringify(execution.capabilitiesUsed),
      new Date().toISOString()
    );
  }

  /**
   * Récupérer les dernières exécutions d'un skill
   */
  getSkillExecutions(skillId: string, limit = 20): Array<{
    id: string;
    input: string;
    output?: string;
    success: boolean;
    error?: string;
    executionTimeMs: number;
    createdAt: Date;
  }> {
    const rows = this.db.prepare(`
      SELECT * FROM skill_executions
      WHERE skill_id = ?
      ORDER BY created_at DESC
      LIMIT ?
    `).all(skillId, limit) as Array<{
      id: string;
      input: string;
      output: string | null;
      success: number;
      error: string | null;
      execution_time_ms: number;
      created_at: string;
    }>;

    return rows.map(row => ({
      id: row.id,
      input: row.input,
      output: row.output || undefined,
      success: row.success === 1,
      error: row.error || undefined,
      executionTimeMs: row.execution_time_ms,
      createdAt: new Date(row.created_at),
    }));
  }

  /**
   * Analyser les patterns d'échecs pour apprentissage (Règle 4: Neo s'améliore)
   * Regroupe les erreurs similaires et identifie les tendances
   */
  getFailurePatterns(skillId?: string, limit = 50): Array<{
    errorPattern: string;
    count: number;
    skillIds: string[];
    skillNames: string[];
    firstSeen: Date;
    lastSeen: Date;
    exampleInputs: string[];
  }> {
    let query = `
      SELECT
        error,
        COUNT(*) as count,
        GROUP_CONCAT(DISTINCT skill_id) as skill_ids,
        GROUP_CONCAT(DISTINCT skill_name) as skill_names,
        MIN(created_at) as first_seen,
        MAX(created_at) as last_seen,
        GROUP_CONCAT(input) as inputs
      FROM skill_executions
      WHERE success = 0 AND error IS NOT NULL
    `;
    const params: unknown[] = [];

    if (skillId) {
      query += ' AND skill_id = ?';
      params.push(skillId);
    }

    query += ' GROUP BY error ORDER BY count DESC LIMIT ?';
    params.push(limit);

    const rows = this.db.prepare(query).all(...params) as Array<{
      error: string;
      count: number;
      skill_ids: string;
      skill_names: string;
      first_seen: string;
      last_seen: string;
      inputs: string;
    }>;

    return rows.map(row => ({
      errorPattern: row.error,
      count: row.count,
      skillIds: row.skill_ids.split(','),
      skillNames: [...new Set(row.skill_names.split(','))],
      firstSeen: new Date(row.first_seen),
      lastSeen: new Date(row.last_seen),
      exampleInputs: row.inputs.split(',').slice(0, 3), // 3 premiers exemples
    }));
  }

  /**
   * Obtenir les insights d'apprentissage depuis les exécutions
   * Pour que Neo puisse améliorer ses skills
   */
  getLearningInsights(skillId: string): {
    successRate: number;
    totalExecutions: number;
    avgExecutionTime: number;
    recentTrend: 'improving' | 'declining' | 'stable';
    commonErrors: Array<{ error: string; count: number; suggestion?: string }>;
    peakUsageHours: number[];
  } {
    // Stats globales
    const stats = this.db.prepare(`
      SELECT
        COUNT(*) as total,
        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
        AVG(execution_time_ms) as avg_time
      FROM skill_executions
      WHERE skill_id = ?
    `).get(skillId) as { total: number; successes: number; avg_time: number };

    // Tendance récente (30 dernières vs 30 précédentes)
    const recent = this.db.prepare(`
      SELECT
        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as rate
      FROM (
        SELECT success FROM skill_executions
        WHERE skill_id = ?
        ORDER BY created_at DESC
        LIMIT 30
      )
    `).get(skillId) as { rate: number | null };

    const older = this.db.prepare(`
      SELECT
        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as rate
      FROM (
        SELECT success FROM skill_executions
        WHERE skill_id = ?
        ORDER BY created_at DESC
        LIMIT 30 OFFSET 30
      )
    `).get(skillId) as { rate: number | null };

    let recentTrend: 'improving' | 'declining' | 'stable' = 'stable';
    if (recent.rate !== null && older.rate !== null) {
      const diff = recent.rate - older.rate;
      if (diff > 0.1) recentTrend = 'improving';
      else if (diff < -0.1) recentTrend = 'declining';
    }

    // Erreurs communes avec suggestions
    const errors = this.db.prepare(`
      SELECT error, COUNT(*) as count
      FROM skill_executions
      WHERE skill_id = ? AND success = 0 AND error IS NOT NULL
      GROUP BY error
      ORDER BY count DESC
      LIMIT 5
    `).all(skillId) as Array<{ error: string; count: number }>;

    const commonErrors = errors.map(e => ({
      error: e.error,
      count: e.count,
      suggestion: this.suggestFixForError(e.error),
    }));

    // Heures de pic d'utilisation
    const hourStats = this.db.prepare(`
      SELECT
        CAST(strftime('%H', created_at) AS INTEGER) as hour,
        COUNT(*) as count
      FROM skill_executions
      WHERE skill_id = ?
      GROUP BY hour
      ORDER BY count DESC
      LIMIT 3
    `).all(skillId) as Array<{ hour: number; count: number }>;

    return {
      successRate: stats.total > 0 ? stats.successes / stats.total : 1,
      totalExecutions: stats.total,
      avgExecutionTime: stats.avg_time || 0,
      recentTrend,
      commonErrors,
      peakUsageHours: hourStats.map(h => h.hour),
    };
  }

  /**
   * Suggérer une correction basée sur l'erreur (heuristiques)
   */
  private suggestFixForError(error: string): string | undefined {
    const lowerError = error.toLowerCase();

    if (lowerError.includes('timeout')) {
      return 'Augmenter le timeout ou optimiser les opérations lentes';
    }
    if (lowerError.includes('rate limit')) {
      return 'Implémenter un backoff exponentiel ou réduire la fréquence';
    }
    if (lowerError.includes('not found') || lowerError.includes('404')) {
      return 'Vérifier l\'URL ou ajouter une validation avant l\'appel';
    }
    if (lowerError.includes('permission') || lowerError.includes('denied')) {
      return 'Vérifier les capabilities requises ou les permissions';
    }
    if (lowerError.includes('json') || lowerError.includes('parse')) {
      return 'Ajouter une validation du format de réponse';
    }
    if (lowerError.includes('memory') || lowerError.includes('limit')) {
      return 'Réduire la taille des données traitées ou paginer';
    }

    return undefined;
  }

  /**
   * Enregistrer une tentative d'amélioration de skill
   * Pour traçabilité de l'évolution autonome
   */
  logSkillImprovement(improvement: {
    skillId: string;
    skillName: string;
    previousVersion: string;
    newVersion: string;
    reason: string;
    changes: string;
    triggeredBy: 'auto' | 'feedback' | 'error_pattern';
  }): string {
    const id = randomUUID();

    // Stocker comme learning
    this.saveLearning({
      id,
      type: 'skill_improvement',
      context: JSON.stringify({
        skillId: improvement.skillId,
        skillName: improvement.skillName,
        previousVersion: improvement.previousVersion,
        newVersion: improvement.newVersion,
        triggeredBy: improvement.triggeredBy,
      }),
      originalResponse: improvement.changes,
      feedback: improvement.reason,
      createdAt: new Date(),
      applied: false,
    });

    console.log(`[Persistence] 📈 Amélioration skill logged: ${improvement.skillName} ${improvement.previousVersion} → ${improvement.newVersion}`);
    return id;
  }

  /**
   * Obtenir l'historique des améliorations d'un skill
   */
  getSkillImprovementHistory(skillId: string): Array<{
    id: string;
    previousVersion: string;
    newVersion: string;
    reason: string;
    changes: string;
    triggeredBy: string;
    createdAt: Date;
  }> {
    const learnings = this.getLearnings({ type: 'skill_improvement' });

    return learnings
      .filter(l => {
        try {
          const context = JSON.parse(l.context);
          return context.skillId === skillId;
        } catch {
          return false;
        }
      })
      .map(l => {
        const context = JSON.parse(l.context);
        return {
          id: l.id,
          previousVersion: context.previousVersion,
          newVersion: context.newVersion,
          reason: l.feedback,
          changes: l.originalResponse,
          triggeredBy: context.triggeredBy,
          createdAt: l.createdAt,
        };
      });
  }

  // ===========================================================================
  // PERSONALITY HISTORY (Traçabilité 10+ ans)
  // ===========================================================================

  /**
   * Logger un changement de personnalité
   */
  logPersonalityChange(trait: string, oldValue: string | null, newValue: string, reason?: string): void {
    const id = randomUUID();
    this.db.prepare(`
      INSERT INTO personality_history (id, trait, old_value, new_value, changed_at, reason)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(id, trait, oldValue, newValue, new Date().toISOString(), reason || null);
  }

  /**
   * Récupérer l'historique des changements de personnalité
   */
  getPersonalityHistory(trait?: string, limit = 50): Array<{
    id: string;
    trait: string;
    oldValue: string | null;
    newValue: string;
    changedAt: Date;
    reason?: string;
  }> {
    let query = 'SELECT * FROM personality_history';
    const params: unknown[] = [];

    if (trait) {
      query += ' WHERE trait = ?';
      params.push(trait);
    }

    query += ' ORDER BY changed_at DESC LIMIT ?';
    params.push(limit);

    const rows = this.db.prepare(query).all(...params) as Array<{
      id: string;
      trait: string;
      old_value: string | null;
      new_value: string;
      changed_at: string;
      reason: string | null;
    }>;

    return rows.map(row => ({
      id: row.id,
      trait: row.trait,
      oldValue: row.old_value,
      newValue: row.new_value,
      changedAt: new Date(row.changed_at),
      reason: row.reason || undefined,
    }));
  }

  // ===========================================================================
  // LONG-TERM MEMORY MANAGEMENT
  // ===========================================================================

  /**
   * Compresser les vieilles conversations (remplacer par synthèses)
   */
  compressOldConversations(olderThanYears: number): number {
    const cutoffDate = new Date();
    cutoffDate.setFullYear(cutoffDate.getFullYear() - olderThanYears);

    // Compter les conversations à compresser
    const count = this.db.prepare(`
      SELECT COUNT(*) as c FROM conversations
      WHERE timestamp < ?
    `).get(cutoffDate.toISOString()) as { c: number };

    if (count.c === 0) return 0;

    // Supprimer les vieilles conversations (les synthèses les ont déjà résumées)
    const result = this.db.prepare(`
      DELETE FROM conversations
      WHERE timestamp < ?
    `).run(cutoffDate.toISOString());

    console.log(`[Persistence] Compressé ${result.changes} vieilles conversations (> ${olderThanYears} ans)`);
    return result.changes;
  }

  /**
   * Récupérer les mémoires par période (pour archivage/analyse)
   */
  getMemoriesByPeriod(startDate: Date, endDate: Date, options?: {
    type?: MemoryType;
    minImportance?: number;
    limit?: number;
  }): MemoryEntry[] {
    let query = 'SELECT * FROM memories WHERE created_at >= ? AND created_at < ?';
    const params: unknown[] = [startDate.toISOString(), endDate.toISOString()];

    if (options?.type) {
      query += ' AND type = ?';
      params.push(options.type);
    }

    if (options?.minImportance !== undefined) {
      query += ' AND importance >= ?';
      params.push(options.minImportance);
    }

    query += ' ORDER BY importance DESC, created_at DESC';

    if (options?.limit) {
      query += ' LIMIT ?';
      params.push(options.limit);
    }

    const rows = this.db.prepare(query).all(...params) as MemoryRow[];
    return rows.map(row => this.rowToMemory(row));
  }

  /**
   * Statistiques de la mémoire par année (pour dashboard long-terme)
   */
  getMemoryStatsByYear(): Array<{
    year: string;
    totalMemories: number;
    avgImportance: number;
    topTypes: Record<string, number>;
  }> {
    const rows = this.db.prepare(`
      SELECT
        substr(created_at, 1, 4) as year,
        COUNT(*) as total,
        AVG(importance) as avg_importance,
        type
      FROM memories
      GROUP BY year, type
      ORDER BY year DESC
    `).all() as Array<{
      year: string;
      total: number;
      avg_importance: number;
      type: string;
    }>;

    // Grouper par année
    const byYear = new Map<string, {
      totalMemories: number;
      avgImportance: number;
      topTypes: Record<string, number>;
    }>();

    for (const row of rows) {
      if (!byYear.has(row.year)) {
        byYear.set(row.year, {
          totalMemories: 0,
          avgImportance: 0,
          topTypes: {},
        });
      }
      const yearData = byYear.get(row.year)!;
      yearData.totalMemories += row.total;
      yearData.avgImportance = (yearData.avgImportance + row.avg_importance) / 2;
      yearData.topTypes[row.type] = row.total;
    }

    return Array.from(byYear.entries()).map(([year, data]) => ({
      year,
      ...data,
    }));
  }

  // ===========================================================================
  // STATS
  // ===========================================================================

  // ===========================================================================
  // BACKUP & INTEGRITY (Règle 1: Neo n'oublie jamais)
  // ===========================================================================

  /**
   * Créer un backup de la base de données
   * Utilise checkpoint WAL + copie pour backup cohérent
   */
  backup(backupPath?: string): string {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupDir = path.join(path.dirname(this.dbPath), 'backups');

    if (!fs.existsSync(backupDir)) {
      fs.mkdirSync(backupDir, { recursive: true });
    }

    const targetPath = backupPath || path.join(backupDir, `memory-${timestamp}.db`);

    // Checkpoint le WAL pour s'assurer que tout est écrit
    this.db.pragma('wal_checkpoint(TRUNCATE)');

    // Copier le fichier principal
    fs.copyFileSync(this.dbPath, targetPath);

    console.log(`[Persistence] 💾 Backup créé: ${targetPath}`);
    return targetPath;
  }

  /**
   * Lister les backups disponibles
   */
  listBackups(): Array<{ path: string; date: Date; sizeMB: number }> {
    const backupDir = path.join(path.dirname(this.dbPath), 'backups');

    if (!fs.existsSync(backupDir)) {
      return [];
    }

    const files = fs.readdirSync(backupDir)
      .filter(f => f.endsWith('.db'))
      .map(f => {
        const fullPath = path.join(backupDir, f);
        const stats = fs.statSync(fullPath);
        return {
          path: fullPath,
          date: stats.mtime,
          sizeMB: stats.size / (1024 * 1024)
        };
      })
      .sort((a, b) => b.date.getTime() - a.date.getTime());

    return files;
  }

  /**
   * Restaurer depuis un backup
   */
  restoreFromBackup(backupPath: string): void {
    if (!fs.existsSync(backupPath)) {
      throw new Error(`Backup not found: ${backupPath}`);
    }

    // Fermer la connexion actuelle
    this.db.close();

    // Créer un backup de sécurité avant restauration
    const safetyBackup = `${this.dbPath}.pre-restore`;
    fs.copyFileSync(this.dbPath, safetyBackup);

    // Restaurer
    fs.copyFileSync(backupPath, this.dbPath);

    // Réouvrir
    this.db = new Database(this.dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');

    console.log(`[Persistence] ✅ Restauré depuis: ${backupPath}`);
  }

  /**
   * Vérifier l'intégrité de la base de données
   */
  checkIntegrity(): { ok: boolean; errors: string[] } {
    const result = this.db.pragma('integrity_check') as Array<{ integrity_check: string }>;
    const errors = result
      .map(r => r.integrity_check)
      .filter(r => r !== 'ok');

    return {
      ok: errors.length === 0,
      errors
    };
  }

  /**
   * Optimiser la base de données (VACUUM + ANALYZE)
   */
  optimize(): void {
    console.log('[Persistence] 🔧 Optimisation en cours...');
    this.db.exec('VACUUM');
    this.db.exec('ANALYZE');
    console.log('[Persistence] ✅ Base optimisée');
  }

  /**
   * Nettoyer les vieux backups (garder les N derniers)
   */
  cleanOldBackups(keepCount: number = 5): number {
    const backups = this.listBackups();
    const toDelete = backups.slice(keepCount);

    for (const backup of toDelete) {
      fs.unlinkSync(backup.path);
    }

    if (toDelete.length > 0) {
      console.log(`[Persistence] 🗑️ ${toDelete.length} vieux backup(s) supprimé(s)`);
    }

    return toDelete.length;
  }

  getStats(): {
    totalMemories: number;
    memoriesByType: Record<string, number>;
    totalTasks: number;
    totalSkills: number;
    totalLearnings: number;
    totalFeedback: number;
    satisfactionRate: number;
    dbSizeMB: number;
  } {
    const totalMemories = (this.db.prepare('SELECT COUNT(*) as c FROM memories').get() as { c: number }).c;
    const totalTasks = (this.db.prepare('SELECT COUNT(*) as c FROM tasks').get() as { c: number }).c;
    const totalSkills = (this.db.prepare('SELECT COUNT(*) as c FROM skills').get() as { c: number }).c;
    const totalLearnings = (this.db.prepare('SELECT COUNT(*) as c FROM learnings').get() as { c: number }).c;

    const typeRows = this.db.prepare(
      'SELECT type, COUNT(*) as c FROM memories GROUP BY type'
    ).all() as Array<{ type: string; c: number }>;

    const memoriesByType: Record<string, number> = {};
    typeRows.forEach(row => {
      memoriesByType[row.type] = row.c;
    });

    // Feedback stats
    const feedbackStats = this.getFeedbackStats();

    let dbSizeMB = 0;
    try {
      const stats = fs.statSync(this.dbPath);
      dbSizeMB = stats.size / (1024 * 1024);
    } catch {
      // DB peut ne pas exister encore
    }

    return {
      totalMemories,
      memoriesByType,
      totalTasks,
      totalSkills,
      totalLearnings,
      totalFeedback: feedbackStats.total,
      satisfactionRate: feedbackStats.satisfactionRate,
      dbSizeMB
    };
  }

  // ===========================================================================
  // HELPERS
  // ===========================================================================

  private rowToMemory(row: MemoryRow): MemoryEntry {
    return {
      id: row.id,
      type: row.type as MemoryType,
      content: row.content,
      embedding: row.embedding ? Array.from(new Float32Array(row.embedding.buffer)) : undefined,
      metadata: {
        source: row.source,
        confidence: row.confidence,
        tags: JSON.parse(row.tags),
        relatedIds: JSON.parse(row.related_ids),
        expiresAt: row.expires_at ? new Date(row.expires_at) : undefined
      },
      createdAt: new Date(row.created_at),
      lastAccessedAt: new Date(row.last_accessed_at),
      accessCount: row.access_count,
      importance: row.importance
    };
  }

  private rowToTask(row: TaskRow): Task {
    return {
      id: row.id,
      title: row.title,
      description: row.description,
      status: row.status as Task['status'],
      priority: row.priority,
      createdAt: new Date(row.created_at),
      updatedAt: new Date(row.updated_at),
      attempts: JSON.parse(row.attempts),
      dependencies: JSON.parse(row.dependencies),
      requiredSkills: JSON.parse(row.required_skills),
      assignedAgent: row.assigned_agent || undefined
    };
  }

  close(): void {
    this.db.close();
  }
}

// Types pour les rows SQLite
interface MemoryRow {
  id: string;
  type: string;
  content: string;
  embedding: Buffer | null;
  source: string;
  confidence: number;
  tags: string;
  related_ids: string;
  importance: number;
  created_at: string;
  last_accessed_at: string;
  access_count: number;
  expires_at: string | null;
  is_archived: number;
}

interface TaskRow {
  id: string;
  title: string;
  description: string;
  status: string;
  priority: number;
  created_at: string;
  updated_at: string;
  attempts: string;
  dependencies: string;
  required_skills: string;
  assigned_agent: string | null;
}

interface SkillRow {
  id: string;
  name: string;
  description: string;
  triggers: string;
  handler: string;
  dependencies: string;
  learned_at: string;
  success_rate: number;
  usage_count: number;
}

interface LearningRow {
  id: string;
  type: string;
  context: string;
  original_response: string;
  correction: string | null;
  feedback: string;
  created_at: string;
  applied: number;
}

interface SynthesisRow {
  id: string;
  period_start: string;
  period_end: string;
  summary: string;
  key_facts: string;
  memory_ids: string;
  created_at: string;
}

// Type pour skills v2 (skills dynamiques exécutables)
export interface SkillV2Row {
  id: string;
  name: string;
  version: string;
  description: string;
  triggers: string;              // JSON array
  required_capabilities: string; // JSON array
  code: string;
  input_schema: string | null;   // JSON schema
  output_schema: string | null;  // JSON schema
  created_at: string;
  updated_at: string;
  created_by: 'user' | 'neo' | 'system';
  success_rate: number;
  usage_count: number;
  is_enabled: boolean;
  is_builtin: boolean;
  last_used_at: string | null;
  last_error: string | null;
}
