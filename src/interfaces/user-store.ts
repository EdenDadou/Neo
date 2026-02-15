/**
 * User Store - Persistance SQLite des utilisateurs
 *
 * Stocke les utilisateurs dans SQLite pour persistance
 * entre les redémarrages du serveur.
 */

import Database from 'better-sqlite3';
import * as path from 'path';
import * as fs from 'fs';

export interface User {
  id: string;
  username: string;
  passwordHash: string;
  createdAt: Date;
  lastLoginAt: Date | null;
}

export class UserStore {
  private db: Database.Database;

  constructor(dataDir: string = './data') {
    // Créer le répertoire si nécessaire
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    const dbPath = path.join(dataDir, 'users.db');
    this.db = new Database(dbPath);

    // Optimisations SQLite
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');

    this.initializeSchema();
    console.log(`[UserStore] 👤 Base utilisateurs initialisée: ${dbPath}`);
  }

  private initializeSchema(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_login_at TEXT
      );

      CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    `);
  }

  /**
   * Créer un nouvel utilisateur
   */
  create(id: string, username: string, passwordHash: string): User {
    const now = new Date();

    this.db.prepare(`
      INSERT INTO users (id, username, password_hash, created_at, last_login_at)
      VALUES (?, ?, ?, ?, NULL)
    `).run(id, username, passwordHash, now.toISOString());

    return {
      id,
      username,
      passwordHash,
      createdAt: now,
      lastLoginAt: null,
    };
  }

  /**
   * Trouver un utilisateur par username
   */
  findByUsername(username: string): User | null {
    const row = this.db.prepare(
      'SELECT * FROM users WHERE username = ?'
    ).get(username) as {
      id: string;
      username: string;
      password_hash: string;
      created_at: string;
      last_login_at: string | null;
    } | undefined;

    if (!row) return null;

    return {
      id: row.id,
      username: row.username,
      passwordHash: row.password_hash,
      createdAt: new Date(row.created_at),
      lastLoginAt: row.last_login_at ? new Date(row.last_login_at) : null,
    };
  }

  /**
   * Trouver un utilisateur par ID
   */
  findById(id: string): User | null {
    const row = this.db.prepare(
      'SELECT * FROM users WHERE id = ?'
    ).get(id) as {
      id: string;
      username: string;
      password_hash: string;
      created_at: string;
      last_login_at: string | null;
    } | undefined;

    if (!row) return null;

    return {
      id: row.id,
      username: row.username,
      passwordHash: row.password_hash,
      createdAt: new Date(row.created_at),
      lastLoginAt: row.last_login_at ? new Date(row.last_login_at) : null,
    };
  }

  /**
   * Mettre à jour la date de dernière connexion
   */
  updateLastLogin(id: string): void {
    this.db.prepare(
      'UPDATE users SET last_login_at = ? WHERE id = ?'
    ).run(new Date().toISOString(), id);
  }

  /**
   * Vérifier si un username existe
   */
  usernameExists(username: string): boolean {
    const row = this.db.prepare(
      'SELECT 1 FROM users WHERE username = ?'
    ).get(username);

    return !!row;
  }

  /**
   * Compter le nombre d'utilisateurs
   */
  count(): number {
    const row = this.db.prepare('SELECT COUNT(*) as count FROM users').get() as { count: number };
    return row.count;
  }

  /**
   * Fermer la connexion
   */
  close(): void {
    this.db.close();
  }
}
