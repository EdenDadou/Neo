/**
 * Gateway HTTP/WebSocket sécurisée
 *
 * Sécurité :
 * - JWT pour authentification
 * - Rate limiting
 * - CORS configuré
 * - Helmet pour headers de sécurité
 * - Validation des entrées
 */

import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import { RateLimiterMemory } from 'rate-limiter-flexible';
import { WebSocketServer, WebSocket } from 'ws';
import { createServer } from 'http';
import { Core } from '../core';
import { randomUUID } from 'crypto';
import { UserStore } from './user-store';
import { logger } from '../utils/logger';

// Configuration
const JWT_SECRET = process.env.JWT_SECRET || 'change-this-in-production-' + randomUUID();
const JWT_EXPIRY = '24h';
const PORT = parseInt(process.env.PORT || '3001');

// Types
interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    username: string;
  };
}

interface WSClient {
  ws: WebSocket;
  userId: string;
  sessionId: string;
  isAlive: boolean;
}

// Rate limiters
const rateLimiter = new RateLimiterMemory({
  points: 100, // Requêtes
  duration: 60, // Par minute
});

const authRateLimiter = new RateLimiterMemory({
  points: 5, // Tentatives de login
  duration: 60 * 15, // Par 15 minutes
});

export class Gateway {
  private app: express.Application;
  private server: ReturnType<typeof createServer>;
  private wss: WebSocketServer;
  private core: Core;
  private clients: Map<string, WSClient> = new Map();
  private userStore: UserStore;

  constructor(core: Core) {
    this.core = core;
    this.app = express();
    this.server = createServer(this.app);
    this.wss = new WebSocketServer({ server: this.server });
    this.userStore = new UserStore('./data');

    this.setupMiddleware();
    this.setupRoutes();
    this.setupWebSocket();
  }

  // ===========================================================================
  // MIDDLEWARE
  // ===========================================================================

  private setupMiddleware(): void {
    // Sécurité headers
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          scriptSrc: ["'self'", "'unsafe-inline'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          connectSrc: ["'self'", "ws:", "wss:"],
        },
      },
    }));

    // CORS
    this.app.use(cors({
      origin: process.env.CORS_ORIGIN || 'http://localhost:5173',
      credentials: true,
    }));

    // Body parser
    this.app.use(express.json({ limit: '10mb' }));

    // Rate limiting global
    this.app.use(async (req: Request, res: Response, next: NextFunction) => {
      try {
        const ip = req.ip || req.socket.remoteAddress || 'unknown';
        await rateLimiter.consume(ip);
        next();
      } catch {
        res.status(429).json({ error: 'Too many requests' });
      }
    });

    // Logging
    this.app.use((req: Request, _res: Response, next: NextFunction) => {
      console.log(`[Gateway] ${req.method} ${req.path}`);
      next();
    });
  }

  // ===========================================================================
  // AUTH MIDDLEWARE
  // ===========================================================================

  private authMiddleware = (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({ error: 'Token required' });
      return;
    }

    const token = authHeader.substring(7);

    try {
      const decoded = jwt.verify(token, JWT_SECRET) as { id: string; username: string };
      req.user = decoded;
      next();
    } catch {
      res.status(401).json({ error: 'Invalid token' });
    }
  };

  // ===========================================================================
  // ROUTES
  // ===========================================================================

  private setupRoutes(): void {
    // Health check
    this.app.get('/health', (_req: Request, res: Response) => {
      res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        version: '0.1.0',
      });
    });

    // ==== AUTH ====

    // Register
    this.app.post('/api/auth/register', async (req: Request, res: Response) => {
      try {
        const { username, password } = req.body;

        if (!username || !password || password.length < 8) {
          res.status(400).json({ error: 'Invalid username or password (min 8 chars)' });
          return;
        }

        // Vérifier si l'utilisateur existe déjà (SQLite)
        if (this.userStore.usernameExists(username)) {
          res.status(409).json({ error: 'Username already exists' });
          return;
        }

        const id = randomUUID();
        const passwordHash = await bcrypt.hash(password, 12);

        // Créer l'utilisateur dans SQLite
        this.userStore.create(id, username, passwordHash);

        const token = jwt.sign({ id, username }, JWT_SECRET, { expiresIn: JWT_EXPIRY });

        console.log(`[Gateway] 👤 Nouvel utilisateur créé: ${username}`);
        res.status(201).json({ token, user: { id, username } });
      } catch (error) {
        console.error('[Gateway] Register error:', error);
        res.status(500).json({ error: 'Registration failed' });
      }
    });

    // Login
    this.app.post('/api/auth/login', async (req: Request, res: Response) => {
      try {
        const ip = req.ip || 'unknown';

        // Rate limit auth
        try {
          await authRateLimiter.consume(ip);
        } catch {
          res.status(429).json({ error: 'Too many login attempts' });
          return;
        }

        const { username, password } = req.body;

        // Chercher l'utilisateur dans SQLite
        const user = this.userStore.findByUsername(username);

        if (!user || !(await bcrypt.compare(password, user.passwordHash))) {
          res.status(401).json({ error: 'Invalid credentials' });
          return;
        }

        // Mettre à jour la date de dernière connexion
        this.userStore.updateLastLogin(user.id);

        const token = jwt.sign(
          { id: user.id, username: user.username },
          JWT_SECRET,
          { expiresIn: JWT_EXPIRY }
        );

        console.log(`[Gateway] 🔓 Connexion: ${username}`);
        res.json({ token, user: { id: user.id, username: user.username } });
      } catch (error) {
        console.error('[Gateway] Login error:', error);
        res.status(500).json({ error: 'Login failed' });
      }
    });

    // ==== CHAT API ====

    // Send message
    this.app.post('/api/chat', this.authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { message } = req.body;

        if (!message || typeof message !== 'string') {
          res.status(400).json({ error: 'Message required' });
          return;
        }

        // Sanitize input
        const sanitizedMessage = message.slice(0, 10000).trim();

        // Process via core
        await this.core.chat(sanitizedMessage);

        res.json({ status: 'processing', message: 'Message received' });
      } catch (error) {
        console.error('[Gateway] Chat error:', error);
        res.status(500).json({ error: 'Failed to process message' });
      }
    });

    // ==== MEMORY API ====

    // Store memory
    this.app.post('/api/memory', this.authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { type, content, tags } = req.body;

        if (!type || !content) {
          res.status(400).json({ error: 'Type and content required' });
          return;
        }

        const validTypes = ['fact', 'preference', 'skill'];
        if (!validTypes.includes(type)) {
          res.status(400).json({ error: `Invalid type. Must be one of: ${validTypes.join(', ')}` });
          return;
        }

        const id = await this.core.remember(type, content, tags);
        res.json({ id, success: true });
      } catch (error) {
        console.error('[Gateway] Memory store error:', error);
        res.status(500).json({ error: 'Failed to store memory' });
      }
    });

    // Search memory
    this.app.get('/api/memory/search', this.authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
      try {
        const query = req.query.q as string;
        const limit = parseInt(req.query.limit as string) || 10;

        if (!query) {
          res.status(400).json({ error: 'Query parameter q required' });
          return;
        }

        const results = await this.core.recall(query, Math.min(limit, 50));
        res.json({ results });
      } catch (error) {
        console.error('[Gateway] Memory search error:', error);
        res.status(500).json({ error: 'Search failed' });
      }
    });

    // ==== STATS ====

    this.app.get('/api/stats', this.authMiddleware, (_req: AuthenticatedRequest, res: Response) => {
      try {
        const metrics = this.core.getMetrics();
        res.json(metrics);
      } catch (error) {
        console.error('[Gateway] Stats error:', error);
        res.status(500).json({ error: 'Failed to get stats' });
      }
    });

    // ==== DASHBOARD ====

    this.app.get('/api/dashboard', this.authMiddleware, (_req: AuthenticatedRequest, res: Response) => {
      try {
        const dashboardData = this.core.getDashboardData();
        res.json(dashboardData);
      } catch (error) {
        console.error('[Gateway] Dashboard error:', error);
        res.status(500).json({ error: 'Failed to get dashboard data' });
      }
    });

    // ==== SYSTEM ACTIONS ====

    this.app.post('/api/system/restart', this.authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
      try {
        console.log('[Gateway] System restart requested');
        res.json({ success: true, message: 'Restart initiated' });
        // Schedule restart after response
        setTimeout(async () => {
          await this.core.stop();
          await this.core.start();
        }, 100);
      } catch (error) {
        console.error('[Gateway] Restart error:', error);
        res.status(500).json({ error: 'Failed to restart' });
      }
    });

    this.app.post('/api/memory/backup', this.authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
      try {
        const backupPath = this.core.backupMemory();
        res.json({ success: true, path: backupPath });
      } catch (error) {
        console.error('[Gateway] Backup error:', error);
        res.status(500).json({ error: 'Failed to backup memory' });
      }
    });

    this.app.post('/api/cache/clear', this.authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
      try {
        this.core.clearCache();
        res.json({ success: true });
      } catch (error) {
        console.error('[Gateway] Cache clear error:', error);
        res.status(500).json({ error: 'Failed to clear cache' });
      }
    });

    // ==== CORRECTION / LEARNING ====

    this.app.post('/api/correct', this.authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { originalResponse, correction, feedback } = req.body;

        if (!originalResponse || !correction || !feedback) {
          res.status(400).json({ error: 'originalResponse, correction and feedback required' });
          return;
        }

        await this.core.correct(originalResponse, correction, feedback);
        res.json({ success: true });
      } catch (error) {
        console.error('[Gateway] Correction error:', error);
        res.status(500).json({ error: 'Failed to record correction' });
      }
    });
  }

  // ===========================================================================
  // WEBSOCKET
  // ===========================================================================

  private setupWebSocket(): void {
    // Ping pour détecter les connexions mortes
    const pingInterval = setInterval(() => {
      this.clients.forEach((client, id) => {
        if (!client.isAlive) {
          client.ws.terminate();
          this.clients.delete(id);
          return;
        }
        client.isAlive = false;
        client.ws.ping();
      });
    }, 30000);

    this.wss.on('close', () => {
      clearInterval(pingInterval);
    });

    this.wss.on('connection', (ws: WebSocket, req) => {
      // Extraire le token de l'URL
      const url = new URL(req.url || '', `http://${req.headers.host}`);
      const token = url.searchParams.get('token');

      if (!token) {
        ws.close(4001, 'Token required');
        return;
      }

      try {
        const decoded = jwt.verify(token, JWT_SECRET) as { id: string; username: string };

        const clientId = randomUUID();
        const client: WSClient = {
          ws,
          userId: decoded.id,
          sessionId: clientId,
          isAlive: true,
        };

        this.clients.set(clientId, client);
        console.log(`[Gateway] 🔌 WebSocket client connected: ${clientId} (user: ${decoded.username})`);

        ws.on('pong', () => {
          client.isAlive = true;
        });

        ws.on('message', async (data) => {
          try {
            const message = JSON.parse(data.toString());
            await this.handleWSMessage(client, message);
          } catch {
            ws.send(JSON.stringify({ type: 'error', message: 'Invalid message format' }));
          }
        });

        ws.on('close', () => {
          this.clients.delete(clientId);
          console.log(`[Gateway] 🔌 WebSocket client disconnected: ${clientId}`);
        });

        // Confirmer la connexion
        ws.send(JSON.stringify({
          type: 'connected',
          sessionId: clientId,
          userId: decoded.id,
          username: decoded.username,
        }));
      } catch {
        ws.close(4002, 'Invalid token');
      }
    });

    // Écouter les réponses du Core pour les broadcaster
    this.core.on('response', (message: string) => {
      // Afficher la réponse dans le terminal avec formatage
      logger.response('Neo', message);

      // Broadcaster aux clients WebSocket
      this.broadcastToClients({
        type: 'response',
        content: message,
        timestamp: new Date().toISOString(),
      });
    });

    // Écouter les événements de streaming (typing indicator, etc.)
    this.core.on('typing', () => {
      this.broadcastToClients({
        type: 'typing',
        timestamp: new Date().toISOString(),
      });
    });

    // Écouter les erreurs
    this.core.on('error', (error: { message: string }) => {
      this.broadcastToClients({
        type: 'error',
        message: error.message,
        timestamp: new Date().toISOString(),
      });
    });
  }

  private async handleWSMessage(client: WSClient, message: { type: string; content?: string; requestId?: string }): Promise<void> {
    switch (message.type) {
      case 'chat':
        if (message.content) {
          // Notifier que le bot "tape"
          this.sendToClient(client, { type: 'typing', timestamp: new Date().toISOString() });

          try {
            await this.core.chat(message.content);
          } catch (error) {
            this.sendToClient(client, {
              type: 'error',
              message: error instanceof Error ? error.message : 'Failed to process message',
              requestId: message.requestId,
            });
          }
        }
        break;

      case 'ping':
        this.sendToClient(client, { type: 'pong', timestamp: new Date().toISOString() });
        break;

      case 'memory_search':
        if (message.content) {
          try {
            const results = await this.core.recall(message.content, 10);
            this.sendToClient(client, {
              type: 'memory_results',
              results,
              requestId: message.requestId,
            });
          } catch (error) {
            this.sendToClient(client, {
              type: 'error',
              message: 'Memory search failed',
              requestId: message.requestId,
            });
          }
        }
        break;

      case 'stats':
        try {
          const metrics = this.core.getMetrics();
          this.sendToClient(client, {
            type: 'stats',
            data: metrics,
            requestId: message.requestId,
          });
        } catch {
          this.sendToClient(client, {
            type: 'error',
            message: 'Failed to get stats',
            requestId: message.requestId,
          });
        }
        break;

      default:
        this.sendToClient(client, { type: 'error', message: `Unknown message type: ${message.type}` });
    }
  }

  /**
   * Envoyer un message à un client spécifique
   */
  private sendToClient(client: WSClient, message: unknown): void {
    if (client.ws.readyState === WebSocket.OPEN) {
      client.ws.send(JSON.stringify(message));
    }
  }

  /**
   * Broadcast à tous les clients connectés
   */
  private broadcastToClients(message: unknown): void {
    const data = JSON.stringify(message);
    let sentCount = 0;
    this.clients.forEach((client) => {
      if (client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(data);
        sentCount++;
      }
    });
    if (sentCount > 0) {
      console.log(`[Gateway] ✅ Message sent to ${sentCount} client(s)`);
    }
  }

  // ===========================================================================
  // START/STOP
  // ===========================================================================

  async start(): Promise<void> {
    return new Promise((resolve) => {
      this.server.listen(PORT, () => {
        console.log(`[Gateway] 🚀 Serveur démarré sur http://localhost:${PORT}`);
        console.log(`[Gateway] 🔌 WebSocket disponible sur ws://localhost:${PORT}`);
        resolve();
      });
    });
  }

  async stop(): Promise<void> {
    return new Promise((resolve) => {
      // Fermer toutes les connexions WebSocket
      this.clients.forEach((client) => {
        client.ws.close(1000, 'Server shutting down');
      });
      this.clients.clear();

      // Fermer le UserStore
      this.userStore.close();

      this.server.close(() => {
        console.log('[Gateway] Serveur arrêté');
        resolve();
      });
    });
  }
}
