/**
 * API Client pour communiquer avec la Gateway
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:3001';

interface AuthResponse {
  token: string;
  user: { id: string; username: string };
}

interface Message {
  type: 'response' | 'error' | 'connected';
  content?: string;
  sessionId?: string;
  timestamp?: string;
}

class ApiClient {
  private token: string | null = null;
  private ws: WebSocket | null = null;
  private messageHandlers: ((message: Message) => void)[] = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  constructor() {
    this.token = localStorage.getItem('token');
  }

  // ===========================================================================
  // AUTH
  // ===========================================================================

  async register(username: string, password: string): Promise<AuthResponse> {
    const res = await fetch(`${API_URL}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.error || 'Registration failed');
    }

    const data = await res.json();
    this.setToken(data.token);
    return data;
  }

  async login(username: string, password: string): Promise<AuthResponse> {
    const res = await fetch(`${API_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.error || 'Login failed');
    }

    const data = await res.json();
    this.setToken(data.token);
    return data;
  }

  logout(): void {
    this.token = null;
    localStorage.removeItem('token');
    this.disconnectWebSocket();
  }

  isAuthenticated(): boolean {
    return !!this.token;
  }

  private setToken(token: string): void {
    this.token = token;
    localStorage.setItem('token', token);
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = { 'Content-Type': 'application/json' };
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    return headers;
  }

  // ===========================================================================
  // CHAT
  // ===========================================================================

  async sendMessage(message: string): Promise<void> {
    // Préférer WebSocket si connecté
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'chat', content: message }));
      return;
    }

    // Fallback HTTP
    const res = await fetch(`${API_URL}/api/chat`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ message }),
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.error || 'Failed to send message');
    }
  }

  // ===========================================================================
  // MEMORY
  // ===========================================================================

  async storeMemory(type: 'fact' | 'preference' | 'skill', content: string, tags?: string[]): Promise<{ id: string }> {
    const res = await fetch(`${API_URL}/api/memory`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ type, content, tags }),
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.error || 'Failed to store memory');
    }

    return res.json();
  }

  async searchMemory(query: string, limit = 10): Promise<unknown[]> {
    const res = await fetch(`${API_URL}/api/memory/search?q=${encodeURIComponent(query)}&limit=${limit}`, {
      headers: this.getHeaders(),
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.error || 'Search failed');
    }

    const data = await res.json();
    return data.results;
  }

  // ===========================================================================
  // STATS
  // ===========================================================================

  async getStats(): Promise<unknown> {
    const res = await fetch(`${API_URL}/api/stats`, {
      headers: this.getHeaders(),
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.error || 'Failed to get stats');
    }

    return res.json();
  }

  // ===========================================================================
  // GENERIC HTTP METHODS
  // ===========================================================================

  async get<T>(path: string): Promise<T> {
    const res = await fetch(`${API_URL}${path}`, {
      headers: this.getHeaders(),
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.error || 'Request failed');
    }

    return res.json();
  }

  async post<T>(path: string, body?: unknown): Promise<T> {
    const res = await fetch(`${API_URL}${path}`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.error || 'Request failed');
    }

    return res.json();
  }

  // ===========================================================================
  // CORRECTION
  // ===========================================================================

  async correct(originalResponse: string, correction: string, feedback: string): Promise<void> {
    const res = await fetch(`${API_URL}/api/correct`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ originalResponse, correction, feedback }),
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.error || 'Failed to record correction');
    }
  }

  // ===========================================================================
  // WEBSOCKET
  // ===========================================================================

  connectWebSocket(): void {
    if (!this.token) {
      console.error('Cannot connect WebSocket without token');
      return;
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }

    this.ws = new WebSocket(`${WS_URL}?token=${this.token}`);

    this.ws.onopen = () => {
      console.log('[WS] Connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as Message;
        this.messageHandlers.forEach(handler => handler(message));
      } catch (error) {
        console.error('[WS] Failed to parse message:', error);
      }
    };

    this.ws.onclose = () => {
      console.log('[WS] Disconnected');
      this.tryReconnect();
    };

    this.ws.onerror = (error) => {
      console.error('[WS] Error:', error);
    };
  }

  disconnectWebSocket(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private tryReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('[WS] Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      if (this.token) {
        this.connectWebSocket();
      }
    }, delay);
  }

  onMessage(handler: (message: Message) => void): () => void {
    this.messageHandlers.push(handler);
    return () => {
      this.messageHandlers = this.messageHandlers.filter(h => h !== handler);
    };
  }
}

export const api = new ApiClient();
export type { Message, AuthResponse };
