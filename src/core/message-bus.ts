/**
 * Message Bus - Communication inter-agents
 * Système de pub/sub pour la communication entre Vox, Memory et Brain
 */

import { EventEmitter } from 'events';
import { AgentMessage, AgentRole, MessageType } from './types';
import { randomUUID } from 'crypto';

export class MessageBus extends EventEmitter {
  private static instance: MessageBus;
  private messageLog: AgentMessage[] = [];
  private readonly maxLogSize = 10000;

  private constructor() {
    super();
    this.setMaxListeners(50); // Support plusieurs agents
  }

  static getInstance(): MessageBus {
    if (!MessageBus.instance) {
      MessageBus.instance = new MessageBus();
    }
    return MessageBus.instance;
  }

  /**
   * Envoyer un message à un agent spécifique ou en broadcast
   */
  send(message: Omit<AgentMessage, 'id' | 'timestamp'>): string {
    const fullMessage: AgentMessage = {
      ...message,
      id: randomUUID(),
      timestamp: new Date(),
    };

    // Log le message
    this.logMessage(fullMessage);

    // Émettre vers la destination
    if (message.to === 'broadcast') {
      this.emit('broadcast', fullMessage);
    } else {
      this.emit(`message:${message.to}`, fullMessage);
    }

    // Émettre aussi un event global pour le monitoring
    this.emit('message', fullMessage);

    return fullMessage.id;
  }

  /**
   * Envoyer et attendre une réponse
   */
  async sendAndWait<T>(
    message: Omit<AgentMessage, 'id' | 'timestamp'>,
    timeoutMs = 30000
  ): Promise<T> {
    const correlationId = randomUUID();
    const messageWithCorrelation = {
      ...message,
      correlationId,
    };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.removeListener(`response:${correlationId}`, handler);
        reject(new Error(`Timeout waiting for response from ${message.to}`));
      }, timeoutMs);

      const handler = (response: AgentMessage) => {
        clearTimeout(timeout);
        resolve(response.payload as T);
      };

      this.once(`response:${correlationId}`, handler);
      this.send(messageWithCorrelation);
    });
  }

  /**
   * Répondre à un message
   */
  reply(originalMessage: AgentMessage, payload: unknown): void {
    if (originalMessage.correlationId) {
      this.emit(`response:${originalMessage.correlationId}`, {
        id: randomUUID(),
        from: originalMessage.to as AgentRole,
        to: originalMessage.from,
        type: originalMessage.type,
        payload,
        timestamp: new Date(),
        correlationId: originalMessage.correlationId,
      });
    }
  }

  /**
   * S'abonner aux messages destinés à un agent
   */
  subscribe(role: AgentRole, handler: (message: AgentMessage) => void): void {
    this.on(`message:${role}`, handler);
    this.on('broadcast', handler);
  }

  /**
   * Se désabonner
   */
  unsubscribe(role: AgentRole, handler: (message: AgentMessage) => void): void {
    this.off(`message:${role}`, handler);
    this.off('broadcast', handler);
  }

  /**
   * Obtenir l'historique des messages
   */
  getHistory(filter?: {
    from?: AgentRole;
    to?: AgentRole;
    type?: MessageType;
    since?: Date;
  }): AgentMessage[] {
    let messages = [...this.messageLog];

    if (filter) {
      if (filter.from) {
        messages = messages.filter((m) => m.from === filter.from);
      }
      if (filter.to) {
        messages = messages.filter((m) => m.to === filter.to);
      }
      if (filter.type) {
        messages = messages.filter((m) => m.type === filter.type);
      }
      if (filter.since) {
        messages = messages.filter((m) => m.timestamp >= filter.since!);
      }
    }

    return messages;
  }

  private logMessage(message: AgentMessage): void {
    this.messageLog.push(message);

    // Rotation du log si trop grand
    if (this.messageLog.length > this.maxLogSize) {
      this.messageLog = this.messageLog.slice(-this.maxLogSize / 2);
    }
  }
}

// Export singleton
export const messageBus = MessageBus.getInstance();
