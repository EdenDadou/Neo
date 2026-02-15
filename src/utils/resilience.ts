/**
 * Resilience Utils - Circuit Breaker & Retry pour Neo
 *
 * Règle 2: Neo ne s'éteint jamais
 * - Circuit breaker pour éviter les cascades d'erreurs
 * - Retry exponential avec jitter
 * - Fallback gracieux
 */

export interface RetryOptions {
  maxRetries?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  exponentialBase?: number;
  jitter?: boolean;
  retryableErrors?: string[];
  onRetry?: (error: Error, attempt: number) => void;
}

export interface CircuitBreakerOptions {
  failureThreshold?: number;
  resetTimeoutMs?: number;
  halfOpenRequests?: number;
  onStateChange?: (state: CircuitState) => void;
}

export type CircuitState = 'closed' | 'open' | 'half-open';

/**
 * Retry avec backoff exponentiel et jitter
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const {
    maxRetries = 3,
    baseDelayMs = 1000,
    maxDelayMs = 30000,
    exponentialBase = 2,
    jitter = true,
    retryableErrors = ['ECONNRESET', 'ETIMEDOUT', 'ENOTFOUND', '429', '500', '502', '503', '504'],
    onRetry
  } = options;

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      const errorMessage = lastError.message || '';
      const errorCode = (lastError as NodeJS.ErrnoException).code || '';

      // Vérifier si l'erreur est retryable
      const isRetryable = retryableErrors.some(e =>
        errorMessage.includes(e) || errorCode.includes(e)
      );

      if (!isRetryable || attempt >= maxRetries) {
        throw lastError;
      }

      // Calculer le délai avec backoff exponentiel
      let delay = Math.min(
        baseDelayMs * Math.pow(exponentialBase, attempt),
        maxDelayMs
      );

      // Ajouter du jitter pour éviter les thundering herds
      if (jitter) {
        delay = delay * (0.5 + Math.random());
      }

      if (onRetry) {
        onRetry(lastError, attempt + 1);
      }

      console.log(`[Retry] Tentative ${attempt + 1}/${maxRetries} après ${Math.round(delay)}ms`);
      await sleep(delay);
    }
  }

  throw lastError;
}

/**
 * Circuit Breaker - Évite les cascades d'erreurs
 */
export class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failures = 0;
  private lastFailureTime = 0;
  private halfOpenSuccesses = 0;

  private readonly failureThreshold: number;
  private readonly resetTimeoutMs: number;
  private readonly halfOpenRequests: number;
  private readonly onStateChange?: (state: CircuitState) => void;

  constructor(options: CircuitBreakerOptions = {}) {
    this.failureThreshold = options.failureThreshold ?? 5;
    this.resetTimeoutMs = options.resetTimeoutMs ?? 60000;
    this.halfOpenRequests = options.halfOpenRequests ?? 2;
    this.onStateChange = options.onStateChange;
  }

  async execute<T>(fn: () => Promise<T>, fallback?: () => T | Promise<T>): Promise<T> {
    // Vérifier l'état du circuit
    this.updateState();

    if (this.state === 'open') {
      console.log('[CircuitBreaker] Circuit ouvert - utilisation du fallback');
      if (fallback) {
        return await fallback();
      }
      throw new Error('Circuit breaker is open');
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();

      if (fallback) {
        console.log('[CircuitBreaker] Erreur - utilisation du fallback');
        return await fallback();
      }

      throw error;
    }
  }

  private updateState(): void {
    if (this.state === 'open') {
      const timeSinceFailure = Date.now() - this.lastFailureTime;
      if (timeSinceFailure >= this.resetTimeoutMs) {
        this.setState('half-open');
        this.halfOpenSuccesses = 0;
      }
    }
  }

  private onSuccess(): void {
    if (this.state === 'half-open') {
      this.halfOpenSuccesses++;
      if (this.halfOpenSuccesses >= this.halfOpenRequests) {
        this.setState('closed');
        this.failures = 0;
      }
    } else {
      this.failures = 0;
    }
  }

  private onFailure(): void {
    this.failures++;
    this.lastFailureTime = Date.now();

    if (this.failures >= this.failureThreshold) {
      this.setState('open');
    }
  }

  private setState(newState: CircuitState): void {
    if (this.state !== newState) {
      console.log(`[CircuitBreaker] ${this.state} -> ${newState}`);
      this.state = newState;
      this.onStateChange?.(newState);
    }
  }

  getState(): CircuitState {
    this.updateState();
    return this.state;
  }

  reset(): void {
    this.failures = 0;
    this.setState('closed');
  }
}

/**
 * Timeout wrapper
 */
export async function withTimeout<T>(
  fn: () => Promise<T>,
  timeoutMs: number,
  timeoutError?: string
): Promise<T> {
  return Promise.race([
    fn(),
    new Promise<never>((_, reject) => {
      setTimeout(() => {
        reject(new Error(timeoutError || `Operation timed out after ${timeoutMs}ms`));
      }, timeoutMs);
    })
  ]);
}

/**
 * Combiner retry + circuit breaker + timeout
 */
export async function withResilience<T>(
  fn: () => Promise<T>,
  options: {
    retry?: RetryOptions;
    circuitBreaker?: CircuitBreaker;
    timeoutMs?: number;
    fallback?: () => T | Promise<T>;
  } = {}
): Promise<T> {
  const { retry, circuitBreaker, timeoutMs, fallback } = options;

  // Wrapper avec timeout si spécifié
  const withTimeoutFn = timeoutMs
    ? () => withTimeout(fn, timeoutMs)
    : fn;

  // Wrapper avec retry si spécifié
  const withRetryFn = retry
    ? () => withRetry(withTimeoutFn, retry)
    : withTimeoutFn;

  // Exécuter avec circuit breaker si spécifié
  if (circuitBreaker) {
    return circuitBreaker.execute(withRetryFn, fallback);
  }

  try {
    return await withRetryFn();
  } catch (error) {
    if (fallback) {
      return await fallback();
    }
    throw error;
  }
}

/**
 * Health check périodique
 */
export class HealthChecker {
  private checks: Map<string, () => Promise<boolean>> = new Map();
  private status: Map<string, { healthy: boolean; lastCheck: Date; error?: string }> = new Map();
  private interval: NodeJS.Timeout | null = null;

  register(name: string, check: () => Promise<boolean>): void {
    this.checks.set(name, check);
    this.status.set(name, { healthy: true, lastCheck: new Date() });
  }

  async checkAll(): Promise<Map<string, { healthy: boolean; lastCheck: Date; error?: string }>> {
    for (const [name, check] of this.checks) {
      try {
        const healthy = await check();
        this.status.set(name, { healthy, lastCheck: new Date() });
      } catch (error) {
        this.status.set(name, {
          healthy: false,
          lastCheck: new Date(),
          error: (error as Error).message
        });
      }
    }
    return this.status;
  }

  startPeriodicChecks(intervalMs: number = 60000): void {
    if (this.interval) {
      clearInterval(this.interval);
    }
    this.interval = setInterval(() => {
      this.checkAll().catch(console.error);
    }, intervalMs);
  }

  stopPeriodicChecks(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  getStatus(): Record<string, { healthy: boolean; lastCheck: Date; error?: string }> {
    return Object.fromEntries(this.status);
  }

  isHealthy(): boolean {
    return Array.from(this.status.values()).every(s => s.healthy);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
