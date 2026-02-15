/**
 * Web Search Service - Recherche Internet native pour Neo
 *
 * Providers supportés:
 * - DuckDuckGo (gratuit, sans API key)
 * - SearXNG (self-hosted, pour VPS)
 * - Brave Search API (optionnel, avec key)
 * - Serper (optionnel, avec key)
 *
 * Conçu pour être facilement déployable sur VPS
 */

export interface SearchResult {
  title: string;
  url: string;
  snippet: string;
  source: string;
}

export interface SearchOptions {
  maxResults?: number;
  language?: string;
  region?: string;
  safeSearch?: boolean;
}

export interface WebSearchConfig {
  provider: 'duckduckgo' | 'searxng' | 'brave' | 'serper';
  apiKey?: string;
  searxngUrl?: string;  // URL de votre instance SearXNG
  timeout?: number;
}

export class WebSearchService {
  private config: WebSearchConfig;
  private readonly defaultTimeout = 10000;

  constructor(config?: Partial<WebSearchConfig>) {
    this.config = {
      provider: config?.provider || this.detectBestProvider(config),
      apiKey: config?.apiKey || process.env.SEARCH_API_KEY,
      searxngUrl: config?.searxngUrl || process.env.SEARXNG_URL,
      timeout: config?.timeout || this.defaultTimeout,
    };

    console.log(`[WebSearch] Provider: ${this.config.provider}`);
  }

  /**
   * Détecter le meilleur provider disponible
   */
  private detectBestProvider(config?: Partial<WebSearchConfig>): WebSearchConfig['provider'] {
    // Priorité: SearXNG (self-hosted) > Brave > Serper > DuckDuckGo
    if (config?.searxngUrl || process.env.SEARXNG_URL) {
      return 'searxng';
    }
    if (process.env.BRAVE_API_KEY) {
      return 'brave';
    }
    if (process.env.SERPER_API_KEY) {
      return 'serper';
    }
    return 'duckduckgo';  // Fallback gratuit
  }

  /**
   * Effectuer une recherche web
   */
  async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    const { maxResults = 5, language = 'fr' } = options;

    console.log(`[WebSearch] Recherche: "${query}" via ${this.config.provider}`);

    try {
      switch (this.config.provider) {
        case 'duckduckgo':
          return await this.searchDuckDuckGo(query, maxResults, language);
        case 'searxng':
          return await this.searchSearXNG(query, maxResults, language);
        case 'brave':
          return await this.searchBrave(query, maxResults, language);
        case 'serper':
          return await this.searchSerper(query, maxResults, language);
        default:
          return await this.searchDuckDuckGo(query, maxResults, language);
      }
    } catch (error) {
      console.error(`[WebSearch] Erreur ${this.config.provider}:`, error);

      // Fallback vers DuckDuckGo si autre provider échoue
      if (this.config.provider !== 'duckduckgo') {
        console.log('[WebSearch] Fallback vers DuckDuckGo...');
        return await this.searchDuckDuckGo(query, maxResults, language);
      }

      return [];
    }
  }

  /**
   * DuckDuckGo - Gratuit, sans API key
   * Utilise l'API HTML de DuckDuckGo Lite
   */
  private async searchDuckDuckGo(
    query: string,
    maxResults: number,
    _language: string
  ): Promise<SearchResult[]> {
    const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; NeoBot/1.0)',
        'Accept': 'text/html',
      },
      signal: AbortSignal.timeout(this.config.timeout!),
    });

    if (!response.ok) {
      throw new Error(`DuckDuckGo error: ${response.status}`);
    }

    const html = await response.text();
    return this.parseDuckDuckGoHTML(html, maxResults);
  }

  /**
   * Parser le HTML de DuckDuckGo
   */
  private parseDuckDuckGoHTML(html: string, maxResults: number): SearchResult[] {
    const results: SearchResult[] = [];

    // Regex pour extraire les résultats
    const resultRegex = /<a class="result__a" href="([^"]+)"[^>]*>([^<]+)<\/a>[\s\S]*?<a class="result__snippet"[^>]*>([^<]*(?:<[^>]*>[^<]*)*)<\/a>/g;

    let match;
    while ((match = resultRegex.exec(html)) !== null && results.length < maxResults) {
      const url = match[1];
      const title = match[2].trim();
      let snippet = match[3]
        .replace(/<[^>]*>/g, '')  // Enlever les tags HTML
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .trim();

      // Ignorer les liens internes DuckDuckGo
      if (url.startsWith('//duckduckgo.com') || url.startsWith('/')) {
        continue;
      }

      results.push({
        title,
        url: url.startsWith('//') ? `https:${url}` : url,
        snippet,
        source: 'duckduckgo',
      });
    }

    // Fallback: parser différemment si pas de résultats
    if (results.length === 0) {
      const altRegex = /<a[^>]*href="(https?:\/\/[^"]+)"[^>]*class="[^"]*result[^"]*"[^>]*>([^<]+)/g;
      while ((match = altRegex.exec(html)) !== null && results.length < maxResults) {
        results.push({
          title: match[2].trim(),
          url: match[1],
          snippet: '',
          source: 'duckduckgo',
        });
      }
    }

    return results;
  }

  /**
   * SearXNG - Self-hosted, parfait pour VPS
   * Agrège plusieurs moteurs de recherche
   */
  private async searchSearXNG(
    query: string,
    maxResults: number,
    language: string
  ): Promise<SearchResult[]> {
    if (!this.config.searxngUrl) {
      throw new Error('SearXNG URL not configured');
    }

    const url = new URL('/search', this.config.searxngUrl);
    url.searchParams.set('q', query);
    url.searchParams.set('format', 'json');
    url.searchParams.set('language', language);
    url.searchParams.set('safesearch', '1');

    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      signal: AbortSignal.timeout(this.config.timeout!),
    });

    if (!response.ok) {
      throw new Error(`SearXNG error: ${response.status}`);
    }

    const data = await response.json() as {
      results: Array<{
        title: string;
        url: string;
        content?: string;
        engine: string;
      }>;
    };

    return data.results.slice(0, maxResults).map(r => ({
      title: r.title,
      url: r.url,
      snippet: r.content || '',
      source: `searxng:${r.engine}`,
    }));
  }

  /**
   * Brave Search API - Bon rapport qualité/prix
   */
  private async searchBrave(
    query: string,
    maxResults: number,
    _language: string
  ): Promise<SearchResult[]> {
    const apiKey = this.config.apiKey || process.env.BRAVE_API_KEY;
    if (!apiKey) {
      throw new Error('Brave API key not configured');
    }

    const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${maxResults}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'X-Subscription-Token': apiKey,
      },
      signal: AbortSignal.timeout(this.config.timeout!),
    });

    if (!response.ok) {
      throw new Error(`Brave error: ${response.status}`);
    }

    const data = await response.json() as {
      web?: {
        results: Array<{
          title: string;
          url: string;
          description: string;
        }>;
      };
    };

    return (data.web?.results || []).map(r => ({
      title: r.title,
      url: r.url,
      snippet: r.description,
      source: 'brave',
    }));
  }

  /**
   * Serper API - Google Search API
   */
  private async searchSerper(
    query: string,
    maxResults: number,
    language: string
  ): Promise<SearchResult[]> {
    const apiKey = this.config.apiKey || process.env.SERPER_API_KEY;
    if (!apiKey) {
      throw new Error('Serper API key not configured');
    }

    const response = await fetch('https://google.serper.dev/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-KEY': apiKey,
      },
      body: JSON.stringify({
        q: query,
        num: maxResults,
        hl: language,
      }),
      signal: AbortSignal.timeout(this.config.timeout!),
    });

    if (!response.ok) {
      throw new Error(`Serper error: ${response.status}`);
    }

    const data = await response.json() as {
      organic: Array<{
        title: string;
        link: string;
        snippet: string;
      }>;
    };

    return (data.organic || []).map(r => ({
      title: r.title,
      url: r.link,
      snippet: r.snippet,
      source: 'serper',
    }));
  }

  /**
   * Récupérer le contenu d'une page web
   */
  async fetchPageContent(url: string): Promise<string> {
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; NeoBot/1.0)',
          'Accept': 'text/html,text/plain',
        },
        signal: AbortSignal.timeout(this.config.timeout!),
      });

      if (!response.ok) {
        throw new Error(`Fetch error: ${response.status}`);
      }

      const html = await response.text();

      // Extraire le texte principal (simpliste mais fonctionnel)
      return this.extractTextFromHTML(html);
    } catch (error) {
      console.error(`[WebSearch] Erreur fetch ${url}:`, error);
      return '';
    }
  }

  /**
   * Extraire le texte d'un HTML (version simple)
   */
  private extractTextFromHTML(html: string): string {
    // Enlever les scripts et styles
    let text = html
      .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
      .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
      .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, '')
      .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, '')
      .replace(/<header[^>]*>[\s\S]*?<\/header>/gi, '');

    // Convertir les tags en espaces
    text = text
      .replace(/<br\s*\/?>/gi, '\n')
      .replace(/<\/p>/gi, '\n\n')
      .replace(/<\/div>/gi, '\n')
      .replace(/<\/li>/gi, '\n')
      .replace(/<[^>]+>/g, ' ');

    // Nettoyer
    text = text
      .replace(/&nbsp;/g, ' ')
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&#\d+;/g, '')
      .replace(/\s+/g, ' ')
      .trim();

    // Limiter la taille
    return text.slice(0, 10000);
  }

  /**
   * Recherche et récupération du contenu en une fois
   */
  async searchAndFetch(
    query: string,
    options: SearchOptions = {}
  ): Promise<Array<SearchResult & { content?: string }>> {
    const results = await this.search(query, options);

    // Fetch le contenu des premiers résultats
    const enriched = await Promise.all(
      results.slice(0, 3).map(async (result) => {
        const content = await this.fetchPageContent(result.url);
        return { ...result, content };
      })
    );

    return [
      ...enriched,
      ...results.slice(3).map(r => ({ ...r, content: undefined })),
    ];
  }

  /**
   * Vérifier si le service est opérationnel
   */
  async healthCheck(): Promise<{
    status: 'ok' | 'error';
    provider: string;
    message?: string;
  }> {
    try {
      const results = await this.search('test', { maxResults: 1 });
      return {
        status: results.length > 0 ? 'ok' : 'error',
        provider: this.config.provider,
        message: results.length > 0 ? undefined : 'No results returned',
      };
    } catch (error) {
      return {
        status: 'error',
        provider: this.config.provider,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Obtenir les infos de configuration (sans secrets)
   */
  getConfig(): {
    provider: string;
    hasApiKey: boolean;
    hasSearxngUrl: boolean;
    timeout: number;
  } {
    return {
      provider: this.config.provider,
      hasApiKey: !!this.config.apiKey,
      hasSearxngUrl: !!this.config.searxngUrl,
      timeout: this.config.timeout!,
    };
  }
}

// Instance singleton pour utilisation globale
let webSearchInstance: WebSearchService | null = null;

export function getWebSearchService(config?: Partial<WebSearchConfig>): WebSearchService {
  if (!webSearchInstance) {
    webSearchInstance = new WebSearchService(config);
  }
  return webSearchInstance;
}
