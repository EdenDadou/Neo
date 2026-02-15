/**
 * Built-in Skill: Web Scraper
 * Scrape et extrait des données depuis une page web
 */

import { SkillCreateInput } from '../types';

export const webScraperSkill: SkillCreateInput = {
  name: 'Web Scraper',
  description: 'Récupère et extrait le contenu d\'une page web',
  triggers: ['scraper', 'extraire', 'récupérer page', 'scraping', 'fetch url'],
  requiredCapabilities: ['web_fetch'],
  code: `
// Skill: Web Scraper v1.0
// Auto-généré par le système

async function execute(input, context) {
  const { webFetch } = context;

  // Récupérer l'URL depuis l'input
  const url = typeof input === 'string' ? input : input.url;

  if (!url) {
    throw new Error('URL requise');
  }

  // Fetch la page
  const response = await webFetch(url);

  // Extraire le titre (basic regex)
  const titleMatch = response.text.match(/<title[^>]*>([^<]+)<\\/title>/i);
  const title = titleMatch ? titleMatch[1].trim() : 'Sans titre';

  // Extraire les meta descriptions
  const metaDescMatch = response.text.match(/<meta[^>]*name=["']description["'][^>]*content=["']([^"']+)["']/i);
  const description = metaDescMatch ? metaDescMatch[1] : null;

  // Compter les liens
  const linkMatches = response.text.match(/<a[^>]*href/gi);
  const linkCount = linkMatches ? linkMatches.length : 0;

  return {
    url,
    title,
    description,
    linkCount,
    contentLength: response.text.length,
    status: response.status,
    timestamp: new Date().toISOString(),
  };
}

return execute(input, { webFetch });
`,
  createdBy: 'system',
};
