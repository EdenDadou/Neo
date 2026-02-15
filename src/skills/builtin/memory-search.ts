/**
 * Built-in Skill: Memory Search
 * Recherche dans la mémoire de Neo
 */

import { SkillCreateInput } from '../types';

export const memorySearchSkill: SkillCreateInput = {
  name: 'Memory Search',
  description: 'Recherche dans la mémoire de Neo pour retrouver des informations',
  triggers: ['chercher mémoire', 'rechercher', 'retrouver', 'mémoire'],
  requiredCapabilities: ['memory_read'],
  code: `
// Skill: Memory Search v1.0

async function execute(input, context) {
  const { memory } = context;

  // Récupérer la query depuis l'input
  const query = typeof input === 'string' ? input : input.query;
  const limit = input.limit || 10;

  if (!query) {
    throw new Error('Query de recherche requise');
  }

  // Rechercher dans la mémoire
  const results = await memory.search(query, limit);

  return {
    query,
    resultCount: results.length,
    results: results.map(r => ({
      id: r.id,
      type: r.type,
      content: r.content,
      importance: r.importance,
      createdAt: r.createdAt,
    })),
    timestamp: new Date().toISOString(),
  };
}

return execute(input, { memory });
`,
  createdBy: 'system',
};
