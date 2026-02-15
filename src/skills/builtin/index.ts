/**
 * Built-in Skills
 * Skills fournis par défaut avec Neo
 */

import { SkillCreateInput } from '../types';
import { webScraperSkill } from './web-scraper';
import { memorySearchSkill } from './memory-search';

export const BUILTIN_SKILLS: SkillCreateInput[] = [
  webScraperSkill,
  memorySearchSkill,
];

export { webScraperSkill, memorySearchSkill };
