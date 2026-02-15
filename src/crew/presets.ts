/**
 * Presets de Crews prédéfinis
 *
 * Ces crews sont optimisés pour des tâches communes et utilisent
 * des modèles locaux (Ollama) par défaut pour minimiser les coûts.
 */

import type { CrewConfig, PresetCrew } from './types';
import { LLM_PRESETS } from './types';

// ===========================================================================
// RESEARCH CREW
// ===========================================================================

export const researchCrew: PresetCrew = {
  id: 'research',
  name: 'Research Crew',
  description: 'Équipe de recherche pour analyser un sujet et produire un rapport',
  requiredInputs: ['topic', 'depth'],

  buildCrew: (inputs): CrewConfig => ({
    name: 'Research Crew',
    description: `Recherche approfondie sur: ${inputs.topic}`,
    process: 'sequential',
    verbose: true,
    memory: true,

    agents: [
      {
        name: 'Researcher',
        role: 'Senior Research Analyst',
        goal: `Conduct comprehensive research on ${inputs.topic}`,
        backstory: 'You are an expert researcher with decades of experience. You excel at finding, analyzing, and synthesizing information from multiple sources.',
        llm: LLM_PRESETS.OLLAMA_BALANCED,
        allowDelegation: false,
        maxIterations: 20,
      },
      {
        name: 'Analyst',
        role: 'Data Analyst',
        goal: 'Analyze research findings and identify key insights',
        backstory: 'You are a skilled analyst who can find patterns and insights in complex data. You provide clear, actionable recommendations.',
        llm: LLM_PRESETS.OLLAMA_FAST,
        allowDelegation: false,
        maxIterations: 15,
      },
      {
        name: 'Writer',
        role: 'Technical Writer',
        goal: 'Create a clear, well-structured report from the analysis',
        backstory: 'You are an experienced technical writer who excels at making complex topics accessible. Your reports are clear, concise, and actionable.',
        llm: LLM_PRESETS.OLLAMA_FAST,
        allowDelegation: false,
        maxIterations: 10,
      },
    ],

    tasks: [
      {
        id: 'research',
        description: `Research the topic: ${inputs.topic}. ${inputs.depth === 'deep' ? 'Conduct a thorough, comprehensive analysis.' : 'Provide a high-level overview.'}`,
        expectedOutput: 'Detailed research findings with sources and key facts',
        agent: 'Researcher',
      },
      {
        id: 'analyze',
        description: 'Analyze the research findings. Identify key insights, trends, and recommendations.',
        expectedOutput: 'Structured analysis with key insights and recommendations',
        agent: 'Analyst',
        context: ['research'],
      },
      {
        id: 'report',
        description: 'Write a comprehensive report based on the research and analysis.',
        expectedOutput: 'Well-structured report with executive summary, findings, and recommendations',
        agent: 'Writer',
        context: ['research', 'analyze'],
      },
    ],
  }),
};

// ===========================================================================
// CODE REVIEW CREW
// ===========================================================================

export const codeReviewCrew: PresetCrew = {
  id: 'code-review',
  name: 'Code Review Crew',
  description: 'Équipe pour analyser et améliorer du code',
  requiredInputs: ['code', 'language', 'focus'],

  buildCrew: (inputs): CrewConfig => ({
    name: 'Code Review Crew',
    description: `Review de code ${inputs.language}`,
    process: 'sequential',
    verbose: true,
    memory: true,

    agents: [
      {
        name: 'SecurityExpert',
        role: 'Security Specialist',
        goal: 'Identify security vulnerabilities and risks in the code',
        backstory: 'You are a cybersecurity expert with deep knowledge of common vulnerabilities (OWASP Top 10, CWEs). You find security issues before they become problems.',
        llm: LLM_PRESETS.OLLAMA_CODE,
        allowDelegation: false,
        maxIterations: 15,
      },
      {
        name: 'Architect',
        role: 'Software Architect',
        goal: 'Review code architecture and suggest improvements',
        backstory: 'You are a seasoned software architect who has designed systems at scale. You focus on maintainability, scalability, and clean code principles.',
        llm: LLM_PRESETS.OLLAMA_CODE,
        allowDelegation: false,
        maxIterations: 15,
      },
      {
        name: 'Reviewer',
        role: 'Code Reviewer',
        goal: 'Synthesize all feedback and provide actionable recommendations',
        backstory: 'You are an experienced code reviewer who can balance different perspectives. You provide clear, prioritized recommendations.',
        llm: LLM_PRESETS.OLLAMA_FAST,
        allowDelegation: false,
        maxIterations: 10,
      },
    ],

    tasks: [
      {
        id: 'security-review',
        description: `Analyze this ${inputs.language} code for security vulnerabilities:\n\n${inputs.code}\n\nFocus on: SQL injection, XSS, authentication issues, data exposure, etc.`,
        expectedOutput: 'List of security issues with severity and remediation suggestions',
        agent: 'SecurityExpert',
      },
      {
        id: 'architecture-review',
        description: `Review the architecture of this ${inputs.language} code:\n\n${inputs.code}\n\nFocus on: ${inputs.focus || 'code quality, maintainability, and best practices'}`,
        expectedOutput: 'Architecture review with improvement suggestions',
        agent: 'Architect',
      },
      {
        id: 'final-review',
        description: 'Synthesize all reviews and provide a prioritized list of recommendations.',
        expectedOutput: 'Final code review report with prioritized action items',
        agent: 'Reviewer',
        context: ['security-review', 'architecture-review'],
      },
    ],
  }),
};

// ===========================================================================
// CONTENT CREATION CREW
// ===========================================================================

export const contentCrew: PresetCrew = {
  id: 'content',
  name: 'Content Creation Crew',
  description: 'Équipe pour créer du contenu de qualité',
  requiredInputs: ['topic', 'type', 'audience'],

  buildCrew: (inputs): CrewConfig => ({
    name: 'Content Creation Crew',
    description: `Création de contenu: ${inputs.type} sur ${inputs.topic}`,
    process: 'sequential',
    verbose: true,
    memory: true,

    agents: [
      {
        name: 'Strategist',
        role: 'Content Strategist',
        goal: `Create a content strategy for ${inputs.audience}`,
        backstory: 'You are a content strategist who understands how to engage different audiences. You create content plans that resonate and convert.',
        llm: LLM_PRESETS.OLLAMA_FAST,
        allowDelegation: false,
        maxIterations: 10,
      },
      {
        name: 'Writer',
        role: 'Content Writer',
        goal: `Write engaging ${inputs.type} content about ${inputs.topic}`,
        backstory: 'You are a versatile writer who can adapt your style to any audience. Your content is engaging, informative, and well-structured.',
        llm: LLM_PRESETS.OLLAMA_BALANCED,
        allowDelegation: false,
        maxIterations: 15,
      },
      {
        name: 'Editor',
        role: 'Content Editor',
        goal: 'Polish and optimize the content for maximum impact',
        backstory: 'You are a meticulous editor who ensures content is error-free, well-structured, and optimized for the target platform.',
        llm: LLM_PRESETS.OLLAMA_FAST,
        allowDelegation: false,
        maxIterations: 10,
      },
    ],

    tasks: [
      {
        id: 'strategy',
        description: `Create a content strategy for ${inputs.type} about "${inputs.topic}" targeting ${inputs.audience}. Include key messages, tone, and structure recommendations.`,
        expectedOutput: 'Content strategy document with key messages and structure',
        agent: 'Strategist',
      },
      {
        id: 'write',
        description: `Write the ${inputs.type} content following the strategy. Make it engaging and valuable for ${inputs.audience}.`,
        expectedOutput: 'Complete draft content',
        agent: 'Writer',
        context: ['strategy'],
      },
      {
        id: 'edit',
        description: 'Edit and polish the content. Fix any errors, improve clarity, and optimize for the target platform.',
        expectedOutput: 'Final, polished content ready for publication',
        agent: 'Editor',
        context: ['strategy', 'write'],
      },
    ],
  }),
};

// ===========================================================================
// DATA ANALYSIS CREW
// ===========================================================================

export const dataAnalysisCrew: PresetCrew = {
  id: 'data-analysis',
  name: 'Data Analysis Crew',
  description: 'Équipe pour analyser des données et produire des insights',
  requiredInputs: ['data', 'question'],

  buildCrew: (inputs): CrewConfig => ({
    name: 'Data Analysis Crew',
    description: `Analyse de données: ${inputs.question}`,
    process: 'sequential',
    verbose: true,
    memory: true,

    agents: [
      {
        name: 'DataEngineer',
        role: 'Data Engineer',
        goal: 'Clean, prepare, and structure the data for analysis',
        backstory: 'You are an experienced data engineer who can work with messy data. You clean, transform, and prepare data for analysis.',
        llm: LLM_PRESETS.OLLAMA_FAST,
        allowDelegation: false,
        maxIterations: 10,
      },
      {
        name: 'Analyst',
        role: 'Data Analyst',
        goal: `Answer the question: ${inputs.question}`,
        backstory: 'You are a skilled data analyst who can find insights in complex datasets. You use statistical methods to answer business questions.',
        llm: LLM_PRESETS.OLLAMA_BALANCED,
        allowDelegation: false,
        maxIterations: 20,
      },
      {
        name: 'Visualizer',
        role: 'Data Visualization Expert',
        goal: 'Create clear visualizations and summaries of the findings',
        backstory: 'You are an expert at communicating data insights. You create clear, compelling summaries that decision-makers can act on.',
        llm: LLM_PRESETS.OLLAMA_FAST,
        allowDelegation: false,
        maxIterations: 10,
      },
    ],

    tasks: [
      {
        id: 'prepare',
        description: `Review and prepare this data for analysis:\n\n${JSON.stringify(inputs.data).slice(0, 2000)}...\n\nIdentify data quality issues and structure.`,
        expectedOutput: 'Data quality report and preparation recommendations',
        agent: 'DataEngineer',
      },
      {
        id: 'analyze',
        description: `Analyze the prepared data to answer: "${inputs.question}". Use appropriate statistical methods and identify key patterns.`,
        expectedOutput: 'Detailed analysis with statistical findings and insights',
        agent: 'Analyst',
        context: ['prepare'],
      },
      {
        id: 'summarize',
        description: 'Create an executive summary of the analysis. Include key findings, visualizations recommendations, and actionable insights.',
        expectedOutput: 'Executive summary with key findings and recommendations',
        agent: 'Visualizer',
        context: ['analyze'],
      },
    ],
  }),
};

// ===========================================================================
// EXPORTS
// ===========================================================================

export const PRESET_CREWS: Record<string, PresetCrew> = {
  research: researchCrew,
  'code-review': codeReviewCrew,
  content: contentCrew,
  'data-analysis': dataAnalysisCrew,
};

/**
 * Retourne un crew preset par son ID
 */
export function getPresetCrew(id: string): PresetCrew | undefined {
  return PRESET_CREWS[id];
}

/**
 * Liste tous les presets disponibles
 */
export function listPresetCrews(): Array<{ id: string; name: string; description: string }> {
  return Object.values(PRESET_CREWS).map((preset) => ({
    id: preset.id,
    name: preset.name,
    description: preset.description,
  }));
}
