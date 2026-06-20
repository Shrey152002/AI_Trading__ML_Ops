export const queryKeys = {
  health: ["health"] as const,
  portfolios: ["portfolios"] as const,
  status: (portfolioId: string) => ["status", portfolioId] as const,
  history: (portfolioId: string) => ["history", portfolioId] as const,
  pipelineRuns: (portfolioId: string) => ["pipelineRuns", portfolioId] as const,
};
