import { z } from "zod";

const riskAppetiteSchema = z.enum(["conservative", "moderate", "aggressive"]);

export const recommendationResponseSchema = z.object({
  recommended_allocation: z.record(z.string(), z.number()),
  expected_return: z.number(),
  expected_volatility: z.number(),
  confidence: z.number(),
  explanation: z.string(),
  model_version: z.string(),
  timestamp: z.string(),
});

export const portfolioStatusResponseSchema = z.object({
  portfolio_id: z.string(),
  model_version: z.string(),
  algorithm: z.string(),
  last_training_date: z.string().nullable(),
  sharpe: z.number(),
  drift_score: z.number(),
  data_freshness_hours: z.number(),
});

export const allocationHistoryEntrySchema = z.object({
  date: z.string(),
  allocation: z.record(z.string(), z.number()),
  realized_return: z.number().nullable(),
});

export const portfolioHistoryResponseSchema = z.object({
  portfolio_id: z.string(),
  history: z.array(allocationHistoryEntrySchema),
});

export const pipelineRunResponseSchema = z.object({
  portfolio_id: z.string(),
  status: z.string(),
  message: z.string(),
});

export const healthResponseSchema = z.object({
  status: z.string(),
  portfolios: z.array(z.string()),
  mlflow: z.string(),
});

export const portfolioConfigResponseSchema = z.object({
  name: z.string(),
  display_name: z.string(),
  tickers: z.array(z.string()),
  risk_appetite: riskAppetiteSchema,
  capital: z.number(),
  rebalance_frequency: z.string(),
});

export const portfolioListResponseSchema = z.object({
  portfolios: z.array(portfolioConfigResponseSchema),
});

export const pipelineRunStepSchema = z
  .object({ step: z.string() })
  .catchall(z.unknown());

export const pipelineRunRecordSchema = z.object({
  portfolio: z.string(),
  started_at: z.string(),
  steps: z.array(pipelineRunStepSchema),
  status: z.string(),
  finished_at: z.string().nullable(),
});

export const pipelineRunsResponseSchema = z.object({
  portfolio_id: z.string(),
  runs: z.array(pipelineRunRecordSchema),
});

export const pipelineLogEntrySchema = z.object({
  seq: z.number(),
  timestamp: z.string(),
  level: z.string(),
  logger: z.string(),
  message: z.string(),
});

export const pipelineLogsResponseSchema = z.object({
  entries: z.array(pipelineLogEntrySchema),
});

export const algoProgressEntrySchema = z.object({
  phase: z.string(),
  trial: z.number(),
  n_trials: z.number(),
  steps_done: z.number(),
  steps_total: z.number(),
});

export const pipelineProgressResponseSchema = z.object({
  portfolio_id: z.string(),
  active: z.boolean(),
  fraction_done: z.number(),
  elapsed_seconds: z.number(),
  eta_seconds: z.number().nullable(),
  algorithms: z.record(z.string(), algoProgressEntrySchema),
});
