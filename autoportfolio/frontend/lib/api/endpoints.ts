import { apiGet, apiPost } from "./client";
import {
  healthResponseSchema,
  pipelineLogsResponseSchema,
  pipelineProgressResponseSchema,
  pipelineRunResponseSchema,
  pipelineRunsResponseSchema,
  portfolioHistoryResponseSchema,
  portfolioListResponseSchema,
  portfolioStatusResponseSchema,
  recommendationResponseSchema,
} from "./schemas";
import type {
  HealthResponse,
  PipelineLogsResponse,
  PipelineProgressResponse,
  PipelineRunRequest,
  PipelineRunResponse,
  PipelineRunsResponse,
  PortfolioHistoryResponse,
  PortfolioListResponse,
  PortfolioStatusResponse,
  RecommendationRequest,
  RecommendationResponse,
} from "./types";

export function getHealth(): Promise<HealthResponse> {
  return apiGet("/health", healthResponseSchema);
}

export function listPortfolios(): Promise<PortfolioListResponse> {
  return apiGet("/portfolios", portfolioListResponseSchema);
}

export function getStatus(portfolioId: string): Promise<PortfolioStatusResponse> {
  return apiGet(`/portfolio/${portfolioId}/status`, portfolioStatusResponseSchema);
}

export function getHistory(portfolioId: string): Promise<PortfolioHistoryResponse> {
  return apiGet(`/portfolio/${portfolioId}/history`, portfolioHistoryResponseSchema);
}

export function getRecommendation(
  body: RecommendationRequest
): Promise<RecommendationResponse> {
  return apiPost("/portfolio/recommendation", recommendationResponseSchema, body);
}

export function runPipeline(body: PipelineRunRequest): Promise<PipelineRunResponse> {
  return apiPost("/pipeline/run", pipelineRunResponseSchema, body);
}

export function getPipelineRuns(portfolioId: string): Promise<PipelineRunsResponse> {
  return apiGet(`/pipeline/runs/${portfolioId}`, pipelineRunsResponseSchema);
}

export function getPipelineLogs(since: number): Promise<PipelineLogsResponse> {
  return apiGet(`/pipeline/logs?since=${since}`, pipelineLogsResponseSchema);
}

export function getPipelineProgress(portfolioId: string): Promise<PipelineProgressResponse> {
  return apiGet(`/pipeline/progress/${portfolioId}`, pipelineProgressResponseSchema);
}
