export type RiskAppetite = "conservative" | "moderate" | "aggressive";

export type Algorithm = "PPO" | "A2C" | "SAC" | "DDPG";

export interface RecommendationRequest {
  portfolio_id: string;
  current_holdings: Record<string, number>;
  risk_appetite: RiskAppetite;
  capital: number;
  sector_constraints: string[];
}

export interface RecommendationResponse {
  recommended_allocation: Record<string, number>;
  expected_return: number;
  expected_volatility: number;
  confidence: number;
  explanation: string;
  model_version: string;
  timestamp: string;
}

export interface PortfolioStatusResponse {
  portfolio_id: string;
  model_version: string;
  algorithm: string;
  last_training_date: string | null;
  sharpe: number;
  drift_score: number;
  data_freshness_hours: number;
}

export interface AllocationHistoryEntry {
  date: string;
  allocation: Record<string, number>;
  realized_return: number | null;
}

export interface PortfolioHistoryResponse {
  portfolio_id: string;
  history: AllocationHistoryEntry[];
}

export interface PipelineRunRequest {
  portfolio_id: string;
  total_timesteps?: number;
  n_trials?: number;
  algorithms?: Algorithm[] | null;
}

export interface PipelineRunResponse {
  portfolio_id: string;
  status: string;
  message: string;
}

export interface HealthResponse {
  status: string;
  portfolios: string[];
  mlflow: string;
}

export interface PortfolioConfigResponse {
  name: string;
  display_name: string;
  tickers: string[];
  risk_appetite: RiskAppetite;
  capital: number;
  rebalance_frequency: string;
}

export interface PortfolioListResponse {
  portfolios: PortfolioConfigResponse[];
}

export interface PipelineRunStep {
  step: string;
  [key: string]: unknown;
}

export interface PipelineRunRecord {
  portfolio: string;
  started_at: string;
  steps: PipelineRunStep[];
  status: string;
  finished_at: string | null;
}

export interface PipelineRunsResponse {
  portfolio_id: string;
  runs: PipelineRunRecord[];
}

export interface PipelineLogEntry {
  seq: number;
  timestamp: string;
  level: string;
  logger: string;
  message: string;
}

export interface PipelineLogsResponse {
  entries: PipelineLogEntry[];
}

export interface AlgoProgressEntry {
  phase: string;
  trial: number;
  n_trials: number;
  steps_done: number;
  steps_total: number;
}

export interface PipelineProgressResponse {
  portfolio_id: string;
  active: boolean;
  fraction_done: number;
  elapsed_seconds: number;
  eta_seconds: number | null;
  algorithms: Record<string, AlgoProgressEntry>;
}
