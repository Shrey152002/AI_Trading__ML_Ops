export interface PipelineStageInfo {
  key: string;
  label: string;
  description: string;
}

export const PIPELINE_STAGES: PipelineStageInfo[] = [
  {
    key: "ingest",
    label: "Ingest",
    description: "Downloads fresh daily price history for every ticker in the portfolio.",
  },
  {
    key: "validate",
    label: "Validate",
    description:
      "Checks for missing rows or stale prices. A hard gate — if this fails, the run stops here.",
  },
  {
    key: "features",
    label: "Features",
    description:
      "Computes and versions the engineered features (returns, volatility, momentum) the agents observe.",
  },
  {
    key: "drift_check",
    label: "Drift check",
    description:
      "Compares the live production model's recent performance to its benchmark. If it's still healthy, training is skipped.",
  },
  {
    key: "train",
    label: "Train",
    description:
      "Trains all four RL agents (PPO, A2C, SAC, DDPG) from scratch, each with its own hyperparameter search.",
  },
  {
    key: "promote",
    label: "Promote",
    description:
      "Benchmarks the newly trained agents and promotes the best one to Production if it beats the current model.",
  },
  {
    key: "report",
    label: "Report",
    description: "Generates an HTML evaluation report comparing all four agents.",
  },
];
