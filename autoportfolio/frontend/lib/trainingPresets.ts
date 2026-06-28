import type { Algorithm } from "./api/types";

export interface TrainingPreset {
  key: string;
  label: string;
  description: string;
  algorithms: Algorithm[];
  nTrials: number;
  totalTimesteps: number;
}

export const TRAINING_PRESETS: TrainingPreset[] = [
  {
    key: "quick",
    label: "Quick",
    description: "PPO + A2C only with a light hyperparameter search. Good for trying the pipeline out.",
    algorithms: ["PPO", "A2C"],
    nTrials: 5,
    totalTimesteps: 20_000,
  },
  {
    key: "balanced",
    label: "Balanced",
    description: "All four algorithms with a moderate search. A reasonable default for real runs.",
    algorithms: ["PPO", "A2C", "SAC", "DDPG"],
    nTrials: 10,
    totalTimesteps: 50_000,
  },
  {
    key: "thorough",
    label: "Thorough",
    description: "All four algorithms with the full search and training budget. Best models, slowest.",
    algorithms: ["PPO", "A2C", "SAC", "DDPG"],
    nTrials: 20,
    totalTimesteps: 100_000,
  },
];

export const ALGORITHM_INFO: Record<Algorithm, { speed: "fast" | "slow"; note: string }> = {
  PPO: { speed: "fast", note: "On-policy — trains quickly, a solid baseline." },
  A2C: { speed: "fast", note: "On-policy — trains quickly, similar speed to PPO." },
  SAC: { speed: "slow", note: "Off-policy — slower to train, often scores the best Sharpe." },
  DDPG: { speed: "slow", note: "Off-policy — slower to train, can be less stable than SAC." },
};

// Each hyperopt trial trains for this many timesteps (training.hyperopt's fixed search budget) —
// must match the backend constant for the estimate to track the real amount of work being done.
const SEARCH_TIMESTEPS = 5000;

// Rough steps/sec throughput observed on a modest machine. These are deliberately approximate —
// labeled as such in the UI — since actual speed depends heavily on CPU and current load.
const THROUGHPUT_STEPS_PER_SEC: Record<Algorithm, number> = {
  PPO: 150,
  A2C: 150,
  SAC: 25,
  DDPG: 25,
};

/** Algorithms now train in parallel worker processes, so wall-clock time is bottlenecked by
 * the slowest selected algorithm, not the sum of all of them. */
export function estimateTrainingSeconds(
  algorithms: Algorithm[],
  nTrials: number,
  totalTimesteps: number
): number {
  if (algorithms.length === 0) return 0;
  const perAlgoSeconds = algorithms.map((algo) => {
    const steps = nTrials * SEARCH_TIMESTEPS + totalTimesteps;
    return steps / THROUGHPUT_STEPS_PER_SEC[algo];
  });
  return Math.max(...perAlgoSeconds);
}

export function formatEstimatedDuration(seconds: number): string {
  if (seconds < 45) return "under a minute";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `~${minutes} min`;
  const hours = minutes / 60;
  return `~${hours.toFixed(1)} hr`;
}
