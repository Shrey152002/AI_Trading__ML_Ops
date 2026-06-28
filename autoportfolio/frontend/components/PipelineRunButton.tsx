"use client";

import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Settings2 } from "lucide-react";
import clsx from "clsx";

import { runPipeline } from "@/lib/api/endpoints";
import { ApiError } from "@/lib/api/client";
import type { Algorithm } from "@/lib/api/types";
import {
  ALGORITHM_INFO,
  TRAINING_PRESETS,
  estimateTrainingSeconds,
  formatEstimatedDuration,
} from "@/lib/trainingPresets";
import { InfoTooltip } from "./InfoTooltip";
import { PipelineStepper } from "./PipelineStepper";
import { PipelineProgressBar } from "./PipelineProgressBar";

const ALL_ALGORITHMS: Algorithm[] = ["PPO", "A2C", "SAC", "DDPG"];

export function PipelineRunButton({
  portfolioId,
  onTriggered,
  variant = "primary",
}: {
  portfolioId: string;
  onTriggered: (submittedAt: string) => void;
  variant?: "primary" | "secondary";
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [activePreset, setActivePreset] = useState<string>("balanced");
  const [algorithms, setAlgorithms] = useState<Algorithm[]>(TRAINING_PRESETS[1].algorithms);
  const [totalTimesteps, setTotalTimesteps] = useState(TRAINING_PRESETS[1].totalTimesteps);
  const [nTrials, setNTrials] = useState(TRAINING_PRESETS[1].nTrials);
  const [isRunning, setIsRunning] = useState(false);

  const estimatedSeconds = useMemo(
    () => estimateTrainingSeconds(algorithms, nTrials, totalTimesteps),
    [algorithms, nTrials, totalTimesteps]
  );

  const mutation = useMutation({
    mutationFn: () =>
      runPipeline({
        portfolio_id: portfolioId,
        total_timesteps: totalTimesteps,
        n_trials: nTrials,
        algorithms,
      }),
    onSuccess: () => {
      setIsRunning(true);
      onTriggered(new Date().toISOString());
    },
  });

  function applyPreset(preset: (typeof TRAINING_PRESETS)[number]) {
    setActivePreset(preset.key);
    setAlgorithms(preset.algorithms);
    setNTrials(preset.nTrials);
    setTotalTimesteps(preset.totalTimesteps);
  }

  function toggleAlgorithm(algo: Algorithm) {
    setActivePreset("custom");
    setAlgorithms((prev) =>
      prev.includes(algo) ? prev.filter((a) => a !== algo) : [...prev, algo]
    );
  }

  return (
    <div>
      <div className="flex items-center gap-3">
        <button
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending || algorithms.length === 0}
          className={
            variant === "primary"
              ? "rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
              : "rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          }
        >
          {mutation.isPending ? "Starting..." : "Run pipeline"}
        </button>
        <span className="text-xs text-slate-500">
          Estimated: {formatEstimatedDuration(estimatedSeconds)}
        </span>
        <button
          onClick={() => setShowAdvanced((v) => !v)}
          className="flex items-center gap-1 text-xs text-slate-500 underline hover:text-slate-700"
        >
          <Settings2 className="h-3 w-3" />
          {showAdvanced ? "Hide options" : "Options"}
        </button>
      </div>

      {showAdvanced && (
        <div className="mt-3 space-y-3 rounded-md bg-slate-50 p-3">
          <p className="text-xs text-slate-500">
            Running the pipeline does seven things in order — ingest fresh data, validate
            it, build features, check if the live model has drifted, retrain (only if
            needed), promote the winner, and write a report. The selected algorithms now
            train in parallel worker processes, so wall-clock time is set by the slowest one
            you pick, not the sum of all of them.
          </p>
          <PipelineStepper />

          <div>
            <p className="text-xs font-medium text-slate-600">Speed preset</p>
            <div className="mt-1.5 flex flex-wrap gap-2">
              {TRAINING_PRESETS.map((preset) => {
                const presetSeconds = estimateTrainingSeconds(
                  preset.algorithms,
                  preset.nTrials,
                  preset.totalTimesteps
                );
                return (
                  <button
                    key={preset.key}
                    onClick={() => applyPreset(preset)}
                    className={clsx(
                      "rounded-md border px-2.5 py-1.5 text-left text-xs",
                      activePreset === preset.key
                        ? "border-slate-900 bg-white shadow-sm"
                        : "border-slate-200 bg-white hover:border-slate-300"
                    )}
                  >
                    <span className="font-medium text-slate-800">{preset.label}</span>
                    <span className="ml-1.5 text-slate-400">
                      {formatEstimatedDuration(presetSeconds)}
                    </span>
                    <p className="mt-0.5 max-w-[16rem] text-slate-500">{preset.description}</p>
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <p className="flex items-center gap-1 text-xs font-medium text-slate-600">
              Algorithms to train
              <InfoTooltip title="Algorithms to train">
                Each selected algorithm trains in its own parallel worker process. PPO and
                A2C are fast (on-policy); SAC and DDPG are slower but often reach a higher
                Sharpe ratio. The best-performing one on held-out test data gets promoted.
              </InfoTooltip>
            </p>
            <div className="mt-1.5 flex flex-wrap gap-3">
              {ALL_ALGORITHMS.map((algo) => (
                <label key={algo} className="flex items-center gap-1.5 text-xs text-slate-700">
                  <input
                    type="checkbox"
                    checked={algorithms.includes(algo)}
                    onChange={() => toggleAlgorithm(algo)}
                    className="rounded border-slate-300"
                  />
                  {algo}
                  <span
                    className={clsx(
                      "rounded-full px-1.5 py-0.5 text-[10px]",
                      ALGORITHM_INFO[algo].speed === "fast"
                        ? "bg-emerald-50 text-emerald-700"
                        : "bg-amber-50 text-amber-700"
                    )}
                  >
                    {ALGORITHM_INFO[algo].speed}
                  </span>
                </label>
              ))}
            </div>
            {algorithms.length === 0 && (
              <p className="mt-1 text-xs text-red-700">Select at least one algorithm.</p>
            )}
          </div>

          <div className="flex flex-wrap gap-6 pt-1 text-xs text-slate-600">
            <label className="flex items-center gap-1.5">
              total_timesteps
              <InfoTooltip title="total_timesteps">
                How many simulated trading days each agent practices on during final
                training. Episodes are 252 days (~1 trading year), so{" "}
                <strong>100,000 ≈ practicing across roughly 400 replayed trading years</strong>.
                Lower it for a faster but less-trained model; raise it for a slower, more
                thoroughly trained one.
              </InfoTooltip>
              <input
                type="number"
                value={totalTimesteps}
                onChange={(e) => {
                  setActivePreset("custom");
                  setTotalTimesteps(Number(e.target.value));
                }}
                className="w-24 rounded border border-slate-300 px-1.5 py-0.5"
              />
            </label>
            <label className="flex items-center gap-1.5">
              n_trials
              <InfoTooltip title="n_trials">
                How many different hyperparameter combinations (learning rate, batch
                size, etc.) Optuna tries per algorithm before picking a winner to fully
                train. With <strong>n_trials = 20</strong>, each selected algorithm gets 20
                short trial runs first before its real, full-length training run begins.
              </InfoTooltip>
              <input
                type="number"
                value={nTrials}
                onChange={(e) => {
                  setActivePreset("custom");
                  setNTrials(Number(e.target.value));
                }}
                className="w-16 rounded border border-slate-300 px-1.5 py-0.5"
              />
            </label>
          </div>
        </div>
      )}

      {mutation.isSuccess && (
        <p className="mt-2 text-sm text-emerald-700">
          {mutation.data.message} Estimated {formatEstimatedDuration(estimatedSeconds)} — open
          &quot;Live backend activity&quot; below to watch it happen, or check the run history
          for a new entry once it finishes.
        </p>
      )}
      {mutation.isError && (
        <p className="mt-2 text-sm text-red-700">
          {mutation.error instanceof ApiError
            ? mutation.error.message
            : "Failed to start the pipeline run."}
        </p>
      )}

      <div className="mt-2">
        <PipelineProgressBar
          portfolioId={portfolioId}
          active={isRunning}
          onFinished={() => setIsRunning(false)}
        />
      </div>
    </div>
  );
}
