"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Settings2 } from "lucide-react";

import { runPipeline } from "@/lib/api/endpoints";
import { ApiError } from "@/lib/api/client";
import { InfoTooltip } from "./InfoTooltip";
import { PipelineStepper } from "./PipelineStepper";

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
  const [totalTimesteps, setTotalTimesteps] = useState(100_000);
  const [nTrials, setNTrials] = useState(20);

  const mutation = useMutation({
    mutationFn: () =>
      runPipeline({
        portfolio_id: portfolioId,
        total_timesteps: totalTimesteps,
        n_trials: nTrials,
      }),
    onSuccess: () => onTriggered(new Date().toISOString()),
  });

  return (
    <div>
      <div className="flex items-center gap-3">
        <button
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending}
          className={
            variant === "primary"
              ? "rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
              : "rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          }
        >
          {mutation.isPending ? "Starting..." : "Run pipeline"}
        </button>
        <button
          onClick={() => setShowAdvanced((v) => !v)}
          className="flex items-center gap-1 text-xs text-slate-500 underline hover:text-slate-700"
        >
          <Settings2 className="h-3 w-3" />
          {showAdvanced ? "Hide options" : "Advanced options"}
        </button>
      </div>

      {showAdvanced && (
        <div className="mt-3 space-y-3 rounded-md bg-slate-50 p-3">
          <p className="text-xs text-slate-500">
            Running the pipeline does seven things in order — ingest fresh data, validate
            it, build features, check if the live model has drifted, retrain (only if
            needed), promote the winner, and write a report:
          </p>
          <PipelineStepper />

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
                onChange={(e) => setTotalTimesteps(Number(e.target.value))}
                className="w-24 rounded border border-slate-300 px-1.5 py-0.5"
              />
            </label>
            <label className="flex items-center gap-1.5">
              n_trials
              <InfoTooltip title="n_trials">
                How many different hyperparameter combinations (learning rate, batch
                size, etc.) Optuna tries per algorithm before picking a winner to fully
                train. With <strong>n_trials = 20</strong>, each of the 4 algorithms
                (PPO/A2C/SAC/DDPG) gets 20 short trial runs first — 80 short runs total —
                before the real, full-length training run begins for each.
              </InfoTooltip>
              <input
                type="number"
                value={nTrials}
                onChange={(e) => setNTrials(Number(e.target.value))}
                className="w-16 rounded border border-slate-300 px-1.5 py-0.5"
              />
            </label>
          </div>
        </div>
      )}

      {mutation.isSuccess && (
        <p className="mt-2 text-sm text-emerald-700">
          {mutation.data.message} This can take several minutes — open &quot;Live backend
          activity&quot; below to watch it happen, or check the run history for a new entry
          once it finishes.
        </p>
      )}
      {mutation.isError && (
        <p className="mt-2 text-sm text-red-700">
          {mutation.error instanceof ApiError
            ? mutation.error.message
            : "Failed to start the pipeline run."}
        </p>
      )}
    </div>
  );
}
