"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { runPipeline } from "@/lib/api/endpoints";
import { ApiError } from "@/lib/api/client";

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
      <div className="flex items-center gap-2">
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
          className="text-xs text-slate-500 underline hover:text-slate-700"
        >
          {showAdvanced ? "Hide options" : "Advanced options"}
        </button>
      </div>

      {showAdvanced && (
        <div className="mt-2 flex gap-4 text-xs text-slate-600">
          <label className="flex items-center gap-1.5">
            total_timesteps
            <input
              type="number"
              value={totalTimesteps}
              onChange={(e) => setTotalTimesteps(Number(e.target.value))}
              className="w-24 rounded border border-slate-300 px-1.5 py-0.5"
            />
          </label>
          <label className="flex items-center gap-1.5">
            n_trials
            <input
              type="number"
              value={nTrials}
              onChange={(e) => setNTrials(Number(e.target.value))}
              className="w-16 rounded border border-slate-300 px-1.5 py-0.5"
            />
          </label>
        </div>
      )}

      {mutation.isSuccess && (
        <p className="mt-2 text-sm text-emerald-700">
          {mutation.data.message} This can take several minutes — check the run history
          below for a new entry (matched by timestamp, since the pipeline doesn&apos;t
          report a run ID back).
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
