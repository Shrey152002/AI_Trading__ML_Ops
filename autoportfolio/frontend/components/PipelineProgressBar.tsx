"use client";

import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

import { getPipelineProgress } from "@/lib/api/endpoints";
import { formatEstimatedDuration } from "@/lib/trainingPresets";
import type { PipelineProgressResponse } from "@/lib/api/types";

const POLL_INTERVAL_MS = 2000;

const PHASE_LABEL: Record<string, string> = {
  pending: "Waiting to start",
  hyperopt: "Searching hyperparameters",
  final_train: "Training",
  done: "Done",
  failed: "Failed",
};

export function PipelineProgressBar({
  portfolioId,
  active,
  onFinished,
}: {
  portfolioId: string;
  active: boolean;
  onFinished?: () => void;
}) {
  const [progress, setProgress] = useState<PipelineProgressResponse | null>(null);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    let sawActive = false;

    async function poll() {
      try {
        const result = await getPipelineProgress(portfolioId);
        if (cancelled) return;
        if (result.active) sawActive = true;
        setProgress(result);
        // Only treat "inactive" as a finish signal once we've actually observed the run
        // active — right after triggering, the first poll or two can still see the
        // previous (already-finished) state before the new run registers itself.
        if (!result.active && sawActive) onFinished?.();
      } catch {
        // best-effort polling — a transient fetch failure just means we try again next tick
      }
    }

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [active, portfolioId, onFinished]);

  if (!active || !progress || !progress.active) return null;

  const percent = Math.round(progress.fraction_done * 100);
  const algoEntries = Object.entries(progress.algorithms);

  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <div className="flex items-center justify-between text-xs font-medium text-slate-700">
        <span className="flex items-center gap-1.5">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Training in progress — {percent}%
        </span>
        <span className="text-slate-500">
          {progress.eta_seconds != null
            ? `ETA ${formatEstimatedDuration(progress.eta_seconds)}`
            : "Estimating time remaining..."}
        </span>
      </div>
      <div className="mt-2 h-1.5 rounded-full bg-slate-200">
        <div
          className="h-1.5 rounded-full bg-slate-900 transition-all"
          style={{ width: `${Math.max(percent, 2)}%` }}
        />
      </div>
      {algoEntries.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-500">
          {algoEntries.map(([algo, entry]) => (
            <span
              key={algo}
              className="rounded-full border border-slate-200 bg-white px-2 py-0.5"
            >
              {algo}: {PHASE_LABEL[entry.phase] ?? entry.phase}
              {entry.phase === "hyperopt" && ` (trial ${entry.trial}/${entry.n_trials})`}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
