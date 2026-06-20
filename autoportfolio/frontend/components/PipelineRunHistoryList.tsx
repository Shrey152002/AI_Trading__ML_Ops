"use client";

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";

import { getPipelineRuns } from "@/lib/api/endpoints";
import { queryKeys } from "@/lib/queryKeys";
import type { PipelineRunRecord } from "@/lib/api/types";

const STATUS_STYLES: Record<string, string> = {
  completed: "text-emerald-700",
  skipped_healthy: "text-slate-500",
  aborted: "text-red-700",
  error: "text-red-700",
};

function RunRow({ run }: { run: PipelineRunRecord }) {
  return (
    <li className="rounded-md border border-slate-200 p-3 text-sm">
      <div className="flex items-center justify-between">
        <span className={STATUS_STYLES[run.status] ?? "text-slate-700"}>
          {run.status.replace(/_/g, " ")}
        </span>
        <span className="text-xs text-slate-400">
          {new Date(run.started_at).toLocaleString()}
        </span>
      </div>
      <div className="mt-1.5 flex flex-wrap gap-1.5">
        {run.steps.map((step, idx) => (
          <span
            key={idx}
            className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-600"
            title={JSON.stringify(step)}
          >
            {step.step}
            {step.step === "train" && typeof step.winner === "string"
              ? `: ${step.winner}`
              : ""}
          </span>
        ))}
      </div>
    </li>
  );
}

export function PipelineRunHistoryList({
  portfolioId,
  pendingSince,
  onRunResolved,
}: {
  portfolioId: string;
  pendingSince: string | null;
  onRunResolved: () => void;
}) {
  const query = useQuery({
    queryKey: queryKeys.pipelineRuns(portfolioId),
    queryFn: () => getPipelineRuns(portfolioId),
    refetchInterval: pendingSince ? 15_000 : false,
  });

  useEffect(() => {
    if (!pendingSince || !query.data) return;
    const hasNewerRun = query.data.runs.some((r) => r.started_at > pendingSince);
    if (hasNewerRun) onRunResolved();
  }, [pendingSince, query.data, onRunResolved]);

  if (query.isLoading) {
    return <p className="text-sm text-slate-500">Loading run history...</p>;
  }
  if (query.isError || !query.data) {
    return <p className="text-sm text-red-700">Could not load pipeline run history.</p>;
  }
  if (query.data.runs.length === 0) {
    return <p className="text-sm text-slate-500">No pipeline runs yet for this portfolio.</p>;
  }

  return (
    <ul className="space-y-2">
      {query.data.runs.map((run, idx) => (
        <RunRow key={`${run.started_at}-${idx}`} run={run} />
      ))}
    </ul>
  );
}
