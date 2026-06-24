"use client";

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertCircle, CheckCircle2, MinusCircle } from "lucide-react";
import clsx from "clsx";

import { getPipelineRuns } from "@/lib/api/endpoints";
import { queryKeys } from "@/lib/queryKeys";
import type { PipelineRunRecord } from "@/lib/api/types";
import { PipelineStepper } from "./PipelineStepper";

const STATUS_CONFIG: Record<string, { text: string; className: string; icon: typeof CheckCircle2 }> = {
  completed: { text: "Completed — model retrained", className: "text-emerald-700", icon: CheckCircle2 },
  skipped_healthy: { text: "Skipped — model already healthy", className: "text-slate-500", icon: MinusCircle },
  aborted: { text: "Aborted — data validation failed", className: "text-red-700", icon: AlertCircle },
  error: { text: "Error", className: "text-red-700", icon: AlertCircle },
};

function RunRow({ run }: { run: PipelineRunRecord }) {
  const config = STATUS_CONFIG[run.status] ?? {
    text: run.status.replace(/_/g, " "),
    className: "text-slate-700",
    icon: CheckCircle2,
  };
  const Icon = config.icon;
  const trainStep = run.steps.find((s) => s.step === "train");
  const winner = trainStep && typeof trainStep.winner === "string" ? trainStep.winner : null;

  return (
    <li className="rounded-md border border-slate-200 p-3 text-sm">
      <div className="flex items-center justify-between">
        <span className={clsx("flex items-center gap-1.5 font-medium", config.className)}>
          <Icon className="h-4 w-4" />
          {config.text}
          {winner && <span className="font-normal text-slate-500">(winner: {winner})</span>}
        </span>
        <span className="text-xs text-slate-400">
          {new Date(run.started_at).toLocaleString()}
        </span>
      </div>
      <div className="mt-2">
        <PipelineStepper steps={run.steps} />
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
