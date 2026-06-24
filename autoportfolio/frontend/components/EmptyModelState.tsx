import { Sparkles } from "lucide-react";

import { PipelineRunButton } from "./PipelineRunButton";

export function EmptyModelState({
  portfolioId,
  onTriggered,
}: {
  portfolioId: string;
  onTriggered: (submittedAt: string) => void;
}) {
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
      <p className="flex items-center gap-1.5 text-sm font-medium text-amber-900">
        <Sparkles className="h-4 w-4" />
        No production model yet
      </p>
      <p className="mt-1 text-sm text-amber-800">
        This portfolio hasn&apos;t been trained yet. Run the pipeline to ingest data, train
        all four agents, and register the best one — usually takes a few minutes.
      </p>
      <div className="mt-3">
        <PipelineRunButton portfolioId={portfolioId} onTriggered={onTriggered} />
      </div>
    </div>
  );
}
