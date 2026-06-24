import { Check, Circle, MinusCircle } from "lucide-react";
import clsx from "clsx";

import { PIPELINE_STAGES } from "@/lib/pipelineStages";
import type { PipelineRunStep } from "@/lib/api/types";
import { InfoTooltip } from "./InfoTooltip";

type StageStatus = "done" | "skipped" | "not-reached" | "info";

function statusFor(stageKey: string, steps?: PipelineRunStep[]): StageStatus {
  if (!steps) return "info";
  const reached = steps.find((s) => s.step === stageKey);
  if (!reached) return "not-reached";
  if (stageKey === "drift_check" && reached.needs_retrain === false) return "done";
  return "done";
}

export function PipelineStepper({ steps }: { steps?: PipelineRunStep[] }) {
  // If drift_check ran and decided no retrain was needed, every later stage was
  // deliberately skipped rather than simply "not reached yet" — worth distinguishing.
  const driftStep = steps?.find((s) => s.step === "drift_check");
  const wasSkippedHealthy = driftStep && driftStep.needs_retrain === false;

  return (
    <ol className="flex flex-wrap gap-2">
      {PIPELINE_STAGES.map((stage, idx) => {
        let status = statusFor(stage.key, steps);
        if (status === "not-reached" && wasSkippedHealthy && idx > 3) status = "skipped";

        return (
          <li
            key={stage.key}
            className={clsx(
              "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs",
              status === "done" && "border-emerald-200 bg-emerald-50 text-emerald-800",
              status === "skipped" && "border-slate-200 bg-slate-50 text-slate-400",
              status === "not-reached" && "border-slate-200 bg-white text-slate-400",
              status === "info" && "border-slate-200 bg-slate-50 text-slate-600"
            )}
          >
            {status === "done" && <Check className="h-3 w-3" />}
            {status === "skipped" && <MinusCircle className="h-3 w-3" />}
            {(status === "not-reached" || status === "info") && (
              <Circle className="h-3 w-3" />
            )}
            {stage.label}
            <InfoTooltip title={stage.label}>{stage.description}</InfoTooltip>
          </li>
        );
      })}
    </ol>
  );
}
