import { formatPercent } from "@/lib/format";
import type { RecommendationResponse } from "@/lib/api/types";
import { InfoTooltip } from "./InfoTooltip";

export function ExplanationPanel({ result }: { result: RecommendationResponse }) {
  return (
    <div className="space-y-3">
      <p className="text-sm leading-relaxed text-slate-700">{result.explanation}</p>

      <div>
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span className="flex items-center gap-1">
            Confidence
            <InfoTooltip title="Confidence">
              How closely the model&apos;s recent live performance matches its
              training-time benchmark, capped at 100%. This is <strong>not</strong> a
              measure of certainty about this specific allocation — it drops when the
              live model has drifted from how it performed during training.
            </InfoTooltip>
          </span>
          <span>{formatPercent(result.confidence, 0)}</span>
        </div>
        <div className="mt-1 h-2 rounded-full bg-slate-100">
          <div
            className="h-2 rounded-full bg-slate-900"
            style={{ width: `${Math.min(result.confidence, 1) * 100}%` }}
          />
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-2 text-xs text-slate-500">
        <div>
          <dt>Model version</dt>
          <dd className="font-medium text-slate-800">{result.model_version}</dd>
        </div>
        <div>
          <dt>Generated</dt>
          <dd className="font-medium text-slate-800">
            {new Date(result.timestamp).toLocaleString()}
          </dd>
        </div>
      </dl>
    </div>
  );
}
