import clsx from "clsx";

export type StatusKind = "live" | "no-model" | "unavailable" | "healthy" | "degraded";

const STYLES: Record<StatusKind, string> = {
  live: "bg-emerald-100 text-emerald-800 border-emerald-300",
  healthy: "bg-emerald-100 text-emerald-800 border-emerald-300",
  "no-model": "bg-amber-100 text-amber-800 border-amber-300",
  degraded: "bg-amber-100 text-amber-800 border-amber-300",
  unavailable: "bg-red-100 text-red-800 border-red-300",
};

const LABELS: Record<StatusKind, string> = {
  live: "Live",
  healthy: "Healthy",
  "no-model": "No model yet",
  degraded: "Drift detected",
  unavailable: "Unavailable",
};

export function StatusBadge({ kind, label }: { kind: StatusKind; label?: string }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        STYLES[kind]
      )}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {label ?? LABELS[kind]}
    </span>
  );
}
