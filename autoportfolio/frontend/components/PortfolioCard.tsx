"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Shield, Scale, Zap } from "lucide-react";

import { getStatus } from "@/lib/api/endpoints";
import { queryKeys } from "@/lib/queryKeys";
import { ApiError } from "@/lib/api/client";
import { formatCurrency } from "@/lib/format";
import type { PortfolioConfigResponse } from "@/lib/api/types";
import { StatusBadge } from "./StatusBadge";
import { Skeleton } from "./Skeleton";

const RISK_ICON = { conservative: Shield, moderate: Scale, aggressive: Zap } as const;

export function PortfolioCard({ config }: { config: PortfolioConfigResponse }) {
  const statusQuery = useQuery({
    queryKey: queryKeys.status(config.name),
    queryFn: () => getStatus(config.name),
    retry: false,
  });

  const noModelYet = statusQuery.error instanceof ApiError && statusQuery.error.status === 404;
  const unavailable = statusQuery.isError && !noModelYet;
  const RiskIcon = RISK_ICON[config.risk_appetite];

  return (
    <Link
      href={`/portfolio/${config.name}`}
      className="group block rounded-xl border border-slate-200 bg-white p-4 shadow-sm transition hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-md"
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="font-semibold text-slate-900">{config.display_name}</h3>
          <p className="mt-0.5 flex items-center gap-1 text-xs text-slate-500">
            <RiskIcon className="h-3 w-3" />
            {config.tickers.length} tickers · {config.risk_appetite} ·{" "}
            {formatCurrency(config.capital)}
          </p>
        </div>
        {statusQuery.isLoading ? (
          <Skeleton className="h-5 w-16" />
        ) : noModelYet ? (
          <StatusBadge kind="no-model" />
        ) : unavailable ? (
          <StatusBadge kind="unavailable" />
        ) : (
          <StatusBadge kind="live" />
        )}
      </div>

      {statusQuery.data && (
        <dl className="mt-3 grid grid-cols-3 gap-2 text-xs">
          <div>
            <dt className="text-slate-400">Sharpe</dt>
            <dd className="font-medium text-slate-800">{statusQuery.data.sharpe.toFixed(2)}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Drift</dt>
            <dd className="font-medium text-slate-800">
              {statusQuery.data.drift_score.toFixed(2)}
            </dd>
          </div>
          <div>
            <dt className="text-slate-400">Algorithm</dt>
            <dd className="font-medium text-slate-800">{statusQuery.data.algorithm}</dd>
          </div>
        </dl>
      )}

      <span className="mt-4 flex items-center gap-1 text-sm font-medium text-slate-900">
        {noModelYet ? "Run pipeline" : "View details"}
        <ArrowRight className="h-3.5 w-3.5 transition group-hover:translate-x-0.5" />
      </span>
    </Link>
  );
}
