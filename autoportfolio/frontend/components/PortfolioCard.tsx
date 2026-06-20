"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";

import { getStatus } from "@/lib/api/endpoints";
import { queryKeys } from "@/lib/queryKeys";
import { ApiError } from "@/lib/api/client";
import { formatCurrency } from "@/lib/format";
import type { PortfolioConfigResponse } from "@/lib/api/types";
import { StatusBadge } from "./StatusBadge";
import { Skeleton } from "./Skeleton";

export function PortfolioCard({ config }: { config: PortfolioConfigResponse }) {
  const statusQuery = useQuery({
    queryKey: queryKeys.status(config.name),
    queryFn: () => getStatus(config.name),
    retry: false,
  });

  const noModelYet = statusQuery.error instanceof ApiError && statusQuery.error.status === 404;
  const unavailable = statusQuery.isError && !noModelYet;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-semibold">{config.display_name}</h3>
          <p className="text-xs text-slate-500">
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
            <dd className="font-medium">{statusQuery.data.sharpe.toFixed(2)}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Drift</dt>
            <dd className="font-medium">{statusQuery.data.drift_score.toFixed(2)}</dd>
          </div>
          <div>
            <dt className="text-slate-400">Algorithm</dt>
            <dd className="font-medium">{statusQuery.data.algorithm}</dd>
          </div>
        </dl>
      )}

      <Link
        href={`/portfolio/${config.name}`}
        className="mt-4 inline-block text-sm font-medium text-slate-900 underline-offset-2 hover:underline"
      >
        {noModelYet ? "Run pipeline →" : "View details →"}
      </Link>
    </div>
  );
}
