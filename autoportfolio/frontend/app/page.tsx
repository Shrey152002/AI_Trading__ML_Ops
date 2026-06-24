"use client";

import { useQuery } from "@tanstack/react-query";

import { listPortfolios } from "@/lib/api/endpoints";
import { queryKeys } from "@/lib/queryKeys";
import { PortfolioCard } from "@/components/PortfolioCard";
import { ErrorState } from "@/components/ErrorState";
import { Skeleton } from "@/components/Skeleton";
import { HeroExplainer } from "@/components/HeroExplainer";

export default function OverviewPage() {
  const query = useQuery({
    queryKey: queryKeys.portfolios,
    queryFn: listPortfolios,
  });

  return (
    <div className="space-y-8">
      <HeroExplainer />

      <div>
        <h2 className="text-lg font-semibold text-slate-900">Portfolios</h2>
        <p className="mt-1 text-sm text-slate-500">
          Live status across every configured portfolio. Click into one to request a
          recommendation, view allocation history, or trigger retraining.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {query.isLoading &&
          Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-36" />)}

        {query.isError && (
          <div className="sm:col-span-2 lg:col-span-4">
            <ErrorState
              message="Could not load the portfolio list."
              onRetry={() => query.refetch()}
            />
          </div>
        )}

        {query.data?.portfolios.map((config) => (
          <PortfolioCard key={config.name} config={config} />
        ))}
      </div>
    </div>
  );
}
