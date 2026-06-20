"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";

import { getHistory, getRecommendation, getStatus, listPortfolios } from "@/lib/api/endpoints";
import { queryKeys } from "@/lib/queryKeys";
import { ApiError } from "@/lib/api/client";
import type { RecommendationRequest, RecommendationResponse } from "@/lib/api/types";
import { StatusBadge } from "@/components/StatusBadge";
import { EmptyModelState } from "@/components/EmptyModelState";
import { ErrorState } from "@/components/ErrorState";
import { Skeleton } from "@/components/Skeleton";
import { RecommendationForm } from "@/components/RecommendationForm";
import { AllocationPieChart } from "@/components/AllocationPieChart";
import { BenchmarkBarChart } from "@/components/BenchmarkBarChart";
import { ExplanationPanel } from "@/components/ExplanationPanel";
import { HistoryChart } from "@/components/HistoryChart";
import { PipelineRunButton } from "@/components/PipelineRunButton";
import { PipelineRunHistoryList } from "@/components/PipelineRunHistoryList";

const PENDING_RUN_TIMEOUT_MS = 10 * 60 * 1000;

export default function PortfolioDetailPage() {
  const params = useParams<{ id: string }>();
  const portfolioId = params.id;
  const queryClient = useQueryClient();

  const [recommendation, setRecommendation] = useState<RecommendationResponse | null>(null);
  const [pendingSince, setPendingSince] = useState<string | null>(null);
  const pendingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const portfoliosQuery = useQuery({
    queryKey: queryKeys.portfolios,
    queryFn: listPortfolios,
  });
  const config = portfoliosQuery.data?.portfolios.find((p) => p.name === portfolioId);

  const statusQuery = useQuery({
    queryKey: queryKeys.status(portfolioId),
    queryFn: () => getStatus(portfolioId),
    retry: false,
  });
  const noModelYet = statusQuery.error instanceof ApiError && statusQuery.error.status === 404;

  const historyQuery = useQuery({
    queryKey: queryKeys.history(portfolioId),
    queryFn: () => getHistory(portfolioId),
  });

  const recommendationMutation = useMutation({
    mutationFn: (request: RecommendationRequest) => getRecommendation(request),
    onSuccess: (result) => setRecommendation(result),
  });

  const handlePipelineTriggered = useCallback(
    (submittedAt: string) => {
      setPendingSince(submittedAt);
      if (pendingTimeoutRef.current) clearTimeout(pendingTimeoutRef.current);
      pendingTimeoutRef.current = setTimeout(() => setPendingSince(null), PENDING_RUN_TIMEOUT_MS);
    },
    []
  );

  const handleRunResolved = useCallback(() => {
    setPendingSince(null);
    if (pendingTimeoutRef.current) clearTimeout(pendingTimeoutRef.current);
    queryClient.invalidateQueries({ queryKey: queryKeys.status(portfolioId) });
    queryClient.invalidateQueries({ queryKey: queryKeys.history(portfolioId) });
  }, [queryClient, portfolioId]);

  useEffect(() => {
    return () => {
      if (pendingTimeoutRef.current) clearTimeout(pendingTimeoutRef.current);
    };
  }, []);

  if (portfoliosQuery.isLoading) {
    return <Skeleton className="h-64" />;
  }
  if (portfoliosQuery.isError || !config) {
    return (
      <ErrorState
        message={`Could not load configuration for "${portfolioId}".`}
        onRetry={() => portfoliosQuery.refetch()}
      />
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <Link href="/" className="text-xs text-slate-500 hover:underline">
          ← All portfolios
        </Link>
        <div className="mt-1 flex items-center gap-3">
          <h1 className="text-xl font-semibold">{config.display_name}</h1>
          {statusQuery.isLoading ? (
            <Skeleton className="h-5 w-16" />
          ) : noModelYet ? (
            <StatusBadge kind="no-model" />
          ) : statusQuery.isError ? (
            <StatusBadge kind="unavailable" />
          ) : (
            <StatusBadge kind="live" />
          )}
        </div>
      </div>

      {/* Status panel / empty state */}
      {noModelYet ? (
        <EmptyModelState portfolioId={portfolioId} onTriggered={handlePipelineTriggered} />
      ) : statusQuery.isError ? (
        <ErrorState
          message="Could not load this portfolio's status."
          onRetry={() => statusQuery.refetch()}
        />
      ) : statusQuery.data ? (
        <dl className="grid grid-cols-2 gap-4 rounded-lg border border-slate-200 bg-white p-4 sm:grid-cols-4">
          <div>
            <dt className="text-xs text-slate-400">Algorithm</dt>
            <dd className="font-medium">{statusQuery.data.algorithm}</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-400">Sharpe (benchmark)</dt>
            <dd className="font-medium">{statusQuery.data.sharpe.toFixed(3)}</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-400">Drift score</dt>
            <dd className="font-medium">{statusQuery.data.drift_score.toFixed(3)}</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-400">Data freshness</dt>
            <dd className="font-medium">{statusQuery.data.data_freshness_hours.toFixed(1)}h</dd>
          </div>
        </dl>
      ) : null}

      {/* Recommendation */}
      <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <h2 className="text-sm font-semibold">Request a recommendation</h2>
          <div className="mt-3">
            <RecommendationForm
              portfolioId={portfolioId}
              tickers={config.tickers}
              defaultRiskAppetite={config.risk_appetite}
              defaultCapital={config.capital}
              onSubmit={(request) => recommendationMutation.mutate(request)}
              isSubmitting={recommendationMutation.isPending}
              errorMessage={
                recommendationMutation.isError
                  ? recommendationMutation.error instanceof ApiError &&
                    recommendationMutation.error.status === 404
                    ? "No production model yet — run the pipeline first."
                    : "Failed to get a recommendation."
                  : null
              }
            />
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <h2 className="text-sm font-semibold">Recommended allocation</h2>
          {recommendation ? (
            <div className="mt-2 space-y-4">
              <AllocationPieChart allocation={recommendation.recommended_allocation} />
              <BenchmarkBarChart
                expectedReturn={recommendation.expected_return}
                expectedVolatility={recommendation.expected_volatility}
              />
              <ExplanationPanel result={recommendation} />
            </div>
          ) : (
            <p className="mt-3 text-sm text-slate-500">
              Submit the form to see a recommended allocation here.
            </p>
          )}
        </div>
      </section>

      {/* History */}
      <section className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="text-sm font-semibold">Allocation history</h2>
        <div className="mt-2">
          {historyQuery.isLoading ? (
            <Skeleton className="h-64" />
          ) : historyQuery.isError ? (
            <ErrorState
              message="Could not load allocation history."
              onRetry={() => historyQuery.refetch()}
            />
          ) : historyQuery.data && historyQuery.data.history.length > 0 ? (
            <HistoryChart history={historyQuery.data.history} />
          ) : (
            <p className="text-sm text-slate-500">No allocation history yet.</p>
          )}
        </div>
      </section>

      {/* Pipeline */}
      {!noModelYet && (
        <section className="rounded-lg border border-slate-200 bg-white p-4">
          <h2 className="text-sm font-semibold">Pipeline runs</h2>
          <div className="mt-2">
            <PipelineRunButton
              portfolioId={portfolioId}
              onTriggered={handlePipelineTriggered}
              variant="secondary"
            />
          </div>
          <div className="mt-4">
            <PipelineRunHistoryList
              portfolioId={portfolioId}
              pendingSince={pendingSince}
              onRunResolved={handleRunResolved}
            />
          </div>
        </section>
      )}
    </div>
  );
}
