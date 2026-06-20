"use client";

import { useQuery } from "@tanstack/react-query";

import { getHealth } from "@/lib/api/endpoints";
import { queryKeys } from "@/lib/queryKeys";

export function HealthBanner() {
  const { isError, data } = useQuery({
    queryKey: queryKeys.health,
    queryFn: getHealth,
    refetchInterval: 30_000,
  });

  if (isError) {
    return (
      <div className="bg-red-600 px-4 py-1.5 text-center text-xs font-medium text-white">
        Can&apos;t reach the AutoPortfolio API — is it running at the configured base URL?
      </div>
    );
  }

  if (data && data.mlflow === "disconnected") {
    return (
      <div className="bg-amber-500 px-4 py-1.5 text-center text-xs font-medium text-white">
        API is up, but MLflow is unreachable — model status may be stale.
      </div>
    );
  }

  return null;
}
