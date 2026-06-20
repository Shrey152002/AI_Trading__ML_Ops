"use client";

import { useState } from "react";

import type { RecommendationRequest, RiskAppetite } from "@/lib/api/types";
import { tickerLabel } from "@/lib/format";

export function RecommendationForm({
  portfolioId,
  tickers,
  defaultRiskAppetite,
  defaultCapital,
  onSubmit,
  isSubmitting,
  errorMessage,
}: {
  portfolioId: string;
  tickers: string[];
  defaultRiskAppetite: RiskAppetite;
  defaultCapital: number;
  onSubmit: (request: RecommendationRequest) => void;
  isSubmitting: boolean;
  errorMessage: string | null;
}) {
  const equalWeight = 1 / tickers.length;
  const [holdings, setHoldings] = useState<Record<string, number>>(
    Object.fromEntries(tickers.map((t) => [t, equalWeight]))
  );
  const [riskAppetite, setRiskAppetite] = useState<RiskAppetite>(defaultRiskAppetite);
  const [capital, setCapital] = useState(defaultCapital);
  const [sectorConstraintsText, setSectorConstraintsText] = useState("");

  const totalWeight = Object.values(holdings).reduce((sum, w) => sum + w, 0);

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit({
          portfolio_id: portfolioId,
          current_holdings: holdings,
          risk_appetite: riskAppetite,
          capital,
          sector_constraints: sectorConstraintsText
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
        });
      }}
      className="space-y-4"
    >
      <div>
        <p className="text-xs font-medium text-slate-600">
          Current holdings (weights, total: {(totalWeight * 100).toFixed(0)}%)
        </p>
        <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3">
          {tickers.map((t) => (
            <label key={t} className="text-xs text-slate-500">
              {tickerLabel(t)}
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={holdings[t]}
                onChange={(e) =>
                  setHoldings((prev) => ({ ...prev, [t]: Number(e.target.value) }))
                }
                className="mt-0.5 w-full rounded border border-slate-300 px-2 py-1 text-sm text-slate-900"
              />
            </label>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <label className="text-xs font-medium text-slate-600">
          Risk appetite
          <select
            value={riskAppetite}
            onChange={(e) => setRiskAppetite(e.target.value as RiskAppetite)}
            className="mt-0.5 w-full rounded border border-slate-300 px-2 py-1.5 text-sm text-slate-900"
          >
            <option value="conservative">Conservative</option>
            <option value="moderate">Moderate</option>
            <option value="aggressive">Aggressive</option>
          </select>
        </label>
        <label className="text-xs font-medium text-slate-600">
          Capital (INR)
          <input
            type="number"
            min={1}
            value={capital}
            onChange={(e) => setCapital(Number(e.target.value))}
            className="mt-0.5 w-full rounded border border-slate-300 px-2 py-1.5 text-sm text-slate-900"
          />
        </label>
      </div>

      <label className="block text-xs font-medium text-slate-600">
        Sector constraints (comma separated, optional)
        <input
          type="text"
          value={sectorConstraintsText}
          onChange={(e) => setSectorConstraintsText(e.target.value)}
          placeholder="e.g. exclude_energy, max_single_30pct"
          className="mt-0.5 w-full rounded border border-slate-300 px-2 py-1.5 text-sm text-slate-900"
        />
      </label>

      <button
        type="submit"
        disabled={isSubmitting || capital <= 0}
        className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
      >
        {isSubmitting ? "Requesting..." : "Get recommendation"}
      </button>

      {errorMessage && <p className="text-sm text-red-700">{errorMessage}</p>}
    </form>
  );
}
