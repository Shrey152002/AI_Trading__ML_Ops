"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { tickerLabel, formatPercent } from "@/lib/format";
import type { AllocationHistoryEntry } from "@/lib/api/types";

const COLORS = [
  "#0f172a",
  "#1d4ed8",
  "#0891b2",
  "#16a34a",
  "#ca8a04",
  "#dc2626",
  "#7c3aed",
  "#db2777",
];

export function HistoryChart({ history }: { history: AllocationHistoryEntry[] }) {
  const tickers = Array.from(new Set(history.flatMap((h) => Object.keys(h.allocation))));

  const data = history
    .slice()
    .sort((a, b) => a.date.localeCompare(b.date))
    .map((entry) => {
      const row: Record<string, string | number> = {
        date: new Date(entry.date).toLocaleDateString(),
      };
      for (const t of tickers) row[t] = entry.allocation[t] ?? 0;
      return row;
    });

  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" tick={{ fontSize: 11 }} />
        <YAxis tickFormatter={(v) => formatPercent(v, 0)} tick={{ fontSize: 11 }} />
        <Tooltip formatter={(value) => formatPercent(Number(value))} />
        <Legend formatter={(value: string) => tickerLabel(value)} />
        {tickers.map((t, idx) => (
          <Area
            key={t}
            type="monotone"
            dataKey={t}
            stackId="1"
            stroke={COLORS[idx % COLORS.length]}
            fill={COLORS[idx % COLORS.length]}
            fillOpacity={0.6}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}
