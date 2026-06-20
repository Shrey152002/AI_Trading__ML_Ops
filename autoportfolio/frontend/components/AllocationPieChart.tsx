"use client";

import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import { tickerLabel, formatPercent } from "@/lib/format";

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

export function AllocationPieChart({ allocation }: { allocation: Record<string, number> }) {
  const data = Object.entries(allocation).map(([ticker, weight]) => ({
    name: tickerLabel(ticker),
    value: weight,
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          innerRadius={50}
          outerRadius={90}
          paddingAngle={2}
        >
          {data.map((_, idx) => (
            <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip formatter={(value) => formatPercent(Number(value))} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}
