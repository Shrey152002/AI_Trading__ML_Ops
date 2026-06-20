"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { formatPercent } from "@/lib/format";

export function BenchmarkBarChart({
  expectedReturn,
  expectedVolatility,
}: {
  expectedReturn: number;
  expectedVolatility: number;
}) {
  const data = [
    { name: "Expected return (annualized)", value: expectedReturn },
    { name: "Expected volatility (annualized)", value: expectedVolatility },
  ];

  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} layout="vertical" margin={{ left: 24 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tickFormatter={(v) => formatPercent(v, 0)} />
        <YAxis type="category" dataKey="name" width={180} tick={{ fontSize: 12 }} />
        <Tooltip formatter={(value) => formatPercent(Number(value))} />
        <Bar dataKey="value" fill="#1d4ed8" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
