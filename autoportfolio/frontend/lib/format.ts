export function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatPercent(value: number, fractionDigits = 1): string {
  return `${(value * 100).toFixed(fractionDigits)}%`;
}

export function formatNumber(value: number, fractionDigits = 2): string {
  return value.toFixed(fractionDigits);
}

export function formatRelativeTime(isoTimestamp: string | null): string {
  if (!isoTimestamp) return "never";
  const then = new Date(isoTimestamp).getTime();
  const diffMs = Date.now() - then;
  const diffHours = diffMs / (1000 * 60 * 60);

  if (diffHours < 1) return "less than an hour ago";
  if (diffHours < 24) return `${Math.round(diffHours)}h ago`;
  return `${Math.round(diffHours / 24)}d ago`;
}

export function tickerLabel(ticker: string): string {
  return ticker.replace(/\.NS$/, "");
}
