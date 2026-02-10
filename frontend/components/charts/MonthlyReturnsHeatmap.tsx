"use client";

interface MonthlyReturn {
  year: number;
  month: number;
  return_pct: number;
  pnl: number;
  trade_count: number;
}

interface MonthlyReturnsHeatmapProps {
  data: MonthlyReturn[];
}

const MONTH_LABELS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/**
 * Monthly returns heatmap: 12 columns (months) x N rows (years).
 * Color scale: deep red (-) → gray (0) → deep green (+).
 * Pure React/CSS grid — no charting library needed.
 */
export default function MonthlyReturnsHeatmap({
  data,
}: MonthlyReturnsHeatmapProps) {
  if (data.length === 0) {
    return (
      <div className="bg-surface-overlay rounded-xl flex items-center justify-center text-muted text-sm h-32">
        No monthly returns data yet.
      </div>
    );
  }

  // Group by year
  const years = Array.from(new Set(data.map((d) => d.year))).sort();
  const byYearMonth = new Map<string, MonthlyReturn>();
  for (const entry of data) {
    byYearMonth.set(`${entry.year}-${entry.month}`, entry);
  }

  // Find max absolute return for color scaling
  const maxAbsReturn = Math.max(
    ...data.map((d) => Math.abs(d.return_pct)),
    1
  );

  function getCellColor(returnPct: number): string {
    const intensity = Math.min(Math.abs(returnPct) / maxAbsReturn, 1);
    if (returnPct > 0) {
      // Green scale
      const g = Math.round(80 + intensity * 117); // 80 → 197
      const r = Math.round(34 * intensity);
      return `rgba(${r}, ${g}, 94, ${0.2 + intensity * 0.6})`;
    } else if (returnPct < 0) {
      // Red scale
      const r = Math.round(120 + intensity * 119); // 120 → 239
      return `rgba(${r}, 68, 68, ${0.2 + intensity * 0.6})`;
    }
    return "rgba(107, 114, 128, 0.15)"; // neutral gray
  }

  // Calculate yearly totals
  const yearlyTotals = years.map((year) => {
    const yearData = data.filter((d) => d.year === year);
    const totalReturn = yearData.reduce((sum, d) => sum + d.return_pct, 0);
    const totalPnl = yearData.reduce((sum, d) => sum + d.pnl, 0);
    return { year, totalReturn, totalPnl };
  });

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr>
            <th className="text-left text-muted font-normal pb-2 pr-2 w-12">
              Year
            </th>
            {MONTH_LABELS.map((m) => (
              <th
                key={m}
                className="text-center text-muted font-normal pb-2 px-0.5"
              >
                {m}
              </th>
            ))}
            <th className="text-center text-muted font-normal pb-2 pl-2 w-16">
              Total
            </th>
          </tr>
        </thead>
        <tbody>
          {years.map((year) => {
            const yearTotal = yearlyTotals.find((y) => y.year === year);
            return (
              <tr key={year}>
                <td className="text-muted font-mono pr-2 py-0.5">{year}</td>
                {Array.from({ length: 12 }, (_, i) => {
                  const key = `${year}-${i + 1}`;
                  const entry = byYearMonth.get(key);
                  return (
                    <td key={i} className="px-0.5 py-0.5">
                      {entry ? (
                        <div
                          className="rounded-sm px-1 py-1.5 text-center font-mono cursor-default transition-opacity hover:opacity-80"
                          style={{ backgroundColor: getCellColor(entry.return_pct) }}
                          title={`${MONTH_LABELS[i]} ${year}: ${entry.return_pct >= 0 ? "+" : ""}${entry.return_pct.toFixed(2)}% ($${entry.pnl.toFixed(0)}) — ${entry.trade_count} trades`}
                        >
                          <span
                            className={
                              entry.return_pct >= 0
                                ? "text-profit"
                                : "text-loss"
                            }
                          >
                            {entry.return_pct >= 0 ? "+" : ""}
                            {entry.return_pct.toFixed(1)}
                          </span>
                        </div>
                      ) : (
                        <div className="rounded-sm px-1 py-1.5 text-center text-surface-border">
                          —
                        </div>
                      )}
                    </td>
                  );
                })}
                <td className="pl-2 py-0.5">
                  <div
                    className={`font-mono text-center font-semibold ${
                      (yearTotal?.totalReturn ?? 0) >= 0
                        ? "text-profit"
                        : "text-loss"
                    }`}
                  >
                    {(yearTotal?.totalReturn ?? 0) >= 0 ? "+" : ""}
                    {(yearTotal?.totalReturn ?? 0).toFixed(1)}%
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
