/**
 * Formatting utilities for the UI.
 * Plain-English, human-readable formatting per PRD.
 */

export function fmtCurrency(value: number | null | undefined): string {
  if (value == null) return "--";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

export function fmtPct(value: number | null | undefined, decimals = 1): string {
  if (value == null) return "--";
  return `${value >= 0 ? "+" : ""}${value.toFixed(decimals)}%`;
}

export function fmtNumber(value: number | null | undefined, decimals = 2): string {
  if (value == null) return "--";
  return value.toFixed(decimals);
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "--";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return "--";
  const d = new Date(iso);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function fmtSignalReason(reason: string | null | undefined): string {
  if (!reason) return "";
  // Translate technical reasons to plain English
  const translations: Record<string, string> = {
    "EMA crossover (EMA20 > EMA50) with close > EMA50": "Short-term trend crossed above long-term trend",
    "Close below EMA50 — trend reversal": "Price fell below the long-term trend",
    "STOP_LOSS": "Automatic stop-loss triggered to limit losses",
    "KILL_SWITCH": "Daily loss limit reached. Trading paused for safety.",
    "End of backtest": "Backtest complete. Final position closed.",
  };
  return translations[reason] || reason;
}

export function trendDirection(value: number): "up" | "down" | "neutral" {
  if (value > 0) return "up";
  if (value < 0) return "down";
  return "neutral";
}
