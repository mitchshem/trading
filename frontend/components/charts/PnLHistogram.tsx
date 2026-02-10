"use client";

interface PnLBar {
  id: number;
  pnl: number;
  symbol: string;
  exit_time: string;
  is_win: boolean;
}

interface PnLHistogramProps {
  trades: PnLBar[];
  height?: number;
}

/**
 * P&L distribution chart rendered as horizontal bars.
 * Green bars for wins, red bars for losses, sorted by trade date.
 * Pure React/CSS — no charting library needed for discrete bars.
 */
export default function PnLHistogram({
  trades,
  height = 300,
}: PnLHistogramProps) {
  if (trades.length === 0) {
    return (
      <div
        className="bg-surface-overlay rounded-xl flex items-center justify-center text-muted text-sm"
        style={{ height }}
      >
        No trade data for distribution chart.
      </div>
    );
  }

  // Sort by exit_time ascending
  const sorted = [...trades].sort(
    (a, b) => new Date(a.exit_time).getTime() - new Date(b.exit_time).getTime()
  );

  const maxPnl = Math.max(...sorted.map((t) => Math.abs(t.pnl)), 1);

  // Stats
  const wins = sorted.filter((t) => t.is_win);
  const losses = sorted.filter((t) => !t.is_win);
  const avgWin = wins.length > 0 ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((s, t) => s + t.pnl, 0) / losses.length : 0;

  return (
    <div style={{ height }} className="flex flex-col">
      {/* Stats row */}
      <div className="flex items-center justify-between text-xs text-muted mb-3 px-1">
        <span>
          <span className="text-profit font-mono">{wins.length}</span> wins
          {wins.length > 0 && (
            <span className="ml-1 text-profit font-mono">
              (avg ${avgWin.toFixed(0)})
            </span>
          )}
        </span>
        <span>
          <span className="text-loss font-mono">{losses.length}</span> losses
          {losses.length > 0 && (
            <span className="ml-1 text-loss font-mono">
              (avg ${Math.abs(avgLoss).toFixed(0)})
            </span>
          )}
        </span>
      </div>

      {/* Bars container */}
      <div
        className="flex-1 overflow-y-auto space-y-0.5"
        style={{ maxHeight: height - 40 }}
      >
        {sorted.map((trade) => {
          const barWidth = Math.max((Math.abs(trade.pnl) / maxPnl) * 100, 2);
          const isWin = trade.pnl >= 0;
          const dateStr = new Date(trade.exit_time).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
          });

          return (
            <div key={trade.id} className="flex items-center gap-2 group">
              {/* Date label */}
              <span className="text-muted text-[10px] font-mono w-12 shrink-0 text-right">
                {dateStr}
              </span>

              {/* Bar */}
              <div className="flex-1 flex items-center h-5">
                {/* Left side (losses) */}
                <div className="w-1/2 flex justify-end">
                  {!isWin && (
                    <div
                      className="h-4 rounded-l-sm bg-loss/80 hover:bg-loss transition-colors"
                      style={{ width: `${barWidth}%` }}
                    />
                  )}
                </div>

                {/* Center line */}
                <div className="w-px h-5 bg-surface-border shrink-0" />

                {/* Right side (wins) */}
                <div className="w-1/2 flex justify-start">
                  {isWin && (
                    <div
                      className="h-4 rounded-r-sm bg-profit/80 hover:bg-profit transition-colors"
                      style={{ width: `${barWidth}%` }}
                    />
                  )}
                </div>
              </div>

              {/* P&L label */}
              <span
                className={`text-[10px] font-mono w-16 shrink-0 ${
                  isWin ? "text-profit" : "text-loss"
                }`}
              >
                {isWin ? "+" : ""}${trade.pnl.toFixed(0)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
