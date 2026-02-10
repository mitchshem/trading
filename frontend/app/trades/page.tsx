"use client";

import { useEffect, useState } from "react";
import Panel from "@/components/Panel";
import StatusBadge from "@/components/StatusBadge";
import ChartContainer from "@/components/charts/ChartContainer";
import PnLHistogram from "@/components/charts/PnLHistogram";
import MonthlyReturnsHeatmap from "@/components/charts/MonthlyReturnsHeatmap";
import {
  getTrades,
  getDecisions,
  getTradeDistribution,
  Trade,
  DecisionEntry,
  TradeDistributionEntry,
} from "@/lib/api";
import { useMonthlyReturns } from "@/lib/useMonthlyReturns";
import {
  fmtCurrency,
  fmtPct,
  fmtDateTime,
  fmtSignalReason,
} from "@/lib/format";

export default function TradesPage() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [decisions, setDecisions] = useState<DecisionEntry[]>([]);
  const [tradeDistribution, setTradeDistribution] = useState<
    TradeDistributionEntry[]
  >([]);
  const [tab, setTab] = useState<"trades" | "decisions" | "analytics">(
    "trades"
  );
  const [expandedTrade, setExpandedTrade] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const monthlyReturns = useMonthlyReturns(60000);

  useEffect(() => {
    async function load() {
      try {
        const [t, d, dist] = await Promise.all([
          getTrades(),
          getDecisions(100).catch(() => ({ decisions: [], total: 0 })),
          getTradeDistribution().catch(() => ({ trades: [] })),
        ]);
        setTrades(t);
        setDecisions(d.decisions);
        setTradeDistribution(dist.trades);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load");
      }
    }
    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, []);

  // Stats
  const closedTrades = trades.filter((t) => t.exit_price != null);
  const wins = closedTrades.filter((t) => (t.pnl ?? 0) > 0).length;
  const losses = closedTrades.filter((t) => (t.pnl ?? 0) <= 0).length;
  const totalPnl = closedTrades.reduce((sum, t) => sum + (t.pnl ?? 0), 0);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Trade Activity</h1>
        <p className="text-muted text-sm mt-1">
          Full history of trades, strategy decisions, and performance analytics
        </p>
      </div>

      {error && (
        <div className="bg-loss/10 border border-loss/30 rounded-xl p-4 text-loss text-sm">
          {error}
        </div>
      )}

      {/* Summary strip */}
      <div className="flex flex-wrap gap-6 text-sm">
        <div>
          <span className="text-muted">Total Trades: </span>
          <span className="text-white font-mono">{closedTrades.length}</span>
        </div>
        <div>
          <span className="text-muted">Wins: </span>
          <span className="text-profit font-mono">{wins}</span>
        </div>
        <div>
          <span className="text-muted">Losses: </span>
          <span className="text-loss font-mono">{losses}</span>
        </div>
        <div>
          <span className="text-muted">Net P&L: </span>
          <span
            className={`font-mono ${totalPnl >= 0 ? "text-profit" : "text-loss"}`}
          >
            {fmtCurrency(totalPnl)}
          </span>
        </div>
      </div>

      {/* Tab Selector */}
      <div className="flex gap-2">
        {(
          [
            { key: "trades", label: "Completed Trades" },
            { key: "decisions", label: "Decision Log" },
            { key: "analytics", label: "Analytics" },
          ] as const
        ).map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              tab === key
                ? "bg-surface-overlay text-white"
                : "text-muted hover:text-white hover:bg-surface-overlay/50"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Trades Tab */}
      {tab === "trades" && (
        <Panel title="Completed Trades" subtitle="Click a trade for details">
          {closedTrades.length > 0 ? (
            <div className="space-y-2">
              {closedTrades
                .slice()
                .reverse()
                .map((trade) => {
                  const isExpanded = expandedTrade === trade.id;
                  const pnl = trade.pnl ?? 0;
                  const entryValue = trade.entry_price * trade.shares;
                  const returnPct =
                    entryValue > 0 ? (pnl / entryValue) * 100 : 0;

                  return (
                    <div
                      key={trade.id}
                      className="bg-surface-overlay rounded-xl overflow-hidden"
                    >
                      {/* Trade row - clickable */}
                      <button
                        onClick={() =>
                          setExpandedTrade(isExpanded ? null : trade.id)
                        }
                        className="w-full flex items-center justify-between p-4 text-left hover:bg-surface-border/30 transition-colors"
                      >
                        <div className="flex items-center gap-4">
                          <div>
                            <p className="text-white font-semibold">
                              {trade.symbol}
                            </p>
                            <p className="text-muted text-xs">
                              {fmtDateTime(trade.entry_time)}
                            </p>
                          </div>
                          <StatusBadge
                            label={pnl >= 0 ? "Win" : "Loss"}
                            variant={pnl >= 0 ? "success" : "danger"}
                          />
                        </div>
                        <div className="text-right">
                          <p
                            className={`font-mono font-semibold ${
                              pnl >= 0 ? "text-profit" : "text-loss"
                            }`}
                          >
                            {fmtCurrency(pnl)}
                          </p>
                          <p className="text-muted text-xs">
                            {fmtPct(returnPct)}
                          </p>
                        </div>
                      </button>

                      {/* Expanded details */}
                      {isExpanded && (
                        <div className="px-4 pb-4 border-t border-surface-border pt-3 space-y-2">
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                            <div>
                              <p className="text-muted text-xs">Entry Price</p>
                              <p className="text-white font-mono">
                                {fmtCurrency(trade.entry_price)}
                              </p>
                            </div>
                            <div>
                              <p className="text-muted text-xs">Exit Price</p>
                              <p className="text-white font-mono">
                                {fmtCurrency(trade.exit_price)}
                              </p>
                            </div>
                            <div>
                              <p className="text-muted text-xs">Shares</p>
                              <p className="text-white font-mono">
                                {trade.shares}
                              </p>
                            </div>
                            <div>
                              <p className="text-muted text-xs">Exit Date</p>
                              <p className="text-white">
                                {fmtDateTime(trade.exit_time)}
                              </p>
                            </div>
                          </div>
                          {trade.reason && (
                            <div className="mt-2 p-3 bg-surface rounded-lg">
                              <p className="text-muted text-xs mb-1">
                                Why this trade closed:
                              </p>
                              <p className="text-white text-sm">
                                {fmtSignalReason(trade.reason)}
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          ) : (
            <p className="text-muted text-sm py-8 text-center">
              No completed trades yet.
            </p>
          )}
        </Panel>
      )}

      {/* Decisions Tab */}
      {tab === "decisions" && (
        <Panel
          title="Decision Log"
          subtitle="Every evaluation the strategy makes, whether or not it trades"
        >
          {decisions.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-muted text-xs uppercase tracking-wider border-b border-surface-border">
                    <th className="text-left pb-2 pr-3">Time</th>
                    <th className="text-left pb-2 pr-3">Symbol</th>
                    <th className="text-left pb-2 pr-3">Close</th>
                    <th className="text-left pb-2 pr-3">Signal</th>
                    <th className="text-left pb-2 pr-3">Reason</th>
                    <th className="text-left pb-2">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-surface-border">
                  {decisions.map((d) => (
                    <tr key={d.id}>
                      <td className="py-2 pr-3 text-muted text-xs whitespace-nowrap">
                        {fmtDateTime(d.timestamp)}
                      </td>
                      <td className="py-2 pr-3 text-white">{d.symbol}</td>
                      <td className="py-2 pr-3 text-white font-mono">
                        {fmtCurrency(d.candle.close)}
                      </td>
                      <td className="py-2 pr-3">
                        <StatusBadge
                          label={d.signal}
                          variant={
                            d.signal === "BUY"
                              ? "success"
                              : d.signal === "EXIT"
                                ? "danger"
                                : "neutral"
                          }
                        />
                      </td>
                      <td className="py-2 pr-3 text-muted text-xs max-w-[200px] truncate">
                        {fmtSignalReason(d.signal_reason) || "--"}
                      </td>
                      <td className="py-2 text-muted text-xs">
                        {d.broker_action || "None"}
                        {d.anomaly_flags && (
                          <span className="ml-1 text-yellow-400">
                            ({d.anomaly_flags})
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-muted text-sm py-8 text-center">
              No decisions recorded yet. Start live trading to begin logging.
            </p>
          )}
        </Panel>
      )}

      {/* Analytics Tab */}
      {tab === "analytics" && (
        <div className="space-y-6">
          {/* P&L Distribution */}
          <Panel
            title="P&L Distribution"
            subtitle="Profit/loss per trade — green wins, red losses"
          >
            <ChartContainer
              height={Math.max(200, Math.min(tradeDistribution.length * 22 + 40, 500))}
              loading={false}
              empty={tradeDistribution.length === 0}
              emptyMessage="No closed trades yet to show distribution."
            >
              {({ height }) => (
                <PnLHistogram trades={tradeDistribution} height={height} />
              )}
            </ChartContainer>
          </Panel>

          {/* Monthly Returns Heatmap */}
          <Panel
            title="Monthly Returns"
            subtitle="Returns by month — hover for details"
          >
            <ChartContainer
              height={monthlyReturns.data.length > 0 ? undefined : 120}
              loading={monthlyReturns.loading}
              error={monthlyReturns.error}
              empty={
                !monthlyReturns.loading && monthlyReturns.data.length === 0
              }
              emptyMessage="No monthly return data yet."
            >
              {() => <MonthlyReturnsHeatmap data={monthlyReturns.data} />}
            </ChartContainer>
          </Panel>

          {/* Signal distribution summary */}
          <Panel
            title="Signal Distribution"
            subtitle="Breakdown of strategy decisions"
          >
            {decisions.length > 0 ? (
              <div className="grid grid-cols-3 gap-4">
                {["BUY", "HOLD", "EXIT"].map((signal) => {
                  const count = decisions.filter(
                    (d) => d.signal === signal
                  ).length;
                  const pct =
                    decisions.length > 0
                      ? ((count / decisions.length) * 100).toFixed(1)
                      : "0";
                  return (
                    <div
                      key={signal}
                      className="bg-surface-overlay rounded-xl p-4 text-center"
                    >
                      <StatusBadge
                        label={signal}
                        variant={
                          signal === "BUY"
                            ? "success"
                            : signal === "EXIT"
                              ? "danger"
                              : "neutral"
                        }
                      />
                      <p className="text-white text-2xl font-mono font-bold mt-2">
                        {count}
                      </p>
                      <p className="text-muted text-xs">{pct}% of decisions</p>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-muted text-sm py-8 text-center">
                No decisions to analyze yet.
              </p>
            )}
          </Panel>
        </div>
      )}
    </div>
  );
}
