"use client";

import { useEffect, useState } from "react";
import MetricCard from "@/components/MetricCard";
import Panel from "@/components/Panel";
import StatusBadge from "@/components/StatusBadge";
import ConnectionStatus from "@/components/ConnectionStatus";
import ChartContainer from "@/components/charts/ChartContainer";
import EquityCurveChart from "@/components/charts/EquityCurveChart";
import DrawdownChart from "@/components/charts/DrawdownChart";
import { useEquityCurve } from "@/lib/useEquityCurve";
import { useDrawdown } from "@/lib/useDrawdown";
import { useSSE } from "@/lib/useSSE";
import {
  getAccount,
  getMetrics,
  getLiveStatus,
  getTrades,
  AccountSummary,
  Metrics,
  LiveStatus,
  Trade,
} from "@/lib/api";
import {
  fmtCurrency,
  fmtPct,
  fmtNumber,
  fmtDateTime,
  trendDirection,
} from "@/lib/format";

export default function OverviewPage() {
  const [account, setAccount] = useState<AccountSummary | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [liveStatus, setLiveStatus] = useState<LiveStatus | null>(null);
  const [recentTrades, setRecentTrades] = useState<Trade[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showDrawdown, setShowDrawdown] = useState(false);

  // Equity curve data for the hero chart
  const equityCurve = useEquityCurve(30000);
  const drawdownData = useDrawdown(equityCurve.data);

  // SSE: real-time updates for equity, trades, and loop status
  const { isConnected, lastEvent } = useSSE([
    "equity_update",
    "trade_executed",
    "loop_status",
  ]);

  // Patch state instantly on SSE events
  useEffect(() => {
    if (!lastEvent) return;

    const { type, payload } = lastEvent;
    const p = payload as Record<string, unknown>;

    if (type === "equity_update") {
      setAccount((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          equity: (p.equity as number) ?? prev.equity,
          daily_pnl: (p.daily_pnl as number) ?? prev.daily_pnl,
        };
      });
    } else if (type === "trade_executed") {
      // Prepend to recent trades (SSE gives us a summary)
      setRecentTrades((prev) => {
        const newTrade: Trade = {
          id: Date.now(),
          symbol: (p.symbol as string) ?? "",
          entry_time: "",
          exit_time: lastEvent.timestamp,
          entry_price: 0,
          exit_price: (p.price as number) ?? 0,
          pnl: null,
          shares: 0,
          reason: (p.reason as string) ?? (p.action as string) ?? "",
        };
        return [newTrade, ...prev].slice(0, 5);
      });
    } else if (type === "loop_status") {
      const status = p.status as string;
      setLiveStatus((prev) => {
        if (!prev) return { status, evaluations_count: 0, trades_count: 0 } as LiveStatus;
        return { ...prev, status };
      });
    }
  }, [lastEvent]);

  useEffect(() => {
    async function load() {
      try {
        const [a, m, ls, t] = await Promise.all([
          getAccount(),
          getMetrics(),
          getLiveStatus().catch(() => null),
          getTrades(),
        ]);
        setAccount(a);
        setMetrics(m);
        setLiveStatus(ls);
        setRecentTrades(t.slice(0, 5));
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data");
      }
    }
    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10">
        <Panel title="Connection Error">
          <p className="text-muted">
            Could not connect to the trading backend. Make sure the server is
            running at <code className="text-accent">localhost:8000</code>.
          </p>
          <p className="text-loss text-sm mt-2">{error}</p>
        </Panel>
      </div>
    );
  }

  const isRunning = liveStatus?.status === "running";

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Overview</h1>
          <p className="text-muted text-sm mt-1">
            Account summary and system status
          </p>
        </div>
        <div className="flex items-center gap-3">
          <ConnectionStatus isConnected={isConnected} />
          <StatusBadge
            label={isRunning ? "Live Trading" : "Idle"}
            variant={isRunning ? "success" : "neutral"}
            pulse={isRunning}
          />
        </div>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Account Value"
          value={fmtCurrency(account?.equity)}
          size="lg"
        />
        <MetricCard
          label="Today's P&L"
          value={fmtCurrency(account?.daily_pnl)}
          trend={trendDirection(account?.daily_pnl ?? 0)}
          size="lg"
        />
        <MetricCard
          label="Total Return"
          value={fmtPct(metrics?.net_return_pct)}
          trend={trendDirection(metrics?.net_return_pct ?? 0)}
        />
        <MetricCard
          label="Win Rate"
          value={metrics ? `${(metrics.win_rate * 100).toFixed(0)}%` : "--"}
          subtitle={
            metrics ? `${metrics.total_trades} trades total` : undefined
          }
        />
      </div>

      {/* Hero Chart with Equity / Drawdown Toggle */}
      <Panel
        title={showDrawdown ? "Drawdown" : "Equity Curve"}
        subtitle={
          showDrawdown
            ? "Decline from peak over time"
            : "Account value over time"
        }
        action={
          <button
            onClick={() => setShowDrawdown(!showDrawdown)}
            className="text-accent text-sm hover:underline"
          >
            {showDrawdown ? "Show Equity" : "Show Drawdown"}
          </button>
        }
      >
        <ChartContainer
          height={300}
          loading={equityCurve.loading}
          error={equityCurve.error}
          empty={!equityCurve.loading && equityCurve.data.length === 0}
          emptyMessage="No equity data yet. Start paper trading to track performance."
        >
          {({ width, height }) =>
            showDrawdown ? (
              <DrawdownChart
                data={drawdownData}
                width={width}
                height={height}
              />
            ) : (
              <EquityCurveChart
                data={equityCurve.data}
                width={width}
                height={height}
                baselineValue={
                  equityCurve.data.length > 0
                    ? equityCurve.data[0].value
                    : undefined
                }
              />
            )
          }
        </ChartContainer>
      </Panel>

      {/* Performance + Positions row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Summary */}
        <Panel title="Performance" subtitle="Key metrics from all trades">
          <div className="grid grid-cols-2 gap-4">
            <MetricCard
              label="Net P&L"
              value={fmtCurrency(metrics?.net_pnl)}
              trend={trendDirection(metrics?.net_pnl ?? 0)}
              size="sm"
            />
            <MetricCard
              label="Profit Factor"
              value={fmtNumber(metrics?.profit_factor)}
              subtitle="Gross profit / gross loss"
              size="sm"
            />
            <MetricCard
              label="Max Drawdown"
              value={fmtPct(
                metrics?.max_drawdown_pct
                  ? -Math.abs(metrics.max_drawdown_pct)
                  : null
              )}
              subtitle={fmtCurrency(
                metrics?.max_drawdown_absolute
                  ? -Math.abs(metrics.max_drawdown_absolute)
                  : null
              )}
              trend="down"
              size="sm"
            />
            <MetricCard
              label="Sharpe Proxy"
              value={fmtNumber(metrics?.sharpe_proxy)}
              subtitle="Risk-adjusted return"
              size="sm"
            />
            <MetricCard
              label="Avg Win"
              value={fmtCurrency(metrics?.avg_win)}
              trend="up"
              size="sm"
            />
            <MetricCard
              label="Avg Loss"
              value={fmtCurrency(
                metrics?.avg_loss ? -Math.abs(metrics.avg_loss) : null
              )}
              trend="down"
              size="sm"
            />
          </div>
        </Panel>

        {/* Open Positions */}
        <Panel
          title="Open Positions"
          subtitle={
            account?.open_positions.length
              ? `${account.open_positions.length} active`
              : "No open positions"
          }
        >
          {account?.open_positions.length ? (
            <div className="space-y-3">
              {account.open_positions.map((pos) => (
                <div
                  key={pos.symbol}
                  className="flex items-center justify-between p-3 bg-surface-overlay rounded-lg"
                >
                  <div>
                    <p className="text-white font-semibold">{pos.symbol}</p>
                    <p className="text-muted text-xs">
                      {pos.shares} shares @ {fmtCurrency(pos.entry_price)}
                    </p>
                    <p className="text-muted text-xs">
                      Stop: {fmtCurrency(pos.stop_price)}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-white text-sm font-mono">
                      {fmtCurrency(pos.current_price)}
                    </p>
                    <p
                      className={`text-sm font-mono ${
                        pos.unrealized_pnl >= 0 ? "text-profit" : "text-loss"
                      }`}
                    >
                      {fmtCurrency(pos.unrealized_pnl)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted text-sm py-6 text-center">
              No open positions. The system will enter trades when conditions
              align.
            </p>
          )}
        </Panel>
      </div>

      {/* Recent Trades */}
      <Panel
        title="Recent Trades"
        subtitle="Last 5 completed trades"
        action={
          <a
            href="/trades"
            className="text-accent text-sm hover:underline"
          >
            View all
          </a>
        }
      >
        {recentTrades.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-muted text-xs uppercase tracking-wider border-b border-surface-border">
                  <th className="text-left pb-2 pr-4">Symbol</th>
                  <th className="text-left pb-2 pr-4">Entry</th>
                  <th className="text-left pb-2 pr-4">Exit</th>
                  <th className="text-right pb-2 pr-4">P&L</th>
                  <th className="text-left pb-2">Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border">
                {recentTrades.map((trade) => (
                  <tr key={trade.id}>
                    <td className="py-2.5 pr-4 text-white font-medium">
                      {trade.symbol}
                    </td>
                    <td className="py-2.5 pr-4 text-muted">
                      {fmtDateTime(trade.entry_time)}
                    </td>
                    <td className="py-2.5 pr-4 text-muted">
                      {trade.exit_time ? fmtDateTime(trade.exit_time) : "--"}
                    </td>
                    <td
                      className={`py-2.5 pr-4 text-right font-mono ${
                        (trade.pnl ?? 0) >= 0 ? "text-profit" : "text-loss"
                      }`}
                    >
                      {fmtCurrency(trade.pnl)}
                    </td>
                    <td className="py-2.5 text-muted text-xs max-w-[200px] truncate">
                      {trade.reason || "--"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-muted text-sm py-6 text-center">
            No trades yet. Start live trading or run a backtest to see results.
          </p>
        )}
      </Panel>

      {/* System Status */}
      <Panel title="System Status" subtitle="Engine and safety checks">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="space-y-1">
            <p className="text-muted text-xs uppercase tracking-wider">
              Kill Switch
            </p>
            <StatusBadge
              label={account?.trade_blocked ? "Activated" : "Normal"}
              variant={account?.trade_blocked ? "danger" : "success"}
            />
          </div>
          <div className="space-y-1">
            <p className="text-muted text-xs uppercase tracking-wider">
              Max Daily Loss
            </p>
            <p className="text-white text-sm font-mono">
              {fmtCurrency(account?.max_daily_loss)}
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-muted text-xs uppercase tracking-wider">
              Trading Engine
            </p>
            <StatusBadge
              label={isRunning ? "Running" : "Stopped"}
              variant={isRunning ? "success" : "neutral"}
              pulse={isRunning}
            />
          </div>
          <div className="space-y-1">
            <p className="text-muted text-xs uppercase tracking-wider">
              Evaluations
            </p>
            <p className="text-white text-sm font-mono">
              {liveStatus?.evaluations_count ?? 0}
            </p>
          </div>
        </div>
      </Panel>
    </div>
  );
}
