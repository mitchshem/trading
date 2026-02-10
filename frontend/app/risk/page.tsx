"use client";

import { useEffect, useState, useCallback } from "react";
import Panel from "@/components/Panel";
import StatusBadge from "@/components/StatusBadge";
import ConnectionStatus from "@/components/ConnectionStatus";
import ChartContainer from "@/components/charts/ChartContainer";
import DrawdownChart from "@/components/charts/DrawdownChart";
import { getRiskState, updateRiskLimits, RiskState, RiskStateLimits } from "@/lib/api";
import { useEquityCurve } from "@/lib/useEquityCurve";
import { useDrawdown } from "@/lib/useDrawdown";
import { useSSE } from "@/lib/useSSE";
import { fmtCurrency, fmtPct } from "@/lib/format";

interface RiskControl {
  name: string;
  description: string;
  currentValue: string;
  limit: string;
  status: "safe" | "warning" | "breached";
  pctUsed: number;
  breached: boolean;
}

// Bounds for editable risk limits (match backend _RISK_LIMIT_BOUNDS)
const LIMIT_BOUNDS: Record<
  string,
  { min: number; max: number; step: number; label: string; asPct: boolean }
> = {
  max_daily_loss_pct: {
    min: 0.1,
    max: 10,
    step: 0.1,
    label: "Daily Loss Limit",
    asPct: true,
  },
  max_weekly_loss_pct: {
    min: 0.5,
    max: 20,
    step: 0.5,
    label: "Weekly Loss Limit",
    asPct: true,
  },
  max_monthly_loss_pct: {
    min: 1,
    max: 30,
    step: 1,
    label: "Monthly Loss Limit",
    asPct: true,
  },
  max_consecutive_losing_days: {
    min: 1,
    max: 20,
    step: 1,
    label: "Max Consecutive Losing Days",
    asPct: false,
  },
  max_drawdown_from_hwm_pct: {
    min: 1,
    max: 50,
    step: 1,
    label: "Max Drawdown from Peak",
    asPct: true,
  },
  max_portfolio_exposure_pct: {
    min: 10,
    max: 100,
    step: 5,
    label: "Max Portfolio Exposure",
    asPct: true,
  },
};

export default function RiskPage() {
  const [riskState, setRiskState] = useState<RiskState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Edit mode state
  const [editing, setEditing] = useState(false);
  const [limitDraft, setLimitDraft] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  // Drawdown chart data
  const equityCurve = useEquityCurve(30000);
  const drawdownData = useDrawdown(equityCurve.data);

  // SSE: instant breach alerts trigger a re-fetch
  const { isConnected, lastEvent } = useSSE([
    "alert_fired",
    "equity_update",
    "risk_limits_updated",
  ]);

  const load = useCallback(async () => {
    try {
      const rs = await getRiskState();
      setRiskState(rs);
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load risk state"
      );
    }
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, [load]);

  // When an alert fires or equity updates, re-fetch risk state immediately
  useEffect(() => {
    if (!lastEvent) return;
    load();
  }, [lastEvent, load]);

  const equity = riskState?.equity ?? 100000;
  const details = riskState?.details;
  const limits = riskState?.limits;

  function controlStatus(
    pctUsed: number,
    breached: boolean
  ): "safe" | "warning" | "breached" {
    if (breached) return "breached";
    if (pctUsed >= 70) return "warning";
    return "safe";
  }

  // Initialize edit mode with current limits as percentages
  function startEditing() {
    if (!limits) return;
    setLimitDraft({
      max_daily_loss_pct: ((limits.max_daily_loss_pct ?? 0.02) * 100).toFixed(1),
      max_weekly_loss_pct: ((limits.max_weekly_loss_pct ?? 0.05) * 100).toFixed(1),
      max_monthly_loss_pct: ((limits.max_monthly_loss_pct ?? 0.10) * 100).toFixed(0),
      max_consecutive_losing_days: String(limits.max_consecutive_losing_days ?? 5),
      max_drawdown_from_hwm_pct: ((limits.max_drawdown_from_hwm_pct ?? 0.15) * 100).toFixed(0),
      max_portfolio_exposure_pct: ((limits.max_portfolio_exposure_pct ?? 0.80) * 100).toFixed(0),
    });
    setEditing(true);
    setSuccess(null);
    setError(null);
  }

  function cancelEditing() {
    setEditing(false);
    setLimitDraft({});
  }

  async function saveLimits() {
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      // Convert percentages back to decimals for the API
      const body: Partial<RiskStateLimits> = {};
      for (const [key, val] of Object.entries(limitDraft)) {
        const num = parseFloat(val);
        if (isNaN(num)) continue;
        const bounds = LIMIT_BOUNDS[key];
        if (!bounds) continue;
        if (bounds.asPct) {
          (body as Record<string, number>)[key] = num / 100;
        } else {
          (body as Record<string, number>)[key] = num;
        }
      }
      const result = await updateRiskLimits(body);
      if (result.error) {
        setError(result.error);
      } else {
        setSuccess("Risk limits updated successfully.");
        setEditing(false);
        load();
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to update risk limits"
      );
    } finally {
      setSaving(false);
    }
  }

  // Compute percentage usage for each limit
  const dailyPnl = details?.daily_pnl ?? 0;
  const dailyLimitAbs = equity * (limits?.max_daily_loss_pct ?? 0.02);
  const dailyPctUsed =
    dailyPnl < 0 && dailyLimitAbs > 0
      ? Math.min((Math.abs(dailyPnl) / dailyLimitAbs) * 100, 100)
      : 0;

  const weeklyPnl = details?.weekly_pnl ?? 0;
  const weeklyLimitAbs = equity * (limits?.max_weekly_loss_pct ?? 0.05);
  const weeklyPctUsed =
    weeklyPnl < 0 && weeklyLimitAbs > 0
      ? Math.min((Math.abs(weeklyPnl) / weeklyLimitAbs) * 100, 100)
      : 0;

  const monthlyPnl = details?.monthly_pnl ?? 0;
  const monthlyLimitAbs = equity * (limits?.max_monthly_loss_pct ?? 0.10);
  const monthlyPctUsed =
    monthlyPnl < 0 && monthlyLimitAbs > 0
      ? Math.min((Math.abs(monthlyPnl) / monthlyLimitAbs) * 100, 100)
      : 0;

  const drawdownPct = details?.drawdown_from_hwm_pct ?? 0;
  const drawdownLimitPct = (limits?.max_drawdown_from_hwm_pct ?? 0.15) * 100;
  const drawdownPctUsed =
    drawdownPct > 0 && drawdownLimitPct > 0
      ? Math.min((drawdownPct / drawdownLimitPct) * 100, 100)
      : 0;

  const consLoseDays = details?.consecutive_losing_days ?? 0;
  const maxConsLoseDays = limits?.max_consecutive_losing_days ?? 5;
  const consLosePctUsed =
    maxConsLoseDays > 0
      ? Math.min((consLoseDays / maxConsLoseDays) * 100, 100)
      : 0;

  const controls: RiskControl[] = [
    {
      name: "Daily Loss Limit",
      description:
        "Maximum loss allowed in a single trading day. Trading stops if breached.",
      currentValue: fmtCurrency(dailyPnl),
      limit: fmtCurrency(-dailyLimitAbs),
      status: controlStatus(dailyPctUsed, riskState?.daily_breached ?? false),
      pctUsed: dailyPctUsed,
      breached: riskState?.daily_breached ?? false,
    },
    {
      name: "Weekly Loss Limit",
      description:
        "Maximum cumulative loss in a calendar week. Limits sustained losing streaks.",
      currentValue: fmtCurrency(weeklyPnl),
      limit: fmtCurrency(-weeklyLimitAbs),
      status: controlStatus(weeklyPctUsed, riskState?.weekly_breached ?? false),
      pctUsed: weeklyPctUsed,
      breached: riskState?.weekly_breached ?? false,
    },
    {
      name: "Monthly Loss Limit",
      description:
        "Maximum cumulative loss in a calendar month. Provides broader safety net.",
      currentValue: fmtCurrency(monthlyPnl),
      limit: fmtCurrency(-monthlyLimitAbs),
      status: controlStatus(
        monthlyPctUsed,
        riskState?.monthly_breached ?? false
      ),
      pctUsed: monthlyPctUsed,
      breached: riskState?.monthly_breached ?? false,
    },
    {
      name: "Max Drawdown from Peak",
      description:
        "Maximum decline from the account's highest value. Hard stop to preserve capital.",
      currentValue: fmtPct(-drawdownPct, 2),
      limit: fmtPct(-drawdownLimitPct, 0),
      status: controlStatus(
        drawdownPctUsed,
        riskState?.hwm_drawdown_breached ?? false
      ),
      pctUsed: drawdownPctUsed,
      breached: riskState?.hwm_drawdown_breached ?? false,
    },
    {
      name: "Consecutive Losing Days",
      description:
        "After consecutive losing days, trading pauses for 1 day to break the streak.",
      currentValue: `${consLoseDays} day${consLoseDays !== 1 ? "s" : ""}`,
      limit: `${maxConsLoseDays} days`,
      status: controlStatus(
        consLosePctUsed,
        riskState?.consecutive_days_breached ?? false
      ),
      pctUsed: consLosePctUsed,
      breached: riskState?.consecutive_days_breached ?? false,
    },
    {
      name: "Portfolio Exposure",
      description:
        "Maximum percentage of capital that can be in open positions at once.",
      currentValue: `${((limits?.max_portfolio_exposure_pct ?? 0.80) * 100).toFixed(0)}% max`,
      limit: "80% of equity",
      status: "safe",
      pctUsed: 0,
      breached: false,
    },
  ];

  const statusVariant = {
    safe: "success" as const,
    warning: "warning" as const,
    breached: "danger" as const,
  };

  const statusLabel = {
    safe: "Normal",
    warning: "Caution",
    breached: "Breached",
  };

  const anyBreached = riskState?.any_breached ?? false;
  const isPaused = riskState?.is_paused ?? false;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Risk Controls</h1>
          <p className="text-muted text-sm mt-1">
            Safety limits that protect your capital automatically
          </p>
        </div>
        <div className="flex items-center gap-3">
          <ConnectionStatus isConnected={isConnected} />
          {riskState && (
            <StatusBadge
              label={
                anyBreached
                  ? "Risk Breached"
                  : isPaused
                    ? "Paused"
                    : "All Clear"
              }
              variant={anyBreached ? "danger" : isPaused ? "warning" : "success"}
              pulse={anyBreached}
            />
          )}
        </div>
      </div>

      {/* Alerts */}
      {error && (
        <div className="bg-loss/10 border border-loss/30 rounded-xl p-4 text-loss text-sm">
          {error}
        </div>
      )}
      {success && (
        <div className="bg-profit/10 border border-profit/30 rounded-xl p-4 text-profit text-sm">
          {success}
        </div>
      )}

      {/* Account Snapshot */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-surface-raised rounded-xl p-4">
          <p className="text-muted text-xs uppercase tracking-wider mb-1">
            Equity
          </p>
          <p className="text-white text-lg font-mono font-semibold">
            {fmtCurrency(equity)}
          </p>
        </div>
        <div className="bg-surface-raised rounded-xl p-4">
          <p className="text-muted text-xs uppercase tracking-wider mb-1">
            High Water Mark
          </p>
          <p className="text-white text-lg font-mono font-semibold">
            {fmtCurrency(details?.high_water_mark ?? equity)}
          </p>
        </div>
        <div className="bg-surface-raised rounded-xl p-4">
          <p className="text-muted text-xs uppercase tracking-wider mb-1">
            Drawdown
          </p>
          <p
            className={`text-lg font-mono font-semibold ${
              drawdownPct > 0 ? "text-loss" : "text-profit"
            }`}
          >
            {fmtPct(-drawdownPct, 2)}
          </p>
        </div>
        <div className="bg-surface-raised rounded-xl p-4">
          <p className="text-muted text-xs uppercase tracking-wider mb-1">
            Trade Status
          </p>
          <StatusBadge
            label={riskState?.trade_blocked ? "Blocked" : "Allowed"}
            variant={riskState?.trade_blocked ? "danger" : "success"}
          />
        </div>
      </div>

      {/* Kill Switch Status */}
      <Panel
        title="Kill Switch"
        action={
          <StatusBadge
            label={riskState?.trade_blocked ? "ACTIVATED" : "Inactive"}
            variant={riskState?.trade_blocked ? "danger" : "success"}
            pulse={riskState?.trade_blocked ?? false}
          />
        }
      >
        <p className="text-muted text-sm leading-relaxed">
          {riskState?.trade_blocked
            ? "Trading has been automatically stopped because a risk limit was breached. No new positions will be opened until the limit resets or is manually cleared."
            : "All safety systems are operating normally. The kill switch will automatically activate if any risk limit is breached."}
        </p>
        {isPaused && details?.pause_until_date && (
          <p className="text-yellow-500 text-sm mt-2">
            Trading paused due to consecutive losing days. Resumes:{" "}
            <span className="font-mono">{details.pause_until_date}</span>
          </p>
        )}
      </Panel>

      {/* Drawdown Chart */}
      <Panel
        title="Drawdown History"
        subtitle="Decline from account peak over time"
      >
        <ChartContainer
          height={200}
          loading={equityCurve.loading}
          error={equityCurve.error}
          empty={!equityCurve.loading && drawdownData.length === 0}
          emptyMessage="No drawdown data yet. Start paper trading to track drawdowns."
        >
          {({ width, height }) => (
            <DrawdownChart
              data={drawdownData}
              width={width}
              height={height}
            />
          )}
        </ChartContainer>
      </Panel>

      {/* Risk Controls Grid */}
      <Panel
        title="Active Limits"
        subtitle="Current risk boundaries"
        action={
          !editing ? (
            <button
              onClick={startEditing}
              className="text-accent text-sm hover:underline"
            >
              Edit Limits
            </button>
          ) : undefined
        }
      >
        {/* Edit Mode: Limit Inputs */}
        {editing && (
          <div className="mb-6 p-4 bg-accent/5 border border-accent/20 rounded-xl">
            <p className="text-accent text-xs uppercase tracking-wider mb-3 font-semibold">
              Editing Risk Limits
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(LIMIT_BOUNDS).map(([key, bounds]) => (
                <div key={key}>
                  <label className="block text-muted text-xs mb-1">
                    {bounds.label}
                    {bounds.asPct && " (%)"}
                  </label>
                  <input
                    type="number"
                    value={limitDraft[key] ?? ""}
                    onChange={(e) =>
                      setLimitDraft((prev) => ({
                        ...prev,
                        [key]: e.target.value,
                      }))
                    }
                    min={bounds.min}
                    max={bounds.max}
                    step={bounds.step}
                    className="w-full bg-surface border border-surface-border rounded-lg px-3 py-1.5 text-white text-sm font-mono focus:outline-none focus:border-accent"
                  />
                  <p className="text-muted text-[10px] mt-0.5">
                    {bounds.min}{bounds.asPct ? "%" : ""} – {bounds.max}{bounds.asPct ? "%" : ""}
                  </p>
                </div>
              ))}
            </div>
            <div className="flex gap-2 mt-4">
              <button
                onClick={saveLimits}
                disabled={saving}
                className="px-4 py-1.5 bg-accent hover:bg-accent-hover text-white text-sm font-semibold rounded-lg transition-colors disabled:opacity-50"
              >
                {saving ? "Updating..." : "Update Limits"}
              </button>
              <button
                onClick={cancelEditing}
                className="px-4 py-1.5 bg-surface-overlay hover:bg-surface-border text-muted text-sm rounded-lg transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        <div className="space-y-4">
          {controls.map((control) => (
            <div
              key={control.name}
              className="p-4 bg-surface-overlay rounded-xl"
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="text-white font-semibold text-sm">
                      {control.name}
                    </h3>
                    <StatusBadge
                      label={statusLabel[control.status]}
                      variant={statusVariant[control.status]}
                    />
                  </div>
                  <p className="text-muted text-xs leading-relaxed">
                    {control.description}
                  </p>
                </div>
              </div>

              <div className="flex items-center justify-between text-xs mt-3">
                <span className="text-muted">
                  Current:{" "}
                  <span className="text-white font-mono">
                    {control.currentValue}
                  </span>
                </span>
                <span className="text-muted">
                  Limit:{" "}
                  <span className="text-white font-mono">{control.limit}</span>
                </span>
              </div>

              {/* Progress bar */}
              <div className="mt-2 h-1.5 bg-surface rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    control.status === "breached"
                      ? "bg-loss"
                      : control.status === "warning"
                        ? "bg-yellow-500"
                        : "bg-profit"
                  }`}
                  style={{ width: `${Math.min(control.pctUsed, 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </Panel>

      {/* Position Sizing */}
      <Panel
        title="Position Sizing"
        subtitle="How trade sizes are determined"
      >
        <div className="space-y-4 text-sm text-muted leading-relaxed">
          <p>
            <strong className="text-white">Base Size:</strong> Each trade risks
            a maximum of 2% of account equity. This means a losing trade can
            only reduce your account by 2% at most.
          </p>
          <p>
            <strong className="text-white">Volatility Adjustment:</strong> When
            the market is more volatile than usual, position sizes are
            automatically reduced. In calm markets, sizes return to normal (but
            never increase beyond the base).
          </p>
          <p>
            <strong className="text-white">Exposure Cap:</strong> The system
            will never invest more than{" "}
            {((limits?.max_portfolio_exposure_pct ?? 0.80) * 100).toFixed(0)}%
            of total equity at once. At least{" "}
            {(100 - (limits?.max_portfolio_exposure_pct ?? 0.80) * 100).toFixed(0)}%
            is always held in cash as a safety buffer.
          </p>
        </div>
      </Panel>

      {/* How Risk Management Works */}
      <Panel
        title="How This Protects You"
        subtitle="Plain-English explanation"
      >
        <div className="space-y-3 text-sm text-muted leading-relaxed">
          <p>
            Think of these controls as a set of circuit breakers. If any single
            limit is hit, trading automatically pauses to prevent further losses.
          </p>
          <p>
            <strong className="text-white">Daily limits</strong> prevent a
            single bad day from causing significant damage.{" "}
            <strong className="text-white">Weekly and monthly limits</strong>{" "}
            catch sustained losing periods.{" "}
            <strong className="text-white">Drawdown from peak</strong> ensures
            that hard-won gains are protected.
          </p>
          <p>
            All limits are calculated as percentages of your account value, so
            they scale automatically as your account grows.
          </p>
        </div>
      </Panel>
    </div>
  );
}
