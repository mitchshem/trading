"use client";

import { useEffect, useState, useCallback } from "react";
import MetricCard from "@/components/MetricCard";
import Panel from "@/components/Panel";
import StatusBadge from "@/components/StatusBadge";
import ConnectionStatus from "@/components/ConnectionStatus";
import { useSSE } from "@/lib/useSSE";
import {
  getLiveStatus,
  getMetrics,
  getStrategies,
  switchStrategy,
  updateParams,
  LiveStatus,
  Metrics,
  StrategyInfo,
} from "@/lib/api";
import { fmtPct, fmtNumber } from "@/lib/format";

export default function StrategyPage() {
  const [liveStatus, setLiveStatus] = useState<LiveStatus | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Switch strategy state
  const [switching, setSwitching] = useState(false);
  const [confirmForce, setConfirmForce] = useState<string | null>(null);

  // Param editing state
  const [editingParams, setEditingParams] = useState(false);
  const [paramDraft, setParamDraft] = useState<Record<string, string>>({});
  const [savingParams, setSavingParams] = useState(false);

  // SSE for real-time updates
  const { isConnected, lastEvent } = useSSE([
    "strategy_switched",
    "params_updated",
    "loop_status",
  ]);

  const load = useCallback(async () => {
    try {
      const [ls, m, s] = await Promise.all([
        getLiveStatus().catch(() => null),
        getMetrics(),
        getStrategies(),
      ]);
      setLiveStatus(ls);
      setMetrics(m);
      setStrategies(s.strategies);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    }
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, [load]);

  // Re-fetch on SSE events
  useEffect(() => {
    if (!lastEvent) return;
    load();
  }, [lastEvent, load]);

  const activeStrategyName = liveStatus?.strategy_name ?? "ema_trend_v1";
  const isRunning = liveStatus?.status === "running";
  const activeStrategy = strategies.find((s) => s.name === activeStrategyName);

  // Initialize param draft when entering edit mode
  function startEditParams() {
    const current = liveStatus?.strategy_params ?? {};
    const draft: Record<string, string> = {};
    for (const [key, val] of Object.entries(current)) {
      draft[key] = String(val);
    }
    setParamDraft(draft);
    setEditingParams(true);
    setSuccess(null);
    setError(null);
  }

  function cancelEditParams() {
    setEditingParams(false);
    setParamDraft({});
  }

  async function saveParams() {
    setSavingParams(true);
    setError(null);
    setSuccess(null);
    try {
      const parsed: Record<string, unknown> = {};
      for (const [key, val] of Object.entries(paramDraft)) {
        const num = parseFloat(val);
        if (val === "true") parsed[key] = true;
        else if (val === "false") parsed[key] = false;
        else if (!isNaN(num)) parsed[key] = num;
        else parsed[key] = val;
      }
      const result = await updateParams(parsed);
      if (result.error) {
        setError(result.error);
      } else {
        setSuccess("Parameters updated successfully.");
        setEditingParams(false);
        load();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update params");
    } finally {
      setSavingParams(false);
    }
  }

  async function handleSwitchStrategy(name: string, force = false) {
    setSwitching(true);
    setError(null);
    setSuccess(null);
    setConfirmForce(null);
    try {
      const result = await switchStrategy({
        strategy_name: name,
        force,
      });
      if (result.error) {
        // If it's a position warning, offer force switch
        if (result.error.includes("open position") && !force) {
          setConfirmForce(name);
          setError(result.error);
        } else {
          setError(result.error);
        }
      } else {
        setSuccess(
          `Switched to ${name}.${result.warning ? ` Warning: ${result.warning}` : ""}`
        );
        setLiveStatus(result.status);
        load();
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to switch strategy";
      // Check if it's a position warning from the backend (400 error)
      if (msg.includes("open position") && !force) {
        setConfirmForce(name);
      }
      setError(msg);
    } finally {
      setSwitching(false);
    }
  }

  // Helper to get range info for a param
  function getParamRange(paramName: string): { min: number; max: number; step: number } | null {
    if (!activeStrategy?.param_ranges?.[paramName]) return null;
    const range = activeStrategy.param_ranges[paramName];
    if (range.length < 2) return null;
    const min = Math.min(...range);
    const max = Math.max(...range);
    // Infer step from range values
    const step = range.length > 1 ? Math.abs(range[1] - range[0]) : 1;
    return { min, max, step };
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Strategy</h1>
          <p className="text-muted text-sm mt-1">
            View and manage your trading strategies
          </p>
        </div>
        <div className="flex items-center gap-3">
          <ConnectionStatus isConnected={isConnected} />
          <StatusBadge
            label={isRunning ? "Running" : "Idle"}
            variant={isRunning ? "success" : "neutral"}
            pulse={isRunning}
          />
        </div>
      </div>

      {/* Alerts */}
      {error && (
        <div className="bg-loss/10 border border-loss/30 rounded-xl p-4 text-loss text-sm">
          {error}
          {confirmForce && (
            <button
              onClick={() => handleSwitchStrategy(confirmForce, true)}
              disabled={switching}
              className="ml-3 px-3 py-1 bg-loss hover:bg-loss/80 text-white text-xs font-semibold rounded-lg transition-colors disabled:opacity-50"
            >
              {switching ? "Switching..." : "Force Switch (close position)"}
            </button>
          )}
        </div>
      )}
      {success && (
        <div className="bg-profit/10 border border-profit/30 rounded-xl p-4 text-profit text-sm">
          {success}
        </div>
      )}

      {/* Active Strategy Highlight */}
      <Panel
        title="Active Strategy"
        action={
          <StatusBadge
            label={isRunning ? "Running" : "Idle"}
            variant={isRunning ? "success" : "neutral"}
            pulse={isRunning}
          />
        }
      >
        <div className="space-y-4">
          <div>
            <h3 className="text-white text-lg font-semibold">
              {activeStrategy?.name ?? activeStrategyName}
            </h3>
            <p className="text-muted text-sm mt-1">
              {activeStrategy?.description ??
                "Custom strategy running on the backend."}
            </p>
          </div>

          {/* Strategy params - editable or read-only */}
          {liveStatus?.strategy_params &&
            Object.keys(liveStatus.strategy_params).length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-muted text-xs uppercase tracking-wider">
                    Parameters
                  </p>
                  {isRunning && !editingParams && (
                    <button
                      onClick={startEditParams}
                      className="text-accent text-xs hover:underline"
                    >
                      Edit
                    </button>
                  )}
                </div>

                {editingParams ? (
                  <div className="space-y-3">
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      {Object.entries(paramDraft).map(([key, val]) => {
                        const range = getParamRange(key);
                        return (
                          <div key={key}>
                            <label className="block text-muted text-xs mb-1 font-mono">
                              {key}
                            </label>
                            <input
                              type="number"
                              value={val}
                              onChange={(e) =>
                                setParamDraft((prev) => ({
                                  ...prev,
                                  [key]: e.target.value,
                                }))
                              }
                              min={range?.min}
                              max={range?.max}
                              step={range?.step}
                              className="w-full bg-surface border border-surface-border rounded-lg px-3 py-1.5 text-white text-sm font-mono focus:outline-none focus:border-accent"
                            />
                            {range && (
                              <p className="text-muted text-[10px] mt-0.5">
                                Range: {range.min} - {range.max}
                              </p>
                            )}
                          </div>
                        );
                      })}
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={saveParams}
                        disabled={savingParams}
                        className="px-4 py-1.5 bg-accent hover:bg-accent-hover text-white text-sm font-semibold rounded-lg transition-colors disabled:opacity-50"
                      >
                        {savingParams ? "Saving..." : "Apply"}
                      </button>
                      <button
                        onClick={cancelEditParams}
                        className="px-4 py-1.5 bg-surface-overlay hover:bg-surface-border text-muted text-sm rounded-lg transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(liveStatus.strategy_params).map(
                      ([key, val]) => (
                        <span
                          key={key}
                          className="bg-surface-overlay px-2.5 py-1 rounded text-xs text-white font-mono"
                        >
                          {key}: {String(val)}
                        </span>
                      )
                    )}
                  </div>
                )}
              </div>
            )}

          {/* Performance metrics for active strategy */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2">
            <MetricCard
              label="Total Return"
              value={fmtPct(metrics?.net_return_pct)}
              trend={
                (metrics?.net_return_pct ?? 0) > 0
                  ? "up"
                  : (metrics?.net_return_pct ?? 0) < 0
                    ? "down"
                    : "neutral"
              }
              size="sm"
            />
            <MetricCard
              label="Win Rate"
              value={
                metrics
                  ? `${(metrics.win_rate * 100).toFixed(0)}%`
                  : "--"
              }
              size="sm"
            />
            <MetricCard
              label="Profit Factor"
              value={fmtNumber(metrics?.profit_factor)}
              size="sm"
            />
            <MetricCard
              label="Total Trades"
              value={metrics?.total_trades ?? "--"}
              size="sm"
            />
          </div>
        </div>
      </Panel>

      {/* All Strategies */}
      <Panel title="Available Strategies" subtitle="Validated strategy library">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {strategies.map((strat) => {
            const isActive = strat.name === activeStrategyName;
            return (
              <div
                key={strat.name}
                className={`p-4 rounded-xl border transition-colors ${
                  isActive
                    ? "bg-accent/5 border-accent/30"
                    : "bg-surface-overlay border-surface-border"
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <h3 className="text-white font-semibold">{strat.name}</h3>
                    <span className="text-muted text-xs">
                      {Object.keys(strat.default_params).length} parameters
                    </span>
                  </div>
                  {isActive ? (
                    <StatusBadge label="Active" variant="accent" />
                  ) : (
                    <button
                      onClick={() => handleSwitchStrategy(strat.name)}
                      disabled={switching || !isRunning}
                      className="px-3 py-1 bg-accent/20 hover:bg-accent/30 text-accent text-xs font-semibold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      title={
                        !isRunning
                          ? "Start trading first to switch strategies"
                          : undefined
                      }
                    >
                      {switching ? "..." : "Activate"}
                    </button>
                  )}
                </div>
                <p className="text-muted text-sm leading-relaxed mb-3">
                  {strat.description}
                </p>
                {/* Default params preview */}
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(strat.default_params).map(([key, val]) => (
                    <span
                      key={key}
                      className="bg-surface px-2 py-0.5 rounded text-[10px] text-muted font-mono"
                    >
                      {key}={String(val)}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
          {strategies.length === 0 && (
            <p className="text-muted text-sm py-6 text-center col-span-2">
              Loading strategies...
            </p>
          )}
        </div>
      </Panel>

      {/* How Strategies Work */}
      <Panel
        title="How It Works"
        subtitle="Understanding the trading process"
      >
        <div className="space-y-4 text-sm text-muted leading-relaxed">
          <p>
            <strong className="text-white">1. Watch the Market.</strong> The
            system checks the price at the end of each trading day.
          </p>
          <p>
            <strong className="text-white">2. Evaluate Conditions.</strong>{" "}
            The active strategy looks at recent price trends and decides whether
            to buy, hold, or sell.
          </p>
          <p>
            <strong className="text-white">3. Manage Risk First.</strong>{" "}
            Before any trade, the system checks daily loss limits, position
            size rules, and safety kill-switches. If any limit is breached,
            no trade happens.
          </p>
          <p>
            <strong className="text-white">4. Execute Next Day.</strong>{" "}
            All buy/sell decisions are queued and executed at the next market
            open. This prevents acting on stale data.
          </p>
        </div>
      </Panel>
    </div>
  );
}
