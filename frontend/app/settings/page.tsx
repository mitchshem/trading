"use client";

import { useEffect, useState, useCallback } from "react";
import Panel from "@/components/Panel";
import StatusBadge from "@/components/StatusBadge";
import ConnectionStatus from "@/components/ConnectionStatus";
import { useSSE } from "@/lib/useSSE";
import {
  getLiveStatus,
  startLive,
  stopLive,
  getStrategies,
  getAlpacaStatus,
  testAlpacaConnection,
  getPromotionStatus,
  getNotificationPrefs,
  updateNotificationPrefs,
  testEmailNotification,
  LiveStatus,
  StrategyInfo,
  AlpacaStatus,
  AlpacaTestResult,
  PromotionStatus,
  NotificationPrefs,
  NotificationPrefsUpdate,
} from "@/lib/api";
import { fmtCurrency, fmtDateTime } from "@/lib/format";

const EVENT_CATEGORIES = [
  { key: "trade_executed", label: "Trade Executed", severity: "info", desc: "BUY, EXIT, or STOP_LOSS actions" },
  { key: "kill_switch", label: "Kill Switch", severity: "critical", desc: "Kill switch activated" },
  { key: "anomaly_detected", label: "Anomaly", severity: "warning", desc: "Data anomaly blocks trading" },
  { key: "risk_limit_breached", label: "Risk Breach", severity: "critical", desc: "Drawdown or loss limit hit" },
  { key: "loop_status_change", label: "Loop Status", severity: "warning", desc: "Trading loop started/stopped" },
  { key: "daily_summary", label: "Daily Summary", severity: "info", desc: "End-of-day equity snapshot" },
  { key: "system_error", label: "System Error", severity: "critical", desc: "Unrecoverable errors" },
];

export default function SettingsPage() {
  const [liveStatus, setLiveStatus] = useState<LiveStatus | null>(null);
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [alpacaStatus, setAlpacaStatus] = useState<AlpacaStatus | null>(null);
  const [alpacaTest, setAlpacaTest] = useState<AlpacaTestResult | null>(null);
  const [testingAlpaca, setTestingAlpaca] = useState(false);
  const [promotion, setPromotion] = useState<PromotionStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Form state for starting live trading
  const [symbol, setSymbol] = useState("SPY");
  const [strategy, setStrategy] = useState("ema_trend_v1");
  const [initialEquity, setInitialEquity] = useState("100000");

  // Notification preferences state
  const [notifPrefs, setNotifPrefs] = useState<NotificationPrefs | null>(null);
  const [notifDraft, setNotifDraft] = useState<NotificationPrefsUpdate>({});
  const [savingNotifs, setSavingNotifs] = useState(false);
  const [testingEmail, setTestingEmail] = useState(false);
  const [emailTestResult, setEmailTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [smtpExpanded, setSmtpExpanded] = useState(false);

  // SSE for real-time updates
  const { isConnected, lastEvent } = useSSE([
    "loop_status",
    "strategy_switched",
    "notification_fired",
  ]);

  const load = useCallback(async () => {
    try {
      const [ls, s, alpStatus, promo, nPrefs] = await Promise.all([
        getLiveStatus().catch(() => null),
        getStrategies().catch(() => ({ strategies: [] })),
        getAlpacaStatus().catch(() => null),
        getPromotionStatus().catch(() => null),
        getNotificationPrefs().catch(() => null),
      ]);
      setLiveStatus(ls);
      setStrategies(s.strategies);
      setAlpacaStatus(alpStatus);
      setPromotion(promo);
      if (nPrefs) setNotifPrefs(nPrefs);
    } catch {
      // Silently fail — server might not be running
    }
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(load, 10000);
    return () => clearInterval(interval);
  }, [load]);

  // Re-fetch on SSE events
  useEffect(() => {
    if (!lastEvent) return;
    load();
  }, [lastEvent, load]);

  const isRunning = liveStatus?.status === "running";

  async function handleStart() {
    setIsLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const result = await startLive({
        symbol,
        strategy_name: strategy,
        initial_equity: parseFloat(initialEquity),
      });
      setLiveStatus(result.status);
      setSuccess("Paper trading started successfully.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleStop() {
    setIsLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const result = await stopLive();
      setLiveStatus(result.status);
      setSuccess("Paper trading stopped.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to stop");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleTestAlpaca() {
    setTestingAlpaca(true);
    setAlpacaTest(null);
    try {
      const result = await testAlpacaConnection();
      setAlpacaTest(result);
    } catch (err) {
      setAlpacaTest({
        success: false,
        message: err instanceof Error ? err.message : "Test failed",
      });
    } finally {
      setTestingAlpaca(false);
    }
  }

  // Notification handlers
  async function saveNotificationPrefs() {
    setSavingNotifs(true);
    setError(null);
    setSuccess(null);
    try {
      const result = await updateNotificationPrefs(notifDraft);
      if (result.prefs) setNotifPrefs(result.prefs);
      setNotifDraft({});
      setSuccess("Notification preferences saved.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save notifications");
    } finally {
      setSavingNotifs(false);
    }
  }

  async function handleTestEmail() {
    setTestingEmail(true);
    setEmailTestResult(null);
    try {
      const result = await testEmailNotification();
      setEmailTestResult(result);
    } catch (err) {
      setEmailTestResult({
        success: false,
        message: err instanceof Error ? err.message : "Test failed",
      });
    } finally {
      setTestingEmail(false);
    }
  }

  function requestBrowserPermission() {
    if (typeof window === "undefined" || !("Notification" in window)) return;
    Notification.requestPermission().then(() => {
      // Force re-render by touching state
      setNotifDraft((prev) => ({ ...prev }));
    });
  }

  // Merged draft values
  const emailEnabled = notifDraft.email_enabled ?? notifPrefs?.email_enabled ?? false;
  const browserEnabled = notifDraft.browser_enabled ?? notifPrefs?.browser_enabled ?? true;
  const minSeverity = notifDraft.min_severity ?? notifPrefs?.min_severity ?? "info";
  const emailCats = { ...(notifPrefs?.email_categories ?? {}), ...(notifDraft.email_categories ?? {}) };
  const browserCats = { ...(notifPrefs?.browser_categories ?? {}), ...(notifDraft.browser_categories ?? {}) };
  const hasDraftChanges = Object.keys(notifDraft).length > 0;
  const browserPermission = typeof window !== "undefined" && "Notification" in window
    ? Notification.permission
    : "default";

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Settings</h1>
          <p className="text-muted text-sm mt-1">
            Configure and control the trading system
          </p>
        </div>
        <div className="flex items-center gap-3">
          <ConnectionStatus isConnected={isConnected} />
          <StatusBadge
            label={isRunning ? "Running" : "Stopped"}
            variant={isRunning ? "success" : "neutral"}
            pulse={isRunning}
          />
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

      {/* Trading Mode */}
      <Panel
        title="Trading Mode"
        action={
          <div className="flex items-center gap-3">
            <StatusBadge label="Paper Trading" variant="accent" />
            <StatusBadge
              label={isRunning ? "Running" : "Stopped"}
              variant={isRunning ? "success" : "neutral"}
              pulse={isRunning}
            />
          </div>
        }
      >
        <div className="bg-accent/5 border border-accent/20 rounded-xl p-4 mb-6">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-accent text-lg font-bold">Paper Trading Mode</span>
          </div>
          <p className="text-muted text-sm leading-relaxed">
            All trades are simulated with virtual money. No real capital is at
            risk. This mode uses the same strategy logic and risk controls that
            would be used with real money, providing an accurate preview of
            system behavior.
          </p>
        </div>

        {liveStatus && isRunning && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div>
              <p className="text-muted text-xs uppercase tracking-wider mb-1">Symbol</p>
              <p className="text-white font-mono">{liveStatus.symbol}</p>
            </div>
            <div>
              <p className="text-muted text-xs uppercase tracking-wider mb-1">Strategy</p>
              <p className="text-white font-mono">{liveStatus.strategy_name}</p>
            </div>
            <div>
              <p className="text-muted text-xs uppercase tracking-wider mb-1">Equity</p>
              <p className="text-white font-mono">{fmtCurrency(liveStatus.broker.equity)}</p>
            </div>
            <div>
              <p className="text-muted text-xs uppercase tracking-wider mb-1">Started At</p>
              <p className="text-white text-sm">{fmtDateTime(liveStatus.started_at)}</p>
            </div>
          </div>
        )}

        {isRunning ? (
          <button
            onClick={handleStop}
            disabled={isLoading}
            className="w-full sm:w-auto px-6 py-3 bg-loss hover:bg-loss/80 text-white font-semibold rounded-xl transition-colors disabled:opacity-50"
          >
            {isLoading ? "Stopping..." : "Stop Paper Trading"}
          </button>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-muted text-xs uppercase tracking-wider mb-1.5">Symbol</label>
                <input
                  type="text"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  className="w-full bg-surface border border-surface-border rounded-lg px-3 py-2 text-white text-sm font-mono focus:outline-none focus:border-accent"
                />
              </div>
              <div>
                <label className="block text-muted text-xs uppercase tracking-wider mb-1.5">Strategy</label>
                <select
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                  className="w-full bg-surface border border-surface-border rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-accent"
                >
                  {strategies.length > 0
                    ? strategies.map((s) => (
                        <option key={s.name} value={s.name}>{s.name}</option>
                      ))
                    : (
                      <>
                        <option value="ema_trend_v1">ema_trend_v1</option>
                        <option value="momentum_breakout_v1">momentum_breakout_v1</option>
                        <option value="mean_reversion_v1">mean_reversion_v1</option>
                        <option value="dual_momentum_v1">dual_momentum_v1</option>
                      </>
                    )}
                </select>
              </div>
              <div>
                <label className="block text-muted text-xs uppercase tracking-wider mb-1.5">Initial Equity ($)</label>
                <input
                  type="number"
                  value={initialEquity}
                  onChange={(e) => setInitialEquity(e.target.value)}
                  className="w-full bg-surface border border-surface-border rounded-lg px-3 py-2 text-white text-sm font-mono focus:outline-none focus:border-accent"
                />
              </div>
            </div>
            <button
              onClick={handleStart}
              disabled={isLoading}
              className="w-full sm:w-auto px-6 py-3 bg-accent hover:bg-accent-hover text-white font-semibold rounded-xl transition-colors disabled:opacity-50"
            >
              {isLoading ? "Starting..." : "Start Paper Trading"}
            </button>
          </div>
        )}
      </Panel>

      {/* Data Connection (Alpaca) */}
      <Panel title="Data Connection" subtitle="Market data provider status">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-white font-semibold">Alpaca Markets</p>
              <p className="text-muted text-xs">Real-time and historical market data</p>
            </div>
            <StatusBadge
              label={alpacaStatus?.configured ? "Configured" : "Not Configured"}
              variant={alpacaStatus?.configured ? "success" : "neutral"}
            />
          </div>

          {alpacaStatus?.configured && alpacaStatus.api_key_masked && (
            <div className="flex justify-between py-2 border-t border-surface-border">
              <span className="text-muted text-sm">API Key</span>
              <span className="text-white text-sm font-mono">{alpacaStatus.api_key_masked}</span>
            </div>
          )}

          <div className="flex items-center gap-3">
            <button
              onClick={handleTestAlpaca}
              disabled={testingAlpaca || !alpacaStatus?.configured}
              className="px-4 py-2 bg-surface-overlay hover:bg-surface-border text-white text-sm rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {testingAlpaca ? "Testing..." : "Test Connection"}
            </button>
            {alpacaTest && (
              <span className={`text-sm ${alpacaTest.success ? "text-profit" : "text-loss"}`}>
                {alpacaTest.success ? "Connected" : "Failed"}: {alpacaTest.message}
              </span>
            )}
          </div>

          {!alpacaStatus?.configured && (
            <div className="bg-surface-overlay rounded-lg p-3 text-muted text-xs leading-relaxed">
              <p className="font-semibold text-white text-xs mb-1">How to configure:</p>
              <p>
                Set <code className="text-accent">ALPACA_API_KEY</code> and{" "}
                <code className="text-accent">ALPACA_SECRET_KEY</code> environment
                variables before starting the backend. Get free API keys at{" "}
                <span className="text-accent">alpaca.markets</span>. Without Alpaca,
                the system falls back to Yahoo Finance (delayed data).
              </p>
            </div>
          )}
        </div>
      </Panel>

      {/* Notifications */}
      <Panel title="Notifications" subtitle="Email and browser alerts for trading events">
        {notifPrefs ? (
          <div className="space-y-6">
            {/* Master Toggles */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="flex items-center justify-between p-3 bg-surface-overlay rounded-lg">
                <div>
                  <p className="text-white text-sm font-semibold">Email Notifications</p>
                  <p className="text-muted text-xs">
                    {notifPrefs.smtp_configured ? "SMTP configured" : "SMTP not configured"}
                  </p>
                </div>
                <button
                  onClick={() => setNotifDraft((d) => ({ ...d, email_enabled: !emailEnabled }))}
                  className={`w-12 h-6 rounded-full transition-colors relative ${
                    emailEnabled ? "bg-accent" : "bg-surface-border"
                  }`}
                >
                  <span
                    className={`block w-5 h-5 bg-white rounded-full transition-transform absolute top-0.5 ${
                      emailEnabled ? "translate-x-6" : "translate-x-0.5"
                    }`}
                  />
                </button>
              </div>

              <div className="flex items-center justify-between p-3 bg-surface-overlay rounded-lg">
                <div>
                  <p className="text-white text-sm font-semibold">Browser Notifications</p>
                  <p className="text-muted text-xs">
                    Permission: {browserPermission}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {browserPermission !== "granted" && (
                    <button
                      onClick={requestBrowserPermission}
                      className="px-2 py-1 bg-accent/20 text-accent text-xs rounded-lg hover:bg-accent/30 transition-colors"
                    >
                      Allow
                    </button>
                  )}
                  <button
                    onClick={() => setNotifDraft((d) => ({ ...d, browser_enabled: !browserEnabled }))}
                    className={`w-12 h-6 rounded-full transition-colors relative ${
                      browserEnabled ? "bg-accent" : "bg-surface-border"
                    }`}
                  >
                    <span
                      className={`block w-5 h-5 bg-white rounded-full transition-transform absolute top-0.5 ${
                        browserEnabled ? "translate-x-6" : "translate-x-0.5"
                      }`}
                    />
                  </button>
                </div>
              </div>
            </div>

            {/* SMTP Configuration (collapsible) */}
            {emailEnabled && (
              <div className="border border-surface-border rounded-xl overflow-hidden">
                <button
                  onClick={() => setSmtpExpanded(!smtpExpanded)}
                  className="w-full flex items-center justify-between p-3 text-left hover:bg-surface-overlay transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-white text-sm font-semibold">SMTP Configuration</span>
                    {notifPrefs.smtp_configured && (
                      <StatusBadge label="Configured" variant="success" />
                    )}
                  </div>
                  <span className="text-muted text-xs">{smtpExpanded ? "Hide" : "Show"}</span>
                </button>
                {smtpExpanded && (
                  <div className="p-4 pt-0 space-y-3">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div>
                        <label className="block text-muted text-xs mb-1">Email Address</label>
                        <input
                          type="email"
                          defaultValue={notifPrefs.email_address ?? ""}
                          onChange={(e) => setNotifDraft((d) => ({ ...d, email_address: e.target.value }))}
                          placeholder="you@example.com"
                          className="w-full bg-surface border border-surface-border rounded-lg px-3 py-1.5 text-white text-sm focus:outline-none focus:border-accent"
                        />
                      </div>
                      <div>
                        <label className="block text-muted text-xs mb-1">SMTP Host</label>
                        <input
                          type="text"
                          defaultValue={notifPrefs.smtp_host ?? ""}
                          onChange={(e) => setNotifDraft((d) => ({ ...d, smtp_host: e.target.value }))}
                          placeholder="smtp.gmail.com"
                          className="w-full bg-surface border border-surface-border rounded-lg px-3 py-1.5 text-white text-sm focus:outline-none focus:border-accent"
                        />
                      </div>
                      <div>
                        <label className="block text-muted text-xs mb-1">SMTP Port</label>
                        <input
                          type="number"
                          defaultValue={notifPrefs.smtp_port ?? 587}
                          onChange={(e) => setNotifDraft((d) => ({ ...d, smtp_port: parseInt(e.target.value) || 587 }))}
                          className="w-full bg-surface border border-surface-border rounded-lg px-3 py-1.5 text-white text-sm font-mono focus:outline-none focus:border-accent"
                        />
                      </div>
                      <div>
                        <label className="block text-muted text-xs mb-1">Username</label>
                        <input
                          type="text"
                          onChange={(e) => setNotifDraft((d) => ({ ...d, smtp_user: e.target.value }))}
                          placeholder="your-username"
                          className="w-full bg-surface border border-surface-border rounded-lg px-3 py-1.5 text-white text-sm focus:outline-none focus:border-accent"
                        />
                      </div>
                      <div>
                        <label className="block text-muted text-xs mb-1">Password</label>
                        <input
                          type="password"
                          onChange={(e) => setNotifDraft((d) => ({ ...d, smtp_password: e.target.value }))}
                          placeholder="app-password"
                          className="w-full bg-surface border border-surface-border rounded-lg px-3 py-1.5 text-white text-sm focus:outline-none focus:border-accent"
                        />
                      </div>
                      <div className="flex items-center gap-2 pt-5">
                        <label className="text-muted text-xs">Use TLS</label>
                        <button
                          onClick={() => setNotifDraft((d) => ({
                            ...d,
                            smtp_use_tls: !(notifDraft.smtp_use_tls ?? notifPrefs.smtp_use_tls),
                          }))}
                          className={`w-10 h-5 rounded-full transition-colors relative ${
                            (notifDraft.smtp_use_tls ?? notifPrefs.smtp_use_tls) ? "bg-accent" : "bg-surface-border"
                          }`}
                        >
                          <span
                            className={`block w-4 h-4 bg-white rounded-full transition-transform absolute top-0.5 ${
                              (notifDraft.smtp_use_tls ?? notifPrefs.smtp_use_tls) ? "translate-x-5" : "translate-x-0.5"
                            }`}
                          />
                        </button>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 pt-2">
                      <button
                        onClick={handleTestEmail}
                        disabled={testingEmail}
                        className="px-4 py-1.5 bg-surface-overlay hover:bg-surface-border text-white text-sm rounded-lg transition-colors disabled:opacity-50"
                      >
                        {testingEmail ? "Sending..." : "Test Email"}
                      </button>
                      {emailTestResult && (
                        <span className={`text-sm ${emailTestResult.success ? "text-profit" : "text-loss"}`}>
                          {emailTestResult.message}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Severity Filter */}
            <div className="flex items-center gap-3">
              <label className="text-muted text-sm">Minimum severity:</label>
              <select
                value={minSeverity}
                onChange={(e) => setNotifDraft((d) => ({ ...d, min_severity: e.target.value }))}
                className="bg-surface border border-surface-border rounded-lg px-3 py-1.5 text-white text-sm focus:outline-none focus:border-accent"
              >
                <option value="info">Info (all events)</option>
                <option value="warning">Warning and above</option>
                <option value="critical">Critical only</option>
              </select>
            </div>

            {/* Event Categories */}
            <div>
              <p className="text-muted text-xs uppercase tracking-wider mb-3">Event Categories</p>
              <div className="space-y-2">
                {EVENT_CATEGORIES.map((cat) => (
                  <div
                    key={cat.key}
                    className="flex items-center justify-between p-3 bg-surface-overlay rounded-lg"
                  >
                    <div className="flex items-center gap-3 flex-1">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-white text-sm">{cat.label}</span>
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-semibold ${
                              cat.severity === "critical"
                                ? "bg-loss/20 text-loss"
                                : cat.severity === "warning"
                                  ? "bg-yellow-500/20 text-yellow-500"
                                  : "bg-accent/20 text-accent"
                            }`}
                          >
                            {cat.severity}
                          </span>
                        </div>
                        <p className="text-muted text-xs">{cat.desc}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <label className="flex items-center gap-1.5 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={emailCats[cat.key] ?? false}
                          onChange={(e) =>
                            setNotifDraft((d) => ({
                              ...d,
                              email_categories: {
                                ...(d.email_categories ?? {}),
                                [cat.key]: e.target.checked,
                              },
                            }))
                          }
                          className="w-4 h-4 rounded bg-surface border-surface-border accent-accent"
                        />
                        <span className="text-muted text-xs">Email</span>
                      </label>
                      <label className="flex items-center gap-1.5 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={browserCats[cat.key] ?? false}
                          onChange={(e) =>
                            setNotifDraft((d) => ({
                              ...d,
                              browser_categories: {
                                ...(d.browser_categories ?? {}),
                                [cat.key]: e.target.checked,
                              },
                            }))
                          }
                          className="w-4 h-4 rounded bg-surface border-surface-border accent-accent"
                        />
                        <span className="text-muted text-xs">Browser</span>
                      </label>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Save button */}
            {hasDraftChanges && (
              <div className="flex items-center gap-3">
                <button
                  onClick={saveNotificationPrefs}
                  disabled={savingNotifs}
                  className="px-6 py-2 bg-accent hover:bg-accent-hover text-white text-sm font-semibold rounded-lg transition-colors disabled:opacity-50"
                >
                  {savingNotifs ? "Saving..." : "Save Preferences"}
                </button>
                <button
                  onClick={() => setNotifDraft({})}
                  className="px-4 py-2 bg-surface-overlay hover:bg-surface-border text-muted text-sm rounded-lg transition-colors"
                >
                  Discard
                </button>
              </div>
            )}
          </div>
        ) : (
          <p className="text-muted text-sm py-4 text-center">Loading notification preferences...</p>
        )}
      </Panel>

      {/* Promotion Readiness */}
      <Panel title="Promotion Readiness" subtitle="Progress toward live capital trading">
        {promotion ? (
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-muted text-sm">
                  {promotion.checks_passed} / {promotion.checks_total} checks passed
                </span>
                <span className="text-white text-sm font-mono font-semibold">
                  {promotion.readiness_pct.toFixed(0)}%
                </span>
              </div>
              <div className="h-2 bg-surface rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    promotion.promoted ? "bg-profit" : promotion.readiness_pct >= 50 ? "bg-yellow-500" : "bg-loss"
                  }`}
                  style={{ width: `${promotion.readiness_pct}%` }}
                />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <StatusBadge
                label={promotion.promoted ? "Ready for Live" : "Not Ready"}
                variant={promotion.promoted ? "success" : "warning"}
              />
              {promotion.promoted && <span className="text-profit text-sm">All promotion criteria met!</span>}
            </div>
            <div className="space-y-2">
              {promotion.reasons.map((reason, i) => {
                const isPassing = promotion.promoted;
                return (
                  <div key={i} className="flex items-start gap-2 text-sm">
                    <span className={isPassing ? "text-profit" : "text-loss"}>
                      {isPassing ? "\u2713" : "\u2717"}
                    </span>
                    <span className="text-muted">{reason}</span>
                  </div>
                );
              })}
              {promotion.reasons.length === 0 && promotion.promoted && (
                <p className="text-profit text-sm">All promotion checks are passing.</p>
              )}
              {promotion.reasons.length === 0 && !promotion.promoted && (
                <p className="text-muted text-sm">Start paper trading to begin tracking promotion criteria.</p>
              )}
            </div>
            {Object.keys(promotion.metrics).length > 0 && (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-2 border-t border-surface-border">
                {Object.entries(promotion.metrics).map(([key, val]) => (
                  <div key={key}>
                    <p className="text-muted text-[10px] uppercase tracking-wider">{key.replace(/_/g, " ")}</p>
                    <p className="text-white text-sm font-mono">
                      {typeof val === "number" ? val.toFixed(2) : String(val)}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <p className="text-muted text-sm py-4 text-center">Loading promotion status...</p>
        )}
      </Panel>

      {/* System Information */}
      <Panel title="System Information">
        <div className="space-y-3 text-sm">
          <div className="flex justify-between py-2 border-b border-surface-border">
            <span className="text-muted">Backend</span>
            <span className="text-white font-mono">localhost:8000</span>
          </div>
          <div className="flex justify-between py-2 border-b border-surface-border">
            <span className="text-muted">Data Source</span>
            <span className="text-white">
              {alpacaStatus?.configured ? "Alpaca Markets (real-time)" : "Yahoo Finance (delayed)"}
            </span>
          </div>
          <div className="flex justify-between py-2 border-b border-surface-border">
            <span className="text-muted">Real-time Feed</span>
            <span className={isConnected ? "text-profit" : "text-muted"}>
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
          <div className="flex justify-between py-2 border-b border-surface-border">
            <span className="text-muted">Trading Hours</span>
            <span className="text-white">9:30 AM - 4:00 PM ET</span>
          </div>
          <div className="flex justify-between py-2 border-b border-surface-border">
            <span className="text-muted">Order Execution</span>
            <span className="text-white">Next-candle open (no lookahead bias)</span>
          </div>
          <div className="flex justify-between py-2">
            <span className="text-muted">Commission Model</span>
            <span className="text-white">$1.00 per trade + 0.01% slippage</span>
          </div>
        </div>
      </Panel>

      {/* Live Capital Notice */}
      <Panel title="About Live Capital">
        <div className="bg-yellow-500/5 border border-yellow-500/20 rounded-xl p-4">
          <p className="text-yellow-400 font-semibold text-sm mb-2">
            {promotion?.promoted ? "Promotion Criteria Met" : "Not Yet Available"}
          </p>
          <p className="text-muted text-sm leading-relaxed">
            {promotion?.promoted
              ? "All promotion criteria have been met. The system has demonstrated consistent paper trading performance across multiple time windows. Review the Promotion Readiness panel above for details."
              : "Live capital trading requires passing all promotion rules: minimum 20 trading days of paper trading, positive walk-forward performance across multiple time windows, and stable risk metrics. The system will notify you when promotion criteria are met."}
          </p>
        </div>
      </Panel>
    </div>
  );
}
