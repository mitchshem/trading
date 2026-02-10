/**
 * Backend API helper functions.
 * All REST calls go through here for centralized error handling.
 */

const API_BASE = "http://localhost:8000";

export async function fetchAPI<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function postAPI<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function patchAPI<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

// ── Type definitions ──

export interface AccountSummary {
  equity: number;
  daily_pnl: number;
  daily_realized_pnl: number;
  open_positions: Position[];
  trade_blocked: boolean;
  max_daily_loss: number;
}

export interface Position {
  symbol: string;
  shares: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  stop_price: number;
}

export interface Trade {
  id: number;
  symbol: string;
  entry_time: string;
  entry_price: number;
  exit_time: string | null;
  exit_price: number | null;
  shares: number;
  pnl: number | null;
  reason: string | null;
}

export interface Metrics {
  net_return_pct: number;
  net_pnl: number;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  expectancy_per_trade: number;
  avg_win: number;
  avg_loss: number;
  sharpe_proxy: number;
  max_drawdown_pct: number;
  max_drawdown_absolute: number;
  consecutive_wins: number;
  consecutive_losses: number;
  exposure_pct: number;
}

export interface EquityPoint {
  timestamp: string;
  equity: number;
}

export interface LiveStatus {
  status: string;
  symbol: string;
  strategy_name: string;
  strategy_params: Record<string, unknown>;
  is_daily: boolean;
  interval_seconds: number;
  evaluations_count: number;
  trades_count: number;
  anomalies_detected: number;
  last_evaluation_time: string | null;
  broker: {
    equity: number;
    cash: number;
    positions: Record<string, { shares: number; entry_price: number; stop_price: number }>;
    pending_orders: number;
    trade_blocked: boolean;
  };
  started_at: string | null;
  stopped_at: string | null;
  error_message: string | null;
}

export interface DecisionEntry {
  id: number;
  timestamp: string;
  symbol: string;
  strategy_name: string;
  candle: { open: number; high: number; low: number; close: number; volume: number };
  signal: string;
  signal_reason: string | null;
  broker_action: string | null;
  broker_action_reason: string | null;
  equity: number | null;
  has_position: boolean;
  anomaly_flags: string | null;
}

// ── Risk State ──

export interface RiskStateDetails {
  daily_pnl: number;
  weekly_pnl: number;
  monthly_pnl: number;
  high_water_mark: number;
  drawdown_from_hwm_pct: number;
  consecutive_losing_days: number;
  pause_until_date: string | null;
}

export interface RiskStateLimits {
  max_daily_loss_pct: number;
  max_weekly_loss_pct: number;
  max_monthly_loss_pct: number;
  max_drawdown_from_hwm_pct: number;
  max_consecutive_losing_days: number;
  max_portfolio_exposure_pct: number;
}

export interface RiskState {
  any_breached: boolean;
  daily_breached: boolean;
  weekly_breached: boolean;
  monthly_breached: boolean;
  hwm_drawdown_breached: boolean;
  consecutive_days_breached: boolean;
  is_paused: boolean;
  details: RiskStateDetails;
  limits: RiskStateLimits;
  equity: number;
  trade_blocked: boolean;
}

// ── Metrics Aggregation ──

export interface MonthlyReturn {
  year: number;
  month: number;
  return_pct: number;
  pnl: number;
  trade_count: number;
}

export interface TradeDistributionEntry {
  id: number;
  pnl: number;
  symbol: string;
  exit_time: string;
  is_win: boolean;
}

// ── API functions ──

export const getAccount = () => fetchAPI<AccountSummary>("/account");
export const getTrades = () => fetchAPI<Trade[]>("/trades");
export const getMetrics = () => fetchAPI<Metrics>("/metrics");
export const getEquityCurve = () => fetchAPI<EquityPoint[]>("/equity-curve");
export const getLiveStatus = () => fetchAPI<LiveStatus>("/live/status");
export const getDecisions = (limit = 50) =>
  fetchAPI<{ decisions: DecisionEntry[]; total: number }>(`/decisions?limit=${limit}`);
export const getRiskState = () => fetchAPI<RiskState>("/risk/state");
export const getMonthlyReturns = () =>
  fetchAPI<{ monthly_returns: MonthlyReturn[] }>("/metrics/monthly-returns");
export const getTradeDistribution = () =>
  fetchAPI<{ trades: TradeDistributionEntry[] }>("/metrics/trade-distribution");

export const startLive = (body: {
  symbol?: string;
  strategy_name?: string;
  strategy_params?: Record<string, unknown>;
  initial_equity?: number;
}) => postAPI<{ message: string; status: LiveStatus }>("/live/start", body);

export const stopLive = () => postAPI<{ message: string; status: LiveStatus }>("/live/stop");

// ── Sprint 4: Strategy Management ──

export interface StrategyInfo {
  name: string;
  description: string;
  default_params: Record<string, number | boolean>;
  param_ranges: Record<string, number[]>;
}

export interface AlpacaStatus {
  configured: boolean;
  api_key_masked: string | null;
}

export interface AlpacaTestResult {
  success: boolean;
  message: string;
}

export interface PromotionStatus {
  promoted: boolean;
  reasons: string[];
  metrics: Record<string, number>;
  checks_passed: number;
  checks_total: number;
  readiness_pct: number;
  thresholds: Record<string, number>;
}

export const getStrategies = () =>
  fetchAPI<{ strategies: StrategyInfo[] }>("/strategies");

export const switchStrategy = (body: {
  strategy_name: string;
  strategy_params?: Record<string, unknown>;
  force?: boolean;
}) =>
  postAPI<{ message: string; warning: string | null; status: LiveStatus; error?: string }>(
    "/live/switch-strategy",
    body,
  );

export const updateParams = (params: Record<string, unknown>) =>
  patchAPI<{
    message: string;
    previous_params: Record<string, unknown>;
    current_params: Record<string, unknown>;
    status: LiveStatus;
    error?: string;
  }>("/live/params", { params });

export const updateRiskLimits = (
  limits: Partial<RiskStateLimits & { vol_adjustment_enabled?: boolean }>,
) =>
  patchAPI<{
    message: string;
    previous_limits: Record<string, unknown>;
    current_limits: Record<string, unknown>;
    error?: string;
  }>("/risk/limits", limits);

export const getAlpacaStatus = () => fetchAPI<AlpacaStatus>("/data/alpaca/status");
export const testAlpacaConnection = () => postAPI<AlpacaTestResult>("/data/alpaca/test");
export const getPromotionStatus = () => fetchAPI<PromotionStatus>("/promotion/status");

// ── Sprint 5: Notifications ──

export interface NotificationPrefs {
  email_enabled: boolean;
  browser_enabled: boolean;
  email_address: string | null;
  smtp_configured: boolean;
  smtp_host: string | null;
  smtp_port: number;
  smtp_use_tls: boolean;
  min_severity: string;
  email_categories: Record<string, boolean>;
  browser_categories: Record<string, boolean>;
  updated_at: string | null;
}

export interface NotificationPrefsUpdate {
  email_enabled?: boolean;
  browser_enabled?: boolean;
  email_address?: string;
  smtp_host?: string;
  smtp_port?: number;
  smtp_user?: string;
  smtp_password?: string;
  smtp_use_tls?: boolean;
  min_severity?: string;
  email_categories?: Record<string, boolean>;
  browser_categories?: Record<string, boolean>;
}

export const getNotificationPrefs = () =>
  fetchAPI<NotificationPrefs>("/notifications/prefs");

export const updateNotificationPrefs = (body: NotificationPrefsUpdate) =>
  patchAPI<{ message: string; prefs: NotificationPrefs }>("/notifications/prefs", body);

export const testEmailNotification = () =>
  postAPI<{ success: boolean; message: string }>("/notifications/test-email");
