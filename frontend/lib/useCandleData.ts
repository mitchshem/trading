import { useEffect, useState } from "react";
import { fetchAPI } from "./api";

interface RawCandle {
  time: number | string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface RawSignal {
  id: number;
  timestamp: string;
  symbol: string;
  strategy_name: string;
  signal: string;
  signal_reason: string | null;
}

export interface CandleChartData {
  time: string; // YYYY-MM-DD
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface SignalMarker {
  time: string;
  position: "aboveBar" | "belowBar";
  color: string;
  shape: "arrowUp" | "arrowDown" | "circle";
  text: string;
}

interface CandleDataResult {
  candles: CandleChartData[];
  markers: SignalMarker[];
  loading: boolean;
  error: string | null;
}

/**
 * Hook that fetches candle data and signal markers for a symbol.
 */
export function useCandleData(
  symbol: string,
  refreshInterval = 60000
): CandleDataResult {
  const [candles, setCandles] = useState<CandleChartData[]>([]);
  const [markers, setMarkers] = useState<SignalMarker[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!symbol) {
      setLoading(false);
      return;
    }

    let mounted = true;

    async function load() {
      try {
        const [candleRes, signalRes] = await Promise.all([
          fetchAPI<{ symbol: string; candles: RawCandle[] }>(
            `/candles?symbol=${symbol}`
          ),
          fetchAPI<{ symbol: string; signals: RawSignal[] }>(
            `/signals?symbol=${symbol}&limit=200`
          ).catch(() => ({ symbol, signals: [] })),
        ]);

        if (!mounted) return;

        // Transform candles to YYYY-MM-DD format
        const chartCandles: CandleChartData[] = candleRes.candles.map((c) => {
          let dateStr: string;
          if (typeof c.time === "number") {
            dateStr = new Date(c.time * 1000).toISOString().split("T")[0];
          } else {
            dateStr = new Date(c.time).toISOString().split("T")[0];
          }
          return {
            time: dateStr,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
            volume: c.volume,
          };
        });

        // Deduplicate by date (keep last)
        const byDate = new Map<string, CandleChartData>();
        for (const candle of chartCandles) {
          byDate.set(candle.time, candle);
        }
        const dedupCandles = Array.from(byDate.values()).sort((a, b) =>
          a.time.localeCompare(b.time)
        );

        // Transform signals to markers
        const chartMarkers: SignalMarker[] = signalRes.signals
          .filter((s) => s.signal === "BUY" || s.signal === "EXIT")
          .map((s) => {
            const dateStr = new Date(s.timestamp).toISOString().split("T")[0];
            const isBuy = s.signal === "BUY";
            return {
              time: dateStr,
              position: isBuy
                ? ("belowBar" as const)
                : ("aboveBar" as const),
              color: isBuy ? "#22c55e" : "#ef4444",
              shape: isBuy
                ? ("arrowUp" as const)
                : ("arrowDown" as const),
              text: isBuy ? "B" : "S",
            };
          });

        setCandles(dedupCandles);
        setMarkers(chartMarkers);
        setError(null);
      } catch (err) {
        if (!mounted) return;
        setError(
          err instanceof Error ? err.message : "Failed to load candle data"
        );
      } finally {
        if (mounted) setLoading(false);
      }
    }

    load();
    const interval = setInterval(load, refreshInterval);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [symbol, refreshInterval]);

  return { candles, markers, loading, error };
}
