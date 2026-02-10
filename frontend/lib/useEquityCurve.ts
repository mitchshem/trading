import { useEffect, useState } from "react";
import { fetchAPI, EquityPoint } from "./api";

interface EquityCurveData {
  data: { time: string; value: number }[];
  loading: boolean;
  error: string | null;
}

/**
 * Hook that fetches equity curve data from /equity-curve
 * and transforms it for lightweight-charts.
 */
export function useEquityCurve(refreshInterval = 30000): EquityCurveData {
  const [data, setData] = useState<{ time: string; value: number }[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    async function load() {
      try {
        const result = await fetchAPI<{
          equity_curve: { id: number; timestamp: string; equity: number }[];
        }>("/equity-curve");

        if (!mounted) return;

        // Transform to chart format: need YYYY-MM-DD for lightweight-charts
        const chartData = result.equity_curve.map((point) => {
          const dt = new Date(point.timestamp);
          const dateStr = dt.toISOString().split("T")[0]; // YYYY-MM-DD
          return { time: dateStr, value: point.equity };
        });

        // Deduplicate by date (keep last value per day)
        const byDate = new Map<string, { time: string; value: number }>();
        for (const point of chartData) {
          byDate.set(point.time, point);
        }
        const deduplicated = Array.from(byDate.values()).sort(
          (a, b) => a.time.localeCompare(b.time)
        );

        setData(deduplicated);
        setError(null);
      } catch (err) {
        if (!mounted) return;
        setError(err instanceof Error ? err.message : "Failed to load equity curve");
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
  }, [refreshInterval]);

  return { data, loading, error };
}
