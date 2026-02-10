import { useEffect, useState } from "react";
import { getMonthlyReturns, MonthlyReturn } from "./api";

interface MonthlyReturnsResult {
  data: MonthlyReturn[];
  loading: boolean;
  error: string | null;
}

/**
 * Hook that fetches monthly returns data from the backend.
 */
export function useMonthlyReturns(refreshInterval = 60000): MonthlyReturnsResult {
  const [data, setData] = useState<MonthlyReturn[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    async function load() {
      try {
        const result = await getMonthlyReturns();
        if (!mounted) return;
        setData(result.monthly_returns);
        setError(null);
      } catch (err) {
        if (!mounted) return;
        setError(
          err instanceof Error ? err.message : "Failed to load monthly returns"
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
  }, [refreshInterval]);

  return { data, loading, error };
}
