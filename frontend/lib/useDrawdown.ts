import { useMemo } from "react";

interface EquityPoint {
  time: string;
  value: number;
}

export interface DrawdownPoint {
  time: string;
  value: number; // Negative percentage
}

/**
 * Computes drawdown series from equity curve data.
 * Client-side computation: dd[i] = (equity[i] - hwm[i]) / hwm[i] * 100
 *
 * @param equityData - The equity curve data from useEquityCurve
 * @returns DrawdownPoint[] with negative percentage values
 */
export function useDrawdown(equityData: EquityPoint[]): DrawdownPoint[] {
  return useMemo(() => {
    if (equityData.length === 0) return [];

    const result: DrawdownPoint[] = [];
    let hwm = equityData[0].value;

    for (const point of equityData) {
      if (point.value > hwm) {
        hwm = point.value;
      }
      const dd = hwm > 0 ? ((point.value - hwm) / hwm) * 100 : 0;
      result.push({
        time: point.time,
        value: parseFloat(dd.toFixed(2)),
      });
    }

    return result;
  }, [equityData]);
}
