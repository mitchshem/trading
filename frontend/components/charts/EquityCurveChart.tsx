"use client";

import { useEffect, useRef } from "react";
import { createChart, IChartApi } from "lightweight-charts";

export interface EquityPoint {
  time: string; // YYYY-MM-DD or Unix timestamp
  value: number;
}

interface EquityCurveChartProps {
  data: EquityPoint[];
  width: number;
  height: number;
  baselineValue?: number;
}

/**
 * Equity curve line chart using lightweight-charts.
 * Shows account value over time with green/red coloring
 * based on whether equity is above or below the baseline.
 */
export default function EquityCurveChart({
  data,
  width,
  height,
  baselineValue,
}: EquityCurveChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return;

    // Clean up previous chart
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(chartContainerRef.current, {
      width,
      height,
      layout: {
        background: { color: "transparent" },
        textColor: "#9ca3af",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#333333" },
        horzLines: { color: "#333333" },
      },
      crosshair: {
        vertLine: { color: "#555", width: 1, style: 2 },
        horzLine: { color: "#555", width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: "#333333",
      },
      timeScale: {
        borderColor: "#333333",
        timeVisible: false,
      },
    });

    chartRef.current = chart;

    const baseline = baselineValue ?? (data.length > 0 ? data[0].value : 100000);

    const lineSeries = chart.addLineSeries({
      color: "#3b82f6",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
    });

    // Format data for lightweight-charts
    const chartData = data.map((point) => ({
      time: point.time as any,
      value: point.value,
    }));

    lineSeries.setData(chartData);

    // Add baseline price line
    lineSeries.createPriceLine({
      price: baseline,
      color: "#6b7280",
      lineWidth: 1,
      lineStyle: 2, // Dashed
      axisLabelVisible: true,
      title: "Initial",
    });

    // Fit content
    chart.timeScale().fitContent();

    return () => {
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [data, width, height, baselineValue]);

  // Handle resize
  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.resize(width, height);
    }
  }, [width, height]);

  if (data.length === 0) {
    return (
      <div
        className="bg-surface-overlay rounded-xl flex items-center justify-center text-muted text-sm"
        style={{ width, height }}
      >
        No equity data yet. Start paper trading to track performance.
      </div>
    );
  }

  return <div ref={chartContainerRef} />;
}
