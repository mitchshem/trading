"use client";

import { useEffect, useRef } from "react";
import { createChart, IChartApi } from "lightweight-charts";

export interface DrawdownPoint {
  time: string; // YYYY-MM-DD
  value: number; // Negative percentage (e.g., -5.2 means 5.2% drawdown)
}

interface DrawdownChartProps {
  data: DrawdownPoint[];
  width: number;
  height: number;
}

/**
 * Drawdown chart showing decline from peak as a filled red area.
 * Values are always <= 0 (0 = at peak, -10 = 10% below peak).
 * Uses lightweight-charts AreaSeries.
 */
export default function DrawdownChart({
  data,
  width,
  height,
}: DrawdownChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return;

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

    const areaSeries = chart.addAreaSeries({
      topColor: "rgba(239, 68, 68, 0.4)",
      bottomColor: "rgba(239, 68, 68, 0.05)",
      lineColor: "#ef4444",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
    });

    const chartData = data.map((point) => ({
      time: point.time as unknown as import("lightweight-charts").UTCTimestamp,
      value: point.value,
    }));

    areaSeries.setData(chartData);

    // Add zero baseline
    areaSeries.createPriceLine({
      price: 0,
      color: "#6b7280",
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: "Peak",
    });

    chart.timeScale().fitContent();

    return () => {
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [data, width, height]);

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
        No drawdown data available.
      </div>
    );
  }

  return <div ref={chartContainerRef} />;
}
