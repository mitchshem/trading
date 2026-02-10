"use client";

import { useEffect, useRef } from "react";
import { createChart, IChartApi } from "lightweight-charts";

export interface CandleData {
  time: string; // YYYY-MM-DD
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface ChartMarker {
  time: string;
  position: "aboveBar" | "belowBar";
  color: string;
  shape: "arrowUp" | "arrowDown" | "circle";
  text: string;
}

interface CandlestickChartProps {
  candles: CandleData[];
  markers?: ChartMarker[];
  width: number;
  height: number;
  showVolume?: boolean;
}

/**
 * Candlestick chart with optional volume histogram and signal markers.
 * Uses lightweight-charts CandlestickSeries.
 */
export default function CandlestickChart({
  candles,
  markers = [],
  width,
  height,
  showVolume = true,
}: CandlestickChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || candles.length === 0) return;

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

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    const candleChartData = candles.map((c) => ({
      time: c.time as unknown as import("lightweight-charts").UTCTimestamp,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));

    candleSeries.setData(candleChartData);

    // Set markers if present
    if (markers.length > 0) {
      const chartMarkers = markers
        .map((m) => ({
          time: m.time as unknown as import("lightweight-charts").UTCTimestamp,
          position: m.position,
          color: m.color,
          shape: m.shape,
          text: m.text,
        }))
        .sort((a, b) => (a.time as unknown as string).localeCompare(b.time as unknown as string));
      candleSeries.setMarkers(chartMarkers);
    }

    // Volume histogram
    if (showVolume && candles.some((c) => c.volume && c.volume > 0)) {
      const volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      });

      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });

      const volumeData = candles
        .filter((c) => c.volume !== undefined)
        .map((c) => ({
          time: c.time as unknown as import("lightweight-charts").UTCTimestamp,
          value: c.volume!,
          color: c.close >= c.open ? "rgba(34, 197, 94, 0.3)" : "rgba(239, 68, 68, 0.3)",
        }));

      volumeSeries.setData(volumeData);
    }

    chart.timeScale().fitContent();

    return () => {
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [candles, markers, width, height, showVolume]);

  // Handle resize
  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.resize(width, height);
    }
  }, [width, height]);

  if (candles.length === 0) {
    return (
      <div
        className="bg-surface-overlay rounded-xl flex items-center justify-center text-muted text-sm"
        style={{ width, height }}
      >
        No candle data available.
      </div>
    );
  }

  return <div ref={chartContainerRef} />;
}
