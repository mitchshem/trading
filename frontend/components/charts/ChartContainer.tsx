"use client";

import { useEffect, useRef, useState, ReactNode } from "react";

interface ChartContainerProps {
  title?: string;
  height?: number;
  loading?: boolean;
  error?: string | null;
  empty?: boolean;
  emptyMessage?: string;
  children: (dimensions: { width: number; height: number }) => ReactNode;
}

/**
 * Shared chart wrapper that handles:
 * - Responsive width via ResizeObserver
 * - Loading skeleton state
 * - Error state
 * - Empty state
 */
export default function ChartContainer({
  title,
  height = 300,
  loading = false,
  error = null,
  empty = false,
  emptyMessage = "No data available yet.",
  children,
}: ChartContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setWidth(entry.contentRect.width);
      }
    });

    observer.observe(containerRef.current);
    // Set initial width
    setWidth(containerRef.current.clientWidth);

    return () => observer.disconnect();
  }, []);

  return (
    <div ref={containerRef} className="w-full">
      {title && (
        <p className="text-muted text-xs uppercase tracking-wider mb-2">
          {title}
        </p>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div
          className="bg-surface-overlay rounded-xl animate-pulse"
          style={{ height }}
        />
      )}

      {/* Error state */}
      {!loading && error && (
        <div
          className="bg-loss/5 border border-loss/20 rounded-xl flex items-center justify-center text-loss text-sm"
          style={{ height }}
        >
          {error}
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && empty && (
        <div
          className="bg-surface-overlay rounded-xl flex items-center justify-center text-muted text-sm"
          style={{ height }}
        >
          {emptyMessage}
        </div>
      )}

      {/* Chart content */}
      {!loading && !error && !empty && width > 0 && children({ width, height })}
    </div>
  );
}
