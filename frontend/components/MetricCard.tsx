"use client";

import { clsx } from "clsx";

interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  trend?: "up" | "down" | "neutral";
  size?: "sm" | "md" | "lg";
}

export default function MetricCard({
  label,
  value,
  subtitle,
  trend = "neutral",
  size = "md",
}: MetricCardProps) {
  return (
    <div className="bg-surface-raised border border-surface-border rounded-xl p-4">
      <p className="text-muted text-xs uppercase tracking-wider mb-1">{label}</p>
      <p
        className={clsx(
          "font-semibold font-mono",
          size === "lg" && "text-3xl",
          size === "md" && "text-xl",
          size === "sm" && "text-base",
          trend === "up" && "text-profit",
          trend === "down" && "text-loss",
          trend === "neutral" && "text-white"
        )}
      >
        {value}
      </p>
      {subtitle && <p className="text-muted text-xs mt-1">{subtitle}</p>}
    </div>
  );
}
