"use client";

import { clsx } from "clsx";

type Variant = "success" | "danger" | "warning" | "neutral" | "accent";

interface StatusBadgeProps {
  label: string;
  variant?: Variant;
  pulse?: boolean;
}

const variantStyles: Record<Variant, string> = {
  success: "bg-profit/20 text-profit",
  danger: "bg-loss/20 text-loss",
  warning: "bg-yellow-500/20 text-yellow-400",
  neutral: "bg-surface-overlay text-muted",
  accent: "bg-accent/20 text-accent",
};

export default function StatusBadge({
  label,
  variant = "neutral",
  pulse = false,
}: StatusBadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium",
        variantStyles[variant]
      )}
    >
      {pulse && (
        <span className="relative flex h-2 w-2">
          <span
            className={clsx(
              "absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping",
              variant === "success" && "bg-profit",
              variant === "danger" && "bg-loss",
              variant === "accent" && "bg-accent"
            )}
          />
          <span
            className={clsx(
              "relative inline-flex h-2 w-2 rounded-full",
              variant === "success" && "bg-profit",
              variant === "danger" && "bg-loss",
              variant === "accent" && "bg-accent"
            )}
          />
        </span>
      )}
      {label}
    </span>
  );
}
