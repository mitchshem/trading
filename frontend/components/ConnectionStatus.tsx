"use client";

interface ConnectionStatusProps {
  /** Whether the SSE connection is active. */
  isConnected: boolean;
}

/**
 * Small dot indicator for SSE connection status.
 *
 * - Green (pulse): connected
 * - Yellow: reconnecting
 * - Gray: disconnected / not available
 */
export default function ConnectionStatus({ isConnected }: ConnectionStatusProps) {
  return (
    <div className="flex items-center gap-1.5" title={isConnected ? "Real-time connected" : "Real-time disconnected"}>
      <span
        className={`inline-block w-2 h-2 rounded-full ${
          isConnected
            ? "bg-profit animate-pulse"
            : "bg-surface-border"
        }`}
      />
      <span className="text-xs text-muted">
        {isConnected ? "Live" : "Offline"}
      </span>
    </div>
  );
}
