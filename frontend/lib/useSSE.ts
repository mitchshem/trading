"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const SSE_URL = "http://localhost:8000/events";

/** Max reconnect backoff in ms. */
const MAX_BACKOFF = 30000;

export interface SSEEvent {
  type: string;
  timestamp: string;
  payload: Record<string, unknown>;
}

interface UseSSEReturn {
  /** Whether the EventSource is currently connected. */
  isConnected: boolean;
  /** The last received event (matching eventTypes filter). */
  lastEvent: SSEEvent | null;
  /** Connection error message, if any. */
  error: string | null;
}

/**
 * React hook that subscribes to the backend SSE stream.
 *
 * @param eventTypes - Optional list of event types to listen to.
 *   If omitted, listens to all events.
 * @returns Connection state, last event, and error.
 *
 * Features:
 * - Auto-reconnect with exponential backoff (1s → 2s → 4s → … → 30s)
 * - Backoff resets on successful connection
 * - Graceful degradation: if /events is unreachable, callers keep polling
 * - Cleans up EventSource on unmount
 */
export function useSSE(eventTypes?: string[]): UseSSEReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<SSEEvent | null>(null);
  const [error, setError] = useState<string | null>(null);

  const esRef = useRef<EventSource | null>(null);
  const backoffRef = useRef(1000);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    // Clean up any existing connection
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }

    try {
      const es = new EventSource(SSE_URL);
      esRef.current = es;

      es.onopen = () => {
        if (!mountedRef.current) return;
        setIsConnected(true);
        setError(null);
        backoffRef.current = 1000; // Reset backoff on success
      };

      es.onerror = () => {
        if (!mountedRef.current) return;
        setIsConnected(false);
        setError("SSE connection lost");

        // Close the errored connection
        es.close();
        esRef.current = null;

        // Schedule reconnect with backoff
        const delay = backoffRef.current;
        backoffRef.current = Math.min(delay * 2, MAX_BACKOFF);

        reconnectTimerRef.current = setTimeout(() => {
          if (mountedRef.current) {
            connect();
          }
        }, delay);
      };

      // Listen for specific event types, or use generic "message" for all
      const handleEvent = (e: MessageEvent) => {
        if (!mountedRef.current) return;
        try {
          const data = JSON.parse(e.data) as SSEEvent;
          if (!eventTypes || eventTypes.length === 0 || eventTypes.includes(data.type)) {
            setLastEvent(data);
          }
        } catch {
          // Ignore malformed events
        }
      };

      // SSE named events: we need to add listeners for each event type
      // The server sends events as "event: <type>\ndata: ...\n\n"
      if (eventTypes && eventTypes.length > 0) {
        for (const type of eventTypes) {
          es.addEventListener(type, handleEvent);
        }
        // Also listen for heartbeat to maintain connection awareness
        es.addEventListener("heartbeat", () => {
          if (mountedRef.current) setIsConnected(true);
        });
      } else {
        // Listen to all named events we know about
        const allTypes = [
          "equity_update",
          "trade_executed",
          "decision_logged",
          "alert_fired",
          "loop_status",
          "risk_state_change",
          "strategy_switched",
          "params_updated",
          "risk_limits_updated",
          "notification_fired",
          "heartbeat",
        ];
        for (const type of allTypes) {
          es.addEventListener(type, handleEvent);
        }
      }
    } catch {
      if (mountedRef.current) {
        setIsConnected(false);
        setError("Failed to create EventSource");
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // Note: eventTypes intentionally excluded from deps to avoid reconnect loops.
  // The hook captures eventTypes on mount. To change filters, remount the component.

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
    };
  }, [connect]);

  return { isConnected, lastEvent, error };
}
