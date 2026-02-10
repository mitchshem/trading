"use client";

import { useEffect, useRef } from "react";
import { useSSE } from "./useSSE";

interface NotificationPayload {
  category: string;
  severity: string;
  title: string;
  body: string;
  details?: Record<string, string>;
}

/**
 * React hook that bridges SSE notification_fired events to the
 * browser Notification API.
 *
 * Usage:
 *   useNotifications();   // Call once in a top-level layout/page
 *
 * Behavior:
 * - Requests notification permission on mount
 * - Listens for "notification_fired" SSE events
 * - Shows native browser notification for each event
 * - Falls back silently if permission denied or API unavailable
 */
export function useNotifications() {
  const { lastEvent } = useSSE(["notification_fired"]);
  const permissionRef = useRef<NotificationPermission>("default");

  // Request permission on mount
  useEffect(() => {
    if (typeof window === "undefined" || !("Notification" in window)) return;

    if (Notification.permission === "granted") {
      permissionRef.current = "granted";
    } else if (Notification.permission !== "denied") {
      Notification.requestPermission().then((perm) => {
        permissionRef.current = perm;
      });
    }
  }, []);

  // Show notification when SSE event arrives
  useEffect(() => {
    if (!lastEvent) return;
    if (typeof window === "undefined" || !("Notification" in window)) return;
    if (permissionRef.current !== "granted") return;

    try {
      const payload = lastEvent.payload as unknown as NotificationPayload;
      if (!payload || !payload.title) return;

      new Notification(payload.title, {
        body: payload.body,
        tag: `trading-${payload.category}-${Date.now()}`,
        silent: payload.severity === "info",
      });
    } catch {
      // Notification API not available or blocked
    }
  }, [lastEvent]);
}
