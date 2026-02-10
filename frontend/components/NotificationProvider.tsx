"use client";

import { useNotifications } from "@/lib/useNotifications";

/**
 * Client component wrapper that activates browser notifications.
 * Place in layout.tsx to enable notifications across all pages.
 */
export function NotificationProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  useNotifications();
  return <>{children}</>;
}
