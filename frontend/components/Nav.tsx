"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clsx } from "clsx";

const links = [
  { href: "/", label: "Overview" },
  { href: "/strategy", label: "Strategy" },
  { href: "/trades", label: "Trades" },
  { href: "/risk", label: "Risk" },
  { href: "/settings", label: "Settings" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav className="bg-surface-raised border-b border-surface-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-14">
          {/* Logo / Title */}
          <div className="flex items-center gap-3">
            <span className="text-white font-bold text-lg tracking-tight">
              AutoTrader
            </span>
            <span className="hidden sm:inline-block px-2 py-0.5 rounded bg-accent/20 text-accent text-xs font-medium">
              Paper Trading
            </span>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center gap-1">
            {links.map((link) => {
              const isActive =
                link.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(link.href);

              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={clsx(
                    "px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
                    isActive
                      ? "bg-surface-overlay text-white"
                      : "text-muted hover:text-white hover:bg-surface-overlay/50"
                  )}
                >
                  {link.label}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
