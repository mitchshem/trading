import type { Metadata } from "next";
import Nav from "@/components/Nav";
import { NotificationProvider } from "@/components/NotificationProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "AutoTrader | Paper Trading System",
  description: "Automated paper trading with risk management and strategy validation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-surface text-white min-h-screen">
        <Nav />
        <NotificationProvider>
          <main>{children}</main>
        </NotificationProvider>
      </body>
    </html>
  );
}
