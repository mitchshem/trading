"use client";

interface PanelProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}

export default function Panel({ title, subtitle, children, action }: PanelProps) {
  return (
    <section className="bg-surface-raised border border-surface-border rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 border-b border-surface-border">
        <div>
          <h2 className="text-white font-semibold text-base">{title}</h2>
          {subtitle && <p className="text-muted text-xs mt-0.5">{subtitle}</p>}
        </div>
        {action && <div>{action}</div>}
      </div>
      <div className="p-5">{children}</div>
    </section>
  );
}
