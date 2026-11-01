import type { ReactNode } from 'react';
import type { Status } from '../types';

export const statusStyles: Record<string, string> = {
  supported: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300',
  partial: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300',
  unsupported: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300',
};

export const toneStyles: Record<string, string> = {
  ...statusStyles,
  neutral: 'bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
  info: 'bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-300',
  violet: 'bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-300',
};

export function statusTone(s: Status | string | null | undefined): string {
  return s && statusStyles[s] ? s : 'neutral';
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: string }) {
  return (
    <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${toneStyles[tone] ?? toneStyles.neutral}`}>
      {children}
    </span>
  );
}

export function Card({
  title,
  right,
  children,
  className = '',
}: {
  title?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900 ${className}`}>
      {(title || right) && (
        <div className="mb-3 flex items-center justify-between gap-2">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">{title}</h2>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-lg bg-slate-100 p-3 dark:bg-slate-800/60">
      <div className="text-xs text-slate-500 dark:text-slate-400">{label}</div>
      <div className="text-lg font-semibold">{value}</div>
      {hint && <div className="text-xs text-slate-500 dark:text-slate-400">{hint}</div>}
    </div>
  );
}

export function Bar({
  value,
  max = 1,
  color = 'bg-sky-500',
  label,
}: {
  value: number;
  max?: number;
  color?: string;
  label?: ReactNode;
}) {
  const width = max > 0 ? Math.min(Math.max(value / max, 0), 1) * 100 : 0;
  return (
    <div className="flex items-center gap-2">
      {label && <span className="w-28 shrink-0 text-xs text-slate-500 dark:text-slate-400">{label}</span>}
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${width}%` }} />
      </div>
      <span className="w-14 shrink-0 text-right text-xs tabular-nums">{value.toFixed(3)}</span>
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300">
      {message}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500 dark:border-slate-700">{children}</p>;
}

export const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
export const usd = (v: number | null | undefined) => (v == null ? 'n/a' : `$${v.toFixed(5)}`);
export const num = (v: number, digits = 3) => v.toFixed(digits);
export const sourceLabel = (filename: string, page: number | null) => (page == null ? filename : `${filename}, p. ${page}`);
