import type { ReactNode } from 'react';
import type { Status } from '../types';

// Status colours keep AA contrast on their tinted backgrounds (800-level text on 50/100 fills).
export const statusStyles: Record<string, string> = {
  supported: 'bg-emerald-50 text-emerald-800 ring-1 ring-inset ring-emerald-200',
  partial: 'bg-amber-50 text-amber-900 ring-1 ring-inset ring-amber-200',
  unsupported: 'bg-red-50 text-red-800 ring-1 ring-inset ring-red-200',
};

export const toneStyles: Record<string, string> = {
  ...statusStyles,
  neutral: 'bg-slate-100 text-slate-700 ring-1 ring-inset ring-slate-200',
  info: 'bg-indigo-50 text-indigo-800 ring-1 ring-inset ring-indigo-200',
  violet: 'bg-violet-50 text-violet-800 ring-1 ring-inset ring-violet-200',
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

export function PageHeader({ title, subtitle, right }: { title: string; subtitle?: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm leading-relaxed text-slate-600">{subtitle}</p>}
      </div>
      {right}
    </div>
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
    <section className={`rounded-xl border border-slate-200 bg-white p-5 ${className}`}>
      {(title || right) && (
        <div className="mb-4 flex items-center justify-between gap-2">
          <h2 className="text-sm font-semibold text-slate-800">{title}</h2>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="mt-0.5 text-lg font-semibold text-slate-900">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

export function Bar({
  value,
  max = 1,
  color = 'bg-indigo-500',
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
      {label && <span className="w-28 shrink-0 text-xs text-slate-500">{label}</span>}
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${width}%` }} />
      </div>
      <span className="w-14 shrink-0 text-right text-xs tabular-nums text-slate-700">{value.toFixed(3)}</span>
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">{message}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-8 text-center text-sm text-slate-500">{children}</p>;
}

export function Skeleton({ className = 'h-4 w-full' }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden />;
}

export function SkeletonCard({ lines = 3 }: { lines?: number }) {
  return (
    <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-5" role="status" aria-label="Loading">
      <Skeleton className="h-4 w-1/4" />
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className={`h-3 ${i === lines - 1 ? 'w-2/3' : 'w-full'}`} />
      ))}
    </div>
  );
}

/** A small "i" that reveals an explanation on hover or keyboard focus. */
export function InfoTip({ children }: { children: ReactNode }) {
  return (
    <span className="group relative inline-block align-middle">
      <button
        type="button"
        aria-label="What does this mean?"
        className="ml-1 inline-flex h-4 w-4 items-center justify-center rounded-full border border-slate-300 bg-white text-[10px] font-semibold leading-none text-slate-500 hover:border-indigo-400 hover:text-indigo-600"
      >
        i
      </button>
      <span
        role="tooltip"
        className="pointer-events-none invisible absolute left-1/2 top-full z-20 mt-1 w-64 -translate-x-1/2 rounded-lg border border-slate-200 bg-white p-3 text-left text-xs font-normal normal-case leading-relaxed text-slate-700 shadow-lg group-focus-within:visible group-hover:visible"
      >
        {children}
      </span>
    </span>
  );
}

export const btnPrimary =
  'rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50';
export const inputBase =
  'w-full rounded-lg border border-slate-300 bg-white p-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-500';

export const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
export const num = (v: number, digits = 3) => v.toFixed(digits);
export const sourceLabel = (filename: string, page: number | null) => (page == null ? filename : `${filename}, p. ${page}`);
