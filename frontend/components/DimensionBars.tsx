'use client';
import { useEffect, useState } from 'react';

interface Dimension {
  score: number;
  max: number;
  value: string | number;
}

interface Props {
  dimensions: Record<string, Dimension>;
}

const LABELS: Record<string, string> = {
  text:       'Text',
  face:       'Face / Persons',
  expression: 'Expression',
  colors:     'Colors',
};

const ICONS: Record<string, string> = {
  text:       '📝',
  face:       '👤',
  expression: '😐',
  colors:     '🎨',
};

function barColor(pct: number) {
  if (pct >= 0.75) return '#22c55e';
  if (pct >= 0.5)  return '#6366f1';
  if (pct >= 0.25) return '#f59e0b';
  return '#ef4444';
}

function formatValue(key: string, value: string | number): string {
  if (key === 'face') return `${value} person${value === 1 ? '' : 's'}`;
  if (key === 'text' && typeof value === 'string')
    return value ? `"${value.slice(0, 30)}${value.length > 30 ? '…' : ''}"` : 'None detected';
  return String(value) || '—';
}

export default function DimensionBars({ dimensions }: Props) {
  const [widths, setWidths] = useState<Record<string, number>>({});

  useEffect(() => {
    const timer = setTimeout(() => {
      const w: Record<string, number> = {};
      for (const [k, d] of Object.entries(dimensions)) w[k] = (d.score / d.max) * 100;
      setWidths(w);
    }, 100);
    return () => clearTimeout(timer);
  }, [dimensions]);

  return (
    <div className="flex flex-col gap-4">
      {Object.entries(dimensions).map(([key, dim]) => {
        const pct   = dim.score / dim.max;
        const color = barColor(pct);
        const width = widths[key] ?? 0;

        return (
          <div key={key}>
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm font-medium flex items-center gap-2">
                <span>{ICONS[key]}</span>
                {LABELS[key]}
              </span>
              <span className="text-sm font-bold" style={{ color }}>
                {dim.score} / {dim.max}
              </span>
            </div>

            {/* Bar */}
            <div className="h-2 rounded-full" style={{ backgroundColor: 'var(--surface2)' }}>
              <div className="h-2 rounded-full"
                style={{
                  width: `${width}%`,
                  backgroundColor: color,
                  transition: 'width 1s cubic-bezier(0.16,1,0.3,1)',
                }} />
            </div>

            {/* Value label */}
            <p className="text-xs mt-1" style={{ color: 'var(--muted)' }}>
              {formatValue(key, dim.value)}
            </p>
          </div>
        );
      })}
    </div>
  );
}
