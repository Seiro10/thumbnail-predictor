'use client';
import { useEffect, useState } from 'react';

const RADIUS = 70;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function scoreColor(score: number) {
  if (score >= 15) return '#22c55e';
  if (score >= 10) return '#6366f1';
  if (score >= 5)  return '#f59e0b';
  return '#ef4444';
}

function scoreLabel(score: number) {
  if (score >= 15) return 'Excellent';
  if (score >= 10) return 'Good';
  if (score >= 5)  return 'Below average';
  return 'Poor';
}

export default function ScoreRing({ score, max = 20 }: { score: number; max?: number }) {
  const [animated, setAnimated] = useState(0);

  useEffect(() => {
    const start = performance.now();
    const duration = 1200;
    const raf = (now: number) => {
      const t = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - t, 3);
      setAnimated(score * ease);
      if (t < 1) requestAnimationFrame(raf);
    };
    requestAnimationFrame(raf);
  }, [score]);

  const pct    = animated / max;
  const offset = CIRCUMFERENCE * (1 - pct);
  const color  = scoreColor(score);

  return (
    <div className="flex flex-col items-center gap-3">
      <svg width="180" height="180" viewBox="0 0 180 180">
        {/* Track */}
        <circle cx="90" cy="90" r={RADIUS} fill="none"
          stroke="var(--surface2)" strokeWidth="14" />
        {/* Progress */}
        <circle cx="90" cy="90" r={RADIUS} fill="none"
          stroke={color} strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          transform="rotate(-90 90 90)"
          style={{ transition: 'stroke 0.4s ease' }}
        />
        {/* Score text */}
        <text x="90" y="85" textAnchor="middle"
          fill={color} fontSize="36" fontWeight="700" fontFamily="Inter, sans-serif">
          {animated.toFixed(1)}
        </text>
        <text x="90" y="108" textAnchor="middle"
          fill="var(--muted)" fontSize="13" fontFamily="Inter, sans-serif">
          / {max}
        </text>
      </svg>
      <span className="text-sm font-semibold tracking-widest uppercase"
        style={{ color }}>
        {scoreLabel(score)}
      </span>
    </div>
  );
}
