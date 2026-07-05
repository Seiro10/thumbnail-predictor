'use client';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer } from 'recharts';

interface Dimension {
  score: number;
  max: number;
}

interface Props {
  dimensions: Record<string, Dimension>;
}

const AXIS_LABELS: Record<string, string> = {
  text:       'Text',
  face:       'Face',
  expression: 'Expression',
  colors:     'Colors',
};

export default function ThumbnailRadar({ dimensions }: Props) {
  const data = Object.entries(dimensions).map(([key, dim]) => ({
    subject: AXIS_LABELS[key] ?? key,
    value:   Math.round((dim.score / dim.max) * 100),
    fullMark: 100,
  }));

  return (
    <ResponsiveContainer width="100%" height={240}>
      <RadarChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
        <PolarGrid stroke="var(--border)" />
        <PolarAngleAxis
          dataKey="subject"
          tick={{ fill: 'var(--muted)', fontSize: 12, fontFamily: 'Inter, sans-serif' }}
        />
        <Radar
          name="Score"
          dataKey="value"
          stroke="#6366f1"
          fill="#6366f1"
          fillOpacity={0.25}
          strokeWidth={2}
          dot={{ fill: '#6366f1', r: 3 }}
          animationBegin={200}
          animationDuration={900}
          animationEasing="ease-out"
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
