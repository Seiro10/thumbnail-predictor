'use client';

interface Vision {
  text_present:  boolean;
  text_length:   number;
  text_content:  string;
  nb_faces:      number;
  expression:    string;
  color:         string;
  background:    string;
  contrast:      string;
  face_conf:     number;
}

interface BadgeProps { label: string; value: string; good?: boolean; neutral?: boolean }

function Badge({ label, value, good, neutral }: BadgeProps) {
  const bg    = good    ? 'rgba(34,197,94,0.12)'  :
                neutral ? 'rgba(99,102,241,0.12)'  :
                          'rgba(239,68,68,0.12)';
  const color = good    ? '#22c55e' :
                neutral ? '#818cf8' :
                          '#f87171';
  return (
    <div className="flex flex-col gap-1 rounded-xl p-3"
      style={{ backgroundColor: bg, border: `1px solid ${color}30` }}>
      <span className="text-xs font-semibold uppercase tracking-wider"
        style={{ color: 'var(--muted)' }}>{label}</span>
      <span className="text-sm font-bold" style={{ color }}>{value}</span>
    </div>
  );
}

export default function VisionBadges({ vision }: { vision: Vision }) {
  const expressionGood = ['Neutre', 'Sourire'].includes(vision.expression);
  const colorGood      = vision.color === 'Froid';
  const colorNeutral   = vision.color === 'Neutre';

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      <Badge label="Text"
        value={vision.text_present ? `✓ ${vision.text_length} chars` : '✗ None'}
        good={vision.text_present} />
      <Badge label="Faces"
        value={vision.nb_faces === 0 ? '✗ None' :
               vision.nb_faces === 1 ? '✓ 1 person' :
               `${vision.nb_faces} people`}
        good={vision.nb_faces === 1}
        neutral={vision.nb_faces === 2} />
      <Badge label="Expression"
        value={vision.expression}
        good={expressionGood}
        neutral={!expressionGood && vision.expression !== 'Aucun'} />
      <Badge label="Colors"
        value={vision.color}
        good={colorGood}
        neutral={colorNeutral} />
      <Badge label="Background"
        value={vision.background}
        neutral={vision.background !== 'Chargé'} />
      <Badge label="Contrast"
        value={vision.contrast}
        neutral />
    </div>
  );
}
