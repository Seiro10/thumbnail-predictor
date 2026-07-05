'use client';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import Image from 'next/image';
import ScoreRing       from '@/components/ScoreRing';
import DimensionBars   from '@/components/DimensionBars';
import ThumbnailRadar  from '@/components/RadarChart';
import VisionBadges    from '@/components/VisionBadges';

interface Dimension { score: number; max: number; value: string | number }
interface ApiResult {
  score:      number;
  score_max:  number;
  dimensions: Record<string, Dimension>;
  vision: {
    text_present:  boolean;
    text_length:   number;
    text_content:  string;
    nb_faces:      number;
    expression:    string;
    color:         string;
    background:    string;
    contrast:      string;
    face_conf:     number;
  };
}

function ChannelForm({ subs, avgPerf, niche, onChange }: {
  subs: number; avgPerf: number; niche: string;
  onChange: (k: string, v: string | number) => void;
}) {
  const inputStyle: React.CSSProperties = {
    backgroundColor: 'var(--surface2)',
    border: '1px solid var(--border)',
    color: 'var(--text)',
    borderRadius: '0.5rem',
    padding: '0.5rem 0.75rem',
    fontSize: '0.875rem',
    outline: 'none',
    width: '100%',
  };
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '0.75rem' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        <label style={{ fontSize: '0.75rem', color: 'var(--muted)', fontWeight: 500 }}>Subscribers</label>
        <input type="number" value={subs} style={inputStyle}
          onChange={e => onChange('subs', parseFloat(e.target.value))} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        <label style={{ fontSize: '0.75rem', color: 'var(--muted)', fontWeight: 500 }}>Avg views/subs</label>
        <input type="number" value={avgPerf} step="0.0001" style={inputStyle}
          onChange={e => onChange('avgPerf', parseFloat(e.target.value))} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        <label style={{ fontSize: '0.75rem', color: 'var(--muted)', fontWeight: 500 }}>Niche</label>
        <select value={niche} style={inputStyle} onChange={e => onChange('niche', e.target.value)}>
          <option value="AI/Tech">AI / Tech</option>
          <option value="Business">Business</option>
        </select>
      </div>
    </div>
  );
}

function DropZone({ onFile, preview }: { onFile: (f: File) => void; preview: string | null }) {
  const onDrop = useCallback((accepted: File[]) => { if (accepted[0]) onFile(accepted[0]); }, [onFile]);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'image/jpeg': [], 'image/png': [], 'image/webp': [] }, maxFiles: 1,
  });

  return (
    <div {...getRootProps()} style={{
      position: 'relative', minHeight: 220, display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', borderRadius: '1rem',
      border: `2px dashed ${isDragActive ? 'var(--accent)' : 'var(--border)'}`,
      backgroundColor: isDragActive ? 'rgba(99,102,241,0.06)' : 'var(--surface)',
      cursor: 'pointer', overflow: 'hidden', transition: 'all 0.2s',
    }}>
      <input {...getInputProps()} />
      {preview ? (
        <>
          <Image src={preview} alt="Thumbnail preview" fill style={{ objectFit: 'contain' }} />
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
            justifyContent: 'center', backgroundColor: 'rgba(0,0,0,0.6)',
            opacity: 0, transition: 'opacity 0.2s',
          }}
            onMouseEnter={e => (e.currentTarget.style.opacity = '1')}
            onMouseLeave={e => (e.currentTarget.style.opacity = '0')}>
            <p style={{ color: 'white', fontSize: '0.875rem', fontWeight: 600 }}>Drop a new image to replace</p>
          </div>
        </>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem', padding: '2rem', textAlign: 'center' }}>
          <span style={{ fontSize: '3rem' }}>🖼️</span>
          <p style={{ fontWeight: 600, color: 'var(--text)' }}>
            {isDragActive ? 'Drop it!' : 'Drag & drop your thumbnail'}
          </p>
          <p style={{ fontSize: '0.875rem', color: 'var(--muted)' }}>or click to browse · JPG, PNG, WebP</p>
        </div>
      )}
    </div>
  );
}

const card: React.CSSProperties = {
  backgroundColor: 'var(--surface)',
  border: '1px solid var(--border)',
  borderRadius: '1rem',
  padding: '1.5rem',
};

export default function Home() {
  const [file,    setFile]    = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result,  setResult]  = useState<ApiResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);
  const [channel, setChannel] = useState({ subs: 100_000, avgPerf: 0.003, niche: 'AI/Tech' });

  function handleFile(f: File) {
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError(null);
  }

  async function handleScore() {
    if (!file) return;
    setLoading(true); setError(null);
    const form = new FormData();
    form.append('file', file);
    form.append('subscriber_count', String(channel.subs));
    form.append('channel_avg_perf', String(channel.avgPerf));
    form.append('niche', channel.niche);
    try {
      const res = await fetch('http://localhost:7800/score', { method: 'POST', body: form });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail ?? 'Server error'); }
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally { setLoading(false); }
  }

  const sectionLabel: React.CSSProperties = {
    fontSize: '0.7rem', fontWeight: 700, letterSpacing: '0.1em',
    textTransform: 'uppercase', color: 'var(--muted)', marginBottom: '1rem',
  };

  return (
    <main style={{ minHeight: '100vh', padding: '2.5rem 1rem', backgroundColor: 'var(--bg)' }}>
      <div style={{ maxWidth: 900, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>

        {/* Header */}
        <div style={{ textAlign: 'center' }}>
          <h1 style={{
            fontSize: '2.5rem', fontWeight: 800, letterSpacing: '-0.02em',
            background: 'linear-gradient(135deg,#6366f1,#8b5cf6)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
            marginBottom: '0.5rem',
          }}>Thumbnail Scorer</h1>
          <p style={{ color: 'var(--muted)', fontSize: '0.9rem' }}>
            AI score based on your channel&apos;s real performance data
          </p>
        </div>

        {/* Input card */}
        <div style={card}>
          <DropZone onFile={handleFile} preview={preview} />
          <div style={{ marginTop: '1.25rem' }}>
            <ChannelForm {...channel} onChange={(k, v) => setChannel(c => ({ ...c, [k]: v }))} />
          </div>
          <button onClick={handleScore} disabled={!file || loading} style={{
            marginTop: '1.25rem', width: '100%', padding: '0.75rem',
            borderRadius: '0.75rem', border: 'none', cursor: file && !loading ? 'pointer' : 'not-allowed',
            fontWeight: 600, fontSize: '0.95rem', color: 'white',
            background: file && !loading ? 'linear-gradient(135deg,#6366f1,#8b5cf6)' : 'var(--surface2)',
            opacity: !file || loading ? 0.5 : 1, transition: 'all 0.2s',
          }}>
            {loading ? 'Analysing…' : 'Score this thumbnail'}
          </button>
          {error && <p style={{ color: '#f87171', textAlign: 'center', fontSize: '0.875rem', marginTop: '0.75rem' }}>{error}</p>}
        </div>

        {/* Results */}
        {result && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }} className="animate-fade-up">

            {/* Score ring + Radar */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
              <div style={{ ...card, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                <p style={sectionLabel}>Overall Score</p>
                <ScoreRing score={result.score} max={result.score_max} />
              </div>
              <div style={card}>
                <p style={sectionLabel}>Dimensions</p>
                <ThumbnailRadar dimensions={result.dimensions} />
              </div>
            </div>

            {/* Dimension bars */}
            <div style={card}>
              <p style={sectionLabel}>Score Breakdown</p>
              <DimensionBars dimensions={result.dimensions} />
            </div>

            {/* Vision badges */}
            <div style={card}>
              <p style={sectionLabel}>Vision API Detection</p>
              <VisionBadges vision={result.vision} />
            </div>

          </div>
        )}
      </div>
    </main>
  );
}
