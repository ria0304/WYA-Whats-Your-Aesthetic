import React, { useState, useEffect, useRef } from 'react';
import { Download, Share2, RefreshCw, Sparkles } from 'lucide-react';
import { api } from '../services/api';

interface AuraData {
  primaryAesthetic: string;
  primaryPercent: number;
  secondaryAesthetic: string;
  secondaryPercent: number;
  tertiaryAesthetic: string;
  tertiaryPercent: number;
  moodTag: string;
  seasonTag: string;
  dominantColors: string[];
  wardrobeCount: number;
  topCategory: string;
}

const AESTHETIC_CONFIGS: Record<string, { emoji: string; vibe: string }> = {
  'Grunge-Core':   { emoji: '🖤', vibe: 'Raw & Unapologetic'  },
  'Minimalist':    { emoji: '🤍', vibe: 'Less Is Everything'  },
  'Cottagecore':   { emoji: '🌸', vibe: 'Soft & Botanical'    },
  'Dark Academia': { emoji: '📚', vibe: 'Intellectual Moody'  },
  'Y2K Futurist':  { emoji: '✨', vibe: 'Born Iconic'          },
  'Old Money':     { emoji: '🏛️', vibe: 'Quietly Luxurious'   },
  'Streetwear':    { emoji: '🔥', vibe: 'Culture Architect'   },
  'Bohemian':      { emoji: '🌙', vibe: 'Free-Spirited Soul'  },
  'Preppy':        { emoji: '⚓', vibe: 'Classic Elevated'    },
  'Avant-Garde':   { emoji: '🎭', vibe: 'Art Walking'         },
  'Classic Chic':  { emoji: '🕊️', vibe: 'Timeless Power'      },
  'Eclectic':      { emoji: '🌈', vibe: 'Beautifully Chaotic' },
};

const getConfig = (aesthetic: string) =>
  AESTHETIC_CONFIGS[aesthetic] || { emoji: '💫', vibe: 'Uniquely You' };

function buildAuraFromData(dna: any, stats: any): AuraData {
  const summary: string = dna?.summary || '';
  const aesthetics = Object.keys(AESTHETIC_CONFIGS);
  const mentioned = aesthetics.filter(a => summary.toLowerCase().includes(a.toLowerCase()));
  return {
    primaryAesthetic:   dna?.primary_aesthetic   || mentioned[0] || 'Classic Chic',
    primaryPercent:     dna?.primary_percent      ?? 60,
    secondaryAesthetic: dna?.secondary_aesthetic  || mentioned[1] || 'Minimalist',
    secondaryPercent:   dna?.secondary_percent    ?? 30,
    tertiaryAesthetic:  dna?.tertiary_aesthetic   || mentioned[2] || 'Eclectic',
    tertiaryPercent:    dna?.tertiary_percent      ?? 10,
    moodTag:            dna?.mood_tag             || 'Effortlessly Curated',
    seasonTag:          dna?.season_tag           || 'Perennial Soul',
    dominantColors:     dna?.dominant_colors      || stats?.dominant_colors || ['#c4a882','#2d2d2d','#f5f0e5','#6b3f2a'],
    wardrobeCount:      stats?.wardrobe_count      || 0,
    topCategory:        stats?.top_category        || 'Outerwear',
  };
}

async function captureCard(el: HTMLElement): Promise<Blob> {
  const html2canvas = (await import('html2canvas')).default;
  const canvas = await html2canvas(el, {
    useCORS: true,
    allowTaint: true,
    scale: 3,
    backgroundColor: null,
    logging: false,
  });
  return new Promise((resolve, reject) => {
    canvas.toBlob(b => b ? resolve(b) : reject(new Error('Blob failed')), 'image/png', 1.0);
  });
}

const downloadBlob = (blob: Blob, filename: string) => {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  a.href     = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
};

const AestheticAura: React.FC = () => {
  const [aura, setAura]       = useState<AuraData | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy]       = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const cardRef               = useRef<HTMLDivElement>(null);

  const fetchAura = async () => {
    setLoading(true); setError(null);
    try {
      const profile      = await api.profile.get();
      const [dna, stats] = await Promise.all([api.style.getDNA(profile.user_id), api.dashboard.getStats()]);
      if (!dna?.has_dna) { setError('Take the Style Quiz first to generate your Aesthetic Aura!'); setLoading(false); return; }
      setAura(buildAuraFromData(dna, stats));
    } catch (e) {
      console.error(e);
      setError('Could not load your aura. Please try again.');
    } finally { setLoading(false); }
  };

  useEffect(() => { fetchAura(); }, []);

  const handleShare = async () => {
    if (!aura || !cardRef.current || busy) return;
    setBusy(true);
    try {
      const blob = await captureCard(cardRef.current);
      const file = new File([blob], 'aesthetic-aura.png', { type: 'image/png' });
      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        await navigator.share({ files: [file], title: 'My Aesthetic Aura — WYA', text: `I'm ${aura.primaryPercent}% ${aura.primaryAesthetic}. What's your aesthetic? ✨` });
      } else if (navigator.share) {
        await navigator.share({ title: 'My Aesthetic Aura — WYA', text: `I'm ${aura.primaryPercent}% ${aura.primaryAesthetic}. What's your aesthetic? ✨`, url: 'https://wya.app' });
      } else {
        downloadBlob(blob, 'aesthetic-aura.png');
      }
    } catch (e: any) { if (e?.name !== 'AbortError') console.error('Share failed', e); }
    finally { setBusy(false); }
  };

  const handleSave = async () => {
    if (!aura || !cardRef.current || busy) return;
    setBusy(true);
    try {
      const blob = await captureCard(cardRef.current);
      downloadBlob(blob, `aesthetic-aura-${aura.primaryAesthetic.toLowerCase().replace(/\s+/g, '-')}.png`);
    } catch (e) { console.error('Save failed', e); }
    finally { setBusy(false); }
  };

  if (loading) return (
    <div className="min-h-full flex flex-col items-center justify-center bg-white p-8 gap-6">
      <div className="relative">
        <div className="w-24 h-24 rounded-full animate-pulse" style={{ background: 'linear-gradient(135deg,#f0b4cc,#d1c1e8,#b8d1eb,#93d9cc)' }} />
        <Sparkles className="w-8 h-8 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-purple-400 animate-spin" />
      </div>
      <div className="text-center">
        <p className="text-[10px] font-black uppercase tracking-[5px] text-slate-400 animate-pulse">Channeling Your Aura...</p>
        <p className="text-xs text-slate-300 mt-2">Scanning your wardrobe DNA</p>
      </div>
    </div>
  );

  if (error) return (
    <div className="min-h-full flex flex-col items-center justify-center bg-white p-8 gap-6 text-center">
      <div className="w-20 h-20 rounded-full flex items-center justify-center" style={{ background: 'linear-gradient(135deg,#f0b4cc44,#d1c1e844)' }}>
        <Sparkles className="w-10 h-10 text-pink-300" />
      </div>
      <p className="text-slate-600 font-bold">{error}</p>
      <button onClick={fetchAura} className="px-8 py-4 gradient-bg text-white rounded-full font-black text-xs uppercase tracking-widest shadow-xl">Retry</button>
    </div>
  );

  if (!aura) return null;
  const cfg = getConfig(aura.primaryAesthetic);

  return (
    <div className="min-h-full bg-white p-6 pb-28">

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl serif text-slate-800">Aesthetic Aura</h1>
          <p className="text-[10px] text-slate-400 uppercase tracking-widest font-bold mt-1">Your style — wrapped</p>
        </div>
        <button onClick={fetchAura} className="p-3 bg-slate-50 rounded-full hover:bg-slate-100 transition-colors">
          <RefreshCw className="w-5 h-5 text-slate-400" />
        </button>
      </div>

      {/* ── AURA CARD ─────────────────────────────────────────────── */}
      <div
        ref={cardRef}
        className="relative rounded-[40px] overflow-hidden shadow-2xl mb-6 aspect-[9/16] max-h-[600px]"
        style={{ background: 'linear-gradient(145deg, #f0b4cc 0%, #d1c1e8 35%, #b8d1eb 68%, #93d9cc 100%)' }}
      >
        {/* Glow orbs */}
        <div className="absolute top-8 -left-12 w-52 h-52 rounded-full blur-3xl opacity-50"
          style={{ background: 'radial-gradient(circle, #ffffff88, transparent)' }} />
        <div className="absolute bottom-16 -right-10 w-60 h-60 rounded-full blur-3xl opacity-40"
          style={{ background: 'radial-gradient(circle, #ffffff66, transparent)' }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-72 h-72 rounded-full blur-3xl opacity-20"
          style={{ background: 'radial-gradient(circle, #e8d5f5, transparent)' }} />

        {/* Inner glass border */}
        <div className="absolute inset-[10px] rounded-[32px] border border-white/50 pointer-events-none" />

        {/* Noise */}
        <div className="absolute inset-0 opacity-[0.04]" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
        }} />

        <div className="relative z-10 h-full flex flex-col justify-between p-8">

          {/* Top */}
          <div>
            <p className="text-[9px] font-black uppercase tracking-[6px] mb-2" style={{ color: 'rgba(255,255,255,0.7)' }}>WYA — What's Your Aesthetic</p>
            <div className="w-10 h-0.5 mb-5" style={{ background: 'linear-gradient(to right,rgba(255,255,255,0.6),transparent)' }} />
            <p className="text-xs font-black uppercase tracking-widest mb-1" style={{ color: 'rgba(80,30,90,0.85)' }}>{aura.moodTag}</p>
            <p className="text-[9px] uppercase tracking-widest font-semibold" style={{ color: 'rgba(80,30,90,0.55)' }}>{aura.seasonTag}</p>
          </div>

          {/* Primary */}
          <div className="text-center py-6">
            <div
              className="inline-flex items-center justify-center w-20 h-20 rounded-full mb-5 shadow-lg"
              style={{ background: 'rgba(255,255,255,0.45)', backdropFilter: 'blur(12px)', border: '1.5px solid rgba(255,255,255,0.7)' }}
            >
              <span className="text-4xl">{cfg.emoji}</span>
            </div>

            <div
              className="inline-block rounded-full px-6 py-2 mb-4"
              style={{ background: 'rgba(255,255,255,0.35)', backdropFilter: 'blur(8px)', border: '1px solid rgba(255,255,255,0.65)' }}
            >
              <span className="text-[9px] font-black uppercase tracking-[4px]" style={{ color: 'rgba(80,30,90,0.7)' }}>You Are</span>
            </div>

            <h2
              className="text-6xl font-black leading-tight tracking-tight"
              style={{ color: 'rgba(255,255,255,0.95)', textShadow: '0 2px 24px rgba(160,80,200,0.3)' }}
            >
              {aura.primaryPercent}%
            </h2>
            <h3
              className="text-2xl font-bold mt-1 serif"
              style={{ color: 'rgba(255,255,255,0.9)', textShadow: '0 2px 12px rgba(160,80,200,0.2)' }}
            >
              {aura.primaryAesthetic}
            </h3>
            <p className="text-[10px] uppercase tracking-widest mt-2 font-bold" style={{ color: 'rgba(80,30,90,0.6)' }}>{cfg.vibe}</p>
          </div>

          {/* Bars */}
          <div className="space-y-3 px-1">
            {[
              { label: aura.primaryAesthetic,   pct: aura.primaryPercent,   primary: true },
              { label: aura.secondaryAesthetic, pct: aura.secondaryPercent, primary: false },
              { label: aura.tertiaryAesthetic,  pct: aura.tertiaryPercent,  primary: false },
            ].map((item) => (
              <div key={item.label}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-[9px] font-black uppercase tracking-widest"
                    style={{ color: item.primary ? 'rgba(70,20,80,0.9)' : 'rgba(70,20,80,0.55)' }}>
                    {item.label}
                  </span>
                  <span className="text-[9px] font-black"
                    style={{ color: item.primary ? 'rgba(70,20,80,0.9)' : 'rgba(70,20,80,0.55)' }}>
                    {item.pct}%
                  </span>
                </div>
                <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.3)' }}>
                  <div className="h-full rounded-full" style={{
                    width: `${item.pct}%`,
                    background: item.primary
                      ? 'linear-gradient(to right, #c084d4, #8bc4e8)'
                      : 'rgba(255,255,255,0.55)',
                  }} />
                </div>
              </div>
            ))}
          </div>

          {/* Palette */}
          <div>
            <p className="text-[8px] uppercase tracking-[4px] font-black mb-3" style={{ color: 'rgba(80,30,90,0.5)' }}>Wardrobe Palette</p>
            <div className="flex gap-2 items-center">
              {aura.dominantColors.slice(0, 5).map((color, i) => (
                <div key={i} className="w-8 h-8 rounded-full shadow-md"
                  style={{ backgroundColor: color, border: '2px solid rgba(255,255,255,0.65)' }} />
              ))}
              <div className="ml-auto text-right">
                <p className="text-[8px] uppercase tracking-widest font-semibold" style={{ color: 'rgba(80,30,90,0.5)' }}>{aura.wardrobeCount} pieces</p>
                <p className="text-[9px] font-bold uppercase" style={{ color: 'rgba(80,30,90,0.7)' }}>{aura.topCategory} Heavy</p>
              </div>
            </div>
          </div>

        </div>
      </div>
      {/* ── END CARD ─────────────────────────────────────────────── */}

      {/* Buttons */}
      <div className="grid grid-cols-2 gap-4">
        <button
          onClick={handleShare}
          disabled={busy}
          className="flex items-center justify-center gap-2 py-5 gradient-bg text-white rounded-full font-black uppercase tracking-[2px] text-[10px] shadow-lg active:scale-95 transition-all disabled:opacity-60"
        >
          {busy ? <><Sparkles className="w-4 h-4 animate-spin" /> Working...</> : <><Share2 className="w-4 h-4" /> Share Story</>}
        </button>
        <button
          onClick={handleSave}
          disabled={busy}
          className="flex items-center justify-center gap-2 py-5 bg-slate-900 text-white rounded-full font-black uppercase tracking-[2px] text-[10px] active:scale-95 transition-all disabled:opacity-60"
        >
          <Download className="w-4 h-4" /> Save Card
        </button>
      </div>

      {/* Stats */}
      <div className="mt-6 grid grid-cols-3 gap-3">
        {[
          { value: `${aura.primaryPercent}%`, label: aura.primaryAesthetic },
          { value: aura.wardrobeCount,         label: 'Pieces' },
          { value: 3,                           label: 'Aesthetics' },
        ].map((s, i) => (
          <div key={i} className="rounded-[24px] p-4 text-center border"
            style={{ background: 'linear-gradient(145deg,#fdf0f6,#f0eafc)', borderColor: '#e8d8f0' }}>
            <p className="text-2xl font-black text-slate-800">{s.value}</p>
            <p className="text-[8px] font-black uppercase tracking-widest text-slate-400 mt-1">{s.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AestheticAura;
