import React, { useState, useEffect, useRef } from 'react';
import { Download, Share2, RefreshCw, Sparkles, Loader2 } from 'lucide-react';
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

const AESTHETIC_CONFIGS: Record<string, { gradient: string; emoji: string; vibe: string }> = {
  'Grunge-Core':   { gradient: 'from-zinc-900 via-stone-700 to-zinc-800',   emoji: '🖤', vibe: 'Raw & Unapologetic' },
  'Minimalist':    { gradient: 'from-stone-100 via-zinc-200 to-slate-300',   emoji: '🤍', vibe: 'Less Is Everything' },
  'Cottagecore':   { gradient: 'from-rose-200 via-amber-100 to-green-200',   emoji: '🌸', vibe: 'Soft & Botanical' },
  'Dark Academia': { gradient: 'from-amber-900 via-stone-800 to-slate-900',  emoji: '📚', vibe: 'Intellectual Moody' },
  'Y2K Futurist':  { gradient: 'from-pink-400 via-purple-400 to-cyan-400',   emoji: '✨', vibe: 'Born Iconic' },
  'Old Money':     { gradient: 'from-amber-50 via-stone-200 to-slate-400',   emoji: '🏛️', vibe: 'Quietly Luxurious' },
  'Streetwear':    { gradient: 'from-slate-900 via-zinc-700 to-orange-500',  emoji: '🔥', vibe: 'Culture Architect' },
  'Bohemian':      { gradient: 'from-amber-300 via-orange-300 to-rose-400',  emoji: '🌙', vibe: 'Free-Spirited Soul' },
  'Preppy':        { gradient: 'from-blue-700 via-blue-500 to-green-400',    emoji: '⚓', vibe: 'Classic Elevated' },
  'Avant-Garde':   { gradient: 'from-purple-900 via-fuchsia-600 to-pink-400',emoji: '🎭', vibe: 'Art Walking' },
  'Classic Chic':  { gradient: 'from-slate-700 via-gray-500 to-slate-300',   emoji: '🕊️', vibe: 'Timeless Power' },
  'Eclectic':      { gradient: 'from-yellow-400 via-pink-500 to-purple-600', emoji: '🌈', vibe: 'Beautifully Chaotic' },
};

const getConfig = (aesthetic: string) =>
  AESTHETIC_CONFIGS[aesthetic] || { gradient: 'from-pink-300 via-purple-300 to-blue-300', emoji: '💫', vibe: 'Uniquely You' };

/** Build aura from Style DNA summary + wardrobe stats */
function buildAuraFromData(dna: any, stats: any): AuraData {
  // The backend should ideally return these directly; this is a robust fallback parser
  const summary: string = dna?.summary || '';
  const aesthetics = Object.keys(AESTHETIC_CONFIGS);

  // Find which aesthetics appear in the summary string
  const mentioned = aesthetics.filter(a => summary.toLowerCase().includes(a.toLowerCase()));

  const primary = mentioned[0] || 'Classic Chic';
  const secondary = mentioned[1] || 'Minimalist';
  const tertiary = mentioned[2] || 'Eclectic';

  // Use backend-provided percentages if available, otherwise distribute
  const primaryPct   = dna?.primary_percent   ?? 60;
  const secondaryPct = dna?.secondary_percent ?? 30;
  const tertiaryPct  = dna?.tertiary_percent  ?? 10;

  // Derive mood + season from summary keywords
  let moodTag = dna?.mood_tag || 'Effortlessly Curated';
  let seasonTag = dna?.season_tag || 'Perennial Soul';

  return {
    primaryAesthetic:   dna?.primary_aesthetic   || primary,
    primaryPercent:     primaryPct,
    secondaryAesthetic: dna?.secondary_aesthetic || secondary,
    secondaryPercent:   secondaryPct,
    tertiaryAesthetic:  dna?.tertiary_aesthetic  || tertiary,
    tertiaryPercent:    tertiaryPct,
    moodTag,
    seasonTag,
    dominantColors:     dna?.dominant_colors     || stats?.dominant_colors || ['#c4a882','#2d2d2d','#f5f0e5','#6b3f2a'],
    wardrobeCount:      stats?.wardrobe_count     || 0,
    topCategory:        stats?.top_category       || 'Outerwear',
  };
}

const AestheticAura: React.FC = () => {
  const [aura, setAura] = useState<AuraData | null>(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);

  const fetchAura = async () => {
    setLoading(true);
    setError(null);
    try {
      // Get user profile for ID
      const profile = await api.profile.get();
      const [dna, stats] = await Promise.all([
        api.style.getDNA(profile.user_id),
        api.dashboard.getStats(),
      ]);

      if (!dna?.has_dna) {
        setError('Take the Style Quiz first to generate your Aesthetic Aura!');
        setLoading(false);
        return;
      }

      setAura(buildAuraFromData(dna, stats));
    } catch (e) {
      console.error('Aura fetch failed', e);
      setError('Could not load your aura. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchAura(); }, []);

  const handleShare = async () => {
    if (!aura) return;
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'My Aesthetic Aura — WYA',
          text: `I'm ${aura.primaryPercent}% ${aura.primaryAesthetic}. What's your aesthetic? ✨`,
          url: 'https://wya.app',
        });
      } catch (e) {
        console.log('Share cancelled');
      }
    }
  };

  if (loading) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center bg-white p-8 gap-6">
        <div className="relative">
          <div className="w-24 h-24 rounded-full bg-gradient-to-br from-pink-200 via-purple-200 to-blue-200 animate-pulse" />
          <Sparkles className="w-8 h-8 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-purple-400 animate-spin" />
        </div>
        <div className="text-center">
          <p className="text-[10px] font-black uppercase tracking-[5px] text-slate-400 animate-pulse">Channeling Your Aura...</p>
          <p className="text-xs text-slate-300 mt-2">Scanning your wardrobe DNA</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center bg-white p-8 gap-6 text-center">
        <div className="w-20 h-20 bg-pink-50 rounded-full flex items-center justify-center">
          <Sparkles className="w-10 h-10 text-pink-200" />
        </div>
        <p className="text-slate-600 font-bold">{error}</p>
        <button onClick={fetchAura} className="px-8 py-4 gradient-bg text-white rounded-full font-black text-xs uppercase tracking-widest shadow-xl">
          Retry
        </button>
      </div>
    );
  }

  if (!aura) return null;
  const cfg = getConfig(aura.primaryAesthetic);

  return (
    <div className="min-h-full bg-white p-6 pb-28">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl serif text-slate-800">Aesthetic Aura</h1>
          <p className="text-[10px] text-slate-400 uppercase tracking-widest font-bold mt-1">Your style — wrapped</p>
        </div>
        <button onClick={fetchAura} className="p-3 bg-slate-50 rounded-full hover:bg-slate-100 transition-colors">
          <RefreshCw className="w-5 h-5 text-slate-400" />
        </button>
      </div>

      {/* SHARE CARD */}
      <div ref={cardRef} className="relative rounded-[40px] overflow-hidden shadow-2xl mb-6 aspect-[9/16] max-h-[600px]">
        <div className={`absolute inset-0 bg-gradient-to-br ${cfg.gradient}`} />

        {/* Blur orbs */}
        <div className="absolute top-10 -left-10 w-48 h-48 rounded-full bg-white/10 blur-3xl" />
        <div className="absolute bottom-20 -right-10 w-56 h-56 rounded-full bg-white/10 blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 rounded-full bg-black/5 blur-3xl" />

        {/* Noise texture */}
        <div className="absolute inset-0 opacity-[0.03]" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        }} />

        <div className="relative z-10 h-full flex flex-col justify-between p-8">
          {/* Header */}
          <div>
            <p className="text-white/60 text-[9px] font-black uppercase tracking-[6px] mb-2">WYA — What's Your Aesthetic</p>
            <div className="w-8 h-0.5 bg-white/40 mb-6" />
            <p className="text-white/80 text-xs font-bold uppercase tracking-widest mb-1">{aura.moodTag}</p>
            <p className="text-white/50 text-[9px] uppercase tracking-widest">{aura.seasonTag}</p>
          </div>

          {/* Primary */}
          <div className="text-center py-8">
            <div className="text-6xl mb-4">{cfg.emoji}</div>
            <div className="inline-block bg-white/15 backdrop-blur-md rounded-full px-6 py-2 mb-4 border border-white/20">
              <span className="text-white/70 text-[9px] font-black uppercase tracking-[4px]">You are</span>
            </div>
            <h2 className="text-5xl font-black text-white leading-tight tracking-tight" style={{ textShadow: '0 4px 20px rgba(0,0,0,0.3)' }}>
              {aura.primaryPercent}%
            </h2>
            <h3 className="text-2xl font-bold text-white/90 mt-1 serif">{aura.primaryAesthetic}</h3>
            <p className="text-white/50 text-[10px] uppercase tracking-widest mt-2 font-bold">{cfg.vibe}</p>
          </div>

          {/* Breakdown bars */}
          <div className="space-y-3">
            {[
              { label: aura.primaryAesthetic,   pct: aura.primaryPercent },
              { label: aura.secondaryAesthetic, pct: aura.secondaryPercent },
              { label: aura.tertiaryAesthetic,  pct: aura.tertiaryPercent },
            ].map((item) => (
              <div key={item.label}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-white/70 text-[9px] font-black uppercase tracking-widest">{item.label}</span>
                  <span className="text-white/70 text-[9px] font-black">{item.pct}%</span>
                </div>
                <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full bg-white/60 rounded-full" style={{ width: `${item.pct}%` }} />
                </div>
              </div>
            ))}
          </div>

          {/* Palette */}
          <div>
            <p className="text-white/40 text-[8px] uppercase tracking-[4px] font-black mb-3">Wardrobe Palette</p>
            <div className="flex gap-2 items-center">
              {aura.dominantColors.slice(0, 5).map((color, i) => (
                <div key={i} className="w-8 h-8 rounded-full border-2 border-white/20 shadow-lg" style={{ backgroundColor: color }} />
              ))}
              <div className="ml-auto text-right">
                <p className="text-white/40 text-[8px] uppercase tracking-widest">{aura.wardrobeCount} pieces</p>
                <p className="text-white/60 text-[9px] font-bold uppercase">{aura.topCategory} Heavy</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Action buttons */}
      <div className="grid grid-cols-2 gap-4">
        <button
          onClick={handleShare}
          className="flex items-center justify-center gap-2 py-5 gradient-bg text-white rounded-full font-black uppercase tracking-[2px] text-[10px] shadow-lg active:scale-95 transition-all"
        >
          {copied ? <><Sparkles className="w-4 h-4" /> Shared!</> : <><Share2 className="w-4 h-4" /> Share Story</>}
        </button>
        <button className="flex items-center justify-center gap-2 py-5 bg-slate-900 text-white rounded-full font-black uppercase tracking-[2px] text-[10px] active:scale-95 transition-all">
          <Download className="w-4 h-4" /> Save Card
        </button>
      </div>

      {/* Stats */}
      <div className="mt-6 grid grid-cols-3 gap-3">
        <div className="bg-slate-50 rounded-[24px] p-4 text-center border border-slate-100">
          <p className="text-2xl font-black text-slate-800">{aura.primaryPercent}%</p>
          <p className="text-[8px] font-black uppercase tracking-widest text-slate-400 mt-1">{aura.primaryAesthetic}</p>
        </div>
        <div className="bg-slate-50 rounded-[24px] p-4 text-center border border-slate-100">
          <p className="text-2xl font-black text-slate-800">{aura.wardrobeCount}</p>
          <p className="text-[8px] font-black uppercase tracking-widest text-slate-400 mt-1">Pieces</p>
        </div>
        <div className="bg-slate-50 rounded-[24px] p-4 text-center border border-slate-100">
          <p className="text-2xl font-black text-slate-800">3</p>
          <p className="text-[8px] font-black uppercase tracking-widest text-slate-400 mt-1">Aesthetics</p>
        </div>
      </div>
    </div>
  );
};

export default AestheticAura;
