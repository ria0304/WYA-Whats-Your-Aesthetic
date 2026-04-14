import React, { useState, useEffect, useRef } from 'react';
import {
  Sparkles, RefreshCw, Box, Loader2, ArrowRight, Heart, Star,
  Layout, Plus, X, Shirt, Image as ImageIcon, Check, Trash2, CheckCircle2, Palette
} from 'lucide-react';
import { api } from '../services/api';
import { WardrobeItem, OutfitSet } from '../types';

// Helper: detect dominant hue family from a hex/name color string
const getColorFamily = (color: string): string => {
  if (!color) return 'neutral';
  const c = color.toLowerCase();
  if (/black|jet|onyx|ebony/.test(c)) return 'black';
  if (/white|ivory|cream|beige|ecru|linen/.test(c)) return 'white';
  if (/red|rose|crimson|wine|burgundy|maroon/.test(c)) return 'red';
  if (/blue|navy|cobalt|azure|indigo|sapphire/.test(c)) return 'blue';
  if (/green|olive|sage|mint|forest|emerald/.test(c)) return 'green';
  if (/brown|camel|tan|khaki|chocolate|coffee/.test(c)) return 'brown';
  if (/pink|blush|coral|salmon|mauve/.test(c)) return 'pink';
  if (/yellow|gold|mustard|lemon|amber/.test(c)) return 'yellow';
  if (/purple|violet|plum|lavender|lilac/.test(c)) return 'purple';
  if (/grey|gray|charcoal|slate/.test(c)) return 'grey';
  if (/orange|terracotta|rust|copper/.test(c)) return 'orange';
  return 'neutral';
};

const HARMONY_BADGES: Record<string, { label: string; color: string }> = {
  complementary: { label: 'Complementary', color: 'bg-violet-50 text-violet-600 border-violet-100' },
  analogous:     { label: 'Analogous',     color: 'bg-amber-50  text-amber-600  border-amber-100' },
  monochrome:    { label: 'Monochrome',    color: 'bg-slate-100 text-slate-600  border-slate-200' },
  neutral:       { label: 'Neutral Mix',   color: 'bg-stone-50  text-stone-600  border-stone-100' },
  contrast:      { label: 'High Contrast', color: 'bg-rose-50   text-rose-600   border-rose-100'  },
};

/** Very simple client-side harmony hint (backend color_harmony.json drives the actual logic) */
function detectHarmony(colors: string[]): string {
  const families = colors.map(getColorFamily);
  const unique = [...new Set(families)];
  if (unique.length === 1) return 'monochrome';
  if (unique.includes('black') && unique.includes('white')) return 'contrast';
  if (unique.every(f => ['black', 'white', 'grey', 'neutral'].includes(f))) return 'neutral';
  const COMPLEMENTARY_PAIRS = [['red', 'green'], ['blue', 'orange'], ['yellow', 'purple'], ['pink', 'green']];
  if (COMPLEMENTARY_PAIRS.some(([a, b]) => unique.includes(a) && unique.includes(b))) return 'complementary';
  return 'analogous';
}

// ─── Main component ─────────────────────────────────────────────────────────
const Curate: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [outfits, setOutfits] = useState<OutfitSet[]>([]);
  const [wardrobe, setWardrobe] = useState<WardrobeItem[]>([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [loggingIds, setLoggingIds] = useState<Set<string>>(new Set());
  const [successIds, setSuccessIds] = useState<Set<string>>(new Set());
  const [savingManual, setSavingManual] = useState(false);

  // Manual Outfit State
  const [manualTop, setManualTop] = useState<WardrobeItem | string | null>(null);
  const [manualBottom, setManualBottom] = useState<WardrobeItem | string | null>(null);
  const [manualName, setManualName] = useState('');

  const topInputRef = useRef<HTMLInputElement>(null);
  const bottomInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { fetchAndCurate(); }, []);

  const fetchAndCurate = async () => {
    setLoading(true);
    try {
      const data = await api.wardrobe.getAll();
      const mapped: WardrobeItem[] = data.map((item: any) => ({
        id: item.item_id || item.id.toString(),
        name: item.name,
        category: item.category,
        color: item.color,
        fabric: item.fabric,
        imageUrl: item.image_url,
        wearCount: item.wear_count || 0,
        isFavorite: false
      }));
      setWardrobe(mapped);

      // ── Load server-persisted outfits first ──────────────────────────────
      let serverOutfits: OutfitSet[] = [];
      try {
        const raw = await api.outfits.getAll();
        serverOutfits = (raw || []).map((o: any) => ({
          id: `server-${o.id}`,
          serverId: o.id,
          name: o.name,
          vibe: o.vibe,
          isManual: o.is_manual,
          isDaily: o.is_daily,
          createdDate: o.created_date,
          wornAt: o.worn_at,
          items: (o.item_ids || []).map((id: string) => mapped.find(m => m.id === id)).filter(Boolean) as WardrobeItem[]
        }));
      } catch (e) {
        console.warn('Could not load server outfits, falling back to localStorage', e);
        // Migrate any legacy localStorage outfits to state
        const legacy: OutfitSet[] = JSON.parse(localStorage.getItem('wya_manual_outfits') || '[]');
        serverOutfits = legacy;
      }

      // ── Daily Drop logic ─────────────────────────────────────────────────
      const today = new Date().toDateString();
      const hasDaily = serverOutfits.find(o => o.isDaily && o.createdDate === today);

      if (!hasDaily && mapped.length > 0) {
        try {
          const aiSuggestions = await api.ai.curateOutfits(mapped);
          if (aiSuggestions.length > 0) {
            const suggestion = aiSuggestions[0];
            const suggestedItems: WardrobeItem[] = (suggestion.item_ids || [])
              .map((id: string) => mapped.find(m => m.id === id))
              .filter(Boolean) as WardrobeItem[];

            if (suggestedItems.length > 0) {
              const newDaily: OutfitSet = {
                id: `daily-${Date.now()}`,
                name: 'Daily Drop',
                vibe: suggestion.vibe,
                items: suggestedItems,
                isDaily: true,
                createdDate: today
              };
              // Persist server-side
              try {
                const saved = await api.outfits.save({
                  name: newDaily.name,
                  vibe: newDaily.vibe,
                  item_ids: suggestedItems.map(i => i.id),
                  is_daily: true,
                  created_date: today
                });
                newDaily.id = `server-${saved.id}`;
                newDaily.serverId = saved.id;
              } catch (e) {
                console.warn('Could not persist daily drop to server');
              }
              serverOutfits = [newDaily, ...serverOutfits];
            }
          }
        } catch (err) {
          console.warn('Daily curate failed', err);
        }
      }

      // ── Generate transient AI sets if we have fewer than 3 ──────────────
      if (serverOutfits.length < 3 && mapped.length > 1) {
        await generateAIOutfits(mapped, serverOutfits);
      } else {
        setOutfits(serverOutfits);
        setLoading(false);
      }
    } catch (e) {
      console.error('Fetch failed', e);
      setLoading(false);
    }
  };

  const generateAIOutfits = async (items: WardrobeItem[], existing: OutfitSet[] = []) => {
    try {
      if (items.length < 2) { setOutfits(existing); setLoading(false); return; }
      const aiSuggestions = await api.ai.curateOutfits(items);
      const aiSets: OutfitSet[] = [];
      for (const suggestion of aiSuggestions) {
        const suggestedItems = (suggestion.item_ids || [])
          .map((id: string) => items.find(i => i.id === id))
          .filter(Boolean) as WardrobeItem[];
        if (suggestedItems.length > 0) {
          aiSets.push({
            id: `ai-${Date.now()}-${Math.random()}`,
            name: suggestion.name,
            vibe: suggestion.vibe,
            items: suggestedItems
          });
        }
      }
      setOutfits([...existing, ...aiSets]);
    } catch (e) {
      console.error('AI Curation failed', e);
      const tops = items.filter(i => i.category === 'Top');
      const bottoms = items.filter(i => i.category === 'Bottom');
      if (tops.length > 0 && bottoms.length > 0) {
        setOutfits([...existing, { id: `fallback-${Date.now()}`, name: 'Classic Combo', vibe: 'Casual', items: [tops[0], bottoms[0]] }]);
      } else {
        setOutfits(existing);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleShuffle = async () => {
    setLoading(true);
    const savedServer = outfits.filter(o => o.serverId !== undefined);
    await generateAIOutfits(wardrobe, savedServer);
  };

  const handleManualImage = (e: React.ChangeEvent<HTMLInputElement>, target: 'top' | 'bottom') => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (target === 'top') setManualTop(reader.result as string);
        else setManualBottom(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const resetForm = () => { setManualTop(null); setManualBottom(null); setManualName(''); };

  const saveManualOutfit = async () => {
    if (!manualTop || !manualBottom) return;
    setSavingManual(true);

    const topUrl  = typeof manualTop   === 'string' ? manualTop   : (manualTop as WardrobeItem).imageUrl;
    const botUrl  = typeof manualBottom === 'string' ? manualBottom : (manualBottom as WardrobeItem).imageUrl;

    const newSet: OutfitSet = {
      id: `manual-${Date.now()}`,
      name: manualName || 'My Custom Look',
      vibe: 'Personalized',
      isManual: true,
      createdDate: new Date().toDateString(),
      items: [
        { id: 'manual-top', name: 'Manual Top', category: 'Top', color: 'Custom', fabric: 'Unknown', imageUrl: topUrl, isFavorite: false },
        { id: 'manual-bottom', name: 'Manual Bottom', category: 'Bottom', color: 'Custom', fabric: 'Unknown', imageUrl: botUrl, isFavorite: false }
      ]
    };

    // Persist to backend
    try {
      const saved = await api.outfits.save({
        name: newSet.name,
        vibe: newSet.vibe,
        item_ids: [],          // manual items have no wardrobe IDs
        image_urls: [topUrl, botUrl],
        is_manual: true,
        created_date: newSet.createdDate
      });
      newSet.id = `server-${saved.id}`;
      newSet.serverId = saved.id;
    } catch (e) {
      console.warn('Could not persist manual outfit to server, saving locally');
      // Fallback: keep in localStorage
      const legacy = JSON.parse(localStorage.getItem('wya_manual_outfits') || '[]');
      localStorage.setItem('wya_manual_outfits', JSON.stringify([newSet, ...legacy]));
    }

    setOutfits([newSet, ...outfits]);
    setShowAddModal(false);
    resetForm();
    setSavingManual(false);
  };

  const deleteOutfit = async (outfit: OutfitSet) => {
    if (outfit.serverId) {
      try { await api.outfits.delete(outfit.serverId); } catch (e) { console.warn('Server delete failed'); }
    } else {
      // Legacy localStorage cleanup
      const legacy = JSON.parse(localStorage.getItem('wya_manual_outfits') || '[]');
      localStorage.setItem('wya_manual_outfits', JSON.stringify(legacy.filter((o: any) => o.id !== outfit.id)));
    }
    setOutfits(outfits.filter(o => o.id !== outfit.id));
  };

  const handleLogWorn = async (outfit: OutfitSet) => {
    if (loggingIds.has(outfit.id)) return;
    const wornAt = new Date().toISOString();

    setLoggingIds(prev => new Set(prev).add(outfit.id));
    try {
      // Log each wardrobe item with worn_at timestamp
      await Promise.all(outfit.items.map(item => {
        if (!item.id.startsWith('manual-')) {
          return api.wardrobe.wear(item.id, wornAt);
        }
        return Promise.resolve();
      }));

      // Log the outfit itself if it has a server id
      if (outfit.serverId) {
        await api.outfits.logWorn(outfit.serverId, wornAt);
      }

      setSuccessIds(prev => new Set(prev).add(outfit.id));
      setTimeout(() => {
        setSuccessIds(prev => { const s = new Set(prev); s.delete(outfit.id); return s; });
      }, 2000);
    } catch (e) {
      console.error('Failed to log worn', e);
      alert('Could not update wear count. Please try again.');
    } finally {
      setLoggingIds(prev => { const s = new Set(prev); s.delete(outfit.id); return s; });
    }
  };

  return (
    <div className="p-6 bg-white min-h-full pb-32">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl serif">Curated Sets</h1>
          <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mt-1">Color Harmony · AI Active</p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => setShowAddModal(true)} className="w-12 h-12 gradient-bg rounded-full flex items-center justify-center text-white shadow-lg active:scale-90 transition-transform">
            <Plus className="w-5 h-5" />
          </button>
          <button onClick={handleShuffle} disabled={loading || wardrobe.length === 0} className="w-12 h-12 bg-pink-50 rounded-full flex items-center justify-center text-pink-500 shadow-sm border border-pink-100 active:scale-90 transition-transform disabled:opacity-50">
            <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-32">
          <div className="w-16 h-16 rounded-full border-4 border-slate-100 border-t-pink-500 animate-spin mb-6" />
          <p className="text-[10px] font-black text-slate-400 uppercase tracking-[5px] animate-pulse">Designing Looks...</p>
        </div>
      ) : outfits.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-32 px-10 text-center">
          <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mb-6">
            <Layout className="w-10 h-10 text-slate-200" />
          </div>
          <h3 className="text-lg font-bold text-slate-700 mb-2">No Sets Yet</h3>
          <p className="text-xs text-slate-400 leading-relaxed mb-8">Create your first outfit set or add items to your closet for AI suggestions.</p>
          <button onClick={() => setShowAddModal(true)} className="px-8 py-4 gradient-bg text-white rounded-full text-[10px] font-black uppercase tracking-widest shadow-xl flex items-center gap-2">
            <Plus className="w-4 h-4" /> Create Outfit
          </button>
        </div>
      ) : (
        <div className="space-y-8 animate-slide-up">
          {outfits.map((set) => {
            const itemColors = set.items.map(i => i.color);
            const harmony = detectHarmony(itemColors);
            const badge = HARMONY_BADGES[harmony];

            return (
              <div key={set.id} className="bg-slate-50 rounded-[45px] p-8 border border-white shadow-sm hover:shadow-md transition-shadow relative">
                <div className="flex justify-between items-start mb-6">
                  <div>
                    <div className="flex items-center gap-2 mb-3 flex-wrap">
                      <span className={`px-4 py-1.5 rounded-full text-[8px] font-black uppercase tracking-[3px] shadow-sm border ${set.isDaily ? 'bg-indigo-50 text-indigo-500 border-indigo-100' : (set.isManual ? 'bg-slate-100 text-slate-500 border-slate-200' : 'bg-white/80 text-pink-500 border-pink-50')}`}>
                        {set.isDaily ? 'Daily Drop' : (set.isManual ? 'Manual Curation' : set.vibe)}
                      </span>
                      {/* Color harmony badge */}
                      <span className={`px-3 py-1 rounded-full text-[7px] font-black uppercase tracking-[2px] border flex items-center gap-1 ${badge.color}`}>
                        <Palette className="w-2.5 h-2.5" /> {badge.label}
                      </span>
                      {set.isManual && <Box className="w-3 h-3 text-slate-300" />}
                      {set.isDaily && <Star className="w-3 h-3 text-indigo-300 fill-indigo-300" />}
                    </div>
                    <h3 className="text-xl serif text-slate-800">{set.name}</h3>
                    {set.wornAt && (
                      <p className="text-[8px] text-slate-400 font-bold mt-1 uppercase tracking-widest">
                        Last worn {new Date(set.wornAt).toLocaleDateString()}
                      </p>
                    )}
                  </div>
                  <button onClick={() => deleteOutfit(set)} className="w-10 h-10 bg-white rounded-full flex items-center justify-center text-slate-300 hover:text-red-400 transition-colors shadow-sm border border-slate-50"><Trash2 className="w-4 h-4" /></button>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {set.items.map((item, i) => (
                    <div key={i} className="bg-white rounded-3xl p-3 shadow-inner group overflow-hidden aspect-[4/5] relative flex flex-col items-center justify-center">
                      <div className="w-full h-full p-2 flex items-center justify-center">
                        {item.imageUrl
                          ? <img src={item.imageUrl} className="w-full h-full object-contain group-hover:scale-110 transition-transform duration-500" alt={item.name} />
                          : <Box className="w-10 h-10 text-slate-100" />
                        }
                      </div>
                      <div className="absolute bottom-3 left-3 right-3 bg-white/90 backdrop-blur-md px-3 py-2 rounded-xl border border-slate-50 shadow-sm transform translate-y-12 group-hover:translate-y-0 transition-transform">
                        <p className="text-[8px] font-black text-slate-800 uppercase truncate">{item.name}</p>
                        <p className={`text-[7px] font-black uppercase tracking-widest ${item.category === 'Top' ? 'text-indigo-400' : 'text-pink-400'}`}>{item.category}</p>
                      </div>
                    </div>
                  ))}
                </div>

                <button
                  onClick={() => handleLogWorn(set)}
                  disabled={loggingIds.has(set.id) || successIds.has(set.id)}
                  className={`w-full mt-6 py-4 rounded-3xl text-[9px] font-black uppercase tracking-[3px] flex items-center justify-center gap-2 transition-all group ${
                    successIds.has(set.id)
                      ? 'bg-emerald-500 text-white border border-emerald-500'
                      : 'bg-white border border-slate-100 text-slate-400 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  {loggingIds.has(set.id) ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    : successIds.has(set.id) ? <>Logged Successfully <CheckCircle2 className="w-3.5 h-3.5" /></>
                    : <>Log as Worn <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" /></>
                  }
                </button>
              </div>
            );
          })}
        </div>
      )}

      {/* Manual Curation Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-[100] flex items-end justify-center">
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => { setShowAddModal(false); resetForm(); }} />
          <div className="relative w-full max-w-md bg-white rounded-t-[50px] p-8 animate-slide-up max-h-[95vh] overflow-y-auto shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl serif text-slate-800">Curation Studio</h2>
              <button onClick={() => { setShowAddModal(false); resetForm(); }} className="p-3 hover:bg-slate-50 rounded-full transition-colors">
                <X className="w-6 h-6" />
              </button>
            </div>
            <div className="space-y-6">
              <input
                type="text"
                placeholder="Name your set (e.g. Sunday Brunch)"
                className="w-full bg-slate-50 border border-slate-100 rounded-2xl p-5 text-sm outline-none focus:ring-2 ring-indigo-100 font-bold"
                value={manualName}
                onChange={e => setManualName(e.target.value)}
              />
              <div className="grid grid-cols-2 gap-5">
                <div onClick={() => topInputRef.current?.click()} className="aspect-[3/4] rounded-[35px] border-2 border-dashed border-slate-100 bg-slate-50 flex flex-col items-center justify-center relative overflow-hidden group cursor-pointer hover:border-indigo-200 transition-colors">
                  {manualTop ? (
                    <img src={typeof manualTop === 'string' ? manualTop : (manualTop as WardrobeItem).imageUrl} className="w-full h-full object-contain p-4" alt="Top Preview" />
                  ) : (
                    <><Shirt className="w-8 h-8 text-slate-300 mb-2" /><span className="text-[10px] font-black uppercase text-slate-400">Add Top</span></>
                  )}
                  <input type="file" ref={topInputRef} className="hidden" accept="image/*" onChange={(e) => handleManualImage(e, 'top')} />
                </div>
                <div onClick={() => bottomInputRef.current?.click()} className="aspect-[3/4] rounded-[35px] border-2 border-dashed border-slate-100 bg-slate-50 flex flex-col items-center justify-center relative overflow-hidden group cursor-pointer hover:border-pink-200 transition-colors">
                  {manualBottom ? (
                    <img src={typeof manualBottom === 'string' ? manualBottom : (manualBottom as WardrobeItem).imageUrl} className="w-full h-full object-contain p-4" alt="Bottom Preview" />
                  ) : (
                    <><ImageIcon className="w-8 h-8 text-slate-300 mb-2" /><span className="text-[10px] font-black uppercase text-slate-400">Add Bottom</span></>
                  )}
                  <input type="file" ref={bottomInputRef} className="hidden" accept="image/*" onChange={(e) => handleManualImage(e, 'bottom')} />
                </div>
              </div>
              <button
                onClick={saveManualOutfit}
                disabled={!manualTop || !manualBottom || savingManual}
                className="w-full py-6 gradient-bg text-white rounded-full font-black uppercase tracking-[3px] text-xs shadow-2xl flex items-center justify-center gap-3 active:scale-95 transition-all disabled:opacity-30"
              >
                {savingManual ? <Loader2 className="w-5 h-5 animate-spin" /> : <><Check className="w-4 h-4" /> Save Curation</>}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Curate;
