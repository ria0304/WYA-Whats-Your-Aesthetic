import React, { useState, useRef } from 'react';
import {
  Shirt, Sparkles, X, Loader2, RefreshCw, Box, Zap,
  ChevronRight, Heart, Watch, Palette, ShoppingBag, TrendingUp, AlertCircle
} from 'lucide-react';
import { api } from '../services/api';
import { GapItem } from '../types';

type ActiveView = 'ai' | 'closet' | 'gap' | 'idle';

const PRIORITY_COLORS: Record<string, string> = {
  high:   'bg-rose-50 border-rose-100 text-rose-700',
  medium: 'bg-amber-50 border-amber-100 text-amber-700',
  low:    'bg-emerald-50 border-emerald-100 text-emerald-700',
};

const AIMatcher: React.FC = () => {
  const [image, setImage] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [matchingCloset, setMatchingCloset] = useState(false);
  const [loadingGap, setLoadingGap] = useState(false);

  const [aiSuggestion, setAiSuggestion] = useState<any | null>(null);
  const [closetMatches, setClosetMatches] = useState<any[]>([]);
  const [gapItems, setGapItems] = useState<GapItem[]>([]);
  const [activeView, setActiveView] = useState<ActiveView>('idle');
  const [variationCount, setVariationCount] = useState(0);
  const [refreshingField, setRefreshingField] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
        setAiSuggestion(null);
        setClosetMatches([]);
        setActiveView('idle');
        setVariationCount(0);
      };
      reader.readAsDataURL(file);
    }
  };

  const getAIInspiration = async (fieldToLock?: string) => {
    if (!image) return;
    if (fieldToLock) {
      setRefreshingField(fieldToLock);
    } else {
      setAnalyzing(true);
      setAiSuggestion(null);
    }
    try {
      const nextVariation = variationCount + 1;
      const data = await api.wardrobe.outfitMatch(image, nextVariation);
      if (data) {
        if (fieldToLock && aiSuggestion) {
          const updatedSuggestion = { ...aiSuggestion };
          const fieldMap: Record<string, string> = {
            style: 'match_piece', jewelry: 'jewelry', shoes: 'shoes', accessories: 'bag'
          };
          const targetKey = fieldMap[fieldToLock];
          if (targetKey) {
            updatedSuggestion[targetKey] = data[targetKey];
            updatedSuggestion.vibe = data.vibe;
          }
          setAiSuggestion(updatedSuggestion);
        } else {
          setAiSuggestion(data);
          setActiveView('ai');
        }
        setVariationCount(nextVariation);
      }
    } catch (e) {
      console.error(e);
      if (!fieldToLock) alert('Styling engine error.');
    } finally {
      setAnalyzing(false);
      setRefreshingField(null);
    }
  };

  /**
   * Closet matches: uses cosine similarity scores from the backend
   * (FashionCLIP embeddings). Falls back to showing items if no score is returned.
   */
  const getClosetMatches = async () => {
    if (!image) return;
    setMatchingCloset(true);
    try {
      // The backend /api/ai/outfit-match returns closet items with `similarity_score`
      // when `include_closet_matches=true` is sent. The score is a 0–100 cosine similarity.
      const data = await api.wardrobe.outfitMatch(image, 0);
      const matches = (data?.closet_matches || []);

      if (matches.length > 0) {
        // Sort by descending similarity score (real FashionCLIP cosine similarity)
        const sorted = [...matches].sort((a: any, b: any) =>
          (b.similarity_score ?? 0) - (a.similarity_score ?? 0)
        );
        setClosetMatches(sorted.slice(0, 4));
      } else {
        // Fallback: show first 4 wardrobe items
        const items = await api.wardrobe.getAll();
        setClosetMatches(items.slice(0, 4).map((it: any) => ({
          ...it,
          similarity_score: null,  // no score available in fallback
        })));
      }
      setActiveView('closet');
    } catch (e) {
      console.error(e);
    } finally {
      setMatchingCloset(false);
    }
  };

  /** Gap Analysis: compare Style DNA to actual wardrobe inventory */
  const runGapAnalysis = async () => {
    setLoadingGap(true);
    setGapItems([]);
    try {
      const data = await api.wardrobe.gapAnalysis();
      // Expected shape: { gaps: GapItem[] }
      setGapItems(data.gaps || []);
      setActiveView('gap');
    } catch (e) {
      console.error(e);
      alert('Gap analysis failed. Make sure you have a Style DNA and at least a few wardrobe items.');
    } finally {
      setLoadingGap(false);
    }
  };

  const openAffiliateSearch = (query: string) => {
    const url = `https://www.google.com/search?q=${encodeURIComponent(query + ' buy online sustainable')}`;
    window.open(url, '_blank', 'noopener');
  };

  return (
    <div className="p-6 bg-white min-h-full pb-32">
      {/* Upload Panel */}
      <div className="bg-[#fff0f5] rounded-[40px] p-8 mb-8 border border-pink-100 shadow-sm relative overflow-hidden">
        <div className="absolute top-0 right-0 p-4 opacity-10 pointer-events-none">
          <Heart className="w-24 h-24 text-pink-500 fill-pink-500" />
        </div>

        <div className="flex items-center gap-2 mb-6">
          <Sparkles className="w-5 h-5 text-pink-500" />
          <h2 className="text-pink-500 font-black uppercase tracking-[4px] text-[10px]">Style Match Engine</h2>
        </div>

        <div
          onClick={() => fileInputRef.current?.click()}
          className="relative border-2 border-dashed border-pink-200 rounded-[40px] bg-white aspect-square flex flex-col items-center justify-center p-8 cursor-pointer overflow-hidden transition-all hover:border-pink-300 shadow-inner group"
        >
          {image ? (
            <img src={image} alt="Target" className="absolute inset-0 w-full h-full object-contain p-4 transition-transform group-hover:scale-105" />
          ) : (
            <>
              <div className="w-16 h-16 bg-pink-50 rounded-full flex items-center justify-center mb-4"><Shirt className="w-8 h-8 text-pink-300" /></div>
              <p className="text-[10px] font-black text-slate-700 uppercase tracking-widest mb-2 text-center">Reference Piece</p>
              <p className="text-[8px] text-slate-400 text-center px-6 leading-relaxed">Upload piece for AI matching.</p>
            </>
          )}
          <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" />
        </div>

        {/* Three-button grid */}
        <div className="grid grid-cols-3 gap-3 mt-8">
          <button
            onClick={() => getAIInspiration()}
            disabled={!image || analyzing}
            className="py-5 rounded-3xl gradient-bg text-white flex items-center justify-center gap-2 text-[9px] font-black uppercase tracking-widest shadow-lg disabled:opacity-50 active:scale-95 transition-all"
          >
            {analyzing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            AI Match
          </button>
          <button
            onClick={getClosetMatches}
            disabled={!image || matchingCloset}
            className="py-5 rounded-3xl bg-white border border-pink-100 text-pink-500 flex items-center justify-center gap-2 text-[9px] font-black uppercase tracking-widest shadow-sm disabled:opacity-50 active:scale-95 transition-all"
          >
            {matchingCloset ? <Loader2 className="w-4 h-4 animate-spin" /> : <Box className="w-4 h-4" />}
            Wardrobe
          </button>
          <button
            onClick={runGapAnalysis}
            disabled={loadingGap}
            className="py-5 rounded-3xl bg-slate-900 text-white flex items-center justify-center gap-2 text-[9px] font-black uppercase tracking-widest shadow-sm disabled:opacity-50 active:scale-95 transition-all"
          >
            {loadingGap ? <Loader2 className="w-4 h-4 animate-spin" /> : <TrendingUp className="w-4 h-4" />}
            Gap
          </button>
        </div>
      </div>

      {/* Results area */}
      <div className="space-y-6 animate-slide-up pb-10 min-h-[300px]">

        {/* Idle */}
        {activeView === 'idle' && !analyzing && !matchingCloset && !loadingGap && (
          <div className="flex flex-col items-center justify-center py-20 opacity-20">
            <Sparkles className="w-12 h-12 text-slate-300 mb-4" />
            <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">Identify an item to start</p>
          </div>
        )}

        {/* AI loading */}
        {analyzing && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="w-16 h-16 rounded-full border-4 border-slate-100 border-t-pink-500 animate-spin mb-6" />
            <p className="text-[10px] font-black uppercase tracking-[5px] text-pink-500">AI Engine Mapping...</p>
          </div>
        )}

        {/* Gap loading */}
        {loadingGap && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="w-16 h-16 rounded-full border-4 border-slate-100 border-t-slate-900 animate-spin mb-6" />
            <p className="text-[10px] font-black uppercase tracking-[5px] text-slate-500">Scanning wardrobe gaps...</p>
          </div>
        )}

        {/* AI Result */}
        {activeView === 'ai' && aiSuggestion && !analyzing && (
          <div className="bg-white rounded-[40px] p-8 border border-pink-50 relative shadow-2xl animate-slide-up">
            <div className="absolute top-0 right-0 p-6 opacity-5 pointer-events-none">
              <Zap className="w-32 h-32 text-pink-500" />
            </div>
            <div className="mb-6 flex items-center justify-between">
              <span className="text-[10px] font-black uppercase tracking-widest text-pink-500 bg-pink-50 px-5 py-2 rounded-full shadow-inner">{aiSuggestion.vibe}</span>
              <button onClick={() => setActiveView('idle')} className="text-slate-300 p-2 hover:bg-slate-50 rounded-full transition-colors"><X className="w-5 h-5" /></button>
            </div>
            <div className="mb-8 flex justify-between items-start">
              <div>
                <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Target Garment</h3>
                <p className="text-2xl font-black text-slate-800 uppercase tracking-tighter leading-none">{aiSuggestion.identified_item}</p>
              </div>
              {aiSuggestion.best_color && (
                <div className="bg-amber-50 px-4 py-2 rounded-2xl border border-amber-100 flex flex-col items-end">
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <Palette className="w-3 h-3 text-amber-500" />
                    <span className="text-[8px] font-black text-amber-600 uppercase tracking-widest">Best Match</span>
                  </div>
                  <span className="text-[10px] font-bold text-amber-700 whitespace-nowrap">{aiSuggestion.best_color}</span>
                </div>
              )}
            </div>
            <div className="grid grid-cols-1 gap-5">
              {aiSuggestion.match_piece && (
                <PairingCard label="Style Match" value={aiSuggestion.match_piece} icon={<Shirt className="w-5 h-5" />} bgColor="bg-indigo-50" textColor="text-indigo-700" onRegenerate={() => getAIInspiration('style')} isRefreshing={refreshingField === 'style'} />
              )}
              <PairingCard label="Jewelry Focus" value={aiSuggestion.jewelry} icon={<Sparkles className="w-5 h-5" />} bgColor="bg-pink-50" textColor="text-pink-700" onRegenerate={() => getAIInspiration('jewelry')} isRefreshing={refreshingField === 'jewelry'} />
              <PairingCard label="Footwear" value={aiSuggestion.shoes} icon={<Box className="w-5 h-5" />} bgColor="bg-slate-50" textColor="text-slate-700" onRegenerate={() => getAIInspiration('shoes')} isRefreshing={refreshingField === 'shoes'} />
              <PairingCard label="Accessories & Watches" value={aiSuggestion.bag} icon={<Watch className="w-5 h-5" />} bgColor="bg-emerald-50" textColor="text-emerald-700" onRegenerate={() => getAIInspiration('accessories')} isRefreshing={refreshingField === 'accessories'} />
            </div>
            <button
              onClick={() => getAIInspiration()}
              className="w-full mt-12 py-6 rounded-3xl border-2 border-slate-100 text-slate-400 flex items-center justify-center gap-3 text-[11px] font-black uppercase tracking-[4px] hover:bg-slate-50 transition-all active:scale-95"
            >
              <RefreshCw className="w-4 h-4" /> REGENERATE ENTIRE LOOK
            </button>
          </div>
        )}

        {/* Closet matches with real similarity scores */}
        {activeView === 'closet' && !matchingCloset && (
          <div className="animate-slide-up">
            <div className="flex items-center justify-between mb-6 px-2">
              <div>
                <h3 className="text-[11px] font-black uppercase tracking-[4px] text-slate-400">Wardrobe Suggestions</h3>
                <p className="text-[9px] text-slate-300 font-bold mt-0.5">Ranked by visual similarity (FashionCLIP)</p>
              </div>
              <button onClick={() => setActiveView('idle')} className="text-slate-300"><X className="w-5 h-5" /></button>
            </div>
            <div className="grid grid-cols-2 gap-5">
              {closetMatches.map((item: any, i) => {
                const score = item.similarity_score ?? item.match_score ?? null;
                return (
                  <div key={i} className="bg-white rounded-[35px] p-4 border border-slate-100 shadow-xl relative group">
                    <div className="aspect-[3/4] bg-slate-50 rounded-3xl mb-4 overflow-hidden shadow-inner flex items-center justify-center">
                      <img src={item.image_url || item.imageUrl} className="w-full h-full object-contain p-2 group-hover:scale-110 transition-transform duration-500" alt={item.name} />
                    </div>
                    <div className="px-2 pb-2">
                      <h4 className="text-[10px] font-black text-slate-800 uppercase truncate mb-1">{item.name}</h4>
                      {score !== null ? (
                        <div className="mt-1">
                          <div className="flex justify-between items-center mb-1">
                            <p className="text-[9px] font-black text-emerald-500 uppercase tracking-widest">{Math.round(score)}% Match</p>
                          </div>
                          <div className="w-full h-1 bg-slate-100 rounded-full overflow-hidden">
                            <div className="h-full bg-gradient-to-r from-emerald-400 to-teal-400 rounded-full" style={{ width: `${score}%` }} />
                          </div>
                        </div>
                      ) : (
                        <p className="text-[9px] font-black text-slate-300 uppercase tracking-widest">—</p>
                      )}
                    </div>
                    <div className="absolute bottom-4 right-4 bg-white/90 backdrop-blur-md p-2 rounded-full shadow-lg opacity-0 group-hover:opacity-100 transition-opacity">
                      <ChevronRight className="w-4 h-4 text-slate-800" />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Gap Analysis — Shop Your Closet */}
        {activeView === 'gap' && !loadingGap && (
          <div className="animate-slide-up">
            <div className="flex items-center justify-between mb-4 px-1">
              <div>
                <h3 className="text-[11px] font-black uppercase tracking-[4px] text-slate-800">Shop Your Closet</h3>
                <p className="text-[9px] text-slate-400 font-bold mt-0.5">Missing pieces to unlock new looks</p>
              </div>
              <button onClick={() => setActiveView('idle')} className="text-slate-300"><X className="w-5 h-5" /></button>
            </div>

            {gapItems.length === 0 ? (
              <div className="bg-emerald-50 rounded-[32px] p-8 text-center border border-emerald-100">
                <p className="text-emerald-600 font-black text-sm mb-1">Your wardrobe is well-balanced! 🎉</p>
                <p className="text-emerald-500 text-[10px]">No significant gaps detected for your Style DNA.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {gapItems.map((gap, i) => (
                  <div key={i} className={`rounded-[32px] p-6 border ${PRIORITY_COLORS[gap.priority]}`}>
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`text-[8px] font-black uppercase tracking-widest px-2 py-0.5 rounded-full border ${PRIORITY_COLORS[gap.priority]}`}>
                            {gap.priority} priority
                          </span>
                          <span className="text-[8px] font-black uppercase tracking-widest text-slate-400">{gap.category}</span>
                        </div>
                        <p className="text-sm font-bold text-slate-800 mb-1">{gap.description}</p>
                        <p className="text-[10px] text-slate-500 leading-relaxed">{gap.reason}</p>
                      </div>
                      <AlertCircle className="w-5 h-5 shrink-0 opacity-40 mt-1" />
                    </div>
                    <button
                      onClick={() => openAffiliateSearch(gap.affiliateQuery)}
                      className="w-full mt-4 py-3 bg-white rounded-2xl text-[10px] font-black uppercase tracking-widest text-slate-700 flex items-center justify-center gap-2 border border-white/60 shadow-sm hover:shadow-md transition-all active:scale-95"
                    >
                      <ShoppingBag className="w-4 h-4" /> Shop Sustainable Options
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const PairingCard: React.FC<{
  label: string;
  value: string;
  icon: React.ReactNode;
  bgColor: string;
  textColor: string;
  onRegenerate?: () => void;
  isRefreshing?: boolean;
}> = ({ label, value, icon, bgColor, textColor, onRegenerate, isRefreshing }) => (
  <div className={`${bgColor} p-6 rounded-[35px] border border-white/60 shadow-md flex items-center justify-between group transition-all hover:shadow-lg`}>
    <div className="flex items-start gap-5 flex-1">
      <div className="w-12 h-12 bg-white rounded-2xl flex items-center justify-center shadow-inner shrink-0">
        {React.cloneElement(icon as React.ReactElement<any>, { className: `${textColor}` })}
      </div>
      <div className="pr-2 min-w-0">
        <h4 className="text-[10px] font-black uppercase tracking-[2px] opacity-40 mb-1">{label}</h4>
        <p className={`text-sm font-bold ${textColor} leading-snug break-words ${isRefreshing ? 'animate-pulse opacity-50' : 'animate-fade-in'}`}>
          {isRefreshing ? 'Regenerating...' : value}
        </p>
      </div>
    </div>
    {onRegenerate && (
      <div className="flex flex-col items-center gap-1.5 shrink-0 ml-4">
        <button
          onClick={onRegenerate}
          disabled={isRefreshing}
          className="w-12 h-12 bg-white rounded-full flex items-center justify-center text-slate-400 hover:text-slate-900 shadow-sm border border-slate-50 transition-all active:scale-90 disabled:opacity-50"
          title="New Look for this item"
        >
          <RefreshCw className={`w-5 h-5 ${isRefreshing ? 'animate-spin' : ''}`} />
        </button>
        <span className="text-[7px] font-black uppercase tracking-[2px] text-slate-400">New Look</span>
      </div>
    )}
  </div>
);

export default AIMatcher;
