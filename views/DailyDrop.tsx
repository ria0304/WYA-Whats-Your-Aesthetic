import React, { useState, useEffect, useCallback } from 'react';
import { Bell, BellOff, Sparkles, Sun, Cloud, Loader2, RefreshCw, Check, ChevronRight, Shirt, Palette } from 'lucide-react';
import { api } from '../services/api';

interface OutfitPiece {
  name: string;
  category: string;
  color: string;
  imageUrl?: string;
}

interface DailyDropData {
  date: string;
  greeting: string;
  weatherSnippet: string;
  outfitName: string;
  outfitVibe: string;
  harmonyType: string;    // e.g. "Analogous", "Complementary"
  pieces: OutfitPiece[];
  styleNote: string;
  dayScore: number;
  colorPalette: string[]; // hex colors
}

const DAYS = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

function getTodayString() {
  const d = new Date();
  return `${DAYS[d.getDay()]}, ${MONTHS[d.getMonth()]} ${d.getDate()}`;
}

const COLOR_PIECE_BG: Record<string, string> = {
  Jacket:      '#fef3c7',
  Top:         '#f0fdf4',
  Trousers:    '#f1f5f9',
  Shoes:       '#fdf4ff',
  Dress:       '#fce7f3',
  Accessories: '#eff6ff',
  Bottom:      '#f0f9ff',
  Jeans:       '#eff6ff',
  Skirt:       '#fdf4ff',
};

const HARMONY_COLORS: Record<string, string> = {
  Complementary: 'bg-violet-100 text-violet-600',
  Analogous:     'bg-amber-100  text-amber-600',
  Monochrome:    'bg-slate-200  text-slate-700',
  'Neutral Mix': 'bg-stone-100  text-stone-600',
  'High Contrast': 'bg-rose-100 text-rose-600',
};

const DROP_CACHE_KEY = 'wya_daily_drop_cache';

const DailyDrop: React.FC = () => {
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [notifTime, setNotifTime] = useState('08:00');
  const [loading, setLoading] = useState(true);
  const [drop, setDrop] = useState<DailyDropData | null>(null);
  const [saved, setSaved] = useState(false);
  const [permissionStatus, setPermissionStatus] = useState<string>('default');
  const [pastDrops, setPastDrops] = useState<{name: string; vibe: string; score: number; date: string}[]>([]);

  const fetchDailyDrop = useCallback(async (force = false) => {
    setLoading(true);
    try {
      const today = new Date().toDateString();

      // Use today's cache unless forcing a refresh
      if (!force) {
        const cached = JSON.parse(localStorage.getItem(DROP_CACHE_KEY) || 'null');
        if (cached && cached.cacheDate === today) {
          setDrop(cached.drop);
          setPastDrops(cached.past || []);
          setLoading(false);
          return;
        }
      }

      // Call backend – uses color_harmony.json logic + wardrobe items
      const data = await api.ai.getDailyDrop();

      const newDrop: DailyDropData = {
        date: getTodayString(),
        greeting: data.greeting || 'Your Daily Drop is ready ✨',
        weatherSnippet: data.weather_snippet || '',
        outfitName: data.outfit_name || 'Today\'s Look',
        outfitVibe: data.outfit_vibe || '',
        harmonyType: data.harmony_type || 'Analogous',
        pieces: (data.pieces || []).map((p: any) => ({
          name: p.name,
          category: p.category,
          color: p.color,
          imageUrl: p.image_url || p.imageUrl,
        })),
        styleNote: data.style_note || '',
        dayScore: data.day_score || 85,
        colorPalette: data.color_palette || [],
      };

      // Cache
      const currentCache = JSON.parse(localStorage.getItem(DROP_CACHE_KEY) || '{}');
      const newPast = [
        ...(currentCache.past || []).slice(0, 2),
      ];
      if (currentCache.drop) {
        newPast.unshift({
          name: currentCache.drop.outfitName,
          vibe: currentCache.drop.outfitVibe,
          score: currentCache.drop.dayScore,
          date: currentCache.drop.date,
        });
      }
      localStorage.setItem(DROP_CACHE_KEY, JSON.stringify({ drop: newDrop, cacheDate: today, past: newPast }));
      setDrop(newDrop);
      setPastDrops(newPast);

    } catch (err) {
      console.error('Daily Drop fetch failed, checking cache', err);
      // Try stale cache as fallback
      const cached = JSON.parse(localStorage.getItem(DROP_CACHE_KEY) || 'null');
      if (cached?.drop) {
        setDrop(cached.drop);
        setPastDrops(cached.past || []);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if ('Notification' in window) {
      setPermissionStatus(Notification.permission);
      setNotificationsEnabled(Notification.permission === 'granted');
    }
    // Load saved notif time
    const savedTime = localStorage.getItem('wya_notif_time');
    if (savedTime) setNotifTime(savedTime);

    fetchDailyDrop();
  }, [fetchDailyDrop]);

  const handleEnableNotifications = async () => {
    if (!('Notification' in window)) { alert('Notifications not supported in this browser.'); return; }
    if (Notification.permission === 'denied') { alert('Notifications are blocked. Please enable them in your browser settings.'); return; }
    const result = await Notification.requestPermission();
    setPermissionStatus(result);
    if (result === 'granted') {
      setNotificationsEnabled(true);
      new Notification('WYA — Daily Drop 🌅', {
        body: 'Your AI outfit for today is ready. Tap to see your look.',
        icon: '/icon.png',
      });
    }
  };

  const handleSaveSettings = () => {
    localStorage.setItem('wya_notif_time', notifTime);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);

    // Schedule via Service Worker if available (production PWA)
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'SCHEDULE_DAILY_DROP',
        time: notifTime,
      });
    }
  };

  return (
    <div className="min-h-full bg-white p-6 pb-28">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl serif text-slate-800">Daily Drop</h1>
          <p className="text-[10px] text-slate-400 uppercase tracking-widest font-bold mt-1">AI Outfit · Color Harmony Engine</p>
        </div>
        <button onClick={() => fetchDailyDrop(true)} className="p-3 bg-slate-50 rounded-full hover:bg-slate-100 transition-colors">
          <RefreshCw className="w-5 h-5 text-slate-400" />
        </button>
      </div>

      {/* Notification Toggle */}
      <div className="bg-slate-900 rounded-[32px] p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${notificationsEnabled ? 'bg-pink-500' : 'bg-slate-700'}`}>
              {notificationsEnabled ? <Bell className="w-5 h-5 text-white" /> : <BellOff className="w-5 h-5 text-slate-400" />}
            </div>
            <div>
              <p className="text-white text-sm font-bold">Morning Alerts</p>
              <p className="text-slate-500 text-[9px] uppercase tracking-widest font-bold">
                {notificationsEnabled ? 'Active' : 'Off'}
              </p>
            </div>
          </div>
          <button
            onClick={notificationsEnabled ? () => setNotificationsEnabled(false) : handleEnableNotifications}
            className={`relative w-14 h-7 rounded-full transition-colors ${notificationsEnabled ? 'bg-pink-500' : 'bg-slate-700'}`}
          >
            <span className={`absolute top-1 w-5 h-5 rounded-full bg-white shadow transition-all ${notificationsEnabled ? 'left-8' : 'left-1'}`} />
          </button>
        </div>

        {notificationsEnabled && (
          <div className="animate-fade-in">
            <p className="text-slate-500 text-[9px] uppercase tracking-widest font-bold mb-3">Drop Time</p>
            <div className="flex gap-3 items-center">
              <input
                type="time"
                value={notifTime}
                onChange={e => setNotifTime(e.target.value)}
                className="bg-slate-800 text-white rounded-2xl px-5 py-3 text-sm font-bold border border-slate-700 outline-none focus:border-pink-500 transition-colors flex-1"
              />
              <button
                onClick={handleSaveSettings}
                className={`px-5 py-3 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all ${saved ? 'bg-emerald-500 text-white' : 'bg-pink-500 text-white hover:bg-pink-400'}`}
              >
                {saved ? <><Check className="w-4 h-4 inline mr-1" />Saved</> : 'Save'}
              </button>
            </div>
          </div>
        )}
        {!notificationsEnabled && (
          <button onClick={handleEnableNotifications} className="w-full mt-2 py-3 bg-white/10 hover:bg-white/15 rounded-2xl text-white text-[10px] font-black uppercase tracking-[3px] transition-colors">
            Enable Notifications
          </button>
        )}
      </div>

      {/* TODAY'S DROP */}
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="w-4 h-4 text-pink-400" />
          <p className="text-[10px] font-black uppercase tracking-[4px] text-slate-400">Today's Drop</p>
          <span className="text-[9px] text-slate-300 font-bold ml-auto">{getTodayString()}</span>
        </div>

        {loading ? (
          <div className="bg-slate-50 rounded-[32px] p-12 flex flex-col items-center gap-4 border border-slate-100">
            <Loader2 className="w-8 h-8 animate-spin text-pink-300" />
            <p className="text-[9px] font-black uppercase tracking-[4px] text-slate-400 animate-pulse">Color Harmony Loading...</p>
          </div>
        ) : drop ? (
          <div className="bg-gradient-to-br from-pink-50 via-purple-50 to-blue-50 rounded-[32px] p-6 border border-white shadow-sm">
            {/* Weather + Score row */}
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-2">
                <Cloud className="w-4 h-4 text-blue-400" />
                <span className="text-[9px] text-slate-500 font-bold">{drop.weatherSnippet || 'Check local weather'}</span>
              </div>
              <div className="bg-white rounded-full px-3 py-1 shadow-sm">
                <span className="text-[9px] font-black text-pink-500">✨ {drop.dayScore} Style Score</span>
              </div>
            </div>

            {/* Outfit name + vibe */}
            <h2 className="text-2xl serif text-slate-800 mb-1">{drop.outfitName}</h2>
            <div className="flex items-center gap-2 mb-5">
              <p className="text-[9px] font-black uppercase tracking-widest text-pink-400">{drop.outfitVibe}</p>
              {/* Color harmony badge */}
              <span className={`text-[7px] font-black uppercase tracking-widest px-2 py-0.5 rounded-full flex items-center gap-1 ${HARMONY_COLORS[drop.harmonyType] || 'bg-slate-100 text-slate-500'}`}>
                <Palette className="w-2.5 h-2.5" /> {drop.harmonyType}
              </span>
            </div>

            {/* Color palette strip */}
            {drop.colorPalette.length > 0 && (
              <div className="flex gap-2 mb-5">
                {drop.colorPalette.map((hex, i) => (
                  <div key={i} className="w-6 h-6 rounded-full border-2 border-white shadow-sm" style={{ backgroundColor: hex }} />
                ))}
                <span className="ml-2 text-[8px] text-slate-400 font-bold self-center uppercase tracking-widest">Palette</span>
              </div>
            )}

            {/* Pieces */}
            <div className="space-y-3 mb-5">
              {drop.pieces.map((piece, i) => (
                <div key={i} className="bg-white rounded-[20px] p-4 flex items-center gap-4 shadow-sm border border-white">
                  <div
                    className="w-10 h-10 rounded-2xl flex items-center justify-center shadow-inner flex-shrink-0 overflow-hidden"
                    style={{ backgroundColor: COLOR_PIECE_BG[piece.category] || '#f8fafc' }}
                  >
                    {piece.imageUrl
                      ? <img src={piece.imageUrl} className="w-full h-full object-contain" alt={piece.name} />
                      : <div className="w-4 h-4 rounded-full" style={{ backgroundColor: piece.color || '#ccc' }} />
                    }
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-[11px] font-black text-slate-800 truncate">{piece.name}</p>
                    <p className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">{piece.category}</p>
                  </div>
                  <ChevronRight className="w-4 h-4 text-slate-200 flex-shrink-0" />
                </div>
              ))}
            </div>

            {/* Style note */}
            <div className="bg-white/60 rounded-[20px] p-4 border border-white">
              <p className="text-[8px] font-black uppercase tracking-[4px] text-slate-400 mb-2">AI Style Note</p>
              <p className="text-[10px] text-slate-600 leading-relaxed italic">{drop.styleNote}</p>
            </div>
          </div>
        ) : (
          <div className="bg-slate-50 rounded-[32px] p-10 text-center border border-slate-100">
            <Shirt className="w-12 h-12 text-slate-200 mx-auto mb-3" />
            <p className="text-sm font-bold text-slate-400">No drop available — add more items to your closet!</p>
          </div>
        )}
      </div>

      {/* Past drops */}
      {pastDrops.length > 0 && (
        <div className="mt-6">
          <p className="text-[10px] font-black uppercase tracking-[4px] text-slate-400 mb-4">Previous Drops</p>
          <div className="space-y-3">
            {pastDrops.map((d, i) => (
              <div key={i} className="bg-slate-50 rounded-[24px] p-4 flex items-center gap-4 border border-slate-100 opacity-70">
                <div className="w-10 h-10 bg-white rounded-2xl flex items-center justify-center shadow-sm">
                  <Shirt className="w-5 h-5 text-slate-300" />
                </div>
                <div className="flex-1">
                  <p className="text-[10px] font-black text-slate-600">{d.name}</p>
                  <p className="text-[8px] text-slate-400 uppercase tracking-widest font-bold">{d.date}</p>
                </div>
                <span className="text-[8px] font-black text-slate-300">{d.score} pts</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DailyDrop;
