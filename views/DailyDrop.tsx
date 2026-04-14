import React, { useState, useEffect } from 'react';
import { Bell, BellOff, Sparkles, Sun, Cloud, Loader2, RefreshCw, Check, ChevronRight, Shirt } from 'lucide-react';

interface OutfitPiece {
  name: string;
  category: string;
  color: string;
}

interface DailyDrop {
  date: string;
  greeting: string;
  weatherSnippet: string;
  outfitName: string;
  outfitVibe: string;
  pieces: OutfitPiece[];
  styleNote: string;
  dayScore: number;
}

const DAYS = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

function getTodayString() {
  const d = new Date();
  return `${DAYS[d.getDay()]}, ${MONTHS[d.getMonth()]} ${d.getDate()}`;
}

const MOCK_DROP: DailyDrop = {
  date: getTodayString(),
  greeting: 'Your Daily Drop is ready ✨',
  weatherSnippet: '22°C · Partly Cloudy · Light breeze',
  outfitName: 'The Quiet Intellectual',
  outfitVibe: 'Dark Academia × Effortless',
  pieces: [
    { name: 'Camel Wool Blazer', category: 'Jacket', color: '#c4a882' },
    { name: 'Cream Ribbed Turtleneck', category: 'Top', color: '#f5f0e5' },
    { name: 'Straight Leg Trousers', category: 'Trousers', color: '#2d2d2d' },
    { name: 'Brown Chelsea Boots', category: 'Shoes', color: '#6b3f2a' },
  ],
  styleNote: 'Layer the blazer open for a relaxed editorial feel. The turtleneck grounds it.',
  dayScore: 87,
};

const COLOR_PIECE_BG: Record<string, string> = {
  Jacket: '#fef3c7',
  Top: '#f0fdf4',
  Trousers: '#f1f5f9',
  Shoes: '#fdf4ff',
  Dress: '#fce7f3',
  Accessories: '#eff6ff',
};

const DailyDrop: React.FC = () => {
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [notifTime, setNotifTime] = useState('08:00');
  const [loading, setLoading] = useState(true);
  const [drop, setDrop] = useState<DailyDrop | null>(null);
  const [saved, setSaved] = useState(false);
  const [permissionStatus, setPermissionStatus] = useState<string>('default');

  useEffect(() => {
    // Check notification permission
    if ('Notification' in window) {
      setPermissionStatus(Notification.permission);
      setNotificationsEnabled(Notification.permission === 'granted');
    }

    // Simulate AI outfit generation
    setTimeout(() => {
      setDrop(MOCK_DROP);
      setLoading(false);
    }, 1500);
  }, []);

  const handleEnableNotifications = async () => {
    if (!('Notification' in window)) {
      alert('Notifications not supported in this browser.');
      return;
    }
    if (Notification.permission === 'denied') {
      alert('Notifications are blocked. Please enable them in your browser settings.');
      return;
    }
    const result = await Notification.requestPermission();
    setPermissionStatus(result);
    if (result === 'granted') {
      setNotificationsEnabled(true);
      // Simulate scheduling
      new Notification('WYA — Daily Drop 🌅', {
        body: 'Your AI outfit for today is ready. Tap to see your look.',
        icon: '/icon.png',
      });
    }
  };

  const handleSaveSettings = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleRefresh = () => {
    setLoading(true);
    setTimeout(() => {
      setDrop({ ...MOCK_DROP, outfitName: 'The Understated Power', dayScore: 91 });
      setLoading(false);
    }, 1200);
  };

  return (
    <div className="min-h-full bg-white p-6 pb-28">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl serif text-slate-800">Daily Drop</h1>
          <p className="text-[10px] text-slate-400 uppercase tracking-widest font-bold mt-1">AI Outfit · Every Morning</p>
        </div>
        <button onClick={handleRefresh} className="p-3 bg-slate-50 rounded-full hover:bg-slate-100 transition-colors">
          <RefreshCw className="w-5 h-5 text-slate-400" />
        </button>
      </div>

      {/* Notification Toggle Card */}
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
                className={`px-5 py-3 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all ${
                  saved ? 'bg-emerald-500 text-white' : 'bg-pink-500 text-white hover:bg-pink-400'
                }`}
              >
                {saved ? <><Check className="w-4 h-4 inline mr-1" />Saved</> : 'Save'}
              </button>
            </div>
          </div>
        )}

        {!notificationsEnabled && (
          <button
            onClick={handleEnableNotifications}
            className="w-full mt-2 py-3 bg-white/10 hover:bg-white/15 rounded-2xl text-white text-[10px] font-black uppercase tracking-[3px] transition-colors"
          >
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
            <p className="text-[9px] font-black uppercase tracking-[4px] text-slate-400 animate-pulse">AI Styling Your Look...</p>
          </div>
        ) : drop ? (
          <div className="bg-gradient-to-br from-pink-50 via-purple-50 to-blue-50 rounded-[32px] p-6 border border-white shadow-sm">
            {/* Weather + Score */}
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-2">
                <Cloud className="w-4 h-4 text-blue-400" />
                <span className="text-[9px] text-slate-500 font-bold">{drop.weatherSnippet}</span>
              </div>
              <div className="bg-white rounded-full px-3 py-1 shadow-sm">
                <span className="text-[9px] font-black text-pink-500">✨ {drop.dayScore} Style Score</span>
              </div>
            </div>

            {/* Outfit name */}
            <h2 className="text-2xl serif text-slate-800 mb-1">{drop.outfitName}</h2>
            <p className="text-[9px] font-black uppercase tracking-widest text-pink-400 mb-5">{drop.outfitVibe}</p>

            {/* Pieces */}
            <div className="space-y-3 mb-5">
              {drop.pieces.map((piece, i) => (
                <div key={i} className="bg-white rounded-[20px] p-4 flex items-center gap-4 shadow-sm border border-white">
                  <div 
                    className="w-10 h-10 rounded-2xl flex items-center justify-center shadow-inner flex-shrink-0"
                    style={{ backgroundColor: COLOR_PIECE_BG[piece.category] || '#f8fafc' }}
                  >
                    <div className="w-4 h-4 rounded-full" style={{ backgroundColor: piece.color }} />
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
        ) : null}
      </div>

      {/* Past drops preview */}
      <div className="mt-6">
        <p className="text-[10px] font-black uppercase tracking-[4px] text-slate-400 mb-4">Previous Drops</p>
        <div className="space-y-3">
          {['Yesterday', '2 days ago', '3 days ago'].map((day, i) => (
            <div key={i} className="bg-slate-50 rounded-[24px] p-4 flex items-center gap-4 border border-slate-100 opacity-60">
              <div className="w-10 h-10 bg-white rounded-2xl flex items-center justify-center shadow-sm">
                <Shirt className="w-5 h-5 text-slate-300" />
              </div>
              <div className="flex-1">
                <p className="text-[10px] font-black text-slate-600">{['The Effortless Edit', 'Monochrome Mood', 'Weekend Soft'][i]}</p>
                <p className="text-[8px] text-slate-400 uppercase tracking-widest font-bold">{day}</p>
              </div>
              <span className="text-[8px] font-black text-slate-300">{[82, 79, 88][i]} pts</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DailyDrop;
