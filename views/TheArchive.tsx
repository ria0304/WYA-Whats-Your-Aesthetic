import React, { useState, useEffect } from 'react';
import { Archive, Heart, Sun, Snowflake, Leaf, Cloud, Clock, Shirt } from 'lucide-react';

interface ArchivedItem {
  id: string;
  name: string;
  category: string;
  color: string;
  fabric: string;
  imageUrl?: string;
  wearCount: number;
  topSeason: string;
  topPairedWith: string;
  archivedDate: string;
  archiveReason: 'sold' | 'worn-out' | 'donated' | 'gifted';
  yearAcquired?: number;
  memoryNote?: string;
}

const MOCK_ARCHIVE: ArchivedItem[] = [
  {
    id: 'arch-1',
    name: 'Vintage Levi\'s 501',
    category: 'Jeans',
    color: 'Indigo Wash',
    fabric: 'Denim',
    wearCount: 42,
    topSeason: 'Summer 2024',
    topPairedWith: 'Oversized White Tee',
    archivedDate: 'March 2025',
    archiveReason: 'sold',
    yearAcquired: 2021,
    memoryNote: 'The jeans that went everywhere — Tulum, Tokyo, Tuesday meetings.',
  },
  {
    id: 'arch-2',
    name: 'Black Leather Moto Jacket',
    category: 'Jacket',
    color: 'Jet Black',
    fabric: 'Faux Leather',
    wearCount: 31,
    topSeason: 'Winter 2023',
    topPairedWith: 'Silk Slip Dress',
    archivedDate: 'January 2025',
    archiveReason: 'worn-out',
    yearAcquired: 2020,
    memoryNote: 'Zipper finally gave out after 4 years of being the cool one.',
  },
  {
    id: 'arch-3',
    name: 'Floral Silk Midi Dress',
    category: 'Dress',
    color: 'Dusty Rose',
    fabric: 'Silk Blend',
    wearCount: 17,
    topSeason: 'Spring 2024',
    topPairedWith: 'White Sneakers',
    archivedDate: 'November 2024',
    archiveReason: 'donated',
    yearAcquired: 2023,
    memoryNote: 'Passed along to a friend who needed it more.',
  },
];

const SEASON_ICONS: Record<string, React.ReactNode> = {
  Spring: <Leaf className="w-3 h-3 text-green-400" />,
  Summer: <Sun className="w-3 h-3 text-amber-400" />,
  Autumn: <Cloud className="w-3 h-3 text-orange-400" />,
  Fall:   <Cloud className="w-3 h-3 text-orange-400" />,
  Winter: <Snowflake className="w-3 h-3 text-blue-400" />,
};

const REASON_LABELS: Record<string, { label: string; color: string }> = {
  sold:      { label: 'Sold', color: 'bg-emerald-50 text-emerald-600' },
  'worn-out':{ label: 'Worn Out', color: 'bg-amber-50 text-amber-600' },
  donated:   { label: 'Donated', color: 'bg-blue-50 text-blue-600' },
  gifted:    { label: 'Gifted', color: 'bg-pink-50 text-pink-600' },
};

const getSeasonIcon = (season: string) => {
  const key = Object.keys(SEASON_ICONS).find(k => season.includes(k));
  return key ? SEASON_ICONS[key] : <Sun className="w-3 h-3 text-slate-400" />;
};

// Color swatch approximation
const COLOR_MAP: Record<string, string> = {
  'Indigo Wash': '#4c5d8a',
  'Jet Black': '#1a1a1a',
  'Dusty Rose': '#d4a0a0',
  'Ivory': '#f5f0e8',
  'Charcoal': '#3d3d3d',
};

const PolaroidCard: React.FC<{ item: ArchivedItem }> = ({ item }) => {
  const [flipped, setFlipped] = useState(false);
  const reason = REASON_LABELS[item.archiveReason];
  const bgColor = COLOR_MAP[item.color] || '#e8e0d8';

  return (
    <div 
      className="relative cursor-pointer select-none"
      onClick={() => setFlipped(!flipped)}
      style={{ perspective: '1000px' }}
    >
      <div 
        className="relative transition-transform duration-500"
        style={{ 
          transformStyle: 'preserve-3d',
          transform: flipped ? 'rotateY(180deg)' : 'rotateY(0deg)'
        }}
      >
        {/* FRONT — Polaroid */}
        <div 
          className="bg-white rounded-sm shadow-[0_8px_24px_rgba(0,0,0,0.12)] p-4 pb-12"
          style={{ backfaceVisibility: 'hidden' }}
        >
          {/* Photo area */}
          <div 
            className="w-full aspect-square rounded-sm mb-4 overflow-hidden flex items-center justify-center relative"
            style={{ backgroundColor: bgColor + '33' }}
          >
            {item.imageUrl ? (
              <img src={item.imageUrl} className="w-full h-full object-contain" alt={item.name} />
            ) : (
              <div className="flex flex-col items-center gap-2">
                <Shirt className="w-16 h-16" style={{ color: bgColor }} />
                <div className="w-8 h-1.5 rounded-full" style={{ backgroundColor: bgColor + '80' }} />
              </div>
            )}
            {/* Color swatch chip */}
            <div 
              className="absolute bottom-2 right-2 w-5 h-5 rounded-full border-2 border-white shadow-sm"
              style={{ backgroundColor: bgColor }}
            />
          </div>
          
          {/* Polaroid caption */}
          <div className="px-1">
            <p className="font-bold text-slate-800 text-sm leading-tight truncate">{item.name}</p>
            <div className="flex items-center gap-2 mt-1">
              <span className={`text-[8px] font-black uppercase tracking-widest px-2 py-0.5 rounded-full ${reason.color}`}>
                {reason.label}
              </span>
              <span className="text-[9px] text-slate-400 font-bold">{item.archivedDate}</span>
            </div>
            <div className="flex items-center gap-1 mt-2">
              <Heart className="w-3 h-3 text-pink-400 fill-pink-400" />
              <span className="text-[10px] font-black text-slate-600">Worn {item.wearCount}×</span>
            </div>
          </div>

          {/* Flip hint */}
          <p className="absolute bottom-3 right-4 text-[7px] text-slate-300 uppercase tracking-widest font-bold">tap to flip</p>
        </div>

        {/* BACK — Stats card */}
        <div 
          className="absolute inset-0 bg-slate-900 rounded-sm shadow-[0_8px_24px_rgba(0,0,0,0.2)] p-5 flex flex-col justify-between"
          style={{ backfaceVisibility: 'hidden', transform: 'rotateY(180deg)' }}
        >
          <div>
            <p className="text-[8px] font-black uppercase tracking-[4px] text-slate-500 mb-1">Memory Lane</p>
            <h3 className="text-white font-bold text-sm leading-tight mb-3">{item.name}</h3>
            
            {item.memoryNote && (
              <p className="text-slate-400 text-[10px] leading-relaxed italic mb-4 border-l-2 border-pink-500/40 pl-3">
                "{item.memoryNote}"
              </p>
            )}
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 text-[9px] uppercase tracking-widest font-bold">Worn</span>
              <span className="text-white text-[10px] font-black">{item.wearCount} times</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500 text-[9px] uppercase tracking-widest font-bold">Top season</span>
              <span className="text-white text-[10px] font-black flex items-center gap-1">
                {getSeasonIcon(item.topSeason)} {item.topSeason}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500 text-[9px] uppercase tracking-widest font-bold">Paired with</span>
              <span className="text-pink-400 text-[9px] font-black truncate ml-2 text-right max-w-[120px]">{item.topPairedWith}</span>
            </div>
            {item.yearAcquired && (
              <div className="flex items-center justify-between">
                <span className="text-slate-500 text-[9px] uppercase tracking-widest font-bold">Acquired</span>
                <span className="text-white text-[10px] font-black">{item.yearAcquired}</span>
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800">
            <p className="text-slate-600 text-[8px] uppercase tracking-[3px] text-center font-bold">Archived {item.archivedDate}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

const TheArchive: React.FC = () => {
  const [archive] = useState<ArchivedItem[]>(MOCK_ARCHIVE);
  const [filter, setFilter] = useState<string>('All');

  const filters = ['All', 'Sold', 'Worn Out', 'Donated', 'Gifted'];
  const filterMap: Record<string, string> = { 'Worn Out': 'worn-out', 'Sold': 'sold', 'Donated': 'donated', 'Gifted': 'gifted' };
  
  const filtered = filter === 'All' 
    ? archive 
    : archive.filter(i => i.archiveReason === filterMap[filter]);

  const totalWears = archive.reduce((sum, i) => sum + i.wearCount, 0);

  return (
    <div className="min-h-full bg-slate-50 p-6 pb-28">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <Archive className="w-5 h-5 text-slate-400" />
          <h1 className="text-3xl serif text-slate-800">The Archive</h1>
        </div>
        <p className="text-[10px] text-slate-400 uppercase tracking-widest font-bold">Memory Lane · Clothing as History</p>
      </div>

      {/* Stats banner */}
      <div className="bg-slate-900 rounded-[32px] p-6 mb-6 flex justify-around">
        <div className="text-center">
          <p className="text-2xl font-black text-white">{archive.length}</p>
          <p className="text-[8px] uppercase tracking-widest text-slate-500 font-bold mt-1">Archived</p>
        </div>
        <div className="w-px bg-slate-700" />
        <div className="text-center">
          <p className="text-2xl font-black text-white">{totalWears}</p>
          <p className="text-[8px] uppercase tracking-widest text-slate-500 font-bold mt-1">Total Wears</p>
        </div>
        <div className="w-px bg-slate-700" />
        <div className="text-center">
          <p className="text-2xl font-black text-pink-400">{Math.round(totalWears / archive.length)}</p>
          <p className="text-[8px] uppercase tracking-widest text-slate-500 font-bold mt-1">Avg/Piece</p>
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-2 overflow-x-auto pb-4 mb-6 custom-scrollbar">
        {filters.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-5 py-2.5 rounded-full text-[9px] font-black uppercase tracking-widest whitespace-nowrap transition-all ${
              filter === f 
                ? 'bg-slate-900 text-white' 
                : 'bg-white text-slate-400 border border-slate-100'
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Polaroid Grid */}
      {filtered.length > 0 ? (
        <div className="grid grid-cols-2 gap-5">
          {filtered.map(item => (
            <PolaroidCard key={item.id} item={item} />
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center py-20 opacity-30">
          <Clock className="w-12 h-12 text-slate-300 mb-3" />
          <p className="text-xs font-bold uppercase tracking-widest text-slate-400">No archived pieces yet</p>
          <p className="text-[9px] text-slate-300 mt-1 text-center max-w-[200px]">When you retire a piece, it lives here forever</p>
        </div>
      )}

      {/* Tip */}
      <div className="mt-8 bg-pink-50 rounded-[24px] p-5 border border-pink-100">
        <p className="text-[9px] font-black text-pink-400 uppercase tracking-widest mb-2">💡 Archive Tip</p>
        <p className="text-[10px] text-pink-700/70 leading-relaxed">
          When deleting an item from your Closet, choose "Archive" to preserve its story here instead of losing it forever.
        </p>
      </div>
    </div>
  );
};

export default TheArchive;
