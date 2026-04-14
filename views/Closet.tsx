import React, { useState, useEffect, useRef } from 'react';
import {
  Box, Plus, X, Loader2, Trash2, Check, Heart,
  Edit2, Archive, Wand2, AlertCircle
} from 'lucide-react';
import { WardrobeItem } from '../types';
import { api } from '../services/api';
import { analyzeImageLocally, LocalAnalysis } from '../services/localML';

// ─── Archive confirmation modal ───────────────────────────────────────────────
interface ArchiveModalProps {
  item: WardrobeItem;
  onConfirm: (reason: string, note: string) => void;
  onHardDelete: () => void;
  onCancel: () => void;
}

const ARCHIVE_REASONS = [
  { key: 'sold',      label: '💸 Sold it' },
  { key: 'worn-out',  label: '🪡 Worn out' },
  { key: 'donated',   label: '💙 Donated' },
  { key: 'gifted',    label: '🎁 Gifted' },
];

const ArchiveModal: React.FC<ArchiveModalProps> = ({ item, onConfirm, onHardDelete, onCancel }) => {
  const [reason, setReason] = useState('sold');
  const [note, setNote] = useState('');

  return (
    <div className="fixed inset-0 z-[200] flex items-end justify-center">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onCancel} />
      <div className="relative w-full max-w-md bg-white rounded-t-[50px] p-8 animate-slide-up shadow-2xl">
        <div className="flex items-start gap-4 mb-6">
          <div className="w-12 h-12 bg-slate-100 rounded-2xl flex items-center justify-center shrink-0">
            <Archive className="w-6 h-6 text-slate-500" />
          </div>
          <div>
            <h2 className="text-xl serif text-slate-800">Retire this piece?</h2>
            <p className="text-[10px] text-slate-400 uppercase tracking-widest font-bold mt-0.5">{item.name}</p>
          </div>
        </div>

        <p className="text-xs text-slate-500 mb-5 leading-relaxed">
          Archiving preserves its story — wear count, season history, and your memory note — forever.
          You can still permanently delete it from The Archive later.
        </p>

        {/* Reason picker */}
        <div className="grid grid-cols-2 gap-3 mb-5">
          {ARCHIVE_REASONS.map(r => (
            <button
              key={r.key}
              onClick={() => setReason(r.key)}
              className={`py-3 px-4 rounded-2xl text-sm font-bold transition-all border-2 ${
                reason === r.key
                  ? 'bg-slate-900 text-white border-slate-900'
                  : 'bg-slate-50 text-slate-600 border-transparent'
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>

        {/* Memory note */}
        <textarea
          placeholder="Add a memory note… (optional)"
          rows={2}
          className="w-full bg-slate-50 border border-slate-100 rounded-2xl p-4 text-sm outline-none focus:ring-2 ring-pink-100 resize-none mb-6"
          value={note}
          onChange={e => setNote(e.target.value)}
        />

        <div className="flex gap-3">
          <button
            onClick={() => onConfirm(reason, note)}
            className="flex-1 py-4 gradient-bg text-white rounded-full font-black text-xs uppercase tracking-widest shadow-lg"
          >
            Archive It
          </button>
          <button
            onClick={onHardDelete}
            className="w-14 h-14 bg-red-50 rounded-full flex items-center justify-center text-red-400 border border-red-100"
            title="Delete permanently without archiving"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
        <button onClick={onCancel} className="w-full mt-3 py-3 text-slate-400 text-xs font-bold uppercase tracking-widest">
          Cancel
        </button>
      </div>
    </div>
  );
};

// ─── Main Closet component ────────────────────────────────────────────────────
const Closet: React.FC = () => {
  const [items, setItems] = useState<WardrobeItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('All');
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [removingBg, setRemovingBg] = useState<string | null>(null); // item id currently processing bg removal
  const [archivingItem, setArchivingItem] = useState<WardrobeItem | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [localAnalysis, setLocalAnalysis] = useState<LocalAnalysis | null>(null);

  const [newItem, setNewItem] = useState<Partial<WardrobeItem>>({
    name: '',
    category: 'Top',
    color: '',
    fabric: ''
  });

  const categories = [
    'All', 'Top', 'Bottom', 'Trousers', 'Jeans', 'Skirt', 'Dress', 'Shorts',
    'T-Shirt', 'Sweater', 'Jacket', 'Outerwear',
    'Shoes', 'Bag', 'Necklace', 'Ring', 'Earrings', 'Watch', 'Accessories'
  ];

  useEffect(() => { fetchItems(); }, []);

  const fetchItems = async () => {
    setLoading(true);
    try {
      const data = await api.wardrobe.getAll();
      setItems(data.map((item: any) => ({
        id: item.item_id || item.id.toString(),
        name: item.name,
        category: item.category,
        color: item.color,
        fabric: item.fabric,
        imageUrl: item.bg_removed_url || item.image_url,
        bgRemovedUrl: item.bg_removed_url,
        wearCount: item.wear_count || 0,
        isFavorite: false
      })));
    } catch (e) {
      console.error('Fetch wardrobe failed:', e);
    } finally {
      setLoading(false);
    }
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64 = reader.result as string;
        setImagePreview(base64);
        setAnalyzing(true);
        try {
          const local = await analyzeImageLocally(base64);
          setLocalAnalysis(local);
          const analysis = await api.wardrobe.scanFabric(base64);
          if (analysis && analysis.success) {
            setNewItem({
              name: analysis.name,
              category: analysis.category,
              color: local.shadeNames[0] || analysis.color,
              fabric: analysis.fabric
            });
          }
        } catch (err) {
          console.error('Backend Autotagging failed', err);
        } finally {
          setAnalyzing(false);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleEdit = (item: WardrobeItem) => {
    setEditingId(item.id);
    setNewItem({ name: item.name, category: item.category, color: item.color, fabric: item.fabric });
    setImagePreview(item.imageUrl || null);
    setShowAddModal(true);
  };

  const handleAddItem = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('name', newItem.name || '');
      formData.append('category', newItem.category || 'Top');
      formData.append('color', newItem.color || '');
      formData.append('fabric', newItem.fabric || '');

      if (editingId) {
        if (imagePreview && !imagePreview.startsWith('http')) {
          formData.append('image_url', imagePreview);
        }
        await api.wardrobe.update(editingId, formData);
      } else {
        if (imagePreview) formData.append('image_url', imagePreview);
        await api.wardrobe.add(formData);
      }

      await fetchItems();
      setShowAddModal(false);
      resetForm();
    } catch (err: any) {
      setError(err.message || 'Failed to save item.');
    } finally {
      setSubmitting(false);
    }
  };

  const resetForm = () => {
    setNewItem({ name: '', category: 'Top', color: '', fabric: '' });
    setImagePreview(null);
    setLocalAnalysis(null);
    setEditingId(null);
  };

  /** Opens the Archive confirmation modal instead of a window.confirm */
  const handleDeleteClick = (item: WardrobeItem) => {
    setArchivingItem(item);
  };

  const handleArchiveConfirm = async (reason: string, note: string) => {
    if (!archivingItem) return;
    try {
      await api.wardrobe.archive(archivingItem.id, reason, note);
      setItems(prev => prev.filter(i => i.id !== archivingItem.id));
    } catch (e) {
      console.error('Archive failed:', e);
      alert('Could not archive item. Please try again.');
    } finally {
      setArchivingItem(null);
    }
  };

  const handleHardDelete = async () => {
    if (!archivingItem) return;
    try {
      const res = await api.wardrobe.delete(archivingItem.id);
      if (res.success) setItems(prev => prev.filter(i => i.id !== archivingItem.id));
    } catch (e) {
      console.error('Delete failed:', e);
      alert('Failed to delete item.');
    } finally {
      setArchivingItem(null);
    }
  };

  const handleWear = async (id: string) => {
    try {
      const res = await api.wardrobe.wear(id, new Date().toISOString());
      if (res.success) {
        setItems(prev => prev.map(item =>
          item.id === id ? { ...item, wearCount: (item.wearCount || 0) + 1 } : item
        ));
      }
    } catch (e) {
      console.error('Wear count update failed:', e);
    }
  };

  /** Trigger background removal for a single item */
  const handleRemoveBg = async (id: string) => {
    setRemovingBg(id);
    try {
      const res = await api.wardrobe.removeBackground(id);
      if (res.bg_removed_url) {
        setItems(prev => prev.map(item =>
          item.id === id ? { ...item, imageUrl: res.bg_removed_url, bgRemovedUrl: res.bg_removed_url } : item
        ));
      }
    } catch (e) {
      console.error('Background removal failed:', e);
      alert('Background removal failed. Please try again.');
    } finally {
      setRemovingBg(null);
    }
  };

  const filteredItems = filter === 'All' ? items : items.filter(i => i.category === filter);

  return (
    <div className="p-6 bg-white min-h-full">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl serif">My Wardrobe</h1>
        <button
          onClick={() => { resetForm(); setShowAddModal(true); }}
          className="w-12 h-12 gradient-bg rounded-full flex items-center justify-center text-white shadow-lg active:scale-90 transition-transform"
        >
          <Plus className="w-6 h-6" />
        </button>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-4 mb-8 custom-scrollbar scroll-smooth">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setFilter(cat)}
            className={`px-6 py-2.5 rounded-full text-[10px] font-black uppercase tracking-widest whitespace-nowrap transition-all shadow-sm ${filter === cat ? 'bg-pink-100 text-pink-500 border border-pink-200' : 'bg-slate-50 text-slate-400 border border-transparent'}`}
          >
            {cat}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex justify-center py-24"><Loader2 className="w-10 h-10 animate-spin text-pink-200" /></div>
      ) : (
        <div className="grid grid-cols-2 gap-5 pb-24">
          {filteredItems.length > 0 ? filteredItems.map(item => (
            <div key={item.id} className="bg-slate-50 rounded-[32px] p-4 relative group shadow-sm border border-slate-100/50">
              <div className="absolute top-3 right-3 flex flex-col gap-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                {/* Archive / Delete button */}
                <button
                  onClick={() => handleDeleteClick(item)}
                  className="p-2 bg-white/90 rounded-full text-slate-400 shadow-sm hover:bg-red-50 hover:text-red-400 transition-colors"
                  title="Archive or delete"
                >
                  <Archive className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => handleEdit(item)}
                  className="p-2 bg-white/90 rounded-full text-blue-400 shadow-sm hover:bg-blue-50 transition-colors"
                >
                  <Edit2 className="w-3.5 h-3.5" />
                </button>
                {/* Background removal */}
                <button
                  onClick={() => handleRemoveBg(item.id)}
                  disabled={removingBg === item.id || !!item.bgRemovedUrl}
                  className={`p-2 bg-white/90 rounded-full shadow-sm transition-colors ${item.bgRemovedUrl ? 'text-emerald-400' : 'text-purple-400 hover:bg-purple-50'}`}
                  title={item.bgRemovedUrl ? 'Background already removed' : 'Remove background (AI)'}
                >
                  {removingBg === item.id
                    ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    : <Wand2 className="w-3.5 h-3.5" />
                  }
                </button>
                <button
                  onClick={() => handleWear(item.id)}
                  className="p-2 bg-white/90 rounded-full text-pink-400 shadow-sm flex items-center gap-1.5 px-3 hover:bg-pink-50 transition-colors"
                >
                  <Heart className="w-3.5 h-3.5 fill-pink-400" />
                  <span className="text-[8px] font-black whitespace-nowrap">Worn {item.wearCount}</span>
                </button>
              </div>

              <div className="w-full aspect-[3/4] bg-white rounded-2xl mb-4 overflow-hidden shadow-inner flex items-center justify-center relative">
                {item.imageUrl ? (
                  <img src={item.imageUrl} className="w-full h-full object-contain p-2" alt={item.name} />
                ) : (
                  <Box className="w-12 h-12 text-slate-100" />
                )}
                {/* BG-removed badge */}
                {item.bgRemovedUrl && (
                  <span className="absolute top-2 left-2 bg-emerald-500 text-white text-[7px] font-black uppercase tracking-widest px-2 py-0.5 rounded-full">
                    Clean BG
                  </span>
                )}
              </div>

              <h3 className="text-[10px] font-black text-slate-800 uppercase truncate px-1">{item.name}</h3>
              <p className="text-[9px] text-pink-400 uppercase font-black tracking-widest mt-1 px-1">{item.category}</p>
            </div>
          )) : (
            <div className="col-span-2 py-20 flex flex-col items-center opacity-30">
              <Box className="w-16 h-16 mb-4 text-slate-200" />
              <p className="text-xs font-bold uppercase tracking-widest text-slate-400">No pieces found in {filter}</p>
            </div>
          )}
        </div>
      )}

      {/* Archive modal */}
      {archivingItem && (
        <ArchiveModal
          item={archivingItem}
          onConfirm={handleArchiveConfirm}
          onHardDelete={handleHardDelete}
          onCancel={() => setArchivingItem(null)}
        />
      )}

      {/* Add / Edit modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-[100] flex items-end justify-center">
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => { setShowAddModal(false); resetForm(); }} />
          <div className="relative w-full max-w-md bg-white rounded-t-[50px] p-8 animate-slide-up max-h-[95vh] overflow-y-auto shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl serif text-slate-800">{editingId ? 'Edit Piece' : 'Wardrobe Entry'}</h2>
              <button onClick={() => { setShowAddModal(false); resetForm(); }} className="p-3 hover:bg-slate-50 rounded-full transition-colors">
                <X className="w-6 h-6" />
              </button>
            </div>

            <form onSubmit={handleAddItem} className="space-y-6">
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-100 rounded-[40px] p-4 flex flex-col items-center justify-center bg-slate-50/50 cursor-pointer relative aspect-[3/4] group transition-colors hover:border-pink-200 overflow-hidden"
              >
                {imagePreview ? (
                  <img src={imagePreview} className="absolute inset-0 w-full h-full object-contain p-4" alt="Preview" />
                ) : (
                  <div className="flex flex-col items-center">
                    <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mb-4 shadow-sm border border-slate-50">
                      <Plus className="w-8 h-8 text-slate-300 group-hover:text-pink-300 transition-colors" />
                    </div>
                    <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Capture Piece</p>
                  </div>
                )}
                {analyzing && (
                  <div className="absolute inset-0 bg-white/80 backdrop-blur-md flex flex-col items-center justify-center text-pink-500 z-20 animate-fade-in">
                    <Loader2 className="w-12 h-12 animate-spin mb-4" />
                    <span className="text-[10px] font-black uppercase tracking-[5px]">AI Classifying...</span>
                  </div>
                )}
                <input type="file" ref={fileInputRef} onChange={handleImageChange} className="hidden" accept="image/*" />
              </div>

              <div className="space-y-4">
                {localAnalysis && (
                  <div className="flex gap-2 animate-fade-in mb-4 justify-center">
                    {localAnalysis.palette.map((color, idx) => (
                      <div
                        key={idx}
                        className="w-7 h-7 rounded-full border border-white shadow-sm"
                        style={{ backgroundColor: color }}
                        title={localAnalysis.shadeNames[idx]}
                      />
                    ))}
                  </div>
                )}

                {/* Background-removal tip */}
                <div className="flex items-center gap-2 bg-purple-50 rounded-2xl px-4 py-3 border border-purple-100">
                  <Wand2 className="w-4 h-4 text-purple-400 shrink-0" />
                  <p className="text-[9px] text-purple-600 font-bold leading-relaxed">
                    After saving, tap the wand icon on any item to remove its background — perfect for lookbook-style Daily Drops.
                  </p>
                </div>

                <div className="relative">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-black text-pink-500 uppercase tracking-widest bg-pink-50 px-3 py-1 rounded-full">{newItem.fabric || 'Scanning...'}</span>
                  </div>
                  <input
                    type="text"
                    placeholder="Item Title (e.g. Vintage Jeans)"
                    required
                    className="w-full bg-slate-50 border border-slate-100 rounded-2xl p-5 text-sm outline-none focus:ring-2 ring-pink-100 font-bold"
                    value={newItem.name}
                    onChange={e => setNewItem({ ...newItem, name: e.target.value })}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <select
                    className="w-full bg-slate-50 border border-slate-100 rounded-2xl p-5 text-sm outline-none font-bold appearance-none"
                    value={newItem.category}
                    onChange={e => setNewItem({ ...newItem, category: e.target.value })}
                  >
                    {categories.filter(c => c !== 'All').map(cat => (
                      <option key={cat}>{cat}</option>
                    ))}
                  </select>
                  <input
                    type="text"
                    placeholder="Material"
                    className="w-full bg-pink-50 border border-pink-100 text-pink-700 rounded-2xl p-5 text-sm outline-none font-bold"
                    value={newItem.fabric}
                    onChange={e => setNewItem({ ...newItem, fabric: e.target.value })}
                  />
                </div>
              </div>

              {error && <p className="text-[10px] text-rose-500 font-bold text-center">{error}</p>}

              <button
                type="submit"
                disabled={submitting || analyzing || !imagePreview}
                className="w-full py-6 gradient-bg text-white rounded-full font-black uppercase tracking-[3px] text-xs shadow-2xl flex items-center justify-center gap-3 active:scale-95 transition-all disabled:opacity-50"
              >
                {submitting ? <Loader2 className="w-5 h-5 animate-spin" /> : (
                  <div className="flex items-center gap-2 font-black">
                    <Check className="w-4 h-4" />
                    {editingId ? 'Update Item' : 'Save to Wardrobe'}
                  </div>
                )}
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Closet;
