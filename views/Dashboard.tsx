import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, Sparkles, Sun, Camera, Plane, Leaf, History,
  User, Palette, Shirt, Layout, Archive, Zap, Bell
} from 'lucide-react';
import { User as UserType } from '../types';
import { api } from '../services/api';

interface DashboardProps {
  user: UserType;
  styleDNA: string | null;
}

const Dashboard: React.FC<DashboardProps> = ({ user, styleDNA }) => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await api.dashboard.getStats();
        setStats(data);
      } catch (e) {
        console.error('Failed to fetch dashboard stats', e);
      }
    };
    fetchStats();
  }, []);

  return (
    <div className="min-h-full gradient-bg overflow-y-auto pb-10">
      <div className="px-8 pt-10 pb-6 flex justify-between items-center text-white">
        <h1 className="text-5xl font-normal serif">hi, {user.name.split(' ')[0]}!</h1>
        <button
          onClick={() => navigate('/me')}
          className="w-12 h-12 rounded-full bg-white/30 backdrop-blur-md flex items-center justify-center border border-white/40 hover:bg-white/40 transition-colors"
        >
          <User className="w-6 h-6" />
        </button>
      </div>

      {/* Style DNA Card */}
      <div className="px-6 mb-8">
        <div className="glass-card rounded-[40px] p-8 shadow-xl">
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-[4px] mb-2">Your Style DNA</p>
          {styleDNA ? (
            <div className="animate-fade-in">
              <h2 className="text-2xl text-slate-700 font-bold mb-4 leading-tight">{stats?.style_archetype || 'DNA Mapped!'}</h2>
              <p className="text-slate-500 text-sm leading-relaxed mb-6">{styleDNA}</p>
              <button
                onClick={() => navigate('/quiz')}
                className="w-full bg-slate-800 text-white rounded-full p-4 text-[10px] font-bold tracking-[3px] uppercase hover:bg-slate-700 transition-colors"
              >
                Retake Style Quiz
              </button>
            </div>
          ) : (
            <>
              <h2 className="text-4xl text-slate-700 serif mb-4 leading-tight">Style Profile Pending</h2>
              <p className="text-slate-500 text-sm mb-6 leading-relaxed">
                Complete your style questionnaire to discover your fashion DNA
              </p>
              <div className="space-y-2 mb-8">
                <div className="flex justify-between text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                  <span>Confidence Level</span>
                  <span>{stats?.style_confidence || 50}%</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-pink-200 to-purple-200 h-full rounded-full transition-all duration-1000"
                    style={{ width: `${stats?.style_confidence || 50}%` }}
                  />
                </div>
              </div>
              <button
                onClick={() => navigate('/quiz')}
                className="w-full bg-white/80 border border-white rounded-full p-4 text-[10px] font-bold text-slate-400 tracking-[3px] uppercase hover:bg-white transition-colors"
              >
                Start Style Quiz
              </button>
            </>
          )}
        </div>
      </div>

      {/* Main Grid */}
      <div className="bg-white rounded-t-[60px] p-8 pb-12 shadow-[0_-20px_40px_rgba(0,0,0,0.05)]">
        <div className="grid grid-cols-2 gap-4">

          {/* Core 4 */}
          <MenuButton icon={<Box className="w-7 h-7" />} label={`Closet (${stats?.wardrobe_count || 0})`} bgColor="bg-indigo-50" iconColor="bg-indigo-100" onClick={() => navigate('/closet')} />
          <MenuButton icon={<Shirt className="w-7 h-7" />} label="AI Matcher" bgColor="bg-pink-50" iconColor="bg-pink-100" onClick={() => navigate('/ai-matcher')} />
          <MenuButton icon={<Palette className="w-7 h-7" />} label="Scan Look" bgColor="bg-emerald-50" iconColor="bg-emerald-100" onClick={() => navigate('/scan-look')} />
          <MenuButton icon={<Sun className="w-7 h-7" />} label="Weather" bgColor="bg-blue-50" iconColor="bg-blue-100" onClick={() => navigate('/weather')} />

          {/* Daily Drop – full width hero */}
          <div className="col-span-2">
            <button
              onClick={() => navigate('/daily-drop')}
              className="w-full bg-gradient-to-br from-violet-50 to-pink-50 p-6 rounded-[32px] flex items-center gap-6 hover:from-violet-100 hover:to-pink-100 transition-all text-left border border-white shadow-sm group"
            >
              <div className="w-14 h-14 bg-white rounded-full flex items-center justify-center text-violet-500 shadow-sm group-hover:scale-110 transition-transform">
                <Bell className="w-7 h-7" />
              </div>
              <div>
                <h3 className="text-violet-600 font-bold uppercase tracking-widest text-xs mb-1">Daily Drop</h3>
                <p className="text-violet-300 text-[10px] font-bold uppercase tracking-wider">Your AI outfit · every morning</p>
              </div>
              <div className="ml-auto">
                <span className="text-[8px] font-black bg-violet-500 text-white px-3 py-1.5 rounded-full uppercase tracking-widest">New</span>
              </div>
            </button>
          </div>

          {/* Travel – full width */}
          <div className="col-span-2">
            <button
              onClick={() => navigate('/travel')}
              className="w-full bg-indigo-50/50 p-6 rounded-[32px] flex items-center gap-6 hover:bg-indigo-50 transition-colors text-left border border-white shadow-sm"
            >
              <div className="w-14 h-14 bg-white rounded-full flex items-center justify-center text-indigo-500 shadow-sm">
                <Plane className="w-8 h-8" />
              </div>
              <div>
                <h3 className="text-indigo-600 font-bold uppercase tracking-widest text-xs mb-1">Vacation Shop</h3>
                <p className="text-indigo-300 text-[10px] font-bold uppercase tracking-wider">Your personal travel stylist</p>
              </div>
            </button>
          </div>

          <MenuButton icon={<Leaf className="w-6 h-6" />} label="Green Score" bgColor="bg-emerald-50/30" iconColor="bg-emerald-50" labelColor="text-emerald-800" onClick={() => navigate('/green-score')} />
          <MenuButton icon={<History className="w-6 h-6" />} label="Evolution" bgColor="bg-blue-50/30" iconColor="bg-blue-50" labelColor="text-blue-800" onClick={() => navigate('/evolution')} />

          {/* Curate – full width */}
          <div className="col-span-2">
            <button
              onClick={() => navigate('/curate')}
              className="w-full bg-pink-50/30 p-6 rounded-[32px] flex items-center gap-6 hover:bg-pink-50 transition-colors text-left border border-white shadow-sm"
            >
              <div className="w-14 h-14 bg-white rounded-full flex items-center justify-center text-pink-500 shadow-sm">
                <Layout className="w-8 h-8" />
              </div>
              <div>
                <h3 className="text-pink-600 font-bold uppercase tracking-widest text-xs mb-1">Curate</h3>
                <p className="text-pink-300 text-[10px] font-bold uppercase tracking-wider">Auto-generated outfit sets</p>
              </div>
            </button>
          </div>

          {/* Aesthetic Aura */}
          <div className="col-span-2">
            <button
              onClick={() => navigate('/aesthetic-aura')}
              className="w-full bg-gradient-to-br from-fuchsia-50 to-amber-50 p-6 rounded-[32px] flex items-center gap-6 hover:from-fuchsia-100 hover:to-amber-100 transition-all text-left border border-white shadow-sm group"
            >
              <div className="w-14 h-14 bg-white rounded-full flex items-center justify-center text-fuchsia-500 shadow-sm group-hover:scale-110 transition-transform">
                <Zap className="w-7 h-7" />
              </div>
              <div>
                <h3 className="text-fuchsia-600 font-bold uppercase tracking-widest text-xs mb-1">Aesthetic Aura</h3>
                <p className="text-fuchsia-300 text-[10px] font-bold uppercase tracking-wider">Your style — Spotify Wrapped</p>
              </div>
            </button>
          </div>

          {/* Archive */}
          <div className="col-span-2">
            <button
              onClick={() => navigate('/archive')}
              className="w-full bg-slate-50 p-6 rounded-[32px] flex items-center gap-6 hover:bg-slate-100 transition-colors text-left border border-white shadow-sm"
            >
              <div className="w-14 h-14 bg-white rounded-full flex items-center justify-center text-slate-500 shadow-sm">
                <Archive className="w-7 h-7" />
              </div>
              <div>
                <h3 className="text-slate-700 font-bold uppercase tracking-widest text-xs mb-1">The Archive</h3>
                <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider">Memory lane · retired pieces</p>
              </div>
            </button>
          </div>

        </div>
      </div>
    </div>
  );
};

const MenuButton: React.FC<{
  icon: React.ReactNode;
  label: string;
  bgColor: string;
  iconColor: string;
  labelColor?: string;
  onClick?: () => void;
}> = ({ icon, label, bgColor, iconColor, labelColor, onClick }) => (
  <button
    onClick={onClick}
    className={`${bgColor} p-6 rounded-[32px] flex flex-col items-center justify-center transition-all active:scale-95 group border border-white shadow-sm hover:shadow-md`}
  >
    <div className={`w-14 h-14 ${iconColor} rounded-full flex items-center justify-center mb-4 transition-transform group-hover:scale-110 shadow-inner`}>
      {icon}
    </div>
    <span className={`${labelColor || 'text-black'} text-[10px] font-black uppercase tracking-widest text-center`}>{label}</span>
  </button>
);

export default Dashboard;
