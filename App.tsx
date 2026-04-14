import React, { useState, useEffect } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { Home, Box, Sparkles, Plane, User, ChevronLeft } from 'lucide-react';
import Login from './views/Login';
import Dashboard from './views/Dashboard';
import Closet from './views/Closet';
import GreenScore from './views/GreenScore';
import WeatherView from './views/Weather';
import ScanLook from './views/ScanLook';
import AIMatcher from './views/AIMatcher';
import VacationShop from './views/VacationShop';
import Evolution from './views/Evolution';
import Profile from './views/Profile';
import StyleQuiz from './views/StyleQuiz';
import Curate from './views/Curate';
import AestheticAura from './views/AestheticAura';
import TheArchive from './views/TheArchive';
import DailyDrop from './views/DailyDrop';
import { AppTab, User as UserType } from './types';
import { api } from './services/api';

const calculateAge = (birthday: string): string => {
  if (!birthday) return '0';
  const birthDate = new Date(birthday);
  const today = new Date();
  let age = today.getFullYear() - birthDate.getFullYear();
  const m = today.getMonth() - birthDate.getMonth();
  if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
    age--;
  }
  return age.toString();
};

const App: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [user, setUser] = useState<UserType | null>(null);
  const [activeTab, setActiveTab] = useState<AppTab>(AppTab.HOME);
  const [styleDNA, setStyleDNA] = useState<string | null>(null);
  const [showSplash, setShowSplash] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('wya_token');
      if (token) {
        try {
          const profile = await api.profile.get();
          setUser({
            id: profile.user_id,
            name: profile.full_name,
            email: profile.email,
            location: profile.location,
            birthday: profile.birthday || '', 
            age: calculateAge(profile.birthday || ''),
            gender: profile.gender,
            isLoggedIn: true,
            emailNotifications: profile.email_notifications === 1
          });
          
          try {
            const dna = await api.style.getDNA(profile.user_id);
            if (dna.has_dna) setStyleDNA(dna.summary);
          } catch (e) {
            console.warn("Style DNA fetch failed, continuing...");
          }
        } catch (e) {
          console.error("Auth check failed", e);
        }
      }
      // Ensure splash clears after attempt
      setTimeout(() => setShowSplash(false), 2000);
    };
    checkAuth();
  }, []);

  useEffect(() => {
    const path = location.pathname.substring(1);
    if (path === '') setActiveTab(AppTab.HOME);
    else if (path.startsWith('closet')) setActiveTab(AppTab.CLOSET);
    else if (path.startsWith('scan-look') || path.startsWith('ai-matcher')) setActiveTab(AppTab.MATCHES);
    else if (path.startsWith('travel')) setActiveTab(AppTab.TRAVEL);
    else if (path.startsWith('me')) setActiveTab(AppTab.ME);
  }, [location]);

  if (showSplash) {
    return (
      <div className="fixed inset-0 z-[1000] flex flex-col items-center justify-center bg-white overflow-hidden">
        <div className="animate-float-splash flex flex-col items-center">
          <div className="w-32 h-32 rounded-full gradient-bg flex items-center justify-center text-white text-6xl serif shadow-[0_20px_40px_rgba(0,0,0,0.05)] border-4 border-white/20 mb-10">
            W
          </div>
          <h1 className="text-5xl font-normal text-slate-800 tracking-wider serif">WYA</h1>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Login onLogin={(u) => {
      setUser(u);
      navigate('/');
    }} />;
  }

  const handleLogout = () => {
    api.auth.logout();
    setUser(null);
    navigate('/');
  };

  const handleTabClick = (tab: AppTab, path: string) => {
    setActiveTab(tab);
    navigate(path);
  };

  const showBackButton = location.pathname !== '/';

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col max-w-md mx-auto relative shadow-2xl overflow-hidden h-[100dvh]">
      {showBackButton && (
        <header className="p-4 flex items-center bg-white border-b sticky top-0 z-50">
          <button onClick={() => navigate(-1)} className="p-2 hover:bg-slate-100 rounded-full transition-colors">
            <ChevronLeft className="w-6 h-6" />
          </button>
          <h1 className="ml-2 text-xl font-semibold capitalize serif">
            {location.pathname.substring(1).split('-').join(' ')}
          </h1>
        </header>
      )}

      <main className="flex-1 overflow-y-auto pb-24 custom-scrollbar">
        <Routes>
          <Route path="/" element={<Dashboard user={user} styleDNA={styleDNA} />} />
          <Route path="/closet" element={<Closet />} />
          <Route path="/green-score" element={<GreenScore />} />
          <Route path="/weather" element={<WeatherView />} />
          <Route path="/scan-look" element={<ScanLook />} />
          <Route path="/ai-matcher" element={<AIMatcher />} />
          <Route path="/travel" element={<VacationShop />} />
          <Route path="/evolution" element={<Evolution />} />
          <Route path="/curate" element={<Curate />} />
          <Route path="/aesthetic-aura" element={<AestheticAura />} />
          <Route path="/archive" element={<TheArchive />} />
          <Route path="/daily-drop" element={<DailyDrop />} />
          <Route path="/me" element={<Profile user={user} onUpdateUser={setUser} onLogout={handleLogout} />} />
          <Route path="/quiz" element={<StyleQuiz onComplete={async (dnaSummary) => { 
            try {
              await api.style.saveDNA({ 
                user_id: user.id, 
                styles: dnaSummary.includes('Minimalist') ? ['minimalist'] : ['classic'], 
                comfort_level: 50,
                summary: dnaSummary
              });
            } catch (e) { console.warn("Saving DNA failed locally", e); }
            setStyleDNA(dnaSummary); 
            navigate('/'); 
          }} userGender={user.gender} />} />
        </Routes>
      </main>

      <nav className="fixed bottom-0 left-0 right-0 max-w-md mx-auto bg-white border-t py-2 px-4 flex justify-around items-center z-50">
        <button onClick={() => handleTabClick(AppTab.HOME, '/')} className={`flex flex-col items-center p-2 transition-colors ${activeTab === AppTab.HOME ? 'text-pink-500' : 'text-slate-400'}`}>
          <Home className="w-6 h-6" />
          <span className="text-[10px] mt-1 font-bold">Home</span>
        </button>
        <button onClick={() => handleTabClick(AppTab.CLOSET, '/closet')} className={`flex flex-col items-center p-2 transition-colors ${activeTab === AppTab.CLOSET ? 'text-pink-500' : 'text-slate-400'}`}>
          <Box className="w-6 h-6" />
          <span className="text-[10px] mt-1 font-bold">Closet</span>
        </button>
        <button onClick={() => handleTabClick(AppTab.MATCHES, '/ai-matcher')} className={`flex flex-col items-center p-2 transition-colors ${activeTab === AppTab.MATCHES ? 'text-pink-500' : 'text-slate-400'}`}>
          <Sparkles className="w-6 h-6" />
          <span className="text-[10px] mt-1 font-bold">Matches</span>
        </button>
        <button onClick={() => handleTabClick(AppTab.TRAVEL, '/travel')} className={`flex flex-col items-center p-2 transition-colors ${activeTab === AppTab.TRAVEL ? 'text-pink-500' : 'text-slate-400'}`}>
          <Plane className="w-6 h-6" />
          <span className="text-[10px] mt-1 font-bold">Travel</span>
        </button>
        <button onClick={() => handleTabClick(AppTab.ME, '/me')} className={`flex flex-col items-center p-2 transition-colors ${activeTab === AppTab.ME ? 'text-pink-500' : 'text-slate-400'}`}>
          <User className="w-6 h-6" />
          <span className="text-[10px] mt-1 font-bold">Me</span>
        </button>
      </nav>
    </div>
  );
};

export default App;
