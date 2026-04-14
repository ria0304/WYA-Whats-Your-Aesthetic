export interface User {
  id: string;
  name: string;
  email: string;
  location: string;
  birthday: string;
  age: string;
  gender: string;
  isLoggedIn: boolean;
  emailNotifications?: boolean;
}

export interface WardrobeItem {
  id: string;
  name: string;
  category: string;
  color: string;
  fabric: string;
  imageUrl?: string;
  isFavorite: boolean;
  wearCount?: number;
  // New: background removal
  bgRemovedUrl?: string;
}

export interface ArchivedItem {
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

export interface OutfitSet {
  id: string;
  name: string;
  vibe: string;
  items: WardrobeItem[];
  isManual?: boolean;
  isDaily?: boolean;
  createdDate?: string;
  // New: server-side persistence
  serverId?: number;
  wornAt?: string | null;
}

export interface GapItem {
  category: string;
  description: string;
  reason: string;
  affiliateQuery: string;
  priority: 'high' | 'medium' | 'low';
}

export interface WeatherData {
  city: string;
  temp: number;
  feelsLike: number;
  condition: string;
  humidity: number;
  wind: number;
}

export enum AppTab {
  HOME = 'home',
  CLOSET = 'closet',
  MATCHES = 'matches',
  TRAVEL = 'travel',
  ME = 'me'
}
