const getBaseUrl = () => {
  if (typeof window !== 'undefined') {
    const { hostname } = window.location;
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return `http://${hostname}:8000`;
    }
    return '';
  }
  return 'http://localhost:8000';
};

const BASE_URL = getBaseUrl();

const storage = {
  get: (key: string) => JSON.parse(localStorage.getItem(`wya_v3_${key}`) || 'null'),
  set: (key: string, val: any) => localStorage.setItem(`wya_v3_${key}`, JSON.stringify(val)),
};

const getAuthHeaders = (isFormData = false) => {
  const token = localStorage.getItem('wya_token');
  const headers: Record<string, string> = {};
  if (token && token !== 'null' && token !== 'undefined') {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (!isFormData) headers['Content-Type'] = 'application/json';
  return headers;
};

const handleResponse = async (response: Response, defaultError: string) => {
  if (!response.ok) {
    if (response.status === 401) {
      localStorage.removeItem('wya_token');
      localStorage.removeItem('wya_v3_user');
      if (window.location.pathname !== '/') {
        window.location.href = '/';
      }
    }
    let errorMessage = defaultError;
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || defaultError;
    } catch (e) {}
    throw new Error(errorMessage);
  }
  return response.json();
};

const apiFetch = async (endpoint: string, options: RequestInit = {}) => {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        ...getAuthHeaders(options.body instanceof FormData),
        ...(options.headers || {})
      }
    });
    return await handleResponse(response, `Request to ${endpoint} failed`);
  } catch (err) {
    console.warn(`API Error (${endpoint}):`, err);
    throw new Error(err instanceof TypeError && err.message === 'Failed to fetch'
      ? 'Unable to connect to server. Ensure the backend is running on port 8000.'
      : (err as Error).message);
  }
};

// ─── Style DNA resilience: local queue ───────────────────────────────────────
const DNA_PENDING_KEY = 'wya_dna_pending';

const flushPendingDNA = async () => {
  const pending = JSON.parse(localStorage.getItem(DNA_PENDING_KEY) || 'null');
  if (!pending) return;
  try {
    await apiFetch('/api/style/dna', {
      method: 'POST',
      body: JSON.stringify(pending),
    });
    localStorage.removeItem(DNA_PENDING_KEY);
  } catch (_) { /* will retry next time */ }
};

// Flush any queued DNA on boot
if (typeof window !== 'undefined') {
  window.addEventListener('online', flushPendingDNA);
  setTimeout(flushPendingDNA, 3000);
}

export const api = {
  auth: {
    login: async (credentials: any) => {
      const data = await apiFetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials),
      });
      if (data.access_token) {
        localStorage.setItem('wya_token', data.access_token);
        storage.set('user', data.user);
      }
      return data;
    },
    register: async (userData: any) => {
      const data = await apiFetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData),
      });
      if (data.access_token) {
        localStorage.setItem('wya_token', data.access_token);
        storage.set('user', data.user);
      }
      return data;
    },
    logout: () => {
      localStorage.removeItem('wya_token');
      localStorage.removeItem('wya_v3_user');
      window.location.href = '/';
    }
  },

  dashboard: {
    getStats: async () => apiFetch('/api/dashboard/stats', {}),
  },

  wardrobe: {
    getAll: async () => apiFetch('/api/wardrobe', {}),
    add: async (formData: FormData) => apiFetch('/api/wardrobe', { method: 'POST', body: formData }),
    update: async (id: string, formData: FormData) => apiFetch(`/api/wardrobe/${id}`, { method: 'PUT', body: formData }),
    delete: async (id: string) => apiFetch(`/api/wardrobe/${id}`, { method: 'DELETE' }),
    /** Log wear with optional ISO timestamp for Evolution tracking */
    wear: async (id: string, wornAt?: string) => apiFetch(`/api/wardrobe/${id}/wear`, {
      method: 'POST',
      body: JSON.stringify({ worn_at: wornAt || new Date().toISOString() }),
    }),
    /** Send item to archive instead of hard-deleting */
    archive: async (id: string, reason: string, memoryNote?: string) =>
      apiFetch(`/api/wardrobe/${id}/archive`, {
        method: 'POST',
        body: JSON.stringify({ reason, memory_note: memoryNote || '' }),
      }),
    /** Remove background from a closet image (returns { bg_removed_url }) */
    removeBackground: async (id: string) =>
      apiFetch(`/api/wardrobe/${id}/remove-bg`, { method: 'POST' }),

    // AI Endpoints
    scanFabric: async (image: string) => apiFetch('/api/ai/fabric-scan', {
      method: 'POST',
      body: JSON.stringify({ image })
    }),
    outfitMatch: async (image: string, variation: number = 0) => apiFetch('/api/ai/outfit-match', {
      method: 'POST',
      body: JSON.stringify({ image, variation })
    }),
    /** Gap analysis: compare Style DNA to actual inventory */
    gapAnalysis: async () => apiFetch('/api/ai/gap-analysis', { method: 'POST' }),
  },

  // ─── Outfits (server-persisted) ────────────────────────────────────────────
  outfits: {
    getAll: async () => apiFetch('/api/outfits', {}),
    save: async (outfit: any) => apiFetch('/api/outfits', {
      method: 'POST',
      body: JSON.stringify(outfit),
    }),
    delete: async (id: number) => apiFetch(`/api/outfits/${id}`, { method: 'DELETE' }),
    /** Log an outfit as worn, records worn_at timestamp */
    logWorn: async (id: number, wornAt?: string) => apiFetch(`/api/outfits/${id}/worn`, {
      method: 'POST',
      body: JSON.stringify({ worn_at: wornAt || new Date().toISOString() }),
    }),
  },

  // ─── Archive ────────────────────────────────────────────────────────────────
  archive: {
    getAll: async () => apiFetch('/api/archive', {}),
    permanentDelete: async (id: string) => apiFetch(`/api/archive/${id}`, { method: 'DELETE' }),
  },

  ai: {
    getWeather: async (city: string) => apiFetch('/api/ai/weather-search', {
      method: 'POST',
      body: JSON.stringify({ city })
    }),
    getGreenAudit: async (brand: string) => apiFetch('/api/ai/green-audit', {
      method: 'POST',
      body: JSON.stringify({ brand })
    }),
    getVacationPlan: async (type: string, days: number, city: string) =>
      apiFetch(`/api/ai/vacation-packer?vacation_type=${type}&duration_days=${days}&city=${encodeURIComponent(city)}`, {}),
    curateOutfits: async (items: any[]) => apiFetch('/api/ai/curate-outfits', {
      method: 'POST',
      body: JSON.stringify({ items })
    }),
    /** Daily Drop – uses color_harmony logic server-side */
    getDailyDrop: async () => apiFetch('/api/ai/daily-drop', { method: 'POST' }),
  },

  profile: {
    get: async () => apiFetch('/api/user/profile', {}),
    update: async (data: any) => apiFetch('/api/user/profile', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),
    getPreferences: async () => apiFetch('/api/user/preferences', {}),
    updatePreferences: async (data: any) => apiFetch('/api/user/preferences', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),
    getActivity: async () => apiFetch('/api/user/activity', {}),
  },

  style: {
    getDNA: async (user_id: string) => apiFetch(`/api/style/dna/${user_id}`, {}),
    /**
     * Save Style DNA with resilience:
     * if the request fails, queue it locally and retry on next online event.
     */
    saveDNA: async (dnaData: any) => {
      try {
        const result = await apiFetch('/api/style/dna', {
          method: 'POST',
          body: JSON.stringify(dnaData),
        });
        localStorage.removeItem(DNA_PENDING_KEY);
        return result;
      } catch (err) {
        // Cache for background retry
        localStorage.setItem(DNA_PENDING_KEY, JSON.stringify(dnaData));
        console.warn('DNA save failed – queued for retry');
        throw err;
      }
    },
    getEvolution: async () => apiFetch('/api/style/evolution', {}),
  }
};
