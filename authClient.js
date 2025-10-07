// authClient.js
import axios from "axios";

const API_BASE = "https://api.humanhelperai.in";

// ---- tiny token “vault” (swap to secure cookie if you want) ----
let accessToken = null;
let refreshToken = null;

export const tokenStore = {
  get access() { return accessToken; },
  get refresh() { return refreshToken; },
  set(access, refresh) { accessToken = access || null; refreshToken = refresh || null; },
  clear() { accessToken = null; refreshToken = null; },
};

// ---- axios instance ----
export const api = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
});

// attach access token
api.interceptors.request.use((config) => {
  if (accessToken) {
    config.headers.Authorization = `Bearer ${accessToken}`;
  }
  return config;
});

// single-flight refresh guard
let isRefreshing = false;
let queued = [];

function flushQueue(err, newToken) {
  queued.forEach(({ resolve, reject }) => (err ? reject(err) : resolve(newToken)));
  queued = [];
}

// refresh + retry once on 401
api.interceptors.response.use(
  (r) => r,
  async (error) => {
    const original = error.config || {};
    const status = error?.response?.status;

    // Not a 401, bubble up
    if (status !== 401 || original._retry) throw error;

    // No refresh token -> hard logout
    if (!refreshToken) throw error;

    original._retry = true;

    try {
      if (isRefreshing) {
        // wait for ongoing refresh
        const newAccess = await new Promise((resolve, reject) => queued.push({ resolve, reject }));
        original.headers.Authorization = `Bearer ${newAccess}`;
        return api(original);
      }

      isRefreshing = true;
      const { data } = await axios.post(`${API_BASE}/auth/refresh`, { refresh: refreshToken }, { timeout: 10000 });
      // API returns: { access, refresh }
      const newAccess = data.access;
      const newRefresh = data.refresh || refreshToken;
      tokenStore.set(newAccess, newRefresh);

      flushQueue(null, newAccess);
      original.headers.Authorization = `Bearer ${newAccess}`;
      return api(original);
    } catch (e) {
      flushQueue(e, null);
      tokenStore.clear();
      throw e;
    } finally {
      isRefreshing = false;
    }
  }
);

// -------- High-level auth API --------
export const auth = {
  async register({ full_name, mobile, password, email, address }) {
    const { data } = await axios.post(`${API_BASE}/auth/register`, {
      full_name, mobile, password, email, address,
    });
    return data; // { message, expires_in_min, mobile }
  },

  async verify({ mobile, code }) {
    const { data } = await axios.post(`${API_BASE}/auth/verify`, { mobile, code });
    return data; // { message: "verified" }
  },

  async resendCode({ mobile }) {
    const { data } = await axios.post(`${API_BASE}/auth/resend-code`, { mobile });
    return data;
  },

  async login({ mobile, password }) {
    const { data } = await axios.post(`${API_BASE}/auth/login`, { mobile, password });
    // { ok, user, access, refresh }
    tokenStore.set(data.access, data.refresh);
    return data;
  },

  async whoami() {
    const { data } = await api.get(`/whoami`);
    return data; // { user_id, mobile }
  },

  async logout() {
    if (refreshToken) {
      try { await axios.post(`${API_BASE}/auth/logout`, { refresh: refreshToken }); }
      catch (_) { /* ignore */ }
    }
    tokenStore.clear();
  },
};
