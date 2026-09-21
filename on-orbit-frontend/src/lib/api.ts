/**
 * Central API access.
 *
 * The backend origin used to be written as the literal "http://localhost:8000"
 * at twenty-five call sites, so the app could only ever talk to a backend on
 * the developer's own machine. It now comes from NEXT_PUBLIC_API_URL (see
 * .env.example), defaulting to localhost for development.
 *
 * `apiFetch` also handles the access/refresh token pair. Login returns both,
 * but only the access token was ever stored, so the refresh endpoint went
 * unused and sessions simply died after 24 hours. A 401 now triggers one
 * refresh attempt and a replay of the original request; only if that fails do
 * we clear the tokens and send the user to /login.
 */

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

const ACCESS_TOKEN_KEY = "token";
const REFRESH_TOKEN_KEY = "refresh_token";

/** Build an absolute API URL from a path such as "/api/cdms/". */
export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(ACCESS_TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(REFRESH_TOKEN_KEY);
}

export function storeTokens(access: string, refresh?: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(ACCESS_TOKEN_KEY, access);
  if (refresh) window.localStorage.setItem(REFRESH_TOKEN_KEY, refresh);
}

export function clearTokens(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(ACCESS_TOKEN_KEY);
  window.localStorage.removeItem(REFRESH_TOKEN_KEY);
  window.localStorage.removeItem("username");
}

/** Exchange the stored refresh token for a new access token. */
async function refreshAccessToken(): Promise<string | null> {
  const refresh = getRefreshToken();
  if (!refresh) return null;

  try {
    const response = await fetch(apiUrl("/api/refresh/"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refresh }),
    });
    if (!response.ok) return null;

    const data = await response.json();
    if (!data?.access) return null;

    storeTokens(data.access);
    return data.access as string;
  } catch {
    return null;
  }
}

export interface ApiFetchOptions extends RequestInit {
  /** Set false for endpoints that take no credentials (login, register). */
  auth?: boolean;
}

/**
 * fetch() against the API, with the bearer token attached and one automatic
 * refresh-and-retry on 401.
 *
 * A 401 that refresh cannot recover clears the stored tokens and is returned
 * to the caller as-is, rather than thrown. Callers already branch on
 * `response.status === 401` to redirect to /login, so that handling keeps
 * working untouched and simply fires less often.
 */
export async function apiFetch(
  path: string,
  options: ApiFetchOptions = {},
): Promise<Response> {
  const { auth = true, headers, ...rest } = options;

  const send = (token: string | null) => {
    const merged = new Headers(headers);
    if (!merged.has("Content-Type") && rest.body) {
      merged.set("Content-Type", "application/json");
    }
    if (token) merged.set("Authorization", `Bearer ${token}`);
    else merged.delete("Authorization");
    return fetch(apiUrl(path), { ...rest, headers: merged });
  };

  const response = await send(auth ? getAccessToken() : null);

  if (response.status !== 401 || !auth) return response;

  const refreshed = await refreshAccessToken();
  if (!refreshed) {
    clearTokens();
    return response;
  }

  const retried = await send(refreshed);
  if (retried.status === 401) clearTokens();
  return retried;
}

/** apiFetch plus JSON decoding, throwing on a non-2xx response. */
export async function apiJson<T = unknown>(
  path: string,
  options: ApiFetchOptions = {},
): Promise<T> {
  const response = await apiFetch(path, options);
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      detail = body?.error || body?.detail || detail;
    } catch {
      /* response had no JSON body; keep the status-based message */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}
