// api.js — thin fetch wrappers around the local FastAPI server.
//
// IMPORTANT: use 127.0.0.1 (matches `server.py` which binds to 127.0.0.1:8000).
// 'localhost' resolves differently in some browsers when the server is bound
// to the v4 loopback only.

export const API_BASE = 'http://127.0.0.1:8000';

async function jsonOrThrow(res) {
  const ct = res.headers.get('content-type') || '';
  const body = ct.includes('application/json') ? await res.json().catch(() => null) : await res.text();
  if (!res.ok) {
    const detail =
      (body && typeof body === 'object' && (body.detail?.message || body.detail)) ||
      (typeof body === 'string' ? body : '') ||
      `HTTP ${res.status}`;
    const err = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return jsonOrThrow(res);
}

export async function listStrategies() {
  const res = await fetch(`${API_BASE}/strategies`);
  return jsonOrThrow(res);
}

export async function getStrategy(strategyId) {
  const res = await fetch(`${API_BASE}/strategies/${encodeURIComponent(strategyId)}`);
  return jsonOrThrow(res);
}

export async function runBacktest(payload) {
  const res = await fetch(`${API_BASE}/backtest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow(res);
}

export async function runSweep(payload) {
  const res = await fetch(`${API_BASE}/backtest/sweep`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow(res);
}
