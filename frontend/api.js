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

export async function setStrategyFavourite(strategyId, favourite) {
  const res = await fetch(`${API_BASE}/strategies/${encodeURIComponent(strategyId)}/favourite`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ favourite }),
  });
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

export async function runEventDriven(payload) {
  const res = await fetch(`${API_BASE}/backtest/event_driven`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow(res);
}

export async function validateStrategy(code) {
  const res = await fetch(`${API_BASE}/strategies/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  });
  return jsonOrThrow(res);
}

export async function translateStrategy(description) {
  const res = await fetch(`${API_BASE}/strategies/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ description }),
  });
  return jsonOrThrow(res);
}

export async function createStrategy(name, description, code) {
  const res = await fetch(`${API_BASE}/strategies`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, code }),
  });
  return jsonOrThrow(res);
}

export async function getLibrary(filters = {}) {
  const qs = new URLSearchParams();
  if (filters.strategy_id) qs.set('strategy_id', filters.strategy_id);
  if (filters.ticker) qs.set('ticker', filters.ticker);
  const url = qs.toString() ? `${API_BASE}/library?${qs}` : `${API_BASE}/library`;
  const res = await fetch(url);
  return jsonOrThrow(res);
}

export async function getRun(runId) {
  const res = await fetch(`${API_BASE}/runs/${encodeURIComponent(runId)}`);
  return jsonOrThrow(res);
}

// ---- T212 (Phase 10: read-only Demo) -----------------------------------

export async function getT212Account() {
  const res = await fetch(`${API_BASE}/t212/account`);
  return jsonOrThrow(res);
}

export async function getT212Positions() {
  const res = await fetch(`${API_BASE}/t212/positions`);
  return jsonOrThrow(res);
}

export async function getT212Orders() {
  const res = await fetch(`${API_BASE}/t212/orders`);
  return jsonOrThrow(res);
}

export async function getT212Pies() {
  const res = await fetch(`${API_BASE}/t212/pies`);
  return jsonOrThrow(res);
}

export async function getT212Instruments({ refresh = false } = {}) {
  const url = refresh
    ? `${API_BASE}/t212/instruments?refresh=true`
    : `${API_BASE}/t212/instruments`;
  const res = await fetch(url);
  return jsonOrThrow(res);
}

// ---- Deployments (Phase 11) --------------------------------------------

export async function listDeployments(filters = {}) {
  const qs = new URLSearchParams();
  if (filters.strategy_id) qs.set('strategy_id', filters.strategy_id);
  if (filters.armed_only) qs.set('armed_only', 'true');
  const url = qs.toString() ? `${API_BASE}/deployments?${qs}` : `${API_BASE}/deployments`;
  const res = await fetch(url);
  return jsonOrThrow(res);
}

export async function getDeployment(id) {
  const res = await fetch(`${API_BASE}/deployments/${encodeURIComponent(id)}`);
  return jsonOrThrow(res);
}

export async function createDeployment(payload) {
  const res = await fetch(`${API_BASE}/deployments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow(res);
}

export async function armDeployment(id, confirmationToken) {
  const res = await fetch(`${API_BASE}/deployments/${encodeURIComponent(id)}/arm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ confirmation_token: confirmationToken }),
  });
  return jsonOrThrow(res);
}

export async function disarmDeployment(id) {
  const res = await fetch(`${API_BASE}/deployments/${encodeURIComponent(id)}/disarm`, {
    method: 'POST',
  });
  return jsonOrThrow(res);
}

export async function runDeployment(id) {
  const res = await fetch(`${API_BASE}/deployments/${encodeURIComponent(id)}/run`, {
    method: 'POST',
  });
  return jsonOrThrow(res);
}

export async function getDeploymentTracking(id) {
  const res = await fetch(`${API_BASE}/deployments/${encodeURIComponent(id)}/tracking`);
  return jsonOrThrow(res);
}

export async function deleteDeployment(id) {
  const res = await fetch(`${API_BASE}/deployments/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
  return jsonOrThrow(res);
}

export async function killSwitch(confirmationToken) {
  const res = await fetch(`${API_BASE}/killswitch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ confirmation_token: confirmationToken }),
  });
  return jsonOrThrow(res);
}
