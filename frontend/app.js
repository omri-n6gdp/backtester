// app.js — single-file React app (no build step).
//
// Phase 4 scope (intentionally minimal):
//   - sidebar: list strategies from /strategies, click to select
//   - right rail: hardcoded SPY/1d/2010-today config + Run button
//   - main canvas: equity curve (strategy + buy & hold) via lightweight-charts,
//     plus a small metrics table below
//
// No JSX (we'd need a build step). We use htm for tagged-template HTML, which
// reads almost identically to JSX.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import htm from 'htm';

import { getHealth, listStrategies, runBacktest } from './api.js';

const html = htm.bind(React.createElement);
const LWC = window.LightweightCharts; // injected by the standalone script in index.html

// ----------------------------- helpers -----------------------------

function fmtPct(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return (x * 100).toFixed(digits) + '%';
}
function fmtNum(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return Number(x).toFixed(digits);
}
function fmtMoney(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return '$' + Math.round(x).toLocaleString();
}
function fmtInt(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return Math.round(x).toLocaleString();
}
function isoToUnixSeconds(iso) {
  // lightweight-charts wants seconds since epoch (UTC) for time-series data.
  return Math.floor(new Date(iso).getTime() / 1000);
}
function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

// Buy-and-hold equity curve from the same close-price series, reusing the
// run's starting capital. We don't have closes in the BacktestResponse
// directly, but we can reconstruct B&H from the per-bar returns *if* we had
// them — except those returns are strategy-shifted. So instead, we derive
// B&H from a fresh /prices fetch... but that adds a roundtrip. The simpler,
// good-enough approximation here: use cumulative product of (1 + bar_return)
// where bar_return is `returns / max(positions, epsilon) when position != 0`,
// which is nonsense for flat periods.
//
// Cleaner approach: compute B&H from the equity curve when positions are flat
// is impossible. So we just compute it from the underlying close, by asking
// the backend... but Phase 4 said keep it minimal.
//
// Final decision: skip B&H here in Phase 4. We'll add it in Phase 7 when the
// run configurator is built out and we have a separate /prices roundtrip.

// ----------------------------- chart ------------------------------

function EquityChart({ timestamps, equity }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  // Create chart once.
  useEffect(() => {
    if (!containerRef.current) return;
    if (!LWC) {
      console.error('lightweight-charts not loaded');
      return;
    }

    const chart = LWC.createChart(containerRef.current, {
      autoSize: true,
      layout: {
        background: { type: 'solid', color: '#0e1014' },
        textColor: '#9aa3b2',
        fontFamily: 'JetBrains Mono, ui-monospace, monospace',
      },
      grid: {
        vertLines: { color: '#181c24' },
        horzLines: { color: '#181c24' },
      },
      rightPriceScale: { borderColor: '#222732' },
      timeScale: { borderColor: '#222732', timeVisible: false, secondsVisible: false },
      crosshair: { mode: 1 },
    });

    // v5 API: addSeries(SeriesType, options)
    const series = chart.addSeries(LWC.AreaSeries, {
      lineColor: '#22c55e',
      topColor: 'rgba(34, 197, 94, 0.28)',
      bottomColor: 'rgba(34, 197, 94, 0.02)',
      lineWidth: 2,
      priceFormat: { type: 'price', precision: 0, minMove: 1 },
    });

    chartRef.current = chart;
    seriesRef.current = series;

    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // Update data whenever it changes.
  useEffect(() => {
    if (!seriesRef.current || !timestamps || !equity) return;
    const data = [];
    let lastTime = -1;
    for (let i = 0; i < timestamps.length; i++) {
      const v = equity[i];
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      const t = isoToUnixSeconds(timestamps[i]);
      if (t <= lastTime) continue; // lightweight-charts requires strictly increasing time
      lastTime = t;
      data.push({ time: t, value: Number(v) });
    }
    seriesRef.current.setData(data);
    chartRef.current && chartRef.current.timeScale().fitContent();
  }, [timestamps, equity]);

  return html`<div ref=${containerRef} class="w-full h-full"></div>`;
}

// ----------------------------- UI bits -----------------------------

function StatusDot({ ok }) {
  const cls = ok
    ? 'bg-accent-500 shadow-[0_0_8px_rgba(34,197,94,0.7)]'
    : 'bg-danger-500';
  return html`<span class="inline-block w-2 h-2 rounded-full ${cls}"></span>`;
}

function Sidebar({ strategies, selectedId, onSelect, loading, error, onRefresh }) {
  return html`
    <aside class="w-72 shrink-0 border-r border-ink-700 bg-ink-900 flex flex-col">
      <div class="px-4 py-3 border-b border-ink-700 flex items-center justify-between">
        <div class="text-xs uppercase tracking-wider text-mute-400">Strategies</div>
        <button
          onClick=${onRefresh}
          class="text-xs text-mute-300 hover:text-mute-100 transition"
          title="Refresh">
          ↻
        </button>
      </div>
      <div class="flex-1 overflow-y-auto">
        ${loading && html`<div class="px-4 py-3 text-mute-400 text-sm">Loading…</div>`}
        ${error && html`<div class="px-4 py-3 text-danger-500 text-sm">Failed: ${error}</div>`}
        ${!loading && !error && strategies.length === 0 && html`
          <div class="px-4 py-6 text-mute-400 text-sm">
            No strategies yet. POST one to <span class="num text-mute-300">/strategies</span>.
          </div>
        `}
        <ul>
          ${strategies.map(s => {
            const active = s.id === selectedId;
            const itemCls = active
              ? 'border-l-2 border-accent-500 bg-ink-850'
              : 'border-l-2 border-transparent hover:bg-ink-850';
            return html`
              <li key=${s.id}>
                <button
                  onClick=${() => onSelect(s.id)}
                  class="w-full text-left px-4 py-3 ${itemCls} transition">
                  <div class="text-sm text-mute-100 font-medium truncate">${s.name}</div>
                  ${s.description && html`
                    <div class="text-xs text-mute-400 truncate mt-0.5">${s.description}</div>
                  `}
                  <div class="text-[10px] text-mute-400 num mt-1">${s.id.slice(0, 8)}…</div>
                </button>
              </li>
            `;
          })}
        </ul>
      </div>
    </aside>
  `;
}

function RightRail({ selectedId, config, setConfig, onRun, running }) {
  const disabled = !selectedId || running;
  const update = (k, v) => setConfig(c => ({ ...c, [k]: v }));

  return html`
    <aside class="w-80 shrink-0 border-l border-ink-700 bg-ink-900 flex flex-col">
      <div class="px-4 py-3 border-b border-ink-700">
        <div class="text-xs uppercase tracking-wider text-mute-400">Run config</div>
      </div>
      <div class="p-4 space-y-4 flex-1 overflow-y-auto">
        <${Field} label="Ticker">
          <input
            class="w-full bg-ink-850 border border-ink-700 rounded px-2 py-1.5 text-sm num text-mute-100 focus:outline-none focus:border-accent-500"
            value=${config.ticker}
            onChange=${e => update('ticker', e.target.value.toUpperCase())} />
        </${Field}>
        <${Field} label="Timeframe">
          <select
            class="w-full bg-ink-850 border border-ink-700 rounded px-2 py-1.5 text-sm num text-mute-100 focus:outline-none focus:border-accent-500"
            value=${config.timeframe}
            onChange=${e => update('timeframe', e.target.value)}>
            <option value="1d">1d</option>
            <option value="1h">1h</option>
            <option value="5m">5m</option>
          </select>
        </${Field}>
        <div class="grid grid-cols-2 gap-3">
          <${Field} label="Start">
            <input type="date"
              class="w-full bg-ink-850 border border-ink-700 rounded px-2 py-1.5 text-sm num text-mute-100 focus:outline-none focus:border-accent-500"
              value=${config.start}
              onChange=${e => update('start', e.target.value)} />
          </${Field}>
          <${Field} label="End">
            <input type="date"
              class="w-full bg-ink-850 border border-ink-700 rounded px-2 py-1.5 text-sm num text-mute-100 focus:outline-none focus:border-accent-500"
              value=${config.end}
              onChange=${e => update('end', e.target.value)} />
          </${Field}>
        </div>
        <${Field} label="Starting capital">
          <input type="number" step="1000" min="1"
            class="w-full bg-ink-850 border border-ink-700 rounded px-2 py-1.5 text-sm num text-mute-100 focus:outline-none focus:border-accent-500"
            value=${config.starting_capital}
            onChange=${e => update('starting_capital', Number(e.target.value))} />
        </${Field}>

        <div class="text-[10px] uppercase tracking-wider text-mute-400 pt-2">
          Costs (defaults; editable in Phase 7)
        </div>
        <div class="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs num text-mute-300">
          <div class="flex justify-between"><span class="text-mute-400">Slippage</span><span>5 bps</span></div>
          <div class="flex justify-between"><span class="text-mute-400">Commission</span><span>0 bps</span></div>
          <div class="flex justify-between"><span class="text-mute-400">FX fee</span><span>15 bps</span></div>
          <div class="flex justify-between"><span class="text-mute-400">Borrow</span><span>0 bps</span></div>
        </div>
      </div>
      <div class="p-4 border-t border-ink-700">
        <button
          onClick=${onRun}
          disabled=${disabled}
          class="w-full px-3 py-2.5 rounded font-medium text-sm transition
                 ${disabled
                    ? 'bg-ink-700 text-mute-400 cursor-not-allowed'
                    : 'bg-accent-500 hover:bg-accent-400 text-ink-950'}">
          ${running ? 'Running…' : 'Run Backtest'}
        </button>
        ${!selectedId && html`
          <div class="text-xs text-mute-400 mt-2 text-center">
            Pick a strategy on the left.
          </div>
        `}
      </div>
    </aside>
  `;
}

function Field({ label, children }) {
  return html`
    <label class="block">
      <div class="text-[10px] uppercase tracking-wider text-mute-400 mb-1">${label}</div>
      ${children}
    </label>
  `;
}

function MetricCard({ label, value, sub, tone }) {
  const valueCls =
    tone === 'good'  ? 'text-accent-400' :
    tone === 'bad'   ? 'text-danger-500' :
                       'text-mute-100';
  return html`
    <div class="bg-ink-900 border border-ink-700 rounded px-4 py-3">
      <div class="text-[10px] uppercase tracking-wider text-mute-400">${label}</div>
      <div class="text-2xl num font-semibold mt-1 ${valueCls}">${value}</div>
      ${sub && html`<div class="text-xs text-mute-400 num mt-0.5">${sub}</div>`}
    </div>
  `;
}

function MetricsRow({ result }) {
  if (!result) return null;
  const s = result.summary || {};
  const m = result.metrics || {};
  const cagr = s.cagr;
  const sharpe = s.sharpe;
  const dd = s.max_drawdown;
  return html`
    <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      <${MetricCard}
        label="CAGR"
        value=${fmtPct(cagr)}
        tone=${cagr > 0 ? 'good' : cagr < 0 ? 'bad' : null} />
      <${MetricCard}
        label="Sharpe"
        value=${fmtNum(sharpe, 2)}
        tone=${sharpe > 1 ? 'good' : sharpe < 0 ? 'bad' : null} />
      <${MetricCard}
        label="Max drawdown"
        value=${fmtPct(dd)}
        tone="bad" />
      <${MetricCard}
        label="Trades"
        value=${fmtInt(s.n_trades)} />
      <${MetricCard}
        label="Win rate"
        value=${fmtPct(s.win_rate, 1)} />
      <${MetricCard}
        label="Final equity"
        value=${fmtMoney(s.final_equity)} />
    </div>
  `;
}

function FullMetricsTable({ result }) {
  if (!result) return null;
  const m = result.metrics || {};
  // Group keys we know about; show everything else under "Other".
  const groups = {
    Return: ['total_return', 'cagr', 'best_month', 'worst_month'],
    Risk: ['volatility', 'max_drawdown', 'max_drawdown_duration', 'avg_drawdown',
           'ulcer_index', 'downside_deviation', 'var_95', 'var_99', 'cvar_95'],
    'Risk-adjusted': ['sharpe', 'sortino', 'calmar', 'omega', 'mar'],
    'Trade-level': ['n_trades', 'win_rate', 'profit_factor', 'expectancy',
                    'avg_win', 'avg_loss', 'avg_holding_period',
                    'max_consecutive_wins', 'max_consecutive_losses'],
  };
  const known = new Set(Object.values(groups).flat());
  const other = Object.keys(m).filter(k => !known.has(k));
  if (other.length) groups['Other'] = other;

  function fmtVal(k, v) {
    if (v === null || v === undefined) return '—';
    if (typeof v !== 'number' || Number.isNaN(v)) return String(v);
    if (k.endsWith('_return') || k === 'cagr' || k.startsWith('best_') || k.startsWith('worst_')
        || k === 'volatility' || k.startsWith('max_drawdown') && k !== 'max_drawdown_duration'
        || k === 'avg_drawdown' || k === 'downside_deviation'
        || k.startsWith('var_') || k.startsWith('cvar_') || k === 'win_rate'
        || k === 'avg_win' || k === 'avg_loss') {
      return fmtPct(v, 2);
    }
    if (k === 'n_trades' || k.startsWith('max_consecutive_')) return fmtInt(v);
    if (k === 'max_drawdown_duration' || k === 'avg_holding_period') return fmtInt(v) + ' bars';
    return fmtNum(v, 3);
  }

  return html`
    <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      ${Object.entries(groups).map(([group, keys]) => html`
        <div key=${group} class="bg-ink-900 border border-ink-700 rounded p-4">
          <div class="text-xs uppercase tracking-wider text-mute-400 mb-2">${group}</div>
          <table class="w-full text-sm">
            <tbody>
              ${keys.filter(k => k in m).map(k => html`
                <tr key=${k} class="border-t border-ink-800 first:border-t-0">
                  <td class="py-1.5 text-mute-300">${k.replaceAll('_', ' ')}</td>
                  <td class="py-1.5 text-right num text-mute-100">${fmtVal(k, m[k])}</td>
                </tr>
              `)}
            </tbody>
          </table>
        </div>
      `)}
    </div>
  `;
}

// ----------------------------- main app ----------------------------

function App() {
  const [serverOk, setServerOk] = useState(null);
  const [serverVersion, setServerVersion] = useState('');

  const [strategies, setStrategies] = useState([]);
  const [stratLoading, setStratLoading] = useState(true);
  const [stratError, setStratError] = useState(null);
  const [selectedId, setSelectedId] = useState(null);

  const [config, setConfig] = useState({
    ticker: 'SPY',
    timeframe: '1d',
    start: '2010-01-01',
    end: todayISO(),
    starting_capital: 100_000,
  });

  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState(null);
  const [result, setResult] = useState(null);

  // health check on mount
  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then(h => { if (!cancelled) { setServerOk(true); setServerVersion(h.version || ''); } })
      .catch(() => { if (!cancelled) setServerOk(false); });
    return () => { cancelled = true; };
  }, []);

  const refreshStrategies = () => {
    setStratLoading(true);
    setStratError(null);
    listStrategies()
      .then(list => {
        setStrategies(list);
        setStratLoading(false);
        // auto-select first if nothing selected
        setSelectedId(prev => prev ?? (list[0]?.id ?? null));
      })
      .catch(err => { setStratError(err.message); setStratLoading(false); });
  };

  useEffect(refreshStrategies, []);

  const selectedStrategy = useMemo(
    () => strategies.find(s => s.id === selectedId) || null,
    [strategies, selectedId]
  );

  const onRun = async () => {
    if (!selectedId) return;
    setRunning(true);
    setRunError(null);
    setResult(null);
    try {
      const payload = {
        strategy_id: selectedId,
        ticker: config.ticker,
        timeframe: config.timeframe,
        start: config.start,
        end: config.end,
        params: {},
        starting_capital: config.starting_capital,
      };
      const res = await runBacktest(payload);
      setResult(res);
    } catch (err) {
      setRunError(err.message);
    } finally {
      setRunning(false);
    }
  };

  return html`
    <div class="h-full flex flex-col">
      <!-- top bar -->
      <header class="h-12 shrink-0 border-b border-ink-700 bg-ink-950 flex items-center px-4 gap-4">
        <div class="text-mute-100 font-semibold tracking-tight">Backtester</div>
        <div class="text-mute-400 text-xs">Phase 4 • minimal end-to-end</div>
        <div class="ml-auto flex items-center gap-2 text-xs text-mute-300">
          <${StatusDot} ok=${serverOk === true} />
          <span class="num">
            ${serverOk === null ? 'connecting…'
              : serverOk ? `server ok · v${serverVersion}`
              : 'server unreachable @ 127.0.0.1:8000'}
          </span>
        </div>
      </header>

      <!-- three-panel layout -->
      <div class="flex-1 min-h-0 flex">
        <${Sidebar}
          strategies=${strategies}
          selectedId=${selectedId}
          onSelect=${setSelectedId}
          loading=${stratLoading}
          error=${stratError}
          onRefresh=${refreshStrategies} />

        <main class="flex-1 min-w-0 overflow-y-auto bg-ink-950">
          <div class="p-6 space-y-6">
            <div>
              <div class="flex items-baseline justify-between">
                <div>
                  <div class="text-xs uppercase tracking-wider text-mute-400">Equity curve</div>
                  <div class="text-lg text-mute-100 font-medium mt-0.5">
                    ${selectedStrategy ? selectedStrategy.name : '— pick a strategy —'}
                  </div>
                </div>
                ${result && html`
                  <div class="text-xs text-mute-400 num">
                    run ${result.run_id.slice(0, 8)}…
                  </div>
                `}
              </div>
              <div class="mt-3 h-[420px] bg-ink-900 border border-ink-700 rounded overflow-hidden">
                ${result
                  ? html`<${EquityChart}
                            timestamps=${result.timestamps}
                            equity=${result.equity} />`
                  : html`
                    <div class="h-full flex items-center justify-center text-mute-400 text-sm">
                      ${running
                        ? 'Running backtest…'
                        : runError
                          ? html`<span class="text-danger-500">Error: ${runError}</span>`
                          : 'Press Run Backtest to see results.'}
                    </div>
                  `}
              </div>
            </div>

            ${result && html`
              <div>
                <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">Summary</div>
                <${MetricsRow} result=${result} />
              </div>
              <div>
                <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">All metrics</div>
                <${FullMetricsTable} result=${result} />
              </div>
            `}
          </div>
        </main>

        <${RightRail}
          selectedId=${selectedId}
          config=${config}
          setConfig=${setConfig}
          onRun=${onRun}
          running=${running} />
      </div>
    </div>
  `;
}

const root = createRoot(document.getElementById('app'));
root.render(html`<${App} />`);
