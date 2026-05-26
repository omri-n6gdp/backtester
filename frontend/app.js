// app.js — Phase 7: full results dashboard + authoring + robustness
//
// Two top-level views: 'run' (backtest/sweep, with optional comparison
// of pinned runs) and 'author' (new strategy). Within 'run', two run
// modes: 'backtest' and 'sweep'.

import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import htm from 'htm';

import { EditorView, basicSetup } from 'codemirror';
import { python } from '@codemirror/lang-python';
import { oneDark } from '@codemirror/theme-one-dark';

import {
  getHealth, listStrategies, runBacktest, runSweep, runEventDriven,
  validateStrategy, translateStrategy, createStrategy, setStrategyFavourite,
  getLibrary, getRun,
  getT212Account, getT212Positions, getT212Orders, getT212Pies,
  listDeployments, createDeployment, armDeployment, disarmDeployment,
  runDeployment, getDeploymentTracking, getDeploymentRunsFeed,
  killSwitch, deleteDeployment,
} from './api.js';

const html = htm.bind(React.createElement);
const LWC = window.LightweightCharts;

const STRATEGY_TEMPLATE = `import pandas as pd

def strategy(data, params):
    # Return a Series of target positions in [-1.0, +1.0] aligned with data.index.
    return pd.Series(0.0, index=data.index)
`;

// Matches backend/engine/costs.py CostConfig field names exactly.
const DEFAULT_COST_CONFIG = {
  slippage_mode: 'fixed',
  slippage_bps: 5.0,
  commission_bps: 0.0,
  spread_bps: 0.0,
  fx_bps: 15.0,
  instrument_is_gbp: false,
  borrow_bps_annual: 0.0,
};

// Up to N pins; pin colors cycle through these.
const PIN_COLORS = ['#22c55e', '#3b82f6', '#f59e0b'];

// Phase 14 — friendly schedule presets for the Deploy form. All in UTC
// because the scheduler is UTC-only; the US close picks summer/winter
// explicitly because cron has no concept of DST.
const SCHEDULE_PRESETS = [
  { label: 'After US close · weekdays · 21:05 UTC (summer/EDT)', cron: '5 21 * * 1-5' },
  { label: 'After US close · weekdays · 22:05 UTC (winter/EST)', cron: '5 22 * * 1-5' },
  { label: 'Before US open · weekdays · 13:25 UTC',              cron: '25 13 * * 1-5' },
  { label: 'After EU close · weekdays · 16:35 UTC',              cron: '35 16 * * 1-5' },
  { label: 'Daily · 00:05 UTC',                                  cron: '5 0 * * *' },
];
const PIN_LIMIT = 3;

const MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

// CLAUDE.md default composite weights for the library view.
const DEFAULT_LIB_WEIGHTS = {
  sharpe: 30,
  calmar: 20,
  consistency: 15,
  dsr: 15,
  breadth: 10,
  recency: 10,
};

// Combine a row's component scores with user weights. Components whose score
// is null (currently just DSR) have their weight redistributed pro rata to the
// available components — so DSR-less runs aren't unfairly zeroed.
function compositeScore(row, weights) {
  const components = [
    ['sharpe',      row.sharpe_score],
    ['calmar',      row.calmar_score],
    ['consistency', row.consistency_score],
    ['dsr',         row.dsr_score],
    ['breadth',     row.breadth_score],
    ['recency',     row.recency_score],
  ];
  let totalActiveWeight = 0;
  for (const [name, val] of components) {
    if (val !== null && val !== undefined) totalActiveWeight += weights[name];
  }
  if (totalActiveWeight === 0) return 0;
  let score = 0;
  for (const [name, val] of components) {
    if (val === null || val === undefined) continue;
    score += (weights[name] / totalActiveWeight) * val;
  }
  return score;
}

// ---- helpers -------------------------------------------------------

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
  return Math.floor(new Date(iso).getTime() / 1000);
}
function todayISO() {
  return new Date().toISOString().slice(0, 10);
}
function fmtMetricVal(k, v) {
  if (v === null || v === undefined) return '—';
  if (typeof v !== 'number' || Number.isNaN(v)) return String(v);
  const isPct = k.endsWith('_return') || k === 'cagr' || k.startsWith('best_') ||
    k.startsWith('worst_') || k === 'volatility' || k === 'avg_drawdown' ||
    k === 'downside_deviation' || k.startsWith('var_') || k.startsWith('cvar_') ||
    k === 'win_rate' || k === 'avg_win' || k === 'avg_loss' ||
    (k.startsWith('max_drawdown') && k !== 'max_drawdown_duration');
  if (isPct) return fmtPct(v);
  if (k === 'n_trades' || k.startsWith('max_consecutive_')) return fmtInt(v);
  if (k === 'max_drawdown_duration' || k === 'avg_holding_period') return fmtInt(v) + ' bars';
  return fmtNum(v, 3);
}

// ---- EquityChart ---------------------------------------------------

// Build a {time, value} series from a curve aligned to `timestamps`,
// dropping nulls/NaNs and enforcing strictly-increasing time (lightweight-
// charts rejects duplicate/decreasing timestamps).
function curveToChartData(timestamps, values) {
  const data = [];
  if (!timestamps || !values) return data;
  let lastTime = -1;
  for (let i = 0; i < timestamps.length; i++) {
    const v = values[i];
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    const t = isoToUnixSeconds(timestamps[i]);
    if (t <= lastTime) continue;
    lastTime = t;
    data.push({ time: t, value: Number(v) });
  }
  return data;
}

function EquityChart({ timestamps, equity, buyHold, title }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  const bhSeriesRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !LWC) return;
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
    // Buy & hold drawn first so the strategy area sits on top of it.
    const bhSeries = chart.addSeries(LWC.LineSeries, {
      color: '#6b7280',
      lineWidth: 1,
      lineStyle: 2,  // dashed
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      priceFormat: { type: 'price', precision: 0, minMove: 1 },
    });
    const series = chart.addSeries(LWC.AreaSeries, {
      lineColor: '#22c55e',
      topColor: 'rgba(34,197,94,0.28)',
      bottomColor: 'rgba(34,197,94,0.02)',
      lineWidth: 2,
      priceFormat: { type: 'price', precision: 0, minMove: 1 },
    });
    chartRef.current = chart;
    seriesRef.current = series;
    bhSeriesRef.current = bhSeries;
    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      bhSeriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!seriesRef.current || !timestamps || !equity) return;
    seriesRef.current.setData(curveToChartData(timestamps, equity));
    chartRef.current && chartRef.current.timeScale().fitContent();
  }, [timestamps, equity]);

  useEffect(() => {
    if (!bhSeriesRef.current) return;
    bhSeriesRef.current.setData(curveToChartData(timestamps, buyHold));
  }, [timestamps, buyHold]);

  const hasBuyHold = !!(buyHold && buyHold.some(v => v !== null && v !== undefined));

  return html`
    <div class="relative w-full h-full">
      <div ref=${containerRef} class="w-full h-full"></div>
      ${hasBuyHold && html`
        <div class="absolute top-1.5 left-2 flex gap-3 text-[10px] text-slate-400
                    pointer-events-none z-10">
          <span class="flex items-center gap-1">
            <span class="inline-block w-3 h-[2px] bg-accent-500"></span>Strategy</span>
          <span class="flex items-center gap-1">
            <span class="inline-block w-3 border-t border-dashed"
                  style=${{ borderColor: '#6b7280' }}></span>Buy & hold</span>
        </div>`}
    </div>`;
}

// ---- UI primitives -------------------------------------------------

function StatusDot({ ok }) {
  const cls = ok
    ? 'bg-accent-500 shadow-[0_0_8px_rgba(34,197,94,0.7)]'
    : 'bg-danger-500';
  return html`<span class="inline-block w-2 h-2 rounded-full ${cls}"></span>`;
}

function Field({ label, children }) {
  return html`
    <label class="block">
      <div class="text-[10px] uppercase tracking-wider text-mute-400 mb-1">${label}</div>
      ${children}
    </label>
  `;
}

function inputCls() {
  return 'w-full bg-ink-850 border border-ink-700 rounded px-2 py-1.5 text-sm num text-mute-100 focus:outline-none focus:border-accent-500';
}

function MetricCard({ label, value, sub, tone }) {
  const valueCls =
    tone === 'good' ? 'text-accent-400' :
    tone === 'bad'  ? 'text-danger-500' :
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
  return html`
    <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      <${MetricCard} label="CAGR" value=${fmtPct(s.cagr)} tone=${s.cagr > 0 ? 'good' : s.cagr < 0 ? 'bad' : null} />
      <${MetricCard} label="Sharpe" value=${fmtNum(s.sharpe, 2)} tone=${s.sharpe > 1 ? 'good' : s.sharpe < 0 ? 'bad' : null} />
      <${MetricCard} label="Max drawdown" value=${fmtPct(s.max_drawdown)} tone="bad" />
      <${MetricCard} label="Trades" value=${fmtInt(s.n_trades)} />
      <${MetricCard} label="Win rate" value=${fmtPct(s.win_rate, 1)} />
      <${MetricCard} label="Final equity" value=${fmtMoney(s.final_equity)} />
    </div>
  `;
}

function FullMetricsTable({ result }) {
  if (!result) return null;
  const m = result.metrics || {};
  const groups = {
    Return: ['total_return','cagr','best_month','worst_month'],
    Risk: ['volatility','max_drawdown','max_drawdown_duration','avg_drawdown',
           'ulcer_index','downside_deviation','var_95','var_99','cvar_95'],
    'Risk-adjusted': ['sharpe','sortino','calmar','omega','mar'],
    'Trade-level': ['n_trades','win_rate','profit_factor','expectancy',
                    'avg_win','avg_loss','avg_holding_period',
                    'max_consecutive_wins','max_consecutive_losses'],
  };
  const known = new Set(Object.values(groups).flat());
  const other = Object.keys(m).filter(k => !known.has(k));
  if (other.length) groups['Other'] = other;

  return html`
    <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      ${Object.entries(groups).map(([group, keys]) => html`
        <div key=${group} class="bg-ink-900 border border-ink-700 rounded p-4">
          <div class="text-xs uppercase tracking-wider text-mute-400 mb-2">${group}</div>
          <table class="w-full text-sm">
            <tbody>
              ${keys.filter(k => k in m).map(k => html`
                <tr key=${k} class="border-t border-ink-800 first:border-t-0">
                  <td class="py-1.5 text-mute-300">${k.replaceAll('_',' ')}</td>
                  <td class="py-1.5 text-right num text-mute-100">${fmtMetricVal(k, m[k])}</td>
                </tr>
              `)}
            </tbody>
          </table>
        </div>
      `)}
    </div>
  `;
}

function Sidebar({ strategies, selectedId, onSelect, loading, error, onRefresh,
                   onNewStrategy, onToggleFavourite }) {
  return html`
    <aside class="w-72 shrink-0 border-r border-ink-700 bg-ink-900 flex flex-col">
      <div class="px-4 py-3 border-b border-ink-700 flex items-center justify-between">
        <div class="text-xs uppercase tracking-wider text-mute-400">Strategies</div>
        <div class="flex items-center gap-2">
          <button onClick=${onNewStrategy}
            class="text-xs px-2 py-0.5 rounded bg-accent-600 hover:bg-accent-500 text-white font-semibold transition"
            title="New strategy">
            + New
          </button>
          <button onClick=${onRefresh} class="text-xs text-mute-300 hover:text-mute-100 transition" title="Refresh">↻</button>
        </div>
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
            const cls = active
              ? 'border-l-2 border-accent-500 bg-ink-850'
              : 'border-l-2 border-transparent hover:bg-ink-850';
            return html`
              <li key=${s.id} class="flex items-stretch ${cls} transition">
                <button onClick=${() => onSelect(s.id)} class="flex-1 min-w-0 text-left px-4 py-3">
                  <div class="text-sm text-mute-100 font-medium truncate">${s.name}</div>
                  ${s.description && html`<div class="text-xs text-mute-400 truncate mt-0.5">${s.description}</div>`}
                  <div class="text-[10px] text-mute-400 num mt-1">${s.id.slice(0,8)}…</div>
                </button>
                <button onClick=${() => onToggleFavourite(s)}
                  class="shrink-0 px-3 text-lg leading-none transition
                         ${s.favourite ? 'text-yellow-400' : 'text-mute-600 hover:text-mute-300'}"
                  title=${s.favourite ? 'Remove from favourites' : 'Add to favourites'}>
                  ${s.favourite ? '★' : '☆'}
                </button>
              </li>
            `;
          })}
        </ul>
      </div>
    </aside>
  `;
}

// ---- Robustness panels ---------------------------------------------

function DSRCard({ dsr }) {
  if (!dsr) return null;
  const gap = dsr.psr !== null && dsr.dsr !== null ? ((dsr.psr - dsr.dsr) * 100).toFixed(1) : '—';
  return html`
    <div class="bg-ink-900 border border-ink-700 rounded p-4">
      <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">Deflated Sharpe Ratio</div>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <div class="text-[10px] text-mute-400">Raw Sharpe</div>
          <div class="text-2xl num font-semibold text-mute-100 mt-1">
            ${dsr.sharpe_annualized !== null ? dsr.sharpe_annualized.toFixed(3) : '—'}
          </div>
        </div>
        <div>
          <div class="text-[10px] text-mute-400">PSR · P(SR&gt;0)</div>
          <div class="text-2xl num font-semibold mt-1 ${dsr.psr > 0.95 ? 'text-accent-400' : dsr.psr > 0.8 ? 'text-mute-100' : 'text-danger-500'}">
            ${dsr.psr !== null ? (dsr.psr * 100).toFixed(1) + '%' : '—'}
          </div>
        </div>
        <div>
          <div class="text-[10px] text-mute-400">DSR · ${dsr.n_trials} trial${dsr.n_trials === 1 ? '' : 's'}</div>
          <div class="text-2xl num font-semibold mt-1 ${dsr.dsr > 0.8 ? 'text-accent-400' : dsr.dsr > 0.5 ? 'text-mute-100' : 'text-danger-500'}">
            ${dsr.dsr !== null ? (dsr.dsr * 100).toFixed(1) + '%' : '—'}
          </div>
        </div>
        <div>
          <div class="text-[10px] text-mute-400">E[max SR | H₀, N=${dsr.n_trials}]</div>
          <div class="text-2xl num font-semibold text-mute-100 mt-1">
            ${dsr.expected_max_sharpe_annualized !== null ? dsr.expected_max_sharpe_annualized.toFixed(3) : '—'}
          </div>
        </div>
      </div>
      <div class="mt-3 text-xs text-mute-400">
        Selection-bias tax PSR→DSR: ${gap}pp across ${dsr.n_trials} trial${dsr.n_trials === 1 ? '' : 's'}.
        Skew ${dsr.skew !== null ? dsr.skew.toFixed(2) : '—'} · excess kurtosis ${dsr.excess_kurtosis !== null ? dsr.excess_kurtosis.toFixed(1) : '—'}.
      </div>
    </div>
  `;
}

function OOSCompare({ oos }) {
  if (!oos) return null;
  const { in_sample: is_, out_of_sample: oos_, split_timestamp } = oos;

  function windowCard(w) {
    const m = w.metrics || {};
    const isOOS = w.label !== 'in_sample';
    return html`
      <div class="bg-ink-900 border ${isOOS ? 'border-accent-600' : 'border-ink-700'} rounded p-4">
        <div class="text-xs uppercase tracking-wider ${isOOS ? 'text-accent-400' : 'text-mute-400'} mb-1">
          ${isOOS ? 'Out-of-sample' : 'In-sample'}
          <span class="ml-2 normal-case text-mute-400 font-normal">${w.n_bars} bars</span>
        </div>
        <div class="text-[10px] text-mute-400 num mb-3">${w.start?.slice(0,10)} → ${w.end?.slice(0,10)}</div>
        <div class="grid grid-cols-3 gap-3">
          ${[['CAGR', fmtPct(m.cagr)], ['Sharpe', fmtNum(m.sharpe,2)], ['Max DD', fmtPct(m.max_drawdown)],
             ['Calmar', fmtNum(m.calmar,2)], ['Win rate', fmtPct(m.win_rate,1)], ['Trades', fmtInt(m.n_trades)]]
            .map(([lbl, val]) => html`
              <div key=${lbl}>
                <div class="text-[10px] text-mute-400">${lbl}</div>
                <div class="text-sm num text-mute-100">${val}</div>
              </div>
            `)}
        </div>
      </div>
    `;
  }

  return html`
    <div>
      <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">
        In-sample vs Out-of-sample
        <span class="ml-2 normal-case font-normal text-mute-400">split ${split_timestamp?.slice(0,10)}</span>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
        ${windowCard(is_)}
        ${windowCard(oos_)}
      </div>
    </div>
  `;
}

function MCCard({ mc }) {
  if (!mc) return null;
  const { n_iterations, n_trades, max_dd_quantiles, original_max_drawdown, original_dd_percentile } = mc;
  const qKeys = Object.keys(max_dd_quantiles || {}).sort((a,b) => parseFloat(a) - parseFloat(b));
  const pctRank = original_dd_percentile !== null ? (original_dd_percentile * 100).toFixed(1) + '%' : '—';
  const verdict = original_dd_percentile !== null && original_dd_percentile >= 0.5
    ? 'Actual ordering was average or better than a random shuffle — consistent with skill.'
    : 'Actual ordering produced worse DD than the median shuffle — ordering drag present.';

  return html`
    <div class="bg-ink-900 border border-ink-700 rounded p-4">
      <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">
        Monte Carlo · ${n_iterations.toLocaleString()} trade-order shuffles · ${n_trades} trades
      </div>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <div class="text-[10px] text-mute-400">Actual max DD</div>
          <div class="text-2xl num font-semibold text-danger-500 mt-1">${fmtPct(original_max_drawdown)}</div>
        </div>
        <div>
          <div class="text-[10px] text-mute-400">DD percentile rank</div>
          <div class="text-2xl num font-semibold mt-1 ${original_dd_percentile >= 0.5 ? 'text-accent-400' : 'text-mute-100'}">
            ${pctRank}
          </div>
        </div>
        ${qKeys.map(q => html`
          <div key=${q}>
            <div class="text-[10px] text-mute-400">Max DD P${Math.round(parseFloat(q)*100)}</div>
            <div class="text-2xl num font-semibold text-mute-100 mt-1">${fmtPct(max_dd_quantiles[q])}</div>
          </div>
        `)}
      </div>
      <div class="mt-3 text-xs text-mute-400">${verdict}</div>
    </div>
  `;
}

function RegimePanel({ regimes }) {
  if (!regimes) return null;

  function axisTable(axis, label, error) {
    if (error) return html`
      <div class="bg-ink-900 border border-ink-700 rounded p-4">
        <div class="text-xs uppercase tracking-wider text-mute-400 mb-2">${label}</div>
        <div class="text-xs text-danger-500">${error}</div>
      </div>
    `;
    if (!axis) return null;
    return html`
      <div class="bg-ink-900 border border-ink-700 rounded p-4">
        <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">
          ${label}
          <span class="ml-2 normal-case font-normal text-mute-400">${axis.n_bars_total} bars total</span>
        </div>
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-ink-700">
              <th class="text-left py-1 text-mute-400 text-xs font-medium">Regime</th>
              <th class="text-right py-1 text-mute-400 text-xs font-medium">% time</th>
              <th class="text-right py-1 text-mute-400 text-xs font-medium">CAGR</th>
              <th class="text-right py-1 text-mute-400 text-xs font-medium">Sharpe</th>
              <th class="text-right py-1 text-mute-400 text-xs font-medium">Max DD</th>
            </tr>
          </thead>
          <tbody>
            ${Object.entries(axis.windows).sort().map(([k, w]) => html`
              <tr key=${k} class="border-t border-ink-800">
                <td class="py-1.5 text-mute-200">${w.label.replace(/_/g,' ')}</td>
                <td class="py-1.5 text-right num text-mute-300">${fmtPct(w.fraction, 1)}</td>
                <td class="py-1.5 text-right num ${w.metrics.cagr > 0 ? 'text-accent-400' : 'text-danger-500'}">${fmtPct(w.metrics.cagr)}</td>
                <td class="py-1.5 text-right num text-mute-200">${fmtNum(w.metrics.sharpe, 2)}</td>
                <td class="py-1.5 text-right num text-danger-500">${fmtPct(w.metrics.max_drawdown)}</td>
              </tr>
            `)}
          </tbody>
        </table>
      </div>
    `;
  }

  return html`
    <div>
      <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">Regime analysis</div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
        ${axisTable(regimes.spy_trend, 'SPY trend (bull / bear)', regimes.spy_trend_error)}
        ${axisTable(regimes.vix_level, 'VIX level', regimes.vix_level_error)}
      </div>
    </div>
  `;
}

// ---- Sweep panels --------------------------------------------------

function HeatmapGrid({ heatmap }) {
  const { x_param, y_param, metric_name, x_values, y_values, grid, missing_mask } = heatmap;

  const allVals = grid.flat().filter(v => v !== null && !Number.isNaN(v));
  const lo = allVals.length ? Math.min(...allVals) : 0;
  const hi = allVals.length ? Math.max(...allVals) : 1;

  function cellColor(v) {
    if (v === null || Number.isNaN(v)) return '#181c24';
    const t = hi === lo ? 0.5 : (v - lo) / (hi - lo);
    return `hsl(${Math.round(t * 120)}, 65%, 32%)`;
  }

  const reversedYIdx = y_values.map((_, i) => y_values.length - 1 - i);
  const cols = x_values.length;
  const gridTemplateColumns = `56px ${'1fr '.repeat(cols).trim()}`;

  return html`
    <div>
      <div class="text-[10px] uppercase tracking-wider text-mute-400 mb-2">
        ${metric_name.replace(/_/g,' ')} · ${x_param} (x) × ${y_param} (y)
      </div>

      <div style=${{ display:'grid', gridTemplateColumns, gap:'2px', marginBottom:'2px' }}>
        <div></div>
        ${x_values.map(x => html`
          <div key=${x} class="text-center text-xs num text-mute-300 pb-1">${x}</div>
        `)}
      </div>

      ${reversedYIdx.map(ri => {
        const y = y_values[ri];
        const row = grid[ri];
        const mask = missing_mask[ri];
        return html`
          <div key=${y} style=${{ display:'grid', gridTemplateColumns, gap:'2px', marginBottom:'2px' }}>
            <div class="flex items-center justify-end pr-2 text-xs num text-mute-300">${y}</div>
            ${x_values.map((_, ci) => {
              const v = mask[ci] ? null : row[ci];
              return html`
                <div key=${ci}
                  style=${{ background: cellColor(v), borderRadius:'3px', padding:'6px 2px', textAlign:'center' }}
                  class="text-xs num text-white font-medium">
                  ${v === null ? '—' : v.toFixed(2)}
                </div>
              `;
            })}
          </div>
        `;
      })}

      <div class="flex items-center gap-2 mt-3">
        <span class="text-[10px] num text-mute-400">${lo.toFixed(2)}</span>
        <div style=${{ flex:1, height:'6px', borderRadius:'3px',
          background:'linear-gradient(to right, hsl(0,65%,32%), hsl(60,65%,32%), hsl(120,65%,32%))' }}></div>
        <span class="text-[10px] num text-mute-400">${hi.toFixed(2)}</span>
      </div>
    </div>
  `;
}

function ComboTable({ combos, bestIndex, selectionMetric, higherIsBetter }) {
  const sorted = [...combos].sort((a, b) => {
    if (a.score === null && b.score === null) return 0;
    if (a.score === null) return 1;
    if (b.score === null) return -1;
    return higherIsBetter ? b.score - a.score : a.score - b.score;
  });

  return html`
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-ink-700">
            <th class="text-left py-2 px-3 text-mute-400 text-xs font-medium">Params</th>
            <th class="text-right py-2 px-3 text-mute-400 text-xs font-medium">${selectionMetric.replace(/_/g,' ')}</th>
            <th class="text-right py-2 px-3 text-mute-400 text-xs font-medium">CAGR</th>
            <th class="text-right py-2 px-3 text-mute-400 text-xs font-medium">Max DD</th>
            <th class="text-right py-2 px-3 text-mute-400 text-xs font-medium">Trades</th>
            <th class="text-right py-2 px-3 text-mute-400 text-xs font-medium">Win rate</th>
          </tr>
        </thead>
        <tbody>
          ${sorted.map(c => {
            const isBest = c.index === bestIndex;
            const m = c.metrics || {};
            return html`
              <tr key=${c.index}
                class="border-t border-ink-800 ${isBest ? 'bg-ink-850' : 'hover:bg-ink-900'} transition">
                <td class="py-2 px-3 num text-mute-200" style=${isBest ? {borderLeft:'2px solid #22c55e'} : {}}>
                  ${Object.entries(c.params).map(([k,v]) => `${k}=${v}`).join(', ')}
                  ${isBest && html` <span class="ml-1 text-accent-400 text-[10px] font-semibold">★ best</span>`}
                  ${c.error && html` <span class="ml-1 text-danger-500 text-[10px]">[error]</span>`}
                </td>
                <td class="py-2 px-3 text-right num ${isBest ? 'text-accent-400 font-semibold' : 'text-mute-100'}">
                  ${c.error
                    ? html`<span class="text-danger-500 text-xs" title=${c.error}>err</span>`
                    : c.score !== null ? c.score.toFixed(3) : '—'}
                </td>
                <td class="py-2 px-3 text-right num text-mute-200">${fmtPct(m.cagr)}</td>
                <td class="py-2 px-3 text-right num text-mute-200">${fmtPct(m.max_drawdown)}</td>
                <td class="py-2 px-3 text-right num text-mute-200">${fmtInt(m.n_trades)}</td>
                <td class="py-2 px-3 text-right num text-mute-200">${fmtPct(m.win_rate, 1)}</td>
              </tr>
            `;
          })}
        </tbody>
      </table>
    </div>
  `;
}

// ---- RightRail (mode-aware) ----------------------------------------

function RightRail({ selectedId, config, setConfig, robustness, setRobustness,
                     costConfig, setCostConfig,
                     eventDriven, setEventDriven,
                     onRun, running }) {
  const disabled = !selectedId || running;
  const upd = (k, v) => setConfig(c => ({ ...c, [k]: v }));
  const updR = (k, v) => setRobustness(r => ({ ...r, [k]: v }));
  const updC = (k, v) => setCostConfig(c => ({ ...c, [k]: v }));
  const updED = (k, v) => setEventDriven(s => ({ ...s, [k]: v }));
  const resetCosts = () => setCostConfig(DEFAULT_COST_CONFIG);

  const ic = inputCls();

  return html`
    <aside class="w-80 shrink-0 border-l border-ink-700 bg-ink-900 flex flex-col">
      <div class="p-4 space-y-4 flex-1 overflow-y-auto text-sm">

        <!-- Shared: ticker / timeframe / dates / capital -->
        <${Field} label="Ticker">
          <input class=${ic} value=${config.ticker}
            onChange=${e => upd('ticker', e.target.value.toUpperCase())} />
        </${Field}>
        <${Field} label="Timeframe">
          <select class=${ic} value=${config.timeframe} onChange=${e => upd('timeframe', e.target.value)}>
            <option value="1d">1d</option>
            <option value="1h">1h</option>
            <option value="5m">5m</option>
          </select>
        </${Field}>
        <div class="grid grid-cols-2 gap-3">
          <${Field} label="Start">
            <input type="date" class=${ic} value=${config.start}
              onChange=${e => upd('start', e.target.value)} />
          </${Field}>
          <${Field} label="End">
            <input type="date" class=${ic} value=${config.end}
              onChange=${e => upd('end', e.target.value)} />
          </${Field}>
        </div>
        <${Field} label="Starting capital">
          <input type="number" step="1000" min="1" class=${ic}
            value=${config.starting_capital}
            onChange=${e => upd('starting_capital', Number(e.target.value))} />
        </${Field}>

        <${Field} label="Strategy params (JSON)">
            <textarea rows="2" class="w-full bg-ink-850 border border-ink-700 rounded px-2 py-1.5 text-xs num text-mute-100 focus:outline-none focus:border-accent-500 font-mono resize-none"
              value=${config.paramsJson ?? '{}'}
              onChange=${e => upd('paramsJson', e.target.value)} />
          </${Field}>

          <div class="pt-1 space-y-3">
            <div class="text-[10px] uppercase tracking-wider text-mute-400">Robustness</div>

            <!-- OOS -->
            <div class="flex items-center gap-2">
              <input type="checkbox" id="oos-cb" class="accent-accent-500"
                checked=${robustness.oos} onChange=${e => updR('oos', e.target.checked)} />
              <label for="oos-cb" class="text-xs text-mute-200 flex-1">OOS split</label>
              ${robustness.oos && html`
                <input type="number" min="0.1" max="0.95" step="0.05"
                  class="w-16 bg-ink-850 border border-ink-700 rounded px-1.5 py-0.5 text-xs num text-mute-100 focus:outline-none focus:border-accent-500"
                  value=${robustness.oosSplit}
                  onChange=${e => updR('oosSplit', Number(e.target.value))} />
              `}
            </div>

            <!-- DSR -->
            <div class="flex items-center gap-2">
              <input type="checkbox" id="dsr-cb" class="accent-accent-500"
                checked=${robustness.dsr} onChange=${e => updR('dsr', e.target.checked)} />
              <label for="dsr-cb" class="text-xs text-mute-200 flex-1">Deflated Sharpe</label>
              ${robustness.dsr && html`
                <input type="number" min="1" step="1" placeholder="N trials"
                  class="w-20 bg-ink-850 border border-ink-700 rounded px-1.5 py-0.5 text-xs num text-mute-100 focus:outline-none focus:border-accent-500"
                  value=${robustness.nTrials}
                  onChange=${e => updR('nTrials', e.target.value)} />
              `}
            </div>

            <!-- Monte Carlo -->
            <div class="flex items-center gap-2">
              <input type="checkbox" id="mc-cb" class="accent-accent-500"
                checked=${robustness.monteCarlo} onChange=${e => updR('monteCarlo', e.target.checked)} />
              <label for="mc-cb" class="text-xs text-mute-200 flex-1">Monte Carlo</label>
              ${robustness.monteCarlo && html`
                <input type="number" min="100" max="5000" step="100"
                  class="w-20 bg-ink-850 border border-ink-700 rounded px-1.5 py-0.5 text-xs num text-mute-100 focus:outline-none focus:border-accent-500"
                  value=${robustness.nShuffles}
                  onChange=${e => updR('nShuffles', Number(e.target.value))} />
              `}
            </div>

            <!-- Regimes -->
            <div class="flex items-center gap-2">
              <input type="checkbox" id="reg-cb" class="accent-accent-500"
                checked=${robustness.regimes} onChange=${e => updR('regimes', e.target.checked)} />
              <label for="reg-cb" class="text-xs text-mute-200">Regime analysis (SPY + VIX)</label>
            </div>
          </div>

          <!-- Event-driven: protective stops -->
          <div class="pt-1 space-y-2">
            <div class="flex items-center gap-2">
              <input type="checkbox" id="ed-cb" class="accent-accent-500"
                checked=${eventDriven.enabled}
                onChange=${e => updED('enabled', e.target.checked)} />
              <label for="ed-cb" class="text-xs text-mute-200">Event-driven (intra-bar stops)</label>
            </div>
            ${eventDriven.enabled && html`
              <div class="grid grid-cols-3 gap-2 pl-6">
                <${Field} label="Stop loss %">
                  <input type="number" min="0" max="99" step="0.5"
                    placeholder="off"
                    class="w-full bg-ink-850 border border-ink-700 rounded px-1.5 py-1 text-xs num text-mute-100 focus:outline-none focus:border-accent-500"
                    value=${eventDriven.stop_loss_pct}
                    onChange=${e => updED('stop_loss_pct', e.target.value)} />
                </${Field}>
                <${Field} label="Take profit %">
                  <input type="number" min="0" max="99" step="0.5"
                    placeholder="off"
                    class="w-full bg-ink-850 border border-ink-700 rounded px-1.5 py-1 text-xs num text-mute-100 focus:outline-none focus:border-accent-500"
                    value=${eventDriven.take_profit_pct}
                    onChange=${e => updED('take_profit_pct', e.target.value)} />
                </${Field}>
                <${Field} label="Trailing %">
                  <input type="number" min="0" max="99" step="0.5"
                    placeholder="off"
                    class="w-full bg-ink-850 border border-ink-700 rounded px-1.5 py-1 text-xs num text-mute-100 focus:outline-none focus:border-accent-500"
                    value=${eventDriven.trailing_stop_pct}
                    onChange=${e => updED('trailing_stop_pct', e.target.value)} />
                </${Field}>
              </div>
              <div class="text-[10px] text-mute-400 pl-6">
                Percent values, e.g. 5 = 5%. Empty disables that stop. Event-driven
                mode disables OOS / MC / DSR / regimes for this run.
              </div>
            `}
          </div>

        <!-- Cost config editor -->
        <div class="pt-1">
          <div class="flex items-center justify-between mb-2">
            <div class="text-[10px] uppercase tracking-wider text-mute-400">Costs (bps)</div>
            <button onClick=${resetCosts}
              class="text-[10px] text-mute-400 hover:text-mute-100 transition">
              reset
            </button>
          </div>
          <div class="space-y-2">
            <${Field} label="Slippage mode">
              <select class=${ic} value=${costConfig.slippage_mode}
                onChange=${e => updC('slippage_mode', e.target.value)}>
                <option value="fixed">fixed</option>
                <option value="vol_scaled">vol_scaled (0.5 × ATR%)</option>
              </select>
            </${Field}>
            <div class="grid grid-cols-2 gap-2">
              <${Field} label="Slippage">
                <input type="number" step="0.5" min="0" class=${ic}
                  value=${costConfig.slippage_bps}
                  onChange=${e => updC('slippage_bps', Number(e.target.value))} />
              </${Field}>
              <${Field} label="Commission">
                <input type="number" step="0.5" min="0" class=${ic}
                  value=${costConfig.commission_bps}
                  onChange=${e => updC('commission_bps', Number(e.target.value))} />
              </${Field}>
              <${Field} label="Spread">
                <input type="number" step="0.5" min="0" class=${ic}
                  value=${costConfig.spread_bps}
                  onChange=${e => updC('spread_bps', Number(e.target.value))} />
              </${Field}>
              <${Field} label="FX fee">
                <input type="number" step="0.5" min="0" class=${ic}
                  value=${costConfig.fx_bps}
                  onChange=${e => updC('fx_bps', Number(e.target.value))}
                  disabled=${costConfig.instrument_is_gbp} />
              </${Field}>
            </div>
            <div class="flex items-center gap-2">
              <input type="checkbox" id="gbp-cb" class="accent-accent-500"
                checked=${costConfig.instrument_is_gbp}
                onChange=${e => updC('instrument_is_gbp', e.target.checked)} />
              <label for="gbp-cb" class="text-xs text-mute-200">GBP-denominated (skip FX fee)</label>
            </div>
            <${Field} label="Borrow cost (annual bps)">
              <input type="number" step="1" min="0" class=${ic}
                value=${costConfig.borrow_bps_annual}
                onChange=${e => updC('borrow_bps_annual', Number(e.target.value))} />
            </${Field}>
          </div>
        </div>
      </div>

      <div class="p-4 border-t border-ink-700">
        <button onClick=${onRun} disabled=${disabled}
          class="w-full px-3 py-2.5 rounded font-semibold text-sm transition
                 ${disabled
                   ? 'bg-ink-700 text-mute-400 cursor-not-allowed'
                   : 'bg-accent-500 hover:bg-accent-400 text-ink-950'}">
          ${running ? 'Running…' : 'Run Backtest'}
        </button>
        ${!selectedId && html`<div class="text-xs text-mute-400 mt-2 text-center">Pick a strategy on the left.</div>`}
      </div>
    </aside>
  `;
}

// ---- DrawdownChart (red, negative) ---------------------------------

function DrawdownChart({ timestamps, drawdown }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !LWC) return;
    const chart = LWC.createChart(containerRef.current, {
      autoSize: true,
      layout: {
        background: { type: 'solid', color: '#0e1014' },
        textColor: '#9aa3b2',
        fontFamily: 'JetBrains Mono, ui-monospace, monospace',
      },
      grid: { vertLines: { color: '#181c24' }, horzLines: { color: '#181c24' } },
      rightPriceScale: { borderColor: '#222732' },
      timeScale: { borderColor: '#222732', timeVisible: false, secondsVisible: false },
      crosshair: { mode: 1 },
    });
    const series = chart.addSeries(LWC.AreaSeries, {
      lineColor: '#ef4444',
      topColor: 'rgba(239,68,68,0.02)',
      bottomColor: 'rgba(239,68,68,0.35)',
      lineWidth: 1,
      priceFormat: { type: 'percent', precision: 2, minMove: 0.0001 },
    });
    chartRef.current = chart;
    seriesRef.current = series;
    return () => { chart.remove(); chartRef.current = null; seriesRef.current = null; };
  }, []);

  useEffect(() => {
    if (!seriesRef.current || !timestamps || !drawdown) return;
    const data = [];
    let lastTime = -1;
    for (let i = 0; i < timestamps.length; i++) {
      const v = drawdown[i];
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      const t = isoToUnixSeconds(timestamps[i]);
      if (t <= lastTime) continue;
      lastTime = t;
      // lightweight-charts treats percent format as already-multiplied (e.g. 0.05 = 0.05%).
      // Our drawdown values are fractional (-0.05 = -5%), so multiply by 100.
      data.push({ time: t, value: Number(v) * 100 });
    }
    seriesRef.current.setData(data);
    chartRef.current && chartRef.current.timeScale().fitContent();
  }, [timestamps, drawdown]);

  return html`<div ref=${containerRef} class="w-full h-full"></div>`;
}

// ---- MultiEquityChart (for pin comparison) -------------------------

function MultiEquityChart({ runs }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !LWC || !runs || runs.length === 0) return;
    const chart = LWC.createChart(containerRef.current, {
      autoSize: true,
      layout: {
        background: { type: 'solid', color: '#0e1014' },
        textColor: '#9aa3b2',
        fontFamily: 'JetBrains Mono, ui-monospace, monospace',
      },
      grid: { vertLines: { color: '#181c24' }, horzLines: { color: '#181c24' } },
      rightPriceScale: { borderColor: '#222732' },
      timeScale: { borderColor: '#222732', timeVisible: false, secondsVisible: false },
      crosshair: { mode: 1 },
    });

    runs.forEach((run, idx) => {
      const color = PIN_COLORS[idx % PIN_COLORS.length];
      const series = chart.addSeries(LWC.LineSeries, {
        color,
        lineWidth: 2,
        priceFormat: { type: 'price', precision: 0, minMove: 1 },
      });
      const data = [];
      let lastTime = -1;
      for (let i = 0; i < run.timestamps.length; i++) {
        const v = run.equity[i];
        if (v === null || v === undefined || Number.isNaN(v)) continue;
        const t = isoToUnixSeconds(run.timestamps[i]);
        if (t <= lastTime) continue;
        lastTime = t;
        data.push({ time: t, value: Number(v) });
      }
      series.setData(data);
    });

    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [runs]);

  return html`<div ref=${containerRef} class="w-full h-full"></div>`;
}

// ---- MonthlyHeatmap ------------------------------------------------

function MonthlyHeatmap({ timestamps, returns }) {
  // Group per-bar returns into months (calendar year × month) and compound them.
  const monthly = useMemo(() => {
    if (!timestamps || !returns) return null;
    const buckets = new Map(); // key 'YYYY-M' -> {year, month, factor}
    for (let i = 0; i < timestamps.length; i++) {
      const r = returns[i];
      if (r === null || r === undefined || Number.isNaN(r)) continue;
      const d = new Date(timestamps[i]);
      const key = `${d.getUTCFullYear()}-${d.getUTCMonth()}`;
      let b = buckets.get(key);
      if (!b) {
        b = { year: d.getUTCFullYear(), month: d.getUTCMonth(), factor: 1.0 };
        buckets.set(key, b);
      }
      b.factor *= 1 + Number(r);
    }
    if (buckets.size === 0) return null;
    const years = [...new Set([...buckets.values()].map(b => b.year))].sort();
    const cells = {}; // year -> [12 values, may be null]
    for (const y of years) cells[y] = Array(12).fill(null);
    for (const b of buckets.values()) cells[b.year][b.month] = b.factor - 1;
    const annual = {}; // year -> compounded total
    for (const y of years) {
      let f = 1.0;
      for (const v of cells[y]) if (v !== null) f *= 1 + v;
      annual[y] = f - 1;
    }
    return { years, cells, annual };
  }, [timestamps, returns]);

  if (!monthly) return null;

  // Find max absolute monthly return for the color scale.
  let absMax = 0;
  for (const y of monthly.years) {
    for (const v of monthly.cells[y]) {
      if (v !== null && Math.abs(v) > absMax) absMax = Math.abs(v);
    }
  }
  if (absMax === 0) absMax = 0.01;

  function cellColor(v) {
    if (v === null || Number.isNaN(v)) return '#181c24';
    const t = Math.min(Math.abs(v) / absMax, 1);
    const hue = v >= 0 ? 140 : 0; // green or red
    const sat = 60;
    const lum = 18 + t * 22; // 18% → 40%
    return `hsl(${hue}, ${sat}%, ${lum}%)`;
  }

  const gridCols = `64px ${'1fr '.repeat(12)}80px`;

  return html`
    <div>
      <div class="text-xs uppercase tracking-wider text-mute-400 mb-2">Monthly returns</div>
      <div class="bg-ink-900 border border-ink-700 rounded p-3">
        <div style=${{ display: 'grid', gridTemplateColumns: gridCols, gap: '2px' }}>
          <div></div>
          ${MONTH_LABELS.map(m => html`
            <div key=${m} class="text-center text-[10px] num text-mute-400 pb-1">${m}</div>
          `)}
          <div class="text-center text-[10px] num text-mute-300 pb-1">Year</div>

          ${monthly.years.map(y => html`
            <div key=${y + '-y'} class="flex items-center justify-end pr-2 text-xs num text-mute-300">${y}</div>
            ${monthly.cells[y].map((v, mi) => html`
              <div key=${y + '-' + mi}
                style=${{ background: cellColor(v), borderRadius: '3px', padding: '4px 2px', textAlign: 'center' }}
                class="text-[10px] num text-white">
                ${v === null ? '' : (v * 100).toFixed(1)}
              </div>
            `)}
            <div key=${y + '-tot'}
              style=${{ background: cellColor(monthly.annual[y]), borderRadius: '3px', padding: '4px 4px', textAlign: 'center' }}
              class="text-xs num text-white font-semibold">
              ${(monthly.annual[y] * 100).toFixed(1)}%
            </div>
          `)}
        </div>
      </div>
    </div>
  `;
}

// ---- TradeList (sortable client-side) ------------------------------

function TradeList({ trades, finalEquity, startingCapital }) {
  const [sortKey, setSortKey] = useState('entry_time');
  const [sortAsc, setSortAsc] = useState(true);
  const [showAll, setShowAll] = useState(false);

  if (!trades || trades.length === 0) {
    return html`
      <div class="bg-ink-900 border border-ink-700 rounded p-4 text-sm text-mute-400">
        No trades recorded.
      </div>
    `;
  }

  const sorted = useMemo(() => {
    const arr = [...trades];
    arr.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      // null / undefined sink to the bottom regardless of direction.
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === 'number') return sortAsc ? av - bv : bv - av;
      if (av < bv) return sortAsc ? -1 : 1;
      if (av > bv) return sortAsc ? 1 : -1;
      return 0;
    });
    return arr;
  }, [trades, sortKey, sortAsc]);

  const visible = showAll ? sorted : sorted.slice(0, 50);
  const baseCap = startingCapital || 100000;
  const hasExitReason = trades.some(t => t.exit_reason != null);

  function exitReasonCell(reason) {
    if (reason == null) return '—';
    const cls = {
      stop_loss: 'text-danger-500',
      take_profit: 'text-accent-400',
      trailing_stop: 'text-yellow-400',
      signal: 'text-mute-300',
    }[reason] || 'text-mute-300';
    return html`<span class="${cls} text-[10px] uppercase tracking-wide">${reason.replace('_', ' ')}</span>`;
  }

  function header(key, label, align = 'left') {
    const active = sortKey === key;
    const arrow = active ? (sortAsc ? '▲' : '▼') : '';
    const click = () => {
      if (sortKey === key) setSortAsc(!sortAsc);
      else { setSortKey(key); setSortAsc(true); }
    };
    return html`
      <th class="py-2 px-3 text-mute-400 text-xs font-medium text-${align} cursor-pointer hover:text-mute-100 select-none"
        onClick=${click}>
        ${label} ${active && html`<span class="text-accent-400 ml-1">${arrow}</span>`}
      </th>
    `;
  }

  return html`
    <div class="bg-ink-900 border border-ink-700 rounded p-4">
      <div class="flex items-baseline justify-between mb-2">
        <div class="text-xs uppercase tracking-wider text-mute-400">Trades</div>
        <div class="text-xs text-mute-400 num">
          ${trades.length} total · showing ${visible.length}
        </div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-ink-700">
              ${header('entry_time', 'Entry')}
              ${header('exit_time', 'Exit')}
              ${header('side', 'Side')}
              ${header('bars_held', 'Bars', 'right')}
              ${header('entry_price', 'Entry $', 'right')}
              ${header('exit_price', 'Exit $', 'right')}
              ${header('pnl_return', 'Return', 'right')}
              ${header('pnl_dollars', 'P&L', 'right')}
              ${hasExitReason && header('exit_reason', 'Why exit')}
            </tr>
          </thead>
          <tbody>
            ${visible.map((t, i) => {
              const ret = t.pnl_return;
              const pnlDollars = t.pnl_dollars !== undefined ? t.pnl_dollars : (ret != null ? baseCap * ret : null);
              const profitable = ret > 0;
              return html`
                <tr key=${i} class="border-t border-ink-800 hover:bg-ink-850 transition">
                  <td class="py-1.5 px-3 num text-mute-200">${t.entry_time?.slice(0,10) ?? '—'}</td>
                  <td class="py-1.5 px-3 num text-mute-200">${t.exit_time?.slice(0,10) ?? '—'}</td>
                  <td class="py-1.5 px-3 ${t.side === 'long' ? 'text-accent-400' : 'text-blue-400'}">${t.side}</td>
                  <td class="py-1.5 px-3 text-right num text-mute-200">${fmtInt(t.bars_held)}</td>
                  <td class="py-1.5 px-3 text-right num text-mute-300">${fmtNum(t.entry_price, 2)}</td>
                  <td class="py-1.5 px-3 text-right num text-mute-300">${fmtNum(t.exit_price, 2)}</td>
                  <td class="py-1.5 px-3 text-right num ${profitable ? 'text-accent-400' : 'text-danger-500'}">${fmtPct(ret, 2)}</td>
                  <td class="py-1.5 px-3 text-right num ${profitable ? 'text-accent-400' : 'text-danger-500'}">${pnlDollars != null ? fmtMoney(pnlDollars) : '—'}</td>
                  ${hasExitReason && html`<td class="py-1.5 px-3">${exitReasonCell(t.exit_reason)}</td>`}
                </tr>
              `;
            })}
          </tbody>
        </table>
      </div>
      ${trades.length > 50 && html`
        <div class="mt-3 text-center">
          <button onClick=${() => setShowAll(s => !s)}
            class="text-xs text-mute-300 hover:text-mute-100 px-3 py-1 rounded bg-ink-800 hover:bg-ink-700 transition">
            ${showAll ? `Show first 50` : `Show all ${trades.length}`}
          </button>
        </div>
      `}
    </div>
  `;
}

// ---- PinStrip + ComparisonView -------------------------------------

function PinStrip({ pins, onUnpin, onCompare, comparing }) {
  if (pins.length === 0) return null;
  return html`
    <div class="bg-ink-900 border border-ink-700 rounded p-3 flex items-center gap-3 flex-wrap">
      <div class="text-[10px] uppercase tracking-wider text-mute-400">Pinned</div>
      ${pins.map((p, i) => html`
        <div key=${p.pinId} class="flex items-center gap-2 bg-ink-850 border border-ink-700 rounded px-2 py-1">
          <span class="inline-block w-2 h-2 rounded-full" style=${{ background: PIN_COLORS[i % PIN_COLORS.length] }}></span>
          <span class="text-xs text-mute-200">${p.label}</span>
          <button onClick=${() => onUnpin(p.pinId)}
            class="text-xs text-mute-400 hover:text-danger-500 transition" title="Unpin">×</button>
        </div>
      `)}
      ${pins.length >= 2 && html`
        <button onClick=${onCompare}
          class="ml-auto text-xs px-3 py-1 rounded font-medium transition
                 ${comparing ? 'bg-accent-500 text-ink-950' : 'bg-ink-800 hover:bg-ink-700 text-mute-100'}">
          ${comparing ? '✓ Comparing' : 'Compare'}
        </button>
      `}
    </div>
  `;
}

function ComparisonView({ pins, onClose }) {
  // Pick a stable set of metric keys to show across runs.
  const metricKeys = ['cagr', 'sharpe', 'max_drawdown', 'volatility', 'sortino', 'calmar', 'n_trades', 'win_rate'];

  return html`
    <div class="p-6 space-y-5">
      <div class="flex items-baseline justify-between">
        <div>
          <div class="text-xs uppercase tracking-wider text-mute-400">Comparison</div>
          <div class="text-lg text-mute-100 font-medium mt-0.5">${pins.length} pinned runs</div>
        </div>
        <button onClick=${onClose}
          class="text-sm text-mute-300 hover:text-mute-100 transition px-2 py-1">
          ← Back to run
        </button>
      </div>

      <div class="h-[420px] bg-ink-900 border border-ink-700 rounded overflow-hidden">
        <${MultiEquityChart} runs=${pins} />
      </div>

      <div class="bg-ink-900 border border-ink-700 rounded p-4 overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-ink-700">
              <th class="text-left py-2 px-3 text-mute-400 text-xs font-medium">Metric</th>
              ${pins.map((p, i) => html`
                <th key=${p.pinId} class="text-right py-2 px-3 text-xs font-medium">
                  <span class="inline-block w-2 h-2 rounded-full mr-2" style=${{ background: PIN_COLORS[i % PIN_COLORS.length] }}></span>
                  <span class="text-mute-100">${p.label}</span>
                </th>
              `)}
            </tr>
          </thead>
          <tbody>
            ${metricKeys.map(k => html`
              <tr key=${k} class="border-t border-ink-800">
                <td class="py-1.5 px-3 text-mute-300">${k.replaceAll('_',' ')}</td>
                ${pins.map(p => html`
                  <td key=${p.pinId} class="py-1.5 px-3 text-right num text-mute-100">
                    ${fmtMetricVal(k, p.metrics?.[k])}
                  </td>
                `)}
              </tr>
            `)}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

// ---- CodeEditor (CodeMirror 6 wrapper) -----------------------------

function CodeEditor({ initialValue, editorRef }) {
  const containerRef = useRef(null);
  const viewRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const view = new EditorView({
      doc: initialValue ?? '',
      extensions: [basicSetup, python(), oneDark],
      parent: containerRef.current,
    });
    viewRef.current = view;
    if (editorRef) {
      editorRef.current = {
        getValue: () => view.state.doc.toString(),
        setValue: (val) => {
          view.dispatch({
            changes: { from: 0, to: view.state.doc.length, insert: val ?? '' },
          });
        },
      };
    }
    return () => {
      view.destroy();
      viewRef.current = null;
      if (editorRef) editorRef.current = null;
    };
  }, []);

  return html`<div ref=${containerRef} class="w-full h-full overflow-auto bg-[#282c34]"></div>`;
}

// ---- AuthorView ----------------------------------------------------

function ValidationPanel({ result }) {
  if (!result) return null;
  if (result.ok && result.warnings.length === 0) {
    return html`
      <div class="bg-accent-500/10 border border-accent-500/30 rounded p-3 text-sm text-accent-400">
        ✓ Validation passed — function "${result.strategy_name ?? 'strategy'}" is safe to save.
      </div>
    `;
  }
  if (result.ok && result.warnings.length > 0) {
    return html`
      <div class="bg-yellow-500/10 border border-yellow-500/30 rounded p-3 text-sm text-yellow-400">
        <div class="font-semibold">✓ Valid, with warnings:</div>
        <ul class="mt-1 list-disc list-inside text-xs space-y-0.5">
          ${result.warnings.map((w, i) => html`<li key=${i}>${w}</li>`)}
        </ul>
      </div>
    `;
  }
  return html`
    <div class="bg-danger-500/10 border border-danger-500/30 rounded p-3 text-sm text-danger-500">
      <div class="font-semibold">✗ Validation failed</div>
      <ul class="mt-1 list-disc list-inside text-xs space-y-0.5">
        ${result.errors.map((e, i) => html`<li key=${i}>${e}</li>`)}
      </ul>
    </div>
  `;
}

function AuthorView({ onCancel, onSaved }) {
  const pushToast = useToast();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [englishPrompt, setEnglishPrompt] = useState('');
  const [generating, setGenerating] = useState(false);
  const [validating, setValidating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [validationResult, setValidationResult] = useState(null);
  const [authorError, setAuthorError] = useState(null);

  const editorRef = useRef(null);

  const handleGenerate = async () => {
    if (!englishPrompt.trim()) return;
    setGenerating(true);
    setAuthorError(null);
    setValidationResult(null);
    try {
      const resp = await translateStrategy(englishPrompt.trim());
      editorRef.current?.setValue(resp.code);
      setValidationResult(resp.validation_result);
    } catch (err) {
      setAuthorError(err.message);
      pushToast('error', `Translate failed: ${err.message}`);
    } finally {
      setGenerating(false);
    }
  };

  const handleValidate = async () => {
    const code = editorRef.current?.getValue() ?? '';
    if (!code.trim()) {
      setAuthorError('Editor is empty — write or generate some code first.');
      return;
    }
    setValidating(true);
    setAuthorError(null);
    try {
      const result = await validateStrategy(code);
      setValidationResult(result);
    } catch (err) {
      setAuthorError(err.message);
      pushToast('error', `Validate failed: ${err.message}`);
    } finally {
      setValidating(false);
    }
  };

  const handleSave = async () => {
    const code = editorRef.current?.getValue() ?? '';
    if (!name.trim()) {
      setAuthorError('Name is required.');
      return;
    }
    if (!code.trim()) {
      setAuthorError('Editor is empty.');
      return;
    }
    setSaving(true);
    setAuthorError(null);
    try {
      // Pre-validate so we don't waste a save attempt on bad code.
      const vresult = await validateStrategy(code);
      setValidationResult(vresult);
      if (!vresult.ok) {
        setAuthorError('Code did not pass validation — fix the errors above and try again.');
        setSaving(false);
        return;
      }
      const created = await createStrategy(name.trim(), description.trim(), code);
      pushToast('success', `Saved strategy "${name.trim()}".`);
      onSaved(created.strategy_id);
    } catch (err) {
      setAuthorError(err.message);
      pushToast('error', `Save failed: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const busy = generating || validating || saving;

  return html`
    <div class="p-6 max-w-5xl mx-auto space-y-5">
      <div class="flex items-baseline justify-between">
        <div>
          <div class="text-xs uppercase tracking-wider text-mute-400">New strategy</div>
          <div class="text-lg text-mute-100 font-medium mt-0.5">Author a strategy</div>
        </div>
        <button onClick=${onCancel} disabled=${busy}
          class="text-sm text-mute-300 hover:text-mute-100 transition px-2 py-1 ${busy ? 'opacity-50 cursor-not-allowed' : ''}">
          ← Cancel
        </button>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <div class="text-[10px] uppercase tracking-wider text-mute-400 mb-1">Name</div>
          <input class=${inputCls()} placeholder="e.g. RSI Mean Reversion"
            value=${name} onChange=${e => setName(e.target.value)} />
        </div>
        <div>
          <div class="text-[10px] uppercase tracking-wider text-mute-400 mb-1">Description (optional)</div>
          <input class=${inputCls()} placeholder="brief note for the strategy library"
            value=${description} onChange=${e => setDescription(e.target.value)} />
        </div>
      </div>

      <div class="bg-ink-900 border border-ink-700 rounded p-4">
        <div class="text-xs uppercase tracking-wider text-mute-400 mb-2">
          Plain English (optional) — Generate fills the editor below
        </div>
        <textarea rows="3"
          class="w-full bg-ink-850 border border-ink-700 rounded px-3 py-2 text-sm text-mute-100 focus:outline-none focus:border-accent-500 resize-none"
          placeholder='Example: "Buy when RSI(14) crosses above 30, sell when above 70."'
          value=${englishPrompt}
          onChange=${e => setEnglishPrompt(e.target.value)} />
        <div class="mt-2 flex justify-end">
          <button onClick=${handleGenerate}
            disabled=${!englishPrompt.trim() || busy}
            class="px-3 py-1.5 rounded text-xs font-medium transition
                   ${!englishPrompt.trim() || busy
                     ? 'bg-ink-700 text-mute-400 cursor-not-allowed'
                     : 'bg-accent-600 hover:bg-accent-500 text-white'}">
            ${generating ? 'Generating…' : 'Generate code'}
          </button>
        </div>
      </div>

      <div>
        <div class="text-xs uppercase tracking-wider text-mute-400 mb-2">Strategy code (Python)</div>
        <div class="h-[420px] border border-ink-700 rounded overflow-hidden">
          <${CodeEditor} initialValue=${STRATEGY_TEMPLATE} editorRef=${editorRef} />
        </div>
      </div>

      ${validationResult && html`<${ValidationPanel} result=${validationResult} />`}

      ${authorError && html`
        <div class="bg-danger-500/10 border border-danger-500/30 rounded p-3 text-sm text-danger-500">
          ${authorError}
        </div>
      `}

      <div class="flex gap-3 justify-end">
        <button onClick=${handleValidate} disabled=${busy}
          class="px-4 py-2 rounded text-sm font-medium transition
                 ${busy ? 'bg-ink-700 text-mute-400 cursor-not-allowed' : 'bg-ink-800 hover:bg-ink-700 text-mute-100'}">
          ${validating ? 'Validating…' : 'Validate'}
        </button>
        <button onClick=${handleSave} disabled=${busy || !name.trim()}
          class="px-4 py-2 rounded text-sm font-semibold transition
                 ${busy || !name.trim() ? 'bg-ink-700 text-mute-400 cursor-not-allowed' : 'bg-accent-500 hover:bg-accent-400 text-ink-950'}">
          ${saving ? 'Saving…' : 'Save strategy'}
        </button>
      </div>
    </div>
  `;
}

// ---- LibraryView (Phase 8) -----------------------------------------

function LibraryView({ onOpenRun, onClose }) {
  const pushToast = useToast();
  const [runs, setRuns] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [weights, setWeights] = useState(DEFAULT_LIB_WEIGHTS);
  const [sortKey, setSortKey] = useState('score');
  const [sortAsc, setSortAsc] = useState(false);
  const [tickerFilter, setTickerFilter] = useState('');

  useEffect(() => {
    let live = true;
    getLibrary()
      .then(body => { if (live) setRuns(body.runs); })
      .catch(err => {
        if (live) {
          setLoadError(err.message);
          pushToast('error', `Could not load library: ${err.message}`);
        }
      });
    return () => { live = false; };
  }, []);

  const scored = useMemo(() => {
    if (!runs) return null;
    return runs.map(r => ({ ...r, score: compositeScore(r, weights) }));
  }, [runs, weights]);

  const filtered = useMemo(() => {
    if (!scored) return null;
    const q = tickerFilter.trim().toUpperCase();
    if (!q) return scored;
    return scored.filter(r => r.ticker.toUpperCase().includes(q));
  }, [scored, tickerFilter]);

  const sorted = useMemo(() => {
    if (!filtered) return null;
    const arr = [...filtered];
    arr.sort((a, b) => {
      // Favourites pin to the top regardless of the active sort column.
      if (!!a.favourite !== !!b.favourite) return a.favourite ? -1 : 1;
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === 'number') return sortAsc ? av - bv : bv - av;
      if (av < bv) return sortAsc ? -1 : 1;
      if (av > bv) return sortAsc ? 1 : -1;
      return 0;
    });
    return arr;
  }, [filtered, sortKey, sortAsc]);

  function header(key, label, align = 'left') {
    const active = sortKey === key;
    const arrow = active ? (sortAsc ? '▲' : '▼') : '';
    const click = () => {
      if (sortKey === key) setSortAsc(!sortAsc);
      else { setSortKey(key); setSortAsc(key !== 'created_at' && key !== 'strategy_name'); }
    };
    return html`
      <th class="py-2 px-3 text-mute-400 text-xs font-medium text-${align} cursor-pointer hover:text-mute-100 select-none"
        onClick=${click}>
        ${label} ${active && html`<span class="text-accent-400 ml-1">${arrow}</span>`}
      </th>
    `;
  }

  function modeBadge(mode) {
    const cls = {
      single: 'bg-ink-700 text-mute-200',
      sweep: 'bg-blue-600/30 text-blue-300',
      walk_forward: 'bg-yellow-600/30 text-yellow-300',
      event_driven: 'bg-purple-600/30 text-purple-300',
    }[mode] || 'bg-ink-700 text-mute-200';
    const label =
      mode === 'walk_forward' ? 'WF' :
      mode === 'sweep' ? 'sweep' :
      mode === 'event_driven' ? 'stops' :
      'single';
    return html`<span class="ml-2 text-[10px] uppercase px-1.5 py-0.5 rounded ${cls}">${label}</span>`;
  }

  return html`
    <div class="p-6 space-y-5">
      <div class="flex items-baseline justify-between">
        <div>
          <div class="text-xs uppercase tracking-wider text-mute-400">Strategy library</div>
          <div class="text-lg text-mute-100 font-medium mt-0.5">
            ${runs ? `${runs.length} stored runs` : 'loading…'}
          </div>
        </div>
        <button onClick=${onClose}
          class="text-sm text-mute-300 hover:text-mute-100 transition px-2 py-1">
          ← Back to run
        </button>
      </div>

      <!-- Weight editor -->
      <div class="bg-ink-900 border border-ink-700 rounded p-4">
        <div class="flex items-baseline justify-between mb-3">
          <div class="text-xs uppercase tracking-wider text-mute-400">Composite weights</div>
          <button onClick=${() => setWeights(DEFAULT_LIB_WEIGHTS)}
            class="text-[10px] text-mute-400 hover:text-mute-100 transition">
            reset to defaults
          </button>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          ${Object.entries(weights).map(([key, val]) => html`
            <div key=${key}>
              <div class="flex items-baseline justify-between text-xs">
                <span class="text-mute-300 capitalize">${key}</span>
                <span class="num text-mute-100 font-semibold">${val}%</span>
              </div>
              <input type="range" min="0" max="100" step="1" value=${val}
                onChange=${e => setWeights(w => ({ ...w, [key]: Number(e.target.value) }))}
                class="w-full accent-accent-500 mt-1" />
            </div>
          `)}
        </div>
        <div class="mt-3 text-[10px] text-mute-400">
          Weights are relative — they don't need to sum to 100. The DSR component is null until DSR is persisted per run; its weight is redistributed pro rata to available components.
        </div>
      </div>

      <!-- Filter -->
      <div class="flex items-center gap-3">
        <div class="text-xs text-mute-400">Filter by ticker</div>
        <input type="text" placeholder="e.g. SPY"
          class="bg-ink-850 border border-ink-700 rounded px-2 py-1 text-sm num text-mute-100 w-32 focus:outline-none focus:border-accent-500"
          value=${tickerFilter}
          onChange=${e => setTickerFilter(e.target.value)} />
        ${tickerFilter && html`
          <button onClick=${() => setTickerFilter('')}
            class="text-xs text-mute-400 hover:text-mute-100">clear</button>
        `}
      </div>

      ${loadError && html`
        <div class="bg-danger-500/10 border border-danger-500/30 rounded p-3 text-sm text-danger-500">
          Failed to load library: ${loadError}
        </div>
      `}

      ${sorted && sorted.length === 0 && html`
        <div class="bg-ink-900 border border-ink-700 rounded p-6 text-center text-mute-400 text-sm">
          ${runs.length === 0
            ? 'No runs yet. Run some backtests or sweeps to populate the library.'
            : 'No runs match the current filter.'}
        </div>
      `}

      ${sorted && sorted.length > 0 && html`
        <div class="bg-ink-900 border border-ink-700 rounded overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-ink-700">
                ${header('strategy_name', 'Strategy')}
                ${header('ticker', 'Ticker')}
                ${header('timeframe', 'TF')}
                <th class="text-left py-2 px-3 text-mute-400 text-xs font-medium">Period</th>
                ${header('created_at', 'Run at')}
                <th class="text-right py-2 px-3 text-mute-400 text-xs font-medium">CAGR</th>
                <th class="text-right py-2 px-3 text-mute-400 text-xs font-medium">Sharpe</th>
                <th class="text-right py-2 px-3 text-mute-400 text-xs font-medium">Calmar</th>
                <th class="text-right py-2 px-3 text-mute-400 text-xs font-medium">Max DD</th>
                ${header('score', '★ Score', 'right')}
              </tr>
            </thead>
            <tbody>
              ${sorted.map(r => html`
                <tr key=${r.run_id}
                  onClick=${() => onOpenRun(r.run_id)}
                  class="border-t border-ink-800 hover:bg-ink-850 transition cursor-pointer">
                  <td class="py-2 px-3 text-mute-100">
                    ${r.favourite && html`<span class="text-yellow-400 mr-1" title="Favourite">★</span>`}
                    ${r.strategy_name}${modeBadge(r.mode)}</td>
                  <td class="py-2 px-3 num text-mute-200">${r.ticker}</td>
                  <td class="py-2 px-3 num text-mute-400">${r.timeframe}</td>
                  <td class="py-2 px-3 num text-mute-400 text-xs">${r.start_date.slice(0,10)} → ${r.end_date.slice(0,10)}</td>
                  <td class="py-2 px-3 num text-mute-400 text-xs">${r.created_at?.slice(0,10) ?? '—'}</td>
                  <td class="py-2 px-3 text-right num ${(r.metrics?.cagr ?? 0) > 0 ? 'text-accent-400' : 'text-danger-500'}">${fmtPct(r.metrics?.cagr)}</td>
                  <td class="py-2 px-3 text-right num text-mute-100">${fmtNum(r.metrics?.sharpe, 2)}</td>
                  <td class="py-2 px-3 text-right num text-mute-200">${fmtNum(r.metrics?.calmar, 2)}</td>
                  <td class="py-2 px-3 text-right num text-danger-500">${fmtPct(r.metrics?.max_drawdown)}</td>
                  <td class="py-2 px-3 text-right num text-accent-400 font-semibold">${(r.score * 100).toFixed(1)}</td>
                </tr>
              `)}
            </tbody>
          </table>
        </div>
      `}
    </div>
  `;
}

// ---- LiveView (Phase 10: T212 Demo read-only) ----------------------

function fmtAcctNum(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return Number(x).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtT212Date(s) {
  if (!s || typeof s !== 'string') return '—';
  return s.slice(0, 10);
}

// Pull a clean error string out of a fetch error. The /t212/* routes
// occasionally send structured detail (e.g. rate-limited with
// retry_after_seconds) — surface that when present.
function t212ErrorString(err) {
  const detail = err?.body?.detail;
  if (detail && typeof detail === 'object') {
    const base = detail.message || JSON.stringify(detail);
    return detail.retry_after_seconds
      ? `${base} (retry after ${detail.retry_after_seconds}s)`
      : base;
  }
  return err?.message || String(err);
}

function LiveTabBar({ tab, setTab, refresh, loading }) {
  const tabs = [
    { key: 'account', label: 'Account' },
    { key: 'positions', label: 'Positions' },
    { key: 'orders', label: 'Orders' },
    { key: 'pies', label: 'Pies' },
  ];
  return html`
    <div class="flex items-center gap-1 border-b border-ink-700">
      ${tabs.map(t => html`
        <button key=${t.key} onClick=${() => setTab(t.key)}
          class="px-4 py-2 text-sm transition border-b-2 -mb-px
                 ${tab === t.key
                   ? 'border-accent-500 text-mute-100 font-medium'
                   : 'border-transparent text-mute-400 hover:text-mute-100'}">
          ${t.label}
        </button>
      `)}
      <div class="ml-auto pb-1.5">
        <button onClick=${() => refresh(tab)} disabled=${loading[tab]}
          class="text-xs px-3 py-1 rounded bg-ink-800 hover:bg-ink-700 text-mute-200
                 disabled:opacity-50 transition">
          ${loading[tab] ? 'refreshing…' : '↻ refresh'}
        </button>
      </div>
    </div>
  `;
}

function LiveLoading({ label }) {
  return html`
    <div class="py-12 text-center text-mute-400 text-sm">Loading ${label}…</div>
  `;
}

function LiveError({ message, onRetry }) {
  return html`
    <div class="py-12 flex flex-col items-center gap-3">
      <div class="text-danger-500 text-sm max-w-md text-center">${message}</div>
      <button onClick=${onRetry}
        class="text-xs px-3 py-1.5 rounded bg-ink-800 hover:bg-ink-700 text-mute-100 transition">
        Try again
      </button>
    </div>
  `;
}

function LiveEmpty({ message }) {
  return html`
    <div class="py-12 text-center text-mute-400 text-sm">${message}</div>
  `;
}

function LiveAccountPanel({ cash }) {
  // T212 cash dict: free, total, ppl (open P&L), result (realized),
  // invested, pieCash, blocked. All in account currency.
  const cards = [
    { label: 'Total',     value: cash.total,     emphasis: true },
    { label: 'Free',      value: cash.free },
    { label: 'Invested',  value: cash.invested },
    { label: 'Open P&L',  value: cash.ppl,       tone: 'pnl' },
    { label: 'Result',    value: cash.result,    tone: 'pnl' },
    { label: 'Pie cash',  value: cash.pieCash },
    { label: 'Blocked',   value: cash.blocked },
  ];
  return html`
    <div class="space-y-4">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        ${cards.map(c => {
          const tone = c.tone === 'pnl'
            ? (c.value > 0 ? 'text-accent-400' : c.value < 0 ? 'text-danger-500' : 'text-mute-100')
            : 'text-mute-100';
          const size = c.emphasis ? 'text-2xl' : 'text-xl';
          return html`
            <div class="bg-ink-900 border border-ink-700 rounded p-3">
              <div class="text-[11px] uppercase tracking-wider text-mute-400 mb-1">${c.label}</div>
              <div class="num ${size} ${tone} font-medium">${fmtAcctNum(c.value)}</div>
            </div>
          `;
        })}
      </div>
      <div class="text-xs text-mute-400">
        Values in account currency. This is a Trading 212 <span class="text-accent-400 font-medium">Demo / Practice</span> account — no real money.
      </div>
    </div>
  `;
}

function LivePositionsPanel({ positions }) {
  if (!positions || positions.length === 0) {
    return html`<${LiveEmpty} message="No open positions yet. Open one in the Trading 212 Demo app and refresh to see it here." />`;
  }
  return html`
    <div class="overflow-x-auto">
      <table class="min-w-full text-sm">
        <thead>
          <tr class="border-b border-ink-700 text-mute-400 text-xs">
            <th class="py-2 px-3 text-left font-medium">Ticker</th>
            <th class="py-2 px-3 text-right font-medium">Quantity</th>
            <th class="py-2 px-3 text-right font-medium">Avg price</th>
            <th class="py-2 px-3 text-right font-medium">Current</th>
            <th class="py-2 px-3 text-right font-medium">Open P&L</th>
            <th class="py-2 px-3 text-right font-medium">FX P&L</th>
            <th class="py-2 px-3 text-left font-medium">Initial fill</th>
          </tr>
        </thead>
        <tbody>
          ${positions.map((p, i) => {
            const ppl = p.ppl ?? 0;
            const pplTone = ppl > 0 ? 'text-accent-400' : ppl < 0 ? 'text-danger-500' : 'text-mute-100';
            return html`
              <tr key=${p.ticker || i} class="border-b border-ink-800 hover:bg-ink-900/60">
                <td class="py-2 px-3 text-mute-100 font-medium">${p.ticker || '—'}</td>
                <td class="py-2 px-3 text-right num text-mute-200">${fmtAcctNum(p.quantity, 4)}</td>
                <td class="py-2 px-3 text-right num text-mute-200">${fmtAcctNum(p.averagePrice)}</td>
                <td class="py-2 px-3 text-right num text-mute-100">${fmtAcctNum(p.currentPrice)}</td>
                <td class="py-2 px-3 text-right num ${pplTone}">${fmtAcctNum(p.ppl)}</td>
                <td class="py-2 px-3 text-right num text-mute-300">${fmtAcctNum(p.fxPpl)}</td>
                <td class="py-2 px-3 text-mute-300 num text-xs">${fmtT212Date(p.initialFillDate)}</td>
              </tr>
            `;
          })}
        </tbody>
      </table>
    </div>
  `;
}

function LiveOrdersPanel({ orders }) {
  if (!orders || orders.length === 0) {
    return html`<${LiveEmpty} message="No working orders. Place a limit/stop order in the Trading 212 Demo app and refresh to see it here." />`;
  }
  return html`
    <div class="overflow-x-auto">
      <table class="min-w-full text-sm">
        <thead>
          <tr class="border-b border-ink-700 text-mute-400 text-xs">
            <th class="py-2 px-3 text-left font-medium">Order ID</th>
            <th class="py-2 px-3 text-left font-medium">Ticker</th>
            <th class="py-2 px-3 text-left font-medium">Type</th>
            <th class="py-2 px-3 text-right font-medium">Quantity</th>
            <th class="py-2 px-3 text-right font-medium">Limit / stop</th>
            <th class="py-2 px-3 text-left font-medium">Status</th>
          </tr>
        </thead>
        <tbody>
          ${orders.map((o, i) => {
            const price = o.limitPrice ?? o.stopPrice ?? null;
            return html`
              <tr key=${o.id || i} class="border-b border-ink-800 hover:bg-ink-900/60">
                <td class="py-2 px-3 text-mute-300 num text-xs">${o.id || '—'}</td>
                <td class="py-2 px-3 text-mute-100 font-medium">${o.ticker || '—'}</td>
                <td class="py-2 px-3 text-mute-200 text-xs uppercase">${o.type || '—'}</td>
                <td class="py-2 px-3 text-right num text-mute-200">${fmtAcctNum(o.quantity, 4)}</td>
                <td class="py-2 px-3 text-right num text-mute-100">${price !== null ? fmtAcctNum(price) : '—'}</td>
                <td class="py-2 px-3 text-mute-300 text-xs uppercase">${o.status || '—'}</td>
              </tr>
            `;
          })}
        </tbody>
      </table>
    </div>
  `;
}

function LivePiesPanel({ pies }) {
  if (!pies || pies.length === 0) {
    return html`<${LiveEmpty} message="No pies set up. Create one in the Trading 212 Demo app and refresh to see it here." />`;
  }
  return html`
    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
      ${pies.map((p, i) => html`
        <div key=${p.id || i} class="bg-ink-900 border border-ink-700 rounded p-3">
          <div class="flex items-baseline justify-between mb-2">
            <div class="text-mute-100 font-medium">${p.name || `Pie #${p.id || i}`}</div>
            <div class="text-mute-400 text-xs num">id ${p.id ?? '—'}</div>
          </div>
          <div class="text-xs text-mute-400">Pie cash</div>
          <div class="num text-lg text-mute-100">${fmtAcctNum(p.cash)}</div>
        </div>
      `)}
    </div>
  `;
}

function LiveView({ onClose }) {
  const pushToast = useToast();
  const [tab, setTab] = useState('account');
  const [data, setData] = useState({});
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState({});

  // Defined inside so each refresh closes over the latest setters; the
  // dependency surface is small enough that this is simpler than useCallback.
  const fetchers = {
    account: getT212Account,
    positions: getT212Positions,
    orders: getT212Orders,
    pies: getT212Pies,
  };

  async function refresh(which) {
    setLoading(prev => ({ ...prev, [which]: true }));
    setErrors(prev => ({ ...prev, [which]: null }));
    try {
      const result = await fetchers[which]();
      setData(prev => ({ ...prev, [which]: result }));
    } catch (err) {
      const msg = t212ErrorString(err);
      setErrors(prev => ({ ...prev, [which]: msg }));
      pushToast('error', `T212 ${which}: ${msg}`);
    } finally {
      setLoading(prev => ({ ...prev, [which]: false }));
    }
  }

  // Lazy-load each tab on first activation. Cached results persist across
  // tab switches; the refresh button re-fetches.
  useEffect(() => {
    if (data[tab] === undefined && errors[tab] == null && !loading[tab]) {
      refresh(tab);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);

  function renderTab() {
    if (loading[tab] && data[tab] === undefined) {
      return html`<${LiveLoading} label=${tab} />`;
    }
    if (errors[tab]) {
      return html`<${LiveError} message=${errors[tab]} onRetry=${() => refresh(tab)} />`;
    }
    if (data[tab] === undefined) {
      return html`<${LiveLoading} label=${tab} />`;
    }
    if (tab === 'account')   return html`<${LiveAccountPanel}   cash=${data.account} />`;
    if (tab === 'positions') return html`<${LivePositionsPanel} positions=${data.positions} />`;
    if (tab === 'orders')    return html`<${LiveOrdersPanel}    orders=${data.orders} />`;
    if (tab === 'pies')      return html`<${LivePiesPanel}      pies=${data.pies} />`;
    return null;
  }

  return html`
    <div class="p-6 space-y-4">
      <div class="flex items-baseline justify-between">
        <div>
          <div class="text-xs uppercase tracking-wider text-mute-400">Trading 212 · Demo</div>
          <div class="text-lg text-mute-100 font-medium mt-0.5 flex items-center gap-2">
            Live account
            <span class="px-2 py-0.5 rounded text-[10px] uppercase tracking-wider bg-accent-500/20 text-accent-400 font-medium">
              Read-only · Demo
            </span>
          </div>
        </div>
        <button onClick=${onClose}
          class="text-sm text-mute-300 hover:text-mute-100 transition px-2 py-1">
          ← Back to run
        </button>
      </div>

      <${LiveTabBar} tab=${tab} setTab=${setTab} refresh=${refresh} loading=${loading} />
      ${renderTab()}
    </div>
  `;
}


// ---- Deploy + Tracking (Phase 11) ----------------------------------

// Modal: enables a live-order action only after the user types ARM.
// Mounted only while active, so each open starts with an empty field.
function ArmModal({ title, body, expect = 'ARM', busy, error, onConfirm, onCancel }) {
  const [token, setToken] = useState('');
  const inputRef = useRef(null);
  useEffect(() => { inputRef.current && inputRef.current.focus(); }, []);
  const armed = token.trim() === expect;
  return html`
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
         onClick=${() => { if (!busy) onCancel(); }}>
      <div class="bg-ink-850 border border-ink-600 rounded-lg w-[460px] max-w-full p-6"
           onClick=${e => e.stopPropagation()}>
        <div class="text-mute-100 font-semibold text-lg">${title}</div>
        <div class="text-sm text-mute-300 mt-2 leading-relaxed">${body}</div>
        <input ref=${inputRef} class="${inputCls()} mt-4 tracking-[0.3em]"
          placeholder=${'Type ' + expect} value=${token}
          onChange=${e => setToken(e.target.value)}
          onKeyDown=${e => { if (e.key === 'Enter' && armed && !busy) onConfirm(); }} />
        ${error && html`<div class="mt-3 text-sm text-danger-500">${error}</div>`}
        <div class="flex justify-end gap-2 mt-5">
          <button onClick=${onCancel} disabled=${busy}
            class="text-sm px-3 py-1.5 rounded bg-ink-800 hover:bg-ink-700 text-mute-200
                   disabled:opacity-50 transition">Cancel</button>
          <button onClick=${() => onConfirm()} disabled=${!armed || busy}
            class="text-sm px-4 py-1.5 rounded font-medium transition
                   ${armed && !busy
                     ? 'bg-danger-500 hover:bg-danger-500/80 text-white'
                     : 'bg-ink-700 text-mute-400 cursor-not-allowed'}">
            ${busy ? 'Working…' : 'Confirm'}
          </button>
        </div>
      </div>
    </div>
  `;
}

function DeployCreateForm({ strategies, busy, onCreate }) {
  const [form, setForm] = useState({
    strategy_id: strategies[0] ? strategies[0].id : '',
    ticker: '',
    timeframe: '1d',
    deployment_capital: '1000',
    order_type: 'market',
    schedule_cron: '5 21 * * 1-5',
    maxPositionValue: '',
    maxLossValue: '',
    paramsJson: '{}',
  });
  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  let parsedParams = {};
  let paramsOk = true;
  try {
    const p = JSON.parse(form.paramsJson || '{}');
    if (p && typeof p === 'object' && !Array.isArray(p)) parsedParams = p;
    else paramsOk = false;
  } catch (e) {
    paramsOk = false;
  }

  const capital = Number(form.deployment_capital);
  // Max position value: empty = no cap; if filled, must be a positive number.
  const maxPosRaw = form.maxPositionValue.trim();
  const maxPos = maxPosRaw === '' ? null : Number(maxPosRaw);
  const maxPosOk = maxPos === null || (Number.isFinite(maxPos) && maxPos > 0);
  // Max loss (circuit breaker): empty = off; if filled, a positive number.
  const maxLossRaw = form.maxLossValue.trim();
  const maxLoss = maxLossRaw === '' ? null : Number(maxLossRaw);
  const maxLossOk = maxLoss === null || (Number.isFinite(maxLoss) && maxLoss > 0);
  const valid =
    !!form.strategy_id && form.ticker.trim() !== '' && form.schedule_cron.trim() !== ''
    && Number.isFinite(capital) && capital > 0 && paramsOk && maxPosOk && maxLossOk;

  function submit() {
    if (!valid) return;
    onCreate({
      strategy_id: form.strategy_id,
      ticker: form.ticker.trim(),
      timeframe: form.timeframe,
      params: parsedParams,
      deployment_capital: capital,
      max_position_value: maxPos,
      max_loss_value: maxLoss,
      order_type: form.order_type,
      schedule_cron: form.schedule_cron.trim(),
    });
  }

  if (strategies.length === 0) {
    return html`
      <div class="bg-ink-900 border border-ink-700 rounded p-4">
        <div class="text-xs uppercase tracking-wider text-mute-400 mb-2">New deployment</div>
        <div class="text-sm text-mute-400">
          Create a strategy first — a deployment runs one of your saved strategies.
        </div>
      </div>
    `;
  }

  return html`
    <div class="bg-ink-900 border border-ink-700 rounded p-4">
      <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">New deployment</div>
      <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
        <${Field} label="Strategy">
          <select class=${inputCls()} value=${form.strategy_id}
            onChange=${e => set('strategy_id', e.target.value)}>
            ${strategies.map(s => html`<option key=${s.id} value=${s.id}>${s.name}</option>`)}
          </select>
        </${Field}>
        <${Field} label="T212 ticker">
          <input class=${inputCls()} placeholder="AAPL_US_EQ" value=${form.ticker}
            onChange=${e => set('ticker', e.target.value)} />
        </${Field}>
        <${Field} label="Timeframe">
          <select class=${inputCls()} value=${form.timeframe}
            onChange=${e => set('timeframe', e.target.value)}>
            <option value="1d">1d</option>
            <option value="1h">1h</option>
            <option value="5m">5m</option>
          </select>
        </${Field}>
        <${Field} label="Capital (account currency)">
          <input class=${inputCls()} type="number" min="1" value=${form.deployment_capital}
            onChange=${e => set('deployment_capital', e.target.value)} />
        </${Field}>
        <${Field} label="Max position (optional)">
          <input class="${inputCls()} ${maxPosOk ? '' : 'border-danger-500'}"
            type="number" min="0" placeholder="no cap" value=${form.maxPositionValue}
            onChange=${e => set('maxPositionValue', e.target.value)} />
        </${Field}>
        <${Field} label="Max loss — circuit breaker (optional)">
          <input class="${inputCls()} ${maxLossOk ? '' : 'border-danger-500'}"
            type="number" min="0" placeholder="off" value=${form.maxLossValue}
            onChange=${e => set('maxLossValue', e.target.value)} />
        </${Field}>
        <${Field} label="Order type">
          <select class=${inputCls()} value=${form.order_type}
            onChange=${e => set('order_type', e.target.value)}>
            <option value="market">market</option>
            <option value="limit">limit</option>
          </select>
        </${Field}>
        <${Field} label="Schedule (cron · UTC)">
          <div class="flex gap-1">
            <select class="${inputCls()} flex-1"
              value=${SCHEDULE_PRESETS.find(p => p.cron === form.schedule_cron)?.cron ?? ''}
              onChange=${e => e.target.value && set('schedule_cron', e.target.value)}
              title="Preset → fills the cron field. The field stays editable.">
              <option value="">— preset —</option>
              ${SCHEDULE_PRESETS.map(p => html`
                <option key=${p.cron} value=${p.cron}>${p.label}</option>`)}
            </select>
            <input class="${inputCls()} flex-1" placeholder="5 21 * * 1-5"
              value=${form.schedule_cron}
              onChange=${e => set('schedule_cron', e.target.value)} />
          </div>
        </${Field}>
      </div>
      <div class="mt-3">
        <${Field} label="Strategy params (JSON)">
          <input class="${inputCls()} ${paramsOk ? '' : 'border-danger-500'}"
            placeholder=${'{"short": 50, "long": 200}'} value=${form.paramsJson}
            onChange=${e => set('paramsJson', e.target.value)} />
        </${Field}>
        ${!paramsOk && html`<div class="text-xs text-danger-500 mt-1">Params must be a JSON object.</div>`}
      </div>
      <div class="flex items-center justify-between mt-3 gap-3">
        <div class="text-xs text-mute-400">
          Ticker is the T212 instrument code (e.g. <span class="num">AAPL_US_EQ</span>).
          Cron is 5 fields in UTC — <span class="num">5 21 * * 1-5</span> is 21:05 on weekdays.
        </div>
        <button onClick=${submit} disabled=${!valid || busy}
          class="text-sm px-4 py-1.5 rounded font-medium whitespace-nowrap transition
                 ${valid && !busy
                   ? 'bg-accent-600 hover:bg-accent-500 text-white'
                   : 'bg-ink-800 text-mute-400 cursor-not-allowed'}">
          Create deployment
        </button>
      </div>
    </div>
  `;
}

function RunResultBanner({ result, onDismiss }) {
  const status = result.status;
  const box =
    status === 'OK'    ? 'bg-accent-500/10 border-accent-500/30 text-accent-400' :
    (status === 'ERROR' || status === 'CIRCUIT_BREAKER')
                       ? 'bg-danger-500/10 border-danger-500/30 text-danger-500' :
                         'bg-ink-800 border-ink-600 text-mute-200';
  let msg;
  if (status === 'OK') {
    msg = `Run complete — ${result.intent_count} order intent(s), ${result.submitted} submitted.`;
  } else if (status === 'DISARMED') {
    msg = 'Deployment is paused — arm it before it can place orders.';
  } else if (status === 'NO_DATA') {
    msg = 'No price data was available for this run — nothing was placed.';
  } else if (status === 'ERROR') {
    msg = `Run failed: ${result.error || 'unknown error'}`;
  } else if (status === 'CIRCUIT_BREAKER') {
    msg = 'Circuit breaker tripped — the open position exceeded the max-loss limit, so the deployment was disarmed.';
  } else {
    msg = `Run status: ${status}`;
  }
  return html`
    <div class="flex items-start gap-3 border rounded px-3 py-2 text-sm ${box}">
      <div class="flex-1">${msg}</div>
      <button onClick=${onDismiss} class="text-mute-400 hover:text-mute-100 transition">✕</button>
    </div>
  `;
}

function DeploymentList({ deployments, loading, error, strategies, busy,
                          onArm, onDisarm, onRun, onTrack, onDelete, onReload }) {
  const nameOf = (id) => {
    const s = strategies.find(x => x.id === id);
    return s ? s.name : id.slice(0, 8);
  };
  return html`
    <div class="bg-ink-900 border border-ink-700 rounded">
      <div class="flex items-center justify-between px-4 py-2.5 border-b border-ink-700">
        <div class="text-xs uppercase tracking-wider text-mute-400">
          Deployments${deployments ? html` <span class="num">· ${deployments.length}</span>` : ''}
        </div>
        <button onClick=${onReload} disabled=${loading}
          class="text-xs px-3 py-1 rounded bg-ink-800 hover:bg-ink-700 text-mute-200
                 disabled:opacity-50 transition">
          ${loading ? 'refreshing…' : '↻ refresh'}
        </button>
      </div>
      ${error
        ? html`<div class="px-4 py-8 text-center text-danger-500 text-sm">${error}</div>`
        : deployments === null
          ? html`<div class="px-4 py-8 text-center text-mute-400 text-sm">Loading deployments…</div>`
          : deployments.length === 0
            ? html`<div class="px-4 py-8 text-center text-mute-400 text-sm">No deployments yet. Create one above.</div>`
            : html`
        <div class="overflow-x-auto">
          <table class="min-w-full text-sm">
            <thead>
              <tr class="border-b border-ink-700 text-mute-400 text-xs">
                <th class="py-2 px-3 text-left font-medium">Strategy</th>
                <th class="py-2 px-3 text-left font-medium">Ticker</th>
                <th class="py-2 px-3 text-right font-medium">Capital</th>
                <th class="py-2 px-3 text-left font-medium">Schedule</th>
                <th class="py-2 px-3 text-left font-medium">Status</th>
                <th class="py-2 px-3 text-left font-medium">Last run</th>
                <th class="py-2 px-3 text-right font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              ${deployments.map(d => html`
                <tr key=${d.id} class="border-b border-ink-800 hover:bg-ink-900/60">
                  <td class="py-2 px-3 text-mute-100">${nameOf(d.strategy_id)}</td>
                  <td class="py-2 px-3 num text-mute-200">
                    ${d.ticker} <span class="text-mute-400 text-xs">${d.timeframe}</span>
                  </td>
                  <td class="py-2 px-3 text-right num text-mute-200">${fmtAcctNum(d.deployment_capital, 0)}</td>
                  <td class="py-2 px-3 num text-mute-300 text-xs">${d.schedule_cron}</td>
                  <td class="py-2 px-3">
                    ${d.armed
                      ? html`<span class="px-2 py-0.5 rounded text-[10px] uppercase tracking-wider bg-accent-500/20 text-accent-400 font-medium">Armed</span>`
                      : html`<span class="px-2 py-0.5 rounded text-[10px] uppercase tracking-wider bg-ink-700 text-mute-400 font-medium">Paused</span>`}
                  </td>
                  <td class="py-2 px-3 num text-mute-400 text-xs">
                    ${d.last_run_at ? fmtT212Date(d.last_run_at) : 'never'}
                  </td>
                  <td class="py-2 px-3">
                    <div class="flex items-center justify-end gap-1.5">
                      ${d.armed
                        ? html`<button onClick=${() => onDisarm(d)} disabled=${busy}
                            class="text-xs px-2 py-1 rounded bg-ink-800 hover:bg-ink-700 text-mute-200
                                   disabled:opacity-50 transition">Pause</button>`
                        : html`<button onClick=${() => onArm(d)} disabled=${busy}
                            class="text-xs px-2 py-1 rounded bg-danger-500/80 hover:bg-danger-500 text-white
                                   disabled:opacity-50 transition">Arm</button>`}
                      <button onClick=${() => onRun(d)} disabled=${busy}
                        class="text-xs px-2 py-1 rounded bg-ink-800 hover:bg-ink-700 text-mute-200
                               disabled:opacity-50 transition">Run now</button>
                      <button onClick=${() => onTrack(d.id)} disabled=${busy}
                        class="text-xs px-2 py-1 rounded bg-ink-800 hover:bg-ink-700 text-mute-200
                               disabled:opacity-50 transition">Tracking</button>
                      <button onClick=${() => onDelete(d)} disabled=${busy}
                        class="text-xs px-2 py-1 rounded bg-ink-800 hover:bg-ink-700 text-danger-500
                               disabled:opacity-50 transition">Delete</button>
                    </div>
                  </td>
                </tr>
              `)}
            </tbody>
          </table>
        </div>
      `}
    </div>
  `;
}

function DeploymentTrackingPanel({ deploymentId, strategies, onBack }) {
  const pushToast = useToast();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setData(await getDeploymentTracking(deploymentId));
    } catch (err) {
      setError(err.message);
      pushToast('error', `Could not load tracking: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => { load(); }, [deploymentId]);

  const dep = data && data.deployment;
  const stratName = dep
    ? ((strategies.find(s => s.id === dep.strategy_id) || {}).name || dep.strategy_id.slice(0, 8))
    : '';
  const m = (data && data.expected_metrics) || {};

  return html`
    <div class="p-6 space-y-5">
      <div class="flex items-baseline justify-between">
        <div>
          <div class="text-xs uppercase tracking-wider text-mute-400">Deployment · Tracking</div>
          <div class="text-lg text-mute-100 font-medium mt-0.5">
            ${dep ? `${stratName} · ${dep.ticker}` : 'Tracking'}
          </div>
        </div>
        <button onClick=${onBack}
          class="text-sm text-mute-300 hover:text-mute-100 transition px-2 py-1">
          ← Back to deployments
        </button>
      </div>

      ${loading
        ? html`<div class="py-12 text-center text-mute-400 text-sm">Loading tracking…</div>`
        : error
          ? html`<${LiveError} message=${error} onRetry=${load} />`
          : html`
        <div class="bg-ink-900 border border-ink-700 rounded p-4 flex flex-wrap gap-x-8 gap-y-3">
          ${[
            ['Status', dep.armed ? 'Armed' : 'Paused'],
            ['Ticker', dep.ticker],
            ['Timeframe', dep.timeframe],
            ['Capital', fmtAcctNum(dep.deployment_capital, 0)],
            ['Order type', dep.order_type],
            ['Schedule', dep.schedule_cron],
            ['Last run', dep.last_run_at ? fmtT212Date(dep.last_run_at) : 'never'],
          ].map(([k, v]) => html`
            <div key=${k}>
              <div class="text-[10px] uppercase tracking-wider text-mute-400">${k}</div>
              <div class="text-sm num text-mute-100 mt-0.5">${v}</div>
            </div>
          `)}
        </div>

        <div>
          <div class="text-xs uppercase tracking-wider text-mute-400 mb-2">
            Expected equity curve
            <span class="normal-case text-mute-400">
              ${data.expected_rebased
                ? '— backtested over the deployment\'s live period, rebased to its capital'
                : '— backtested over the full lookback window (no live bars yet)'}
            </span>
          </div>
          ${data.expected_error
            ? html`<div class="bg-ink-900 border border-ink-700 rounded px-4 py-8 text-center text-mute-400 text-sm">
                Expected curve unavailable — ${data.expected_error}
              </div>`
            : data.expected_timestamps.length === 0
              ? html`<div class="bg-ink-900 border border-ink-700 rounded px-4 py-8 text-center text-mute-400 text-sm">
                  No expected curve to show.
                </div>`
              : html`
                <div class="h-[300px] bg-ink-900 border border-ink-700 rounded overflow-hidden">
                  <${EquityChart} timestamps=${data.expected_timestamps} equity=${data.expected_equity}
                                  buyHold=${data.expected_buy_hold} />
                </div>
                <div class="grid grid-cols-3 gap-3 mt-3">
                  <${MetricCard} label="Expected CAGR" value=${fmtPct(m.cagr)}
                    tone=${m.cagr > 0 ? 'good' : m.cagr < 0 ? 'bad' : null} />
                  <${MetricCard} label="Expected Sharpe" value=${fmtNum(m.sharpe, 2)} />
                  <${MetricCard} label="Expected max drawdown" value=${fmtPct(m.max_drawdown)} tone="bad" />
                </div>
              `}
        </div>

        <div>
          <div class="text-xs uppercase tracking-wider text-mute-400 mb-2">
            Orders placed <span class="num">· ${data.orders.length}</span>
            <span class="normal-case text-mute-400">— what this deployment actually sent</span>
          </div>
          ${data.orders.length === 0
            ? html`<div class="bg-ink-900 border border-ink-700 rounded px-4 py-8 text-center text-mute-400 text-sm">
                No orders placed yet. Orders appear here once the deployment runs.
              </div>`
            : html`
            <div class="bg-ink-900 border border-ink-700 rounded overflow-x-auto">
              <table class="min-w-full text-sm">
                <thead>
                  <tr class="border-b border-ink-700 text-mute-400 text-xs">
                    <th class="py-2 px-3 text-left font-medium">As of</th>
                    <th class="py-2 px-3 text-left font-medium">Side</th>
                    <th class="py-2 px-3 text-right font-medium">Quantity</th>
                    <th class="py-2 px-3 text-left font-medium">Type</th>
                    <th class="py-2 px-3 text-left font-medium">Status</th>
                    <th class="py-2 px-3 text-right font-medium">T212 order</th>
                  </tr>
                </thead>
                <tbody>
                  ${data.orders.map(o => {
                    const sideTone = o.side === 'BUY' ? 'text-accent-400' : 'text-danger-500';
                    const statusTone = o.status === 'SUBMITTED' ? 'text-accent-400'
                      : o.status === 'FAILED' ? 'text-danger-500' : 'text-mute-300';
                    return html`
                      <tr key=${o.id} class="border-b border-ink-800">
                        <td class="py-2 px-3 num text-mute-300 text-xs">${fmtT212Date(o.as_of_timestamp)}</td>
                        <td class="py-2 px-3 font-medium ${sideTone}">${o.side}</td>
                        <td class="py-2 px-3 text-right num text-mute-200">${fmtAcctNum(o.qty, 4)}</td>
                        <td class="py-2 px-3 text-mute-300 text-xs uppercase">${o.order_type}</td>
                        <td class="py-2 px-3 text-xs uppercase ${statusTone}">${o.status}</td>
                        <td class="py-2 px-3 text-right num text-mute-400 text-xs">${o.t212_order_id || '—'}</td>
                      </tr>
                    `;
                  })}
                </tbody>
              </table>
            </div>
          `}
        </div>

        <!-- Phase 14 — per-run history. The audit trail of every fire,
             not just the last one. Read-only; rendered below the orders
             so the user sees what was placed first, then the broader
             context of how often the deployment has been running and how
             it's behaved. -->
        <div>
          <div class="text-xs uppercase tracking-wider text-mute-400 mb-2">
            Recent runs <span class="num">· ${data.runs?.length ?? 0}</span>
            <span class="normal-case text-mute-400">— every fire of the scheduler, most recent first</span>
          </div>
          ${!data.runs || data.runs.length === 0
            ? html`<div class="bg-ink-900 border border-ink-700 rounded px-4 py-8 text-center text-mute-400 text-sm">
                No runs yet. History appears here once the deployment fires (cron or "Run now").
              </div>`
            : html`
              <div class="bg-ink-900 border border-ink-700 rounded overflow-x-auto">
                <table class="min-w-full text-sm">
                  <thead>
                    <tr class="border-b border-ink-700 text-mute-400 text-xs">
                      <th class="py-2 px-3 text-left font-medium">Ran at (UTC)</th>
                      <th class="py-2 px-3 text-left font-medium">Status</th>
                      <th class="py-2 px-3 text-left font-medium">As-of bar</th>
                      <th class="py-2 px-3 text-right font-medium">Intents</th>
                      <th class="py-2 px-3 text-right font-medium">Submitted</th>
                      <th class="py-2 px-3 text-left font-medium">Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${data.runs.map(r => {
                      const statusTone =
                        r.status === 'OK' ? 'text-accent-400' :
                        r.status === 'ERROR' || r.status === 'CIRCUIT_BREAKER' ? 'text-danger-500' :
                        r.status === 'INSUFFICIENT_CASH' ? 'text-yellow-400' :
                        'text-mute-300';
                      return html`
                        <tr key=${r.id} class="border-t border-ink-800">
                          <td class="py-2 px-3 num text-mute-200 text-xs">${r.ran_at.slice(0, 19).replace('T', ' ')}</td>
                          <td class="py-2 px-3 num ${statusTone}">${r.status}</td>
                          <td class="py-2 px-3 num text-mute-400 text-xs">${r.as_of_timestamp?.slice(0, 10) ?? '—'}</td>
                          <td class="py-2 px-3 text-right num text-mute-200">${r.intent_count}</td>
                          <td class="py-2 px-3 text-right num text-mute-100">${r.submitted_count}</td>
                          <td class="py-2 px-3 text-xs text-danger-500 truncate max-w-xs">${r.error ?? ''}</td>
                        </tr>`;
                    })}
                  </tbody>
                </table>
              </div>
            `}
        </div>
      `}
    </div>
  `;
}

function DeployView({ strategies, onClose }) {
  const pushToast = useToast();
  const acknowledgeRuns = useContext(AcknowledgeRunsContext);
  const [deployments, setDeployments] = useState(null);
  const [listError, setListError] = useState(null);
  const [listLoading, setListLoading] = useState(false);
  const [armModal, setArmModal] = useState(null);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState(null);
  const [runResult, setRunResult] = useState(null);
  const [trackingId, setTrackingId] = useState(null);
  const [formKey, setFormKey] = useState(0);
  const [killResult, setKillResult] = useState(null);

  async function loadDeployments() {
    setListLoading(true);
    setListError(null);
    try {
      const res = await listDeployments();
      setDeployments(res.deployments);
    } catch (err) {
      setListError(err.message);
      setDeployments([]);
      pushToast('error', `Failed to load deployments: ${err.message}`);
    } finally {
      setListLoading(false);
    }
  }
  useEffect(() => { loadDeployments(); }, []);

  function requestCreate(payload) {
    setActionError(null);
    setArmModal({
      title: 'Arm new deployment',
      body: html`<span>This places real orders into your Trading 212 <span class="text-accent-400 font-medium">Demo</span> account on the schedule you set. Type <span class="num text-mute-100">ARM</span> to confirm.</span>`,
      onConfirm: async () => {
        setBusy(true);
        setActionError(null);
        try {
          await createDeployment({ ...payload, confirmation_token: 'ARM' });
          setArmModal(null);
          setFormKey(k => k + 1);
          await loadDeployments();
          pushToast('success', `Armed deployment for ${payload.ticker}.`);
        } catch (err) {
          setActionError(err.message);
          pushToast('error', `Could not arm deployment: ${err.message}`);
        } finally {
          setBusy(false);
        }
      },
    });
  }

  function requestArm(dep) {
    setActionError(null);
    setArmModal({
      title: 'Arm deployment',
      body: html`<span>This resumes placing real orders into your Trading 212 <span class="text-accent-400 font-medium">Demo</span> account. Type <span class="num text-mute-100">ARM</span> to confirm.</span>`,
      onConfirm: async () => {
        setBusy(true);
        setActionError(null);
        try {
          await armDeployment(dep.id, 'ARM');
          setArmModal(null);
          await loadDeployments();
          pushToast('success', `Armed: ${dep.ticker}.`);
        } catch (err) {
          setActionError(err.message);
          pushToast('error', `Arm failed: ${err.message}`);
        } finally {
          setBusy(false);
        }
      },
    });
  }

  async function doDisarm(dep) {
    setActionError(null);
    setBusy(true);
    try {
      await disarmDeployment(dep.id);
      await loadDeployments();
      pushToast('info', `Paused: ${dep.ticker}.`);
    } catch (err) {
      setActionError(err.message);
      pushToast('error', `Pause failed: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function doDelete(dep) {
    if (!window.confirm(
      'Delete this deployment? Any working order is cancelled and its order history is removed.'
    )) return;
    setActionError(null);
    setBusy(true);
    try {
      await deleteDeployment(dep.id);
      await loadDeployments();
      pushToast('success', `Deleted: ${dep.ticker}.`);
    } catch (err) {
      setActionError(err.message);
      pushToast('error', `Delete failed: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function doRun(dep) {
    setActionError(null);
    setRunResult(null);
    setBusy(true);
    try {
      const res = await runDeployment(dep.id);
      setRunResult(res);
      await loadDeployments();
      // Fast-forward the notifier watermark so the toast we just pushed
      // for this manual run isn't re-pushed by the 30s poller.
      try {
        const feed = await getDeploymentRunsFeed(0);
        acknowledgeRuns(feed.latest_id);
      } catch { /* poller will catch up next tick */ }
      // Status-aware toast: OK is success, anything else is informational
      // or an error depending on severity.
      if (res.status === 'OK') {
        pushToast('success', `${dep.ticker}: ran (${res.submitted ?? 0} order(s) submitted).`);
      } else if (res.status === 'ERROR' || res.status === 'CIRCUIT_BREAKER') {
        pushToast('error', `${dep.ticker}: ${res.status}${res.error ? ' — ' + res.error : ''}`);
      } else {
        pushToast('warning', `${dep.ticker}: ${res.status}`);
      }
    } catch (err) {
      setActionError(err.message);
      pushToast('error', `Run failed: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  function requestKill() {
    setActionError(null);
    setArmModal({
      title: 'Kill switch',
      expect: 'KILL',
      body: html`<span>This disarms <span class="text-mute-100 font-medium">every deployment</span> and places market orders to <span class="text-mute-100 font-medium">flatten every open position</span> in your Trading 212 Demo account. Type <span class="num text-mute-100">KILL</span> to confirm.</span>`,
      onConfirm: async () => {
        setBusy(true);
        setActionError(null);
        try {
          const res = await killSwitch('KILL');
          setArmModal(null);
          setKillResult(res);
          await loadDeployments();
          pushToast('warning', `Kill switch fired — disarmed ${res.disarmed_count ?? 0}, flattened ${res.flattened_count ?? 0}.`);
        } catch (err) {
          setActionError(err.message);
          pushToast('error', `Kill switch failed: ${err.message}`);
        } finally {
          setBusy(false);
        }
      },
    });
  }

  if (trackingId) {
    return html`<${DeploymentTrackingPanel}
      deploymentId=${trackingId} strategies=${strategies}
      onBack=${() => setTrackingId(null)} />`;
  }

  return html`
    <div class="p-6 space-y-4">
      <div class="flex items-baseline justify-between">
        <div>
          <div class="text-xs uppercase tracking-wider text-mute-400">Trading 212 · Demo</div>
          <div class="text-lg text-mute-100 font-medium mt-0.5 flex items-center gap-2">
            Deployments
            <span class="px-2 py-0.5 rounded text-[10px] uppercase tracking-wider bg-accent-500/20 text-accent-400 font-medium">
              Demo
            </span>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <button onClick=${requestKill} disabled=${busy}
            class="text-xs px-3 py-1.5 rounded font-medium bg-danger-500/90
                   hover:bg-danger-500 text-white disabled:opacity-50 transition">
            ⚠ Kill switch
          </button>
          <button onClick=${onClose}
            class="text-sm text-mute-300 hover:text-mute-100 transition px-2 py-1">
            ← Back to run
          </button>
        </div>
      </div>

      <div class="text-xs text-mute-400 leading-relaxed">
        A deployment runs a saved strategy on a schedule and places orders into your Trading 212
        Demo account. A deployment is created <span class="text-accent-400">armed</span> — you confirm
        by typing ARM. Pause it any time; re-arming asks for ARM again.
      </div>

      <${DeployCreateForm} key=${formKey} strategies=${strategies}
        busy=${busy} onCreate=${requestCreate} />

      ${actionError && !armModal && html`
        <div class="bg-danger-500/10 border border-danger-500/30 rounded px-3 py-2 text-sm text-danger-500">
          ${actionError}
        </div>
      `}

      ${runResult && html`<${RunResultBanner} result=${runResult} onDismiss=${() => setRunResult(null)} />`}

      ${killResult && html`
        <div class="flex items-start gap-3 border rounded px-3 py-2 text-sm
                    bg-danger-500/10 border-danger-500/30 text-danger-500">
          <div class="flex-1">
            Kill switch fired — ${killResult.deployments_disarmed} deployment(s) disarmed,
            ${' '}${killResult.orders.length} position(s) sent closing orders.${' '}
            ${killResult.error || ''}
          </div>
          <button onClick=${() => setKillResult(null)}
            class="text-mute-400 hover:text-mute-100 transition">✕</button>
        </div>
      `}

      <${DeploymentList}
        deployments=${deployments} loading=${listLoading} error=${listError}
        strategies=${strategies} busy=${busy}
        onArm=${requestArm} onDisarm=${doDisarm} onRun=${doRun} onDelete=${doDelete}
        onTrack=${id => setTrackingId(id)} onReload=${loadDeployments} />

      ${armModal && html`<${ArmModal}
        title=${armModal.title} body=${armModal.body} expect=${armModal.expect}
        busy=${busy} error=${actionError}
        onConfirm=${armModal.onConfirm}
        onCancel=${() => { setArmModal(null); setActionError(null); }} />`}
    </div>
  `;
}


// ---- Optimise flow (Phase 15.5) ------------------------------------

// Heatmap axes for an optimise sweep: the first two grid keys that vary
// (a list of >1 value). Phase 15.1 — previously hardcoded to SMA
// short/long, which made the sweep 500 for any other strategy.
function gridHeatmapAxes(grid, metric) {
  if (!grid || Array.isArray(grid) || typeof grid !== 'object') return [];
  const varied = Object.keys(grid).filter(
    k => Array.isArray(grid[k]) && grid[k].length > 1);
  if (varied.length < 2) return [];
  return [{ x_param: varied[0], y_param: varied[1], metric_name: metric }];
}

// Phase 15.6 — the best-combo params re-tested across other assets.
function MultiAssetTable({ rows }) {
  return html`
    <div class="bg-ink-900 border border-ink-700 rounded p-4">
      <div class="text-xs uppercase tracking-wider text-mute-400 mb-1">
        Best params across other assets
      </div>
      <div class="text-[11px] text-mute-400 mb-3">
        The winning parameters re-tested unchanged — do they generalise, or
        are they fit to this one ticker?
      </div>
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-ink-700 text-mute-400 text-xs">
            <th class="text-left py-2 px-3 font-medium">Ticker</th>
            <th class="text-right py-2 px-3 font-medium">CAGR</th>
            <th class="text-right py-2 px-3 font-medium">Sharpe</th>
            <th class="text-right py-2 px-3 font-medium">Max DD</th>
            <th class="text-right py-2 px-3 font-medium">Calmar</th>
            <th class="text-right py-2 px-3 font-medium">Trades</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(r => html`
            <tr key=${r.ticker} class="border-t border-ink-800">
              <td class="py-2 px-3 num text-mute-100">${r.ticker}</td>
              ${r.error
                ? html`<td colspan="5" class="py-2 px-3 text-xs text-danger-500">${r.error}</td>`
                : html`
                    <td class="py-2 px-3 text-right num ${(r.metrics?.cagr ?? 0) >= 0 ? 'text-accent-400' : 'text-danger-500'}">
                      ${fmtPct(r.metrics?.cagr)}</td>
                    <td class="py-2 px-3 text-right num text-mute-100">${fmtNum(r.metrics?.sharpe, 2)}</td>
                    <td class="py-2 px-3 text-right num text-danger-500">${fmtPct(r.metrics?.max_drawdown)}</td>
                    <td class="py-2 px-3 text-right num text-mute-200">${fmtNum(r.metrics?.calmar, 2)}</td>
                    <td class="py-2 px-3 text-right num text-mute-400">${r.metrics?.n_trades ?? '—'}</td>`}
            </tr>`)}
        </tbody>
      </table>
    </div>`;
}

// The result of an optimisation: the best combo's equity curve overlaid
// on the baseline backtest, plus the heatmap, DSR, full combo table, and
// (Phase 15.6) the best combo's regime breakdown and multi-asset re-test.
function OptimiseResult({ baseline, sweep }) {
  const bestCombo = (sweep.best_index !== null && sweep.best_index !== undefined)
    ? sweep.combos.find(c => c.index === sweep.best_index)
    : null;

  // Memoized so MultiEquityChart's [runs] effect doesn't remount the chart
  // on every unrelated App re-render.
  const overlayRuns = useMemo(() => {
    const runs = [];
    if (baseline?.timestamps && baseline?.equity) {
      runs.push({
        label: 'Baseline (current params)',
        timestamps: baseline.timestamps, equity: baseline.equity,
      });
    }
    if (sweep.best_combo_timestamps && sweep.best_combo_equity) {
      runs.push({
        label: 'Optimised (best combo)',
        timestamps: sweep.best_combo_timestamps, equity: sweep.best_combo_equity,
      });
    }
    return runs;
  }, [baseline, sweep]);

  const lastValid = (arr) => {
    if (!arr) return null;
    for (let i = arr.length - 1; i >= 0; i--) {
      if (arr[i] !== null && arr[i] !== undefined && !Number.isNaN(arr[i])) return arr[i];
    }
    return null;
  };
  const baseFinal = lastValid(baseline?.equity);
  const optFinal = lastValid(sweep.best_combo_equity);
  const delta = (baseFinal && optFinal) ? (optFinal / baseFinal - 1) : null;

  return html`
    <div class="space-y-5">
      <div class="text-sm text-mute-200">
        ${sweep.n_combinations} combinations tested.
        ${bestCombo
          ? html` Best: <span class="num text-accent-400">${
              Object.entries(bestCombo.params).map(([k, v]) => `${k}=${v}`).join(', ')
            }</span>`
          : ' No best combo found.'}
        ${delta !== null && html`
          <span class="num ${delta >= 0 ? 'text-accent-400' : 'text-danger-500'}">
            · final equity ${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)}% vs baseline
          </span>`}
      </div>

      ${overlayRuns.length > 0 && html`
        <div>
          <div class="text-xs uppercase tracking-wider text-mute-400 mb-2">
            Optimised vs baseline equity
          </div>
          <div class="h-[300px] bg-ink-900 border border-ink-700 rounded overflow-hidden">
            <${MultiEquityChart} runs=${overlayRuns} />
          </div>
          <div class="flex gap-4 mt-2 text-[11px] text-mute-300">
            ${overlayRuns.map((r, i) => html`
              <span key=${i} class="flex items-center gap-1">
                <span class="inline-block w-3 h-[2px]"
                      style=${{ background: PIN_COLORS[i % PIN_COLORS.length] }}></span>
                ${r.label}</span>`)}
          </div>
        </div>`}

      ${sweep.heatmaps?.length > 0 && html`
        <div class="bg-ink-900 border border-ink-700 rounded p-4">
          <div class="text-xs uppercase tracking-wider text-mute-400 mb-4">Parameter heatmap</div>
          ${sweep.heatmaps.map((hm, i) => html`<div key=${i}><${HeatmapGrid} heatmap=${hm} /></div>`)}
        </div>`}

      ${sweep.deflated_sharpe && html`<${DSRCard} dsr=${sweep.deflated_sharpe} />`}

      ${sweep.regimes && html`<${RegimePanel} regimes=${sweep.regimes} />`}

      ${sweep.multi_asset?.length > 0 && html`<${MultiAssetTable} rows=${sweep.multi_asset} />`}

      <div class="bg-ink-900 border border-ink-700 rounded p-4">
        <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">
          All combos · sorted by ${sweep.selection_metric.replace(/_/g, ' ')}
        </div>
        <${ComboTable}
          combos=${sweep.combos}
          bestIndex=${sweep.best_index}
          selectionMetric=${sweep.selection_metric}
          higherIsBetter=${sweep.higher_is_better} />
      </div>
    </div>`;
}

// The Optimise section that hangs below a backtest result: an Optimise
// button, a collapsible param-grid panel, and (once run) the results.
function OptimiseSection({ baseline, sweepCfg, setSweepCfg, sweepResult,
                           optimising, show, onShow, onHide, onOptimise, error }) {
  const updS = (k, v) => setSweepCfg(s => ({ ...s, [k]: v }));
  const ic = inputCls();

  const panel = html`
    <div class="bg-ink-900 border border-ink-700 rounded p-4 space-y-3">
      <div class="text-xs text-mute-400">
        Backtests every parameter combination over the same ticker and window
        and keeps the best one by the selection metric (deflated for the number
        of trials, so a lucky combo can't fool it). The winner also gets a
        regime breakdown and a re-test on the other assets below.
      </div>
      <${Field} label="Parameter grid (JSON)">
        <textarea rows="4"
          class="w-full bg-ink-850 border border-ink-700 rounded px-2 py-1.5 text-xs num text-mute-100 focus:outline-none focus:border-accent-500 font-mono resize-none"
          value=${sweepCfg.paramGridJson}
          onChange=${e => updS('paramGridJson', e.target.value)} />
      </${Field}>
      <div class="w-48">
        <${Field} label="Selection metric">
          <select class=${ic} value=${sweepCfg.selectionMetric}
            onChange=${e => updS('selectionMetric', e.target.value)}>
            <option value="sharpe">Sharpe</option>
            <option value="cagr">CAGR</option>
            <option value="calmar">Calmar</option>
            <option value="sortino">Sortino</option>
            <option value="max_drawdown">Max drawdown</option>
          </select>
        </${Field}>
      </div>
      <${Field} label="Re-test best params on (comma-separated tickers)">
        <input class=${ic} value=${sweepCfg.multiAssetTickers}
          onChange=${e => updS('multiAssetTickers', e.target.value)} />
      </${Field}>
      <div class="flex gap-2 pt-1">
        <button onClick=${onOptimise} disabled=${optimising}
          class="px-4 py-2 rounded font-semibold text-sm transition
                 ${optimising
                   ? 'bg-ink-700 text-mute-400 cursor-not-allowed'
                   : 'bg-accent-500 hover:bg-accent-400 text-ink-950'}">
          ${optimising ? 'Optimising…' : 'Run optimisation'}
        </button>
        <button onClick=${onHide} disabled=${optimising}
          class="px-4 py-2 rounded text-sm text-mute-300 hover:text-mute-100 transition">
          Cancel
        </button>
      </div>
    </div>`;

  return html`
    <div class="space-y-4">
      <div class="flex items-baseline justify-between">
        <div class="text-xs uppercase tracking-wider text-mute-400">Optimise</div>
        ${(sweepResult || show) && html`
          <button onClick=${show ? onHide : onShow}
            class="text-xs text-mute-300 hover:text-mute-100 transition">
            ${show ? 'hide settings' : 'adjust & re-run'}
          </button>`}
      </div>

      ${!show && !sweepResult && html`
        <div class="space-y-2">
          <button onClick=${onShow}
            class="px-4 py-2 rounded font-semibold text-sm bg-accent-600 hover:bg-accent-500 text-white transition">
            ⚙ Optimise this strategy
          </button>
          <div class="text-xs text-mute-400">
            Sweep a grid of parameters and see if a better combination beats the run above.
          </div>
        </div>`}

      ${show && panel}

      ${error && !optimising && html`
        <div class="bg-danger-500/10 border border-danger-500/30 rounded p-3 text-sm text-danger-500">
          ${error}
        </div>`}

      ${optimising && !sweepResult && html`
        <div class="text-sm text-mute-400">Optimising — backtesting every combination…</div>`}

      ${sweepResult && html`<${OptimiseResult} baseline=${baseline} sweep=${sweepResult} />`}
    </div>`;
}


// ---- Toast notifications (Phase 14) --------------------------------
//
// One global queue, rendered top-right, used by any view that wants to
// surface a transient message. Errors get a manual close button; all
// kinds auto-dismiss after their TTL. The `pushToast` ref pattern lets
// non-component code (e.g. event handlers fired from setTimeout chains)
// reach the queue without re-rendering on every push.

const ToastContext = createContext(null);

const TOAST_TTL = { error: 8000, warning: 6000, success: 4000, info: 4000 };

const TOAST_STYLE = {
  error:   { border: 'border-danger-500/40',  bg: 'bg-danger-500/10',  text: 'text-danger-500',  label: 'Error'   },
  warning: { border: 'border-yellow-400/40',  bg: 'bg-yellow-400/10',  text: 'text-yellow-300',  label: 'Warning' },
  success: { border: 'border-accent-500/40',  bg: 'bg-accent-500/10',  text: 'text-accent-400',  label: 'Done'    },
  info:    { border: 'border-mute-500/40',    bg: 'bg-ink-800',        text: 'text-mute-200',    label: 'Info'    },
};

function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const idRef = useRef(0);

  const dismiss = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const push = useCallback((arg1, arg2) => {
    // Two call shapes: pushToast('error', 'message') or
    // pushToast({ kind, message, ttl }). The first is shorter at the
    // catch site; the second lets callers tune ttl.
    const opts = typeof arg1 === 'string' ? { kind: arg1, message: arg2 } : (arg1 || {});
    const kind = opts.kind && TOAST_STYLE[opts.kind] ? opts.kind : 'info';
    const message = String(opts.message ?? '').trim();
    if (!message) return;
    const id = ++idRef.current;
    const ttl = opts.ttl ?? TOAST_TTL[kind];
    setToasts(prev => [...prev, { id, kind, message }]);
    if (ttl > 0) setTimeout(() => dismiss(id), ttl);
  }, [dismiss]);

  return html`
    <${ToastContext.Provider} value=${push}>
      ${children}
      <div class="fixed top-3 right-3 z-50 flex flex-col gap-2 max-w-sm pointer-events-none">
        ${toasts.map(t => {
          const s = TOAST_STYLE[t.kind];
          return html`
            <div key=${t.id}
                 class="pointer-events-auto ${s.bg} ${s.text} border ${s.border} rounded shadow-lg px-3 py-2 text-sm flex items-start gap-2">
              <div class="flex-1 min-w-0">
                <div class="text-xs uppercase tracking-wider opacity-70">${s.label}</div>
                <div class="break-words">${t.message}</div>
              </div>
              <button class="text-mute-400 hover:text-mute-100 text-xs leading-none px-1"
                      onClick=${() => dismiss(t.id)} aria-label="Dismiss">✕</button>
            </div>`;
        })}
      </div>
    </${ToastContext.Provider}>
  `;
}

// Used by every component that wants to push a toast. Safe to call from
// anywhere inside the Provider — returns a no-op outside it (so unit
// helpers don't crash).
function useToast() {
  return useContext(ToastContext) || (() => {});
}


// ---- Deployment notifier (Phase 14) --------------------------------
//
// App owns a watermark (lastSeenRunId) and polls /deployments/runs
// every 30s. For each new row the poller pushes a toast AND fires a
// browser Notification (if the user has granted permission). The
// AcknowledgeRunsContext lets DeployView fast-forward the watermark
// after a manual "Run now" so the manual toast isn't duplicated 30s
// later by the poller.

const AcknowledgeRunsContext = createContext(() => {});

const NOTIFIER_POLL_MS = 30_000;

// What we say to the user, per status. INSUFFICIENT_CASH is a benign
// transient (the deployment stays armed); ERROR / CIRCUIT_BREAKER are
// real failures that warrant the OS notification.
function _runMessage(row) {
  const tk = row.ticker || row.deployment_id?.slice(0, 6) || 'deployment';
  if (row.status === 'OK') {
    if ((row.submitted_count || 0) > 0) {
      const n = row.submitted_count;
      return { kind: 'success', notify: true, text: `${tk}: submitted ${n} order${n === 1 ? '' : 's'}.` };
    }
    return { kind: 'info', notify: false, text: `${tk}: ran (no orders needed).` };
  }
  if (row.status === 'ERROR') {
    return { kind: 'error', notify: true, text: `${tk}: error — ${row.error || 'unknown'}` };
  }
  if (row.status === 'CIRCUIT_BREAKER') {
    return { kind: 'error', notify: true, text: `${tk}: circuit breaker — auto-disarmed.` };
  }
  if (row.status === 'INSUFFICIENT_CASH') {
    return { kind: 'warning', notify: true, text: `${tk}: not enough cash to place the order.` };
  }
  // NO_DATA and anything else: surface quietly.
  return { kind: 'info', notify: false, text: `${tk}: ${row.status}` };
}

function _fireNotification(text) {
  if (typeof window === 'undefined' || !('Notification' in window)) return;
  if (Notification.permission !== 'granted') return;
  try {
    new Notification('Backtester', { body: text, tag: 'backtester-deployment' });
  } catch { /* the OS can reject; toast still covers it */ }
}


// ---- App -----------------------------------------------------------

function App() {
  const pushToast = useToast();
  const [serverOk, setServerOk] = useState(null);
  const [serverVersion, setServerVersion] = useState('');
  const [strategies, setStrategies] = useState([]);
  const [stratLoading, setStratLoading] = useState(true);
  const [stratError, setStratError] = useState(null);
  const [selectedId, setSelectedId] = useState(null);

  // Top-level view: 'run' (backtest/sweep) or 'author' (new strategy).
  const [view, setView] = useState('run');

  const [config, setConfig] = useState({
    ticker: 'SPY', timeframe: '1d',
    start: '2010-01-01', end: todayISO(),
    starting_capital: 100_000,
    paramsJson: '{}',
  });

  const [robustness, setRobustness] = useState({
    oos: false, oosSplit: 0.7,
    dsr: false, nTrials: '',
    monteCarlo: false, nShuffles: 1000,
    regimes: false,
  });

  // Optimise (Phase 15.5/15.6). higher_is_better and the heatmap axes are
  // derived at submit time — the metric implies the direction and the
  // grid keys imply the axes — so they're not stored here.
  const [sweepCfg, setSweepCfg] = useState({
    paramGridJson: '{"short":[10,20,50],"long":[100,150,200]}',
    selectionMetric: 'sharpe',
    multiAssetTickers: 'SPY, QQQ, AAPL, GLD',
  });

  const [costConfig, setCostConfig] = useState({ ...DEFAULT_COST_CONFIG });

  // Event-driven (Phase 9) stops. Empty strings on the percent fields
  // mean "stop disabled"; non-empty parses to a percent number.
  const [eventDriven, setEventDriven] = useState({
    enabled: false,
    stop_loss_pct: '',
    take_profit_pct: '',
    trailing_stop_pct: '',
  });

  const [running, setRunning] = useState(false);
  // Synchronous double-submit guard. `disabled=${running}` only blocks the
  // button after React re-renders, leaving a window where a fast double-click
  // fires onRun twice (~20ms-apart pairs were visible in the DB). The ref
  // flips immediately, so the second call bails before any request goes out.
  const runLockRef = useRef(false);
  const [runError, setRunError] = useState(null);
  const [result, setResult] = useState(null);

  // Optimise state. sweepResult is the sweep output shown below the
  // baseline backtest; showOptimise toggles the param-grid panel;
  // optimiseLockRef is the same synchronous double-submit guard as runLockRef.
  const [sweepResult, setSweepResult] = useState(null);
  const [showOptimise, setShowOptimise] = useState(false);
  const [optimising, setOptimising] = useState(false);
  const optimiseLockRef = useRef(false);

  // Pinned runs for comparison mode. Each pin holds the minimum payload
  // needed to overlay: timestamps, equity, metrics, and a display label.
  const [pins, setPins] = useState([]);
  const [comparing, setComparing] = useState(false);

  const handlePin = () => {
    if (!result || pins.length >= PIN_LIMIT) return;
    const label = `${selectedStrategy?.name ?? 'run'} · ${config.ticker} · ${result.run_id.slice(0,6)}`;
    setPins([...pins, {
      pinId: result.run_id,
      label,
      timestamps: result.timestamps,
      equity: result.equity,
      metrics: result.metrics,
      summary: result.summary,
    }]);
  };
  const handleUnpin = (pinId) => {
    const next = pins.filter(p => p.pinId !== pinId);
    setPins(next);
    if (next.length < 2) setComparing(false);
  };
  const alreadyPinned = result && pins.some(p => p.pinId === result.run_id);
  const canPin = !!result && !alreadyPinned && pins.length < PIN_LIMIT;

  useEffect(() => {
    let live = true;
    getHealth()
      .then(h => { if (live) { setServerOk(true); setServerVersion(h.version || ''); } })
      .catch(() => { if (live) setServerOk(false); });
    return () => { live = false; };
  }, []);

  // Deployment notifier (Phase 14). Watermark + browser Notification
  // permission + 30s poller. Manual "Run now" calls acknowledgeRuns to
  // skip the duplicate toast 30s later.
  const lastSeenRunIdRef = useRef(0);
  const baselineReadyRef = useRef(false);
  const acknowledgeRuns = useCallback((latestId) => {
    if (typeof latestId === 'number' && latestId > lastSeenRunIdRef.current) {
      lastSeenRunIdRef.current = latestId;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    // Ask once at startup if the user hasn't decided. Default and denied
    // are both fine — toast still pops; we just won't fire the OS layer.
    if (typeof window !== 'undefined' && 'Notification' in window
        && Notification.permission === 'default') {
      Notification.requestPermission().catch(() => { /* ignore */ });
    }

    // Set the baseline so we don't notify about runs that happened before
    // the user opened the app. After the first tick the baseline is the
    // server's current max id.
    async function baseline() {
      try {
        const body = await getDeploymentRunsFeed(0);
        if (cancelled) return;
        lastSeenRunIdRef.current = body.latest_id ?? 0;
        baselineReadyRef.current = true;
      } catch { /* server may be down on first load — poller will retry */ }
    }

    async function tick() {
      if (!baselineReadyRef.current) {
        await baseline();
        return;
      }
      try {
        const body = await getDeploymentRunsFeed(lastSeenRunIdRef.current);
        if (cancelled) return;
        // Server returns latest-first; reverse so notifications appear
        // in chronological order if multiple fired in one window.
        const rows = (body.runs || []).slice().reverse();
        for (const row of rows) {
          const m = _runMessage(row);
          pushToast(m.kind, m.text);
          if (m.notify) _fireNotification(m.text);
        }
        if (typeof body.latest_id === 'number'
            && body.latest_id > lastSeenRunIdRef.current) {
          lastSeenRunIdRef.current = body.latest_id;
        }
      } catch { /* keep polling on transient errors */ }
    }

    baseline();
    const id = setInterval(tick, NOTIFIER_POLL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [pushToast]);

  const refreshStrategies = () => {
    setStratLoading(true);
    setStratError(null);
    listStrategies()
      .then(list => {
        setStrategies(list);
        setStratLoading(false);
        setSelectedId(prev => prev ?? (list[0]?.id ?? null));
      })
      .catch(err => {
        setStratError(err.message);
        setStratLoading(false);
        pushToast('error', `Could not load strategies: ${err.message}`);
      });
  };
  useEffect(refreshStrategies, []);

  const onToggleFavourite = async (s) => {
    // Optimistic flip so the star responds instantly; refresh re-sorts
    // (favourites pin to the top) and reconciles with the server.
    setStrategies(prev => prev.map(
      x => x.id === s.id ? { ...x, favourite: !x.favourite } : x));
    try {
      await setStrategyFavourite(s.id, !s.favourite);
    } catch (err) {
      setStratError(err.message);
    }
    refreshStrategies();
  };

  const selectedStrategy = useMemo(
    () => strategies.find(s => s.id === selectedId) || null,
    [strategies, selectedId]
  );

  const onRun = async () => {
    if (!selectedId || runLockRef.current) return;
    runLockRef.current = true;
    setRunning(true);
    setRunError(null);
    setResult(null);
    // A fresh backtest invalidates any optimisation built from the old run.
    setSweepResult(null);
    setShowOptimise(false);
    try {
      let params = {};
      try { params = JSON.parse(config.paramsJson || '{}'); } catch { /* use empty */ }

      // Event-driven mode: hits a separate route. Robustness analyses
      // (OOS / MC / DSR / regimes) aren't wired into it.
      if (eventDriven.enabled) {
        const stopPct = (v) => {
          const s = String(v ?? '').trim();
          if (s === '') return null;
          const n = Number(s);
          if (!Number.isFinite(n) || n < 0) return null;
          return n / 100.0;
        };
        const stop_config = {
          stop_loss_pct: stopPct(eventDriven.stop_loss_pct),
          take_profit_pct: stopPct(eventDriven.take_profit_pct),
          trailing_stop_pct: stopPct(eventDriven.trailing_stop_pct),
        };
        const payload = {
          strategy_id: selectedId,
          ticker: config.ticker, timeframe: config.timeframe,
          start: config.start, end: config.end,
          params,
          starting_capital: config.starting_capital,
          cost_config: costConfig,
          stop_config,
        };
        setResult(await runEventDriven(payload));
        return; // finally below clears the running flag.
      }

      const payload = {
        strategy_id: selectedId,
        ticker: config.ticker, timeframe: config.timeframe,
        start: config.start, end: config.end,
        params,
        starting_capital: config.starting_capital,
        cost_config: costConfig,
      };
      if (robustness.oos) payload.oos_split = robustness.oosSplit;
      if (robustness.dsr && robustness.nTrials) payload.n_trials = parseInt(robustness.nTrials, 10);
      if (robustness.monteCarlo) payload.monte_carlo = { n_iterations: robustness.nShuffles };
      if (robustness.regimes) payload.include_regimes = true;

      setResult(await runBacktest(payload));
    } catch (err) {
      setRunError(err.message);
      pushToast('error', `Backtest failed: ${err.message}`);
    } finally {
      setRunning(false);
      runLockRef.current = false;
    }
  };

  // Optimise: sweep the param grid for the current strategy/ticker/window
  // and surface the best combo alongside the baseline backtest above.
  // Anti-overfitting is non-negotiable, so DSR is always requested; the
  // metric implies the direction and the grid keys imply the heatmap axes.
  const onOptimise = async () => {
    if (!selectedId || optimiseLockRef.current) return;

    let paramGrid;
    try {
      paramGrid = JSON.parse(sweepCfg.paramGridJson);
    } catch {
      const msg = 'Param grid JSON is invalid — check the syntax and try again.';
      setRunError(msg);
      pushToast('error', msg);
      return;
    }

    optimiseLockRef.current = true;
    setOptimising(true);
    setRunError(null);
    setSweepResult(null);
    try {
      // Phase 15.6 — Optimise is the full robustness picture: the best
      // combo also gets a regime check and a re-test on other assets.
      const multiAssetTickers = (sweepCfg.multiAssetTickers || '')
        .split(',').map(t => t.trim().toUpperCase()).filter(Boolean);

      const payload = {
        strategy_id: selectedId,
        ticker: config.ticker, timeframe: config.timeframe,
        start: config.start, end: config.end,
        param_grid: paramGrid,
        selection_metric: sweepCfg.selectionMetric,
        higher_is_better: sweepCfg.selectionMetric !== 'max_drawdown',
        heatmaps: gridHeatmapAxes(paramGrid, sweepCfg.selectionMetric),
        include_deflated_sharpe: true,
        include_regimes: true,
        multi_asset_tickers: multiAssetTickers,
        starting_capital: config.starting_capital,
        cost_config: costConfig,
      };
      setSweepResult(await runSweep(payload));
      setShowOptimise(false);
    } catch (err) {
      setRunError(err.message);
      pushToast('error', `Optimise failed: ${err.message}`);
    } finally {
      setOptimising(false);
      optimiseLockRef.current = false;
    }
  };

  // ---- main canvas content ------------------------------------------

  const backtestCanvas = result && html`
    <div class="p-6 space-y-6">
      <${PinStrip} pins=${pins} onUnpin=${handleUnpin}
        onCompare=${() => setComparing(true)} comparing=${comparing} />

      <div>
        <div class="flex items-baseline justify-between">
          <div>
            <div class="text-xs uppercase tracking-wider text-mute-400">Equity curve</div>
            <div class="text-lg text-mute-100 font-medium mt-0.5">
              ${selectedStrategy?.name ?? '—'}
            </div>
          </div>
          <div class="flex items-center gap-3">
            <div class="text-xs text-mute-400 num">run ${result.run_id.slice(0,8)}…</div>
            <button onClick=${handlePin} disabled=${!canPin}
              class="text-xs px-3 py-1 rounded transition
                     ${!canPin
                       ? 'bg-ink-800 text-mute-400 cursor-not-allowed'
                       : 'bg-accent-600 hover:bg-accent-500 text-white font-medium'}">
              ${alreadyPinned ? '✓ Pinned' : pins.length >= PIN_LIMIT ? 'Pin limit reached' : '📌 Pin this run'}
            </button>
          </div>
        </div>
        <div class="mt-3 h-[340px] bg-ink-900 border border-ink-700 rounded overflow-hidden">
          <${EquityChart} timestamps=${result.timestamps} equity=${result.equity}
                          buyHold=${result.buy_hold_equity} />
        </div>
      </div>

      <div>
        <div class="text-xs uppercase tracking-wider text-mute-400 mb-2">Drawdown</div>
        <div class="h-[160px] bg-ink-900 border border-ink-700 rounded overflow-hidden">
          <${DrawdownChart} timestamps=${result.timestamps} drawdown=${result.drawdown} />
        </div>
      </div>

      <div>
        <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">Summary</div>
        <${MetricsRow} result=${result} />
      </div>

      ${result.deflated_sharpe && html`<${DSRCard} dsr=${result.deflated_sharpe} />`}
      ${result.oos && html`<${OOSCompare} oos=${result.oos} />`}
      ${result.monte_carlo && html`<${MCCard} mc=${result.monte_carlo} />`}
      ${result.regimes && html`<${RegimePanel} regimes=${result.regimes} />`}

      <${MonthlyHeatmap} timestamps=${result.timestamps} returns=${result.returns} />

      <${TradeList}
        trades=${result.trades}
        finalEquity=${result.summary?.final_equity}
        startingCapital=${config.starting_capital} />

      <div>
        <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">All metrics</div>
        <${FullMetricsTable} result=${result} />
      </div>

      <div class="border-t border-ink-700 pt-6">
        <${OptimiseSection}
          baseline=${result}
          sweepCfg=${sweepCfg} setSweepCfg=${setSweepCfg}
          sweepResult=${sweepResult}
          optimising=${optimising}
          show=${showOptimise}
          onShow=${() => setShowOptimise(true)}
          onHide=${() => setShowOptimise(false)}
          onOptimise=${onOptimise}
          error=${runError} />
      </div>
    </div>
  `;

  const emptyCanvas = html`
    <div class="h-full flex items-center justify-center text-mute-400 text-sm">
      ${running
        ? html`<span>Running backtest…</span>`
        : runError
          ? html`<span class="text-danger-500 max-w-md text-center">Error: ${runError}</span>`
          : html`<span>Press Run Backtest to see results.</span>`}
    </div>
  `;

  const showCanvas = !!result;

  return html`
    <${AcknowledgeRunsContext.Provider} value=${acknowledgeRuns}>
    <div class="h-full flex flex-col">
      <header class="h-12 shrink-0 border-b border-ink-700 bg-ink-950 flex items-center px-4 gap-4">
        <div class="text-mute-100 font-semibold tracking-tight">Backtester</div>
        <div class="text-mute-400 text-xs">Phase 11 · deploy (Demo)</div>
        <div class="ml-auto flex items-center gap-3 text-xs text-mute-300">
          <button onClick=${() => setView(view === 'deploy' ? 'run' : 'deploy')}
            class="px-3 py-1 rounded transition font-medium
                   ${view === 'deploy'
                     ? 'bg-accent-500 text-ink-950'
                     : 'bg-ink-800 hover:bg-ink-700 text-mute-100'}">
            Deploy
          </button>
          <button onClick=${() => setView(view === 'live' ? 'run' : 'live')}
            class="px-3 py-1 rounded transition font-medium
                   ${view === 'live'
                     ? 'bg-accent-500 text-ink-950'
                     : 'bg-ink-800 hover:bg-ink-700 text-mute-100'}">
            Live
          </button>
          <button onClick=${() => setView(view === 'library' ? 'run' : 'library')}
            class="px-3 py-1 rounded transition font-medium
                   ${view === 'library'
                     ? 'bg-accent-500 text-ink-950'
                     : 'bg-ink-800 hover:bg-ink-700 text-mute-100'}">
            Library
          </button>
          <div class="flex items-center gap-2">
            <${StatusDot} ok=${serverOk === true} />
            <span class="num">
              ${serverOk === null ? 'connecting…'
                : serverOk ? `server ok · v${serverVersion}`
                : 'server unreachable @ 127.0.0.1:8000'}
            </span>
          </div>
        </div>
      </header>

      <div class="flex-1 min-h-0 flex">
        <${Sidebar}
          strategies=${strategies} selectedId=${selectedId} onSelect=${setSelectedId}
          loading=${stratLoading} error=${stratError} onRefresh=${refreshStrategies}
          onNewStrategy=${() => setView('author')} onToggleFavourite=${onToggleFavourite} />

        <main class="flex-1 min-w-0 overflow-y-auto bg-ink-950">
          ${view === 'author'
            ? html`<${AuthorView}
                onCancel=${() => setView('run')}
                onSaved=${async (newId) => {
                  const list = await listStrategies();
                  setStrategies(list);
                  setSelectedId(newId);
                  setView('run');
                }} />`
            : view === 'library'
              ? html`<${LibraryView}
                  onClose=${() => setView('run')}
                  onOpenRun=${async (runId) => {
                    try {
                      const r = await getRun(runId);
                      setResult(r);
                      setSweepResult(null);
                      setShowOptimise(false);
                      setRunError(null);
                      setView('run');
                      if (r.strategy_id) setSelectedId(r.strategy_id);
                    } catch (err) {
                      setRunError(`Failed to load run: ${err.message}`);
                      pushToast('error', `Failed to load run: ${err.message}`);
                      setView('run');
                    }
                  }} />`
              : view === 'live'
                ? html`<${LiveView} onClose=${() => setView('run')} />`
              : view === 'deploy'
                ? html`<${DeployView} strategies=${strategies} onClose=${() => setView('run')} />`
              : comparing
                ? html`<${ComparisonView} pins=${pins} onClose=${() => setComparing(false)} />`
                : html`
                  ${runError && !showCanvas && html`
                    <div class="px-6 pt-6">
                      <div class="bg-danger-500/10 border border-danger-500/30 rounded p-3 text-sm text-danger-500">
                        ${runError}
                      </div>
                    </div>
                  `}
                  ${showCanvas ? backtestCanvas : emptyCanvas}
                `}
        </main>

        ${view === 'run' && !comparing && html`<${RightRail}
          selectedId=${selectedId} config=${config} setConfig=${setConfig}
          robustness=${robustness} setRobustness=${setRobustness}
          costConfig=${costConfig} setCostConfig=${setCostConfig}
          eventDriven=${eventDriven} setEventDriven=${setEventDriven}
          onRun=${onRun} running=${running} />`}
      </div>
    </div>
    </${AcknowledgeRunsContext.Provider}>
  `;
}

const root = createRoot(document.getElementById('app'));
root.render(html`<${ToastProvider}><${App} /></${ToastProvider}>`);
