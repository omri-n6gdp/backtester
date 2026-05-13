// app.js — Phase 5.8: robustness dashboard
//
// Two run modes:
//   Backtest — single run with optional OOS / DSR / Monte Carlo / regimes
//   Sweep    — parameter grid sweep with heatmap, per-combo table, DSR

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import htm from 'htm';

import { getHealth, listStrategies, runBacktest, runSweep } from './api.js';

const html = htm.bind(React.createElement);
const LWC = window.LightweightCharts;

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

function EquityChart({ timestamps, equity, title }) {
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
      grid: {
        vertLines: { color: '#181c24' },
        horzLines: { color: '#181c24' },
      },
      rightPriceScale: { borderColor: '#222732' },
      timeScale: { borderColor: '#222732', timeVisible: false, secondsVisible: false },
      crosshair: { mode: 1 },
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
    return () => { chart.remove(); chartRef.current = null; seriesRef.current = null; };
  }, []);

  useEffect(() => {
    if (!seriesRef.current || !timestamps || !equity) return;
    const data = [];
    let lastTime = -1;
    for (let i = 0; i < timestamps.length; i++) {
      const v = equity[i];
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      const t = isoToUnixSeconds(timestamps[i]);
      if (t <= lastTime) continue;
      lastTime = t;
      data.push({ time: t, value: Number(v) });
    }
    seriesRef.current.setData(data);
    chartRef.current && chartRef.current.timeScale().fitContent();
  }, [timestamps, equity]);

  return html`<div ref=${containerRef} class="w-full h-full"></div>`;
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

function Sidebar({ strategies, selectedId, onSelect, loading, error, onRefresh }) {
  return html`
    <aside class="w-72 shrink-0 border-r border-ink-700 bg-ink-900 flex flex-col">
      <div class="px-4 py-3 border-b border-ink-700 flex items-center justify-between">
        <div class="text-xs uppercase tracking-wider text-mute-400">Strategies</div>
        <button onClick=${onRefresh} class="text-xs text-mute-300 hover:text-mute-100 transition" title="Refresh">↻</button>
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
              <li key=${s.id}>
                <button onClick=${() => onSelect(s.id)} class="w-full text-left px-4 py-3 ${cls} transition">
                  <div class="text-sm text-mute-100 font-medium truncate">${s.name}</div>
                  ${s.description && html`<div class="text-xs text-mute-400 truncate mt-0.5">${s.description}</div>`}
                  <div class="text-[10px] text-mute-400 num mt-1">${s.id.slice(0,8)}…</div>
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
                     sweepCfg, setSweepCfg, runMode, setRunMode, onRun, running }) {
  const disabled = !selectedId || running;
  const upd = (k, v) => setConfig(c => ({ ...c, [k]: v }));
  const updR = (k, v) => setRobustness(r => ({ ...r, [k]: v }));
  const updS = (k, v) => setSweepCfg(s => ({ ...s, [k]: v }));

  const ic = inputCls();

  return html`
    <aside class="w-80 shrink-0 border-l border-ink-700 bg-ink-900 flex flex-col">
      <div class="px-4 py-3 border-b border-ink-700">
        <div class="text-[10px] uppercase tracking-wider text-mute-400 mb-2">Mode</div>
        <div class="flex rounded overflow-hidden border border-ink-700">
          ${['backtest','sweep'].map(mode => html`
            <button key=${mode} onClick=${() => setRunMode(mode)}
              class="flex-1 py-1.5 text-xs font-semibold transition
                     ${runMode === mode
                       ? 'bg-accent-600 text-white'
                       : 'bg-ink-850 text-mute-300 hover:bg-ink-800'}">
              ${mode === 'backtest' ? 'Backtest' : 'Sweep'}
            </button>
          `)}
        </div>
      </div>

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

        ${runMode === 'backtest' && html`
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
        `}

        ${runMode === 'sweep' && html`
          <${Field} label="Param grid (JSON)">
            <textarea rows="4"
              class="w-full bg-ink-850 border border-ink-700 rounded px-2 py-1.5 text-xs num text-mute-100 focus:outline-none focus:border-accent-500 font-mono resize-none"
              value=${sweepCfg.paramGridJson}
              onChange=${e => updS('paramGridJson', e.target.value)} />
          </${Field}>

          <${Field} label="Selection metric">
            <select class=${ic} value=${sweepCfg.selectionMetric}
              onChange=${e => updS('selectionMetric', e.target.value)}>
              <option value="sharpe">Sharpe</option>
              <option value="cagr">CAGR</option>
              <option value="calmar">Calmar</option>
              <option value="sortino">Sortino</option>
              <option value="max_drawdown">Max drawdown (lower=better)</option>
            </select>
          </${Field}>

          <div class="flex items-center gap-2">
            <input type="checkbox" id="hib-cb" class="accent-accent-500"
              checked=${sweepCfg.higherIsBetter}
              onChange=${e => updS('higherIsBetter', e.target.checked)} />
            <label for="hib-cb" class="text-xs text-mute-200">Higher metric = better</label>
          </div>

          <div class="pt-1 space-y-2">
            <div class="text-[10px] uppercase tracking-wider text-mute-400">Heatmap axes</div>
            <div class="grid grid-cols-3 gap-2">
              <${Field} label="X param">
                <input class="w-full bg-ink-850 border border-ink-700 rounded px-1.5 py-1 text-xs num text-mute-100 focus:outline-none focus:border-accent-500"
                  value=${sweepCfg.xParam} onChange=${e => updS('xParam', e.target.value)} />
              </${Field}>
              <${Field} label="Y param">
                <input class="w-full bg-ink-850 border border-ink-700 rounded px-1.5 py-1 text-xs num text-mute-100 focus:outline-none focus:border-accent-500"
                  value=${sweepCfg.yParam} onChange=${e => updS('yParam', e.target.value)} />
              </${Field}>
              <${Field} label="Metric">
                <input class="w-full bg-ink-850 border border-ink-700 rounded px-1.5 py-1 text-xs num text-mute-100 focus:outline-none focus:border-accent-500"
                  value=${sweepCfg.hmMetric} onChange=${e => updS('hmMetric', e.target.value)} />
              </${Field}>
            </div>
          </div>

          <div class="flex items-center gap-2">
            <input type="checkbox" id="dsr-sweep-cb" class="accent-accent-500"
              checked=${sweepCfg.includeDsr}
              onChange=${e => updS('includeDsr', e.target.checked)} />
            <label for="dsr-sweep-cb" class="text-xs text-mute-200">Deflated Sharpe (DSR)</label>
          </div>
        `}

        <!-- Static cost display -->
        <div class="pt-1">
          <div class="text-[10px] uppercase tracking-wider text-mute-400 mb-1">Costs (defaults)</div>
          <div class="grid grid-cols-2 gap-x-3 gap-y-1 text-xs num text-mute-300">
            <div class="flex justify-between"><span class="text-mute-400">Slippage</span><span>5 bps</span></div>
            <div class="flex justify-between"><span class="text-mute-400">Commission</span><span>0 bps</span></div>
            <div class="flex justify-between"><span class="text-mute-400">FX fee</span><span>15 bps</span></div>
            <div class="flex justify-between"><span class="text-mute-400">Borrow</span><span>0 bps</span></div>
          </div>
        </div>
      </div>

      <div class="p-4 border-t border-ink-700">
        <button onClick=${onRun} disabled=${disabled}
          class="w-full px-3 py-2.5 rounded font-semibold text-sm transition
                 ${disabled
                   ? 'bg-ink-700 text-mute-400 cursor-not-allowed'
                   : 'bg-accent-500 hover:bg-accent-400 text-ink-950'}">
          ${running
            ? (runMode === 'sweep' ? 'Sweeping…' : 'Running…')
            : (runMode === 'sweep' ? 'Run Sweep' : 'Run Backtest')}
        </button>
        ${!selectedId && html`<div class="text-xs text-mute-400 mt-2 text-center">Pick a strategy on the left.</div>`}
      </div>
    </aside>
  `;
}

// ---- App -----------------------------------------------------------

function App() {
  const [serverOk, setServerOk] = useState(null);
  const [serverVersion, setServerVersion] = useState('');
  const [strategies, setStrategies] = useState([]);
  const [stratLoading, setStratLoading] = useState(true);
  const [stratError, setStratError] = useState(null);
  const [selectedId, setSelectedId] = useState(null);

  const [runMode, setRunMode] = useState('backtest');

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

  const [sweepCfg, setSweepCfg] = useState({
    paramGridJson: '{"short":[10,20,50],"long":[100,150,200]}',
    selectionMetric: 'sharpe',
    higherIsBetter: true,
    xParam: 'short',
    yParam: 'long',
    hmMetric: 'sharpe',
    includeDsr: true,
  });

  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState(null);
  const [result, setResult] = useState(null);
  const [sweepResult, setSweepResult] = useState(null);

  useEffect(() => {
    let live = true;
    getHealth()
      .then(h => { if (live) { setServerOk(true); setServerVersion(h.version || ''); } })
      .catch(() => { if (live) setServerOk(false); });
    return () => { live = false; };
  }, []);

  const refreshStrategies = () => {
    setStratLoading(true);
    setStratError(null);
    listStrategies()
      .then(list => {
        setStrategies(list);
        setStratLoading(false);
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

    if (runMode === 'backtest') {
      setResult(null);
      try {
        let params = {};
        try { params = JSON.parse(config.paramsJson || '{}'); } catch { /* use empty */ }

        const payload = {
          strategy_id: selectedId,
          ticker: config.ticker, timeframe: config.timeframe,
          start: config.start, end: config.end,
          params,
          starting_capital: config.starting_capital,
        };
        if (robustness.oos) payload.oos_split = robustness.oosSplit;
        if (robustness.dsr && robustness.nTrials) payload.n_trials = parseInt(robustness.nTrials, 10);
        if (robustness.monteCarlo) payload.monte_carlo = { n_iterations: robustness.nShuffles };
        if (robustness.regimes) payload.include_regimes = true;

        setResult(await runBacktest(payload));
      } catch (err) {
        setRunError(err.message);
      } finally {
        setRunning(false);
      }
    } else {
      setSweepResult(null);
      let paramGrid;
      try {
        paramGrid = JSON.parse(sweepCfg.paramGridJson);
      } catch {
        setRunError('Param grid JSON is invalid — check the syntax and try again.');
        setRunning(false);
        return;
      }

      try {
        const heatmaps =
          sweepCfg.xParam && sweepCfg.yParam && sweepCfg.hmMetric
            ? [{ x_param: sweepCfg.xParam, y_param: sweepCfg.yParam, metric_name: sweepCfg.hmMetric }]
            : [];

        const payload = {
          strategy_id: selectedId,
          ticker: config.ticker, timeframe: config.timeframe,
          start: config.start, end: config.end,
          param_grid: paramGrid,
          selection_metric: sweepCfg.selectionMetric,
          higher_is_better: sweepCfg.higherIsBetter,
          heatmaps,
          include_deflated_sharpe: sweepCfg.includeDsr,
          starting_capital: config.starting_capital,
        };

        setSweepResult(await runSweep(payload));
      } catch (err) {
        setRunError(err.message);
      } finally {
        setRunning(false);
      }
    }
  };

  // ---- main canvas content ------------------------------------------

  const backtestCanvas = result && runMode === 'backtest' && html`
    <div class="p-6 space-y-6">
      <div>
        <div class="flex items-baseline justify-between">
          <div>
            <div class="text-xs uppercase tracking-wider text-mute-400">Equity curve</div>
            <div class="text-lg text-mute-100 font-medium mt-0.5">
              ${selectedStrategy?.name ?? '—'}
            </div>
          </div>
          <div class="text-xs text-mute-400 num">run ${result.run_id.slice(0,8)}…</div>
        </div>
        <div class="mt-3 h-[380px] bg-ink-900 border border-ink-700 rounded overflow-hidden">
          <${EquityChart} timestamps=${result.timestamps} equity=${result.equity} />
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

      <div>
        <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">All metrics</div>
        <${FullMetricsTable} result=${result} />
      </div>
    </div>
  `;

  const sweepCanvas = sweepResult && runMode === 'sweep' && html`
    <div class="p-6 space-y-6">
      <div class="flex items-baseline justify-between">
        <div>
          <div class="text-xs uppercase tracking-wider text-mute-400">Parameter sweep</div>
          <div class="text-lg text-mute-100 font-medium mt-0.5">
            ${selectedStrategy?.name ?? '—'}
            <span class="ml-2 text-sm font-normal text-mute-400 num">
              ${sweepResult.n_combinations} combos · best index ${sweepResult.best_index ?? '—'}
            </span>
          </div>
        </div>
        <div class="text-xs text-mute-400 num">run ${sweepResult.run_id?.slice(0,8)}…</div>
      </div>

      ${sweepResult.heatmaps?.length > 0 && html`
        <div class="bg-ink-900 border border-ink-700 rounded p-4">
          <div class="text-xs uppercase tracking-wider text-mute-400 mb-4">Parameter heatmap</div>
          ${sweepResult.heatmaps.map((hm, i) => html`
            <div key=${i}><${HeatmapGrid} heatmap=${hm} /></div>
          `)}
        </div>
      `}

      ${sweepResult.deflated_sharpe && html`<${DSRCard} dsr=${sweepResult.deflated_sharpe} />`}

      <div class="bg-ink-900 border border-ink-700 rounded p-4">
        <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">
          All combos · sorted by ${sweepResult.selection_metric.replace(/_/g,' ')}
          ${sweepResult.higher_is_better ? '(↓ best first)' : '(↑ best first)'}
        </div>
        <${ComboTable}
          combos=${sweepResult.combos}
          bestIndex=${sweepResult.best_index}
          selectionMetric=${sweepResult.selection_metric}
          higherIsBetter=${sweepResult.higher_is_better} />
      </div>

      ${sweepResult.best_combo_timestamps && sweepResult.best_combo_equity && html`
        <div>
          <div class="text-xs uppercase tracking-wider text-mute-400 mb-3">
            Best combo equity curve
            ${sweepResult.best_index !== null && sweepResult.combos[sweepResult.best_index]
              ? html` <span class="ml-2 normal-case font-normal text-mute-400 num">
                  (${Object.entries(sweepResult.combos.find(c => c.index === sweepResult.best_index)?.params ?? {})
                    .map(([k,v]) => `${k}=${v}`).join(', ')})
                </span>`
              : null}
          </div>
          <div class="h-[340px] bg-ink-900 border border-ink-700 rounded overflow-hidden">
            <${EquityChart}
              timestamps=${sweepResult.best_combo_timestamps}
              equity=${sweepResult.best_combo_equity} />
          </div>
        </div>
      `}
    </div>
  `;

  const emptyCanvas = html`
    <div class="h-full flex items-center justify-center text-mute-400 text-sm">
      ${running
        ? html`<span>${runMode === 'sweep' ? 'Running sweep…' : 'Running backtest…'}</span>`
        : runError
          ? html`<span class="text-danger-500 max-w-md text-center">Error: ${runError}</span>`
          : html`<span>Press ${runMode === 'sweep' ? 'Run Sweep' : 'Run Backtest'} to see results.</span>`}
    </div>
  `;

  const showCanvas = runMode === 'backtest' ? !!result : !!sweepResult;

  return html`
    <div class="h-full flex flex-col">
      <header class="h-12 shrink-0 border-b border-ink-700 bg-ink-950 flex items-center px-4 gap-4">
        <div class="text-mute-100 font-semibold tracking-tight">Backtester</div>
        <div class="text-mute-400 text-xs">Phase 5 · robustness dashboard</div>
        <div class="ml-auto flex items-center gap-2 text-xs text-mute-300">
          <${StatusDot} ok=${serverOk === true} />
          <span class="num">
            ${serverOk === null ? 'connecting…'
              : serverOk ? `server ok · v${serverVersion}`
              : 'server unreachable @ 127.0.0.1:8000'}
          </span>
        </div>
      </header>

      <div class="flex-1 min-h-0 flex">
        <${Sidebar}
          strategies=${strategies} selectedId=${selectedId} onSelect=${setSelectedId}
          loading=${stratLoading} error=${stratError} onRefresh=${refreshStrategies} />

        <main class="flex-1 min-w-0 overflow-y-auto bg-ink-950">
          ${runError && !showCanvas && html`
            <div class="px-6 pt-6">
              <div class="bg-danger-500/10 border border-danger-500/30 rounded p-3 text-sm text-danger-500">
                ${runError}
              </div>
            </div>
          `}
          ${showCanvas ? (runMode === 'backtest' ? backtestCanvas : sweepCanvas) : emptyCanvas}
        </main>

        <${RightRail}
          selectedId=${selectedId} config=${config} setConfig=${setConfig}
          robustness=${robustness} setRobustness=${setRobustness}
          sweepCfg=${sweepCfg} setSweepCfg=${setSweepCfg}
          runMode=${runMode} setRunMode=${setRunMode}
          onRun=${onRun} running=${running} />
      </div>
    </div>
  `;
}

const root = createRoot(document.getElementById('app'));
root.render(html`<${App} />`);
