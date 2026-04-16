"use client";

import { useState, useEffect } from "react";
import { Key, Cpu, Shield, Check, ToggleLeft, ToggleRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface RoverSettings {
  alpaca_key: string;
  alpaca_secret: string;
  place_order: boolean;
  alpaca_paper: boolean;
  reallocate_full: boolean;
  top_allocate_count: number;
  invest_total_pct: number;
  symbols: string;
  extended_hours: boolean;
  use_web_research: boolean;
  openai_model: string;
  news_window_hours: number;
}

function Toggle({
  value,
  onChange,
  label,
  hint,
  danger,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
  danger?: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-4 py-3 border-b border-zinc-800/60 last:border-0">
      <div className="flex-1 min-w-0">
        <p className="text-sm text-zinc-200">{label}</p>
        {hint && <p className="text-xs text-zinc-500 mt-0.5">{hint}</p>}
      </div>
      <button
        onClick={() => onChange(!value)}
        className={cn(
          "shrink-0 transition-colors",
          value
            ? danger
              ? "text-red-400 hover:text-red-300"
              : "text-emerald-400 hover:text-emerald-300"
            : "text-zinc-600 hover:text-zinc-400"
        )}
      >
        {value ? (
          <ToggleRight className="w-7 h-7" />
        ) : (
          <ToggleLeft className="w-7 h-7" />
        )}
      </button>
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="py-3 border-b border-zinc-800/60 last:border-0">
      <label className="text-xs font-medium text-zinc-400 block mb-1.5">{label}</label>
      {children}
      {hint && <p className="text-xs text-zinc-600 mt-1">{hint}</p>}
    </div>
  );
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<RoverSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    fetch("/api/settings")
      .then((r) => r.json())
      .then(setSettings);
  }, []);

  function update<K extends keyof RoverSettings>(key: K, value: RoverSettings[K]) {
    setSettings((prev) => (prev ? { ...prev, [key]: value } : prev));
  }

  async function save() {
    if (!settings) return;
    setSaving(true);
    await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    setSaving(false);
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  }

  if (!settings) {
    return (
      <div className="space-y-4 max-w-2xl">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-xl h-48 animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold text-zinc-100">Settings</h1>
        <p className="text-sm text-zinc-500 mt-0.5">
          Configure your trading engine and risk parameters
        </p>
      </div>

      {/* Alpaca Keys */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-8 h-8 rounded-lg bg-zinc-800 flex items-center justify-center shrink-0">
            <Key className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-zinc-200">Alpaca Brokerage</h2>
            <p className="text-xs text-zinc-500">
              Your own paper or live trading account.{" "}
              <a
                href="https://alpaca.markets"
                target="_blank"
                rel="noopener noreferrer"
                className="text-indigo-400 hover:text-indigo-300"
              >
                Get free keys at alpaca.markets →
              </a>
            </p>
          </div>
        </div>
        <div className="space-y-3">
          <Field label="API Key ID" hint="Starts with PK... (paper) or AK... (live)">
            <input
              type="text"
              value={settings.alpaca_key}
              onChange={(e) => update("alpaca_key", e.target.value)}
              placeholder="PKTJ7S..."
              autoComplete="off"
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-300 font-mono placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/50"
            />
          </Field>
          <Field label="Secret Key">
            <input
              type="password"
              value={settings.alpaca_secret}
              onChange={(e) => update("alpaca_secret", e.target.value)}
              placeholder="••••••••••••••••"
              autoComplete="new-password"
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-300 font-mono placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/50"
            />
          </Field>
        </div>
      </div>

      {/* Trading Engine */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-8 h-8 rounded-lg bg-zinc-800 flex items-center justify-center shrink-0">
            <Cpu className="w-4 h-4 text-zinc-400" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-zinc-200">Trading Engine</h2>
            <p className="text-xs text-zinc-500">Controls bot execution behavior</p>
          </div>
        </div>

        <Toggle
          value={settings.alpaca_paper}
          onChange={(v) => update("alpaca_paper", v)}
          label="Paper Trading Mode"
          hint="Route all orders to Alpaca paper trading — no real capital at risk"
        />
        <Toggle
          value={settings.place_order}
          onChange={(v) => update("place_order", v)}
          label="Place Orders"
          hint="When disabled, the bot evaluates signals but doesn't execute trades"
          danger
        />
        <Toggle
          value={settings.reallocate_full}
          onChange={(v) => update("reallocate_full", v)}
          label="Full Reallocation"
          hint="Liquidate all existing positions before reallocating each run"
          danger
        />
        <Toggle
          value={settings.extended_hours}
          onChange={(v) => update("extended_hours", v)}
          label="Extended Hours Trading"
          hint="Allow pre-market and after-hours order placement"
        />
        <Toggle
          value={settings.use_web_research}
          onChange={(v) => update("use_web_research", v)}
          label="Web Research"
          hint="Fetch live news and analyst ratings during each run (slower)"
        />

        <Field
          label="Investment % per Run"
          hint="Fraction of buying power to deploy each run (0.05 = 5%)"
        >
          <div className="flex items-center bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 gap-1 focus-within:border-indigo-500/50 transition-colors max-w-xs">
            <input
              type="number"
              min={0.001}
              max={1}
              step={0.001}
              value={settings.invest_total_pct}
              onChange={(e) => update("invest_total_pct", parseFloat(e.target.value) || 0.05)}
              className="flex-1 bg-transparent text-sm text-zinc-300 focus:outline-none tabular-nums"
            />
            <span className="text-xs text-zinc-500">
              ({(settings.invest_total_pct * 100).toFixed(1)}%)
            </span>
          </div>
        </Field>

        <Field
          label="Top Symbols to Allocate"
          hint="Number of highest-ranked symbols to spread capital across each run"
        >
          <input
            type="number"
            min={1}
            max={20}
            value={settings.top_allocate_count}
            onChange={(e) => update("top_allocate_count", parseInt(e.target.value) || 3)}
            className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-300 focus:outline-none focus:border-indigo-500/50 max-w-xs w-full"
          />
        </Field>

        <Field
          label="News Window (hours)"
          hint="Hours of news history to scan per analysis run"
        >
          <input
            type="number"
            min={1}
            max={168}
            value={settings.news_window_hours}
            onChange={(e) => update("news_window_hours", parseInt(e.target.value) || 36)}
            className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-300 focus:outline-none focus:border-indigo-500/50 max-w-xs w-full"
          />
        </Field>
      </div>

      {/* Symbols */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-8 h-8 rounded-lg bg-zinc-800 flex items-center justify-center shrink-0">
            <Key className="w-4 h-4 text-zinc-400" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-zinc-200">Symbol Watchlist</h2>
            <p className="text-xs text-zinc-500">Comma-separated tickers the bot analyzes</p>
          </div>
        </div>
        <textarea
          value={settings.symbols}
          onChange={(e) => update("symbols", e.target.value)}
          rows={3}
          className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2.5 text-sm text-zinc-300 font-mono placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/50 transition-colors resize-none"
          placeholder="AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA"
        />
        <p className="text-xs text-zinc-600 mt-1">
          {settings.symbols.split(",").filter(Boolean).length} symbols configured
        </p>
      </div>

      {/* AI Model */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-8 h-8 rounded-lg bg-zinc-800 flex items-center justify-center shrink-0">
            <Cpu className="w-4 h-4 text-violet-400" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-zinc-200">AI Model</h2>
            <p className="text-xs text-zinc-500">Model used for scoring and sentiment analysis</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {["gpt-4o-mini", "gpt-4o", "claude-haiku-4-5-20251001", "claude-sonnet-4-6"].map((m) => (
            <button
              key={m}
              onClick={() => update("openai_model", m)}
              className={cn(
                "text-xs px-3 py-1.5 rounded-lg border transition-colors font-mono",
                settings.openai_model === m
                  ? "bg-indigo-500/20 border-indigo-500/40 text-indigo-300"
                  : "bg-zinc-800 border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200"
              )}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Save */}
      <div className="flex items-center justify-between">
        <p className="text-xs text-zinc-600">
          Settings are saved to your account — Alpaca keys are per-user.
        </p>
        <Button
          onClick={save}
          disabled={saving}
          className={cn(
            "gap-2 text-sm transition-colors",
            saved
              ? "bg-emerald-600 hover:bg-emerald-600 text-white"
              : "bg-indigo-500 hover:bg-indigo-600 text-white"
          )}
        >
          {saved ? (
            <><Check className="w-3.5 h-3.5" />Saved</>
          ) : saving ? (
            "Saving…"
          ) : (
            "Save Settings"
          )}
        </Button>
      </div>

      {/* Paper trading notice */}
      <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-2">
          <Shield className="w-4 h-4 text-amber-400 shrink-0" />
          <p className="text-sm font-medium text-amber-400">
            {settings.alpaca_paper ? "Paper Trading Mode Active" : "Live Trading Mode"}
          </p>
        </div>
        <p className="text-xs text-amber-400/70 leading-relaxed">
          {settings.alpaca_paper
            ? "All orders are routed to Alpaca's paper trading environment. No real capital is at risk."
            : "Live trading is enabled. Orders will execute with real capital. Use caution."}
        </p>
      </div>
    </div>
  );
}
