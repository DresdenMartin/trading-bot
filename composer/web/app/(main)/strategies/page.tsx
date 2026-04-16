"use client";

import { useState, useEffect, useCallback } from "react";
import { Plus, TrendingUp, Trash2, Rocket, PauseCircle, StopCircle, Globe, History, CheckCircle2, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import StrategyBuilder from "@/components/strategy-builder/StrategyBuilder";
import type { Strategy } from "@/components/strategy-builder/types";
import { cn } from "@/lib/utils";

interface DbStrategy {
  id: string;
  name: string;
  description: string;
  logic_json: Strategy;
  is_public: boolean;
  created_at: string;
  deploy_status?: "active" | "paused" | "stopped" | null;
}

interface StrategyRun {
  id: string;
  strategy_id: string;
  status: "completed" | "error";
  signals_fired: number;
  orders_placed: { side: string; symbol: string; shares: number; price: number }[];
  error_msg: string | null;
  created_at: string;
  strategies: { name: string } | null;
}

const STORAGE_KEY = "composer_strategies"; // fallback when Supabase not configured

function isSupabaseConfigured() {
  return Boolean(
    process.env.NEXT_PUBLIC_SUPABASE_URL &&
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  );
}

export default function StrategiesPage() {
  const [strategies, setStrategies] = useState<DbStrategy[]>([]);
  const [building, setBuilding] = useState(false);
  const [editing, setEditing] = useState<DbStrategy | null>(null);
  const [loading, setLoading] = useState(true);
  const [deployingId, setDeployingId] = useState<string | null>(null);
  const [runs, setRuns] = useState<StrategyRun[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);

  const useDb = isSupabaseConfigured();

  // Load strategies
  const fetchStrategies = useCallback(async () => {
    if (!useDb) {
      try {
        const local: Strategy[] = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
        setStrategies(
          local.map((s, i) => ({
            id: String(i),
            name: s.name,
            description: s.description,
            logic_json: s,
            is_public: false,
            created_at: new Date().toISOString(),
          }))
        );
      } catch {}
      setLoading(false);
      return;
    }

    try {
      const [stratRes, deployRes] = await Promise.all([
        fetch("/api/strategies"),
        fetch("/api/active-strategies").catch(() => null),
      ]);
      const data: DbStrategy[] = stratRes.ok ? await stratRes.json() : [];
      const deploys = deployRes?.ok ? await deployRes.json() : [];
      const deployMap = Object.fromEntries(
        (deploys as { strategy_id: string; status: string }[]).map((d) => [d.strategy_id, d.status])
      );
      setStrategies(
        data.map((s) => ({
          ...s,
          deploy_status: (deployMap[s.id] as DbStrategy["deploy_status"]) ?? null,
        }))
      );
    } catch {}
    setLoading(false);
  }, [useDb]);

  useEffect(() => { fetchStrategies(); }, [fetchStrategies]);

  const fetchRuns = useCallback(async () => {
    if (!useDb) return;
    setRunsLoading(true);
    try {
      const res = await fetch("/api/strategy-runs");
      if (res.ok) setRuns(await res.json());
    } catch {}
    setRunsLoading(false);
  }, [useDb]);

  useEffect(() => { fetchRuns(); }, [fetchRuns]);

  // Load pending strategy from Rover AI chat
  useEffect(() => {
    try {
      const pending = localStorage.getItem("rover_pending_strategy");
      if (pending) {
        const parsed: Strategy = JSON.parse(pending);
        localStorage.removeItem("rover_pending_strategy");
        setEditing({
          id: "",
          name: parsed.name,
          description: parsed.description,
          logic_json: parsed,
          is_public: false,
          created_at: new Date().toISOString(),
        });
        setBuilding(true);
      }
    } catch {}
  }, []);

  async function handleSave(s: Strategy) {
    if (!useDb) {
      // localStorage fallback
      const local: Strategy[] = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
      const updated = editing
        ? local.map((x) => (x.name === editing.name ? s : x))
        : [...local, s];
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } else if (editing?.id) {
      await fetch(`/api/strategies/${editing.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: s.name, description: s.description, logic_json: s }),
      });
    } else {
      await fetch("/api/strategies", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: s.name, description: s.description, logic_json: s }),
      });
    }
    setBuilding(false);
    setEditing(null);
    fetchStrategies();
  }

  async function handleDelete(s: DbStrategy) {
    if (!useDb) {
      const local: Strategy[] = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
      localStorage.setItem(STORAGE_KEY, JSON.stringify(local.filter((x) => x.name !== s.name)));
    } else {
      await fetch(`/api/strategies/${s.id}`, { method: "DELETE" });
    }
    fetchStrategies();
  }

  async function togglePublic(s: DbStrategy) {
    if (!useDb) return;
    await fetch(`/api/strategies/${s.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ is_public: !s.is_public }),
    });
    fetchStrategies();
  }

  async function handleDeploy(s: DbStrategy) {
    if (!useDb) return;
    setDeployingId(s.id);
    await fetch(`/api/strategies/${s.id}/deploy`, { method: "POST" });
    setDeployingId(null);
    fetchStrategies();
  }

  async function handleDeployStatus(s: DbStrategy, status: "active" | "paused" | "stopped") {
    if (!useDb) return;
    await fetch(`/api/strategies/${s.id}/deploy`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
    fetchStrategies();
  }

  const symbols = (s: DbStrategy) =>
    [
      ...new Set(
        s.logic_json.rules.flatMap((r) =>
          [...r.thenActions, ...r.elseActions]
            .filter((a) => a.type !== "hold_cash" && a.asset)
            .map((a) => a.asset)
        )
      ),
    ].slice(0, 5);

  return (
    <div className="space-y-6 max-w-7xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-100">Strategies</h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            Build and manage your automated trading symphonies
          </p>
        </div>
        {!building && (
          <Button
            onClick={() => { setEditing(null); setBuilding(true); }}
            className="bg-indigo-500 hover:bg-indigo-600 text-white gap-2 text-sm"
          >
            <Plus className="w-4 h-4" />
            New Strategy
          </Button>
        )}
      </div>

      {/* Saved strategy cards */}
      {!building && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {loading
            ? Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 animate-pulse h-40" />
              ))
            : strategies.map((s) => (
                <div
                  key={s.id}
                  className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 hover:border-zinc-700 transition-colors group flex flex-col"
                >
                  {/* Header */}
                  <div className="flex items-start justify-between mb-2">
                    <div
                      className="flex items-center gap-2.5 cursor-pointer flex-1 min-w-0"
                      onClick={() => { setEditing(s); setBuilding(true); }}
                    >
                      <div className="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center shrink-0">
                        <TrendingUp className="w-4 h-4 text-indigo-400" />
                      </div>
                      <h3 className="text-sm font-semibold text-zinc-100 group-hover:text-white transition-colors truncate">
                        {s.name}
                      </h3>
                    </div>
                    <div className="flex items-center gap-1 ml-2">
                      {s.deploy_status === "active" && (
                        <Badge className="text-[10px] bg-emerald-500/10 text-emerald-400 border-emerald-500/20">
                          Live
                        </Badge>
                      )}
                      {s.is_public && (
                        <span title="Public">
                          <Globe className="w-3 h-3 text-zinc-500" />
                        </span>
                      )}
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDelete(s); }}
                        className="text-zinc-700 hover:text-red-400 transition-colors ml-1"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  <p className="text-xs text-zinc-500 mb-3 line-clamp-2 flex-1">
                    {s.description || `${s.logic_json.rules.length} rule${s.logic_json.rules.length !== 1 ? "s" : ""}`}
                  </p>

                  <div className="flex gap-1 flex-wrap mb-3">
                    {symbols(s).map((sym) => (
                      <span key={sym} className="text-xs bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded-md font-medium">
                        {sym}
                      </span>
                    ))}
                  </div>

                  {/* Action row */}
                  {useDb && (
                    <div className="flex items-center gap-1.5 pt-2 border-t border-zinc-800/60">
                      {!s.deploy_status || s.deploy_status === "stopped" ? (
                        <button
                          onClick={() => handleDeploy(s)}
                          disabled={deployingId === s.id}
                          className="flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
                        >
                          <Rocket className="w-3 h-3" />
                          {deployingId === s.id ? "Deploying…" : "Deploy"}
                        </button>
                      ) : (
                        <>
                          {s.deploy_status === "active" ? (
                            <button
                              onClick={() => handleDeployStatus(s, "paused")}
                              className="flex items-center gap-1 text-xs text-amber-400 hover:text-amber-300 transition-colors"
                            >
                              <PauseCircle className="w-3 h-3" />
                              Pause
                            </button>
                          ) : (
                            <button
                              onClick={() => handleDeployStatus(s, "active")}
                              className="flex items-center gap-1 text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
                            >
                              <Rocket className="w-3 h-3" />
                              Resume
                            </button>
                          )}
                          <button
                            onClick={() => handleDeployStatus(s, "stopped")}
                            className="flex items-center gap-1 text-xs text-zinc-500 hover:text-red-400 transition-colors ml-auto"
                          >
                            <StopCircle className="w-3 h-3" />
                            Stop
                          </button>
                        </>
                      )}
                      <button
                        onClick={() => togglePublic(s)}
                        className={cn(
                          "text-xs transition-colors ml-auto",
                          s.deploy_status && s.deploy_status !== "stopped" ? "" : "ml-auto",
                          s.is_public
                            ? "text-emerald-400 hover:text-zinc-400"
                            : "text-zinc-600 hover:text-zinc-400"
                        )}
                        title={s.is_public ? "Make private" : "Share publicly"}
                      >
                        {s.is_public ? "Public" : "Private"}
                      </button>
                    </div>
                  )}
                </div>
              ))}

          {/* New strategy card */}
          {!loading && (
            <button
              onClick={() => { setEditing(null); setBuilding(true); }}
              className="bg-zinc-900/40 border border-dashed border-zinc-800 rounded-xl p-5 flex flex-col items-center justify-center gap-3 cursor-pointer hover:border-zinc-700 hover:bg-zinc-900 transition-colors min-h-[160px]"
            >
              <div className="w-10 h-10 rounded-xl bg-zinc-800 flex items-center justify-center">
                <Plus className="w-5 h-5 text-zinc-500" />
              </div>
              <div className="text-center">
                <p className="text-sm font-medium text-zinc-400">New strategy</p>
                <p className="text-xs text-zinc-600 mt-0.5">Use the visual builder</p>
              </div>
            </button>
          )}
        </div>
      )}

      {/* Builder panel */}
      {building && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="text-sm font-semibold text-zinc-300">
                {editing ? "Edit Strategy" : "Strategy Builder"}
              </h2>
              <p className="text-xs text-zinc-500 mt-0.5">
                Define conditions and actions, or generate with AI
              </p>
            </div>
            <button
              onClick={() => { setBuilding(false); setEditing(null); }}
              className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              Cancel
            </button>
          </div>
          <StrategyBuilder
            initial={editing?.logic_json}
            onSave={handleSave}
          />
        </div>
      )}

      {/* Run History */}
      {!building && useDb && (runs.length > 0 || runsLoading) && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-4">
            <History className="w-4 h-4 text-zinc-500" />
            <h2 className="text-sm font-semibold text-zinc-300">Recent Runs</h2>
            <span className="text-xs text-zinc-600 ml-auto">
              Last {Math.min(runs.length, 20)} of {runs.length}
            </span>
          </div>
          {runsLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="h-10 bg-zinc-800 rounded-lg animate-pulse" />
              ))}
            </div>
          ) : (
            <div className="overflow-x-auto -mx-1">
              <table className="w-full text-xs min-w-[500px]">
                <thead>
                  <tr className="border-b border-zinc-800">
                    {["Strategy", "Status", "Signals", "Orders", "Time"].map((col) => (
                      <th key={col} className="text-left font-medium text-zinc-500 pb-2 pr-4 last:pr-0">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/50">
                  {runs.slice(0, 20).map((r) => (
                    <tr key={r.id} className="hover:bg-zinc-800/30 transition-colors">
                      <td className="py-2.5 pr-4 font-medium text-zinc-200">
                        {r.strategies?.name ?? "—"}
                      </td>
                      <td className="py-2.5 pr-4">
                        {r.status === "completed" ? (
                          <span className="flex items-center gap-1 text-emerald-400">
                            <CheckCircle2 className="w-3 h-3" />
                            Completed
                          </span>
                        ) : (
                          <span className="flex items-center gap-1 text-red-400">
                            <XCircle className="w-3 h-3" />
                            Error
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 pr-4 text-zinc-400 tabular-nums">
                        {r.signals_fired}
                      </td>
                      <td className="py-2.5 pr-4 text-zinc-400">
                        {r.orders_placed?.length > 0 ? (
                          <span className="text-zinc-300">
                            {r.orders_placed.map((o) => `${o.side} ${o.symbol}`).join(", ")}
                          </span>
                        ) : (
                          <span className="text-zinc-600">No orders</span>
                        )}
                      </td>
                      <td className="py-2.5 text-zinc-500">
                        {new Date(r.created_at).toLocaleString("en-US", {
                          month: "short", day: "numeric",
                          hour: "numeric", minute: "2-digit",
                        })}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!building && !loading && strategies.length === 0 && (
        <div className="bg-zinc-900 border border-dashed border-zinc-800 rounded-xl p-12 flex flex-col items-center gap-4">
          <TrendingUp className="w-10 h-10 text-zinc-700" />
          <div className="text-center">
            <p className="text-sm font-medium text-zinc-400">No strategies yet</p>
            <p className="text-xs text-zinc-600 mt-1">
              Click &quot;New Strategy&quot; or describe one in plain English
            </p>
          </div>
          <Button
            onClick={() => setBuilding(true)}
            className="bg-indigo-500 hover:bg-indigo-600 text-white gap-2 text-sm mt-2"
          >
            <Plus className="w-4 h-4" />
            Build your first strategy
          </Button>
        </div>
      )}
    </div>
  );
}
