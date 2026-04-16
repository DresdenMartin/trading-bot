"use client";

import { useState, useEffect } from "react";
import { Search, TrendingUp, Users, Copy, CheckCircle } from "lucide-react";

interface PublicStrategy {
  id: string;
  name: string;
  description: string;
  logic_json: {
    rules: {
      thenActions: { type: string; asset: string }[];
      elseActions: { type: string; asset: string }[];
    }[];
  };
  created_at: string;
  user_id: string;
}

export default function CommunityPage() {
  const [strategies, setStrategies] = useState<PublicStrategy[]>([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [cloning, setCloning] = useState<string | null>(null);
  const [cloned, setCloned] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/community")
      .then((r) => r.json())
      .then((data) => {
        setStrategies(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  async function handleClone(id: string) {
    setCloning(id);
    const res = await fetch("/api/community", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ strategy_id: id }),
    });
    setCloning(null);
    if (res.ok) {
      setCloned(id);
      setTimeout(() => setCloned(null), 3000);
    }
  }

  const filtered = strategies.filter(
    (s) =>
      !search ||
      s.name.toLowerCase().includes(search.toLowerCase()) ||
      s.description?.toLowerCase().includes(search.toLowerCase())
  );

  function getSymbols(s: PublicStrategy) {
    return [
      ...new Set(
        s.logic_json.rules.flatMap((r) =>
          [...(r.thenActions ?? []), ...(r.elseActions ?? [])]
            .filter((a) => a.type !== "hold_cash" && a.asset)
            .map((a) => a.asset)
        )
      ),
    ].slice(0, 4);
  }

  function handle(userId: string) {
    return userId.slice(0, 8);
  }

  return (
    <div className="space-y-6 max-w-7xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-100">Community</h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            Discover and clone public strategies
          </p>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-zinc-500">
          <Users className="w-3.5 h-3.5" />
          <span>{filtered.length} strategies</span>
        </div>
      </div>

      {/* Search */}
      <div className="relative max-w-sm">
        <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500 pointer-events-none" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search strategies…"
          className="w-full bg-zinc-900 border border-zinc-800 rounded-lg pl-10 pr-4 py-2 text-sm text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/50 transition-colors"
        />
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 animate-pulse h-44" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="bg-zinc-900 border border-dashed border-zinc-800 rounded-xl p-12 flex flex-col items-center gap-3">
          <TrendingUp className="w-10 h-10 text-zinc-700" />
          <p className="text-sm text-zinc-500">
            {search ? "No strategies match your search" : "No public strategies yet — be the first to share one!"}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {filtered.map((s) => {
            const syms = getSymbols(s);
            const isCloned = cloned === s.id;
            const isCloning = cloning === s.id;
            return (
              <div
                key={s.id}
                className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 hover:border-zinc-700 transition-colors flex flex-col"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="min-w-0">
                    <h3 className="text-sm font-semibold text-zinc-100 truncate">
                      {s.name}
                    </h3>
                    <p className="text-xs text-zinc-600 mt-0.5">
                      by @{handle(s.user_id)}
                    </p>
                  </div>
                  <p className="text-xs text-zinc-600 shrink-0 ml-2">
                    {new Date(s.created_at).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                    })}
                  </p>
                </div>

                <p className="text-xs text-zinc-400 mb-3 line-clamp-2 flex-1">
                  {s.description ||
                    `${s.logic_json.rules.length} rule${s.logic_json.rules.length !== 1 ? "s" : ""}`}
                </p>

                <div className="flex gap-1 flex-wrap mb-3">
                  {syms.map((sym) => (
                    <span
                      key={sym}
                      className="text-xs bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded-md font-medium"
                    >
                      {sym}
                    </span>
                  ))}
                </div>

                <div className="flex items-center justify-end pt-2 border-t border-zinc-800/60">
                  <button
                    onClick={() => handleClone(s.id)}
                    disabled={isCloning || isCloned}
                    className="flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 transition-colors font-medium disabled:opacity-60"
                  >
                    {isCloned ? (
                      <>
                        <CheckCircle className="w-3 h-3 text-emerald-400" />
                        <span className="text-emerald-400">Cloned!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3" />
                        {isCloning ? "Cloning…" : "Clone →"}
                      </>
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
