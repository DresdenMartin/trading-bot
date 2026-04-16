// Shared TypeScript types for the Composer trading platform.

// ── Market data ──────────────────────────────────────────────────────────────

export interface SymbolQuote {
  ticker: string;
  name?: string;
  sector?: string;
  price: number | null;
  change: number | null;
  changePercent: number | null;
  volume: number | null;
  timestamp: string | null;
}

// ── Portfolio ────────────────────────────────────────────────────────────────

export interface PortfolioMetrics {
  equity: number;
  cash: number;
  buyingPower: number;
  dayPL: number;
  dayPLPercent: number;
  totalPL: number;
  totalPLPercent: number;
}

export interface PortfolioSnapshot {
  ts: number;
  timestamp: string;
  equity: number | null;
  cash: number | null;
  positions: PortfolioPosition[];
}

export interface PortfolioPosition {
  symbol: string;
  qty: number;
  marketValue: number;
  unrealizedPL: number;
  unrealizedPLPercent: number;
  currentPrice: number;
  avgEntryPrice: number;
}

// ── Strategies ───────────────────────────────────────────────────────────────

export interface Strategy {
  id: string;
  name: string;
  description: string;
  status: "active" | "paused" | "draft";
  symbols: string[];
  createdAt: string;
  updatedAt: string;
  returnPct?: number;
  sharpe?: number;
}

// ── Scoring / Analysis ───────────────────────────────────────────────────────

export interface StrategyScore {
  symbol: string;
  score: number;
  scoreRaw: number;
  confidence: number;
  decision: "buy" | "sell" | "hold";
  rank: number;
  rationale: string;
  rationale_bullets: string[];
  details: {
    news_sent: number;
    news_count: number;
    news_confidence: number;
    top_catalysts: string[];
    latest_price: number;
    indicators: Record<string, number | null>;
  };
  analysis?: {
    summary: string;
    suggested_action: "buy" | "sell" | "hold" | null;
    confidence: number | null;
    rationale: string | null;
  };
}

export interface Mag7Analysis {
  timestamp: string;
  scores: StrategyScore[];
  top: StrategyScore[];
  account?: Record<string, unknown>;
  positions?: unknown[];
  note?: string;
  error?: string;
}

// ── Backtesting ──────────────────────────────────────────────────────────────

export interface BacktestConfig {
  strategyName: string;
  symbols: string[];
  startDate: string;
  endDate: string;
  startingCapital: number;
}

export interface BacktestResult {
  config: BacktestConfig;
  endingCapital: number;
  totalReturn: number;
  totalReturnPct: number;
  sharpeRatio: number | null;
  maxDrawdown: number;
  maxDrawdownPct: number;
  winRate: number;
  totalTrades: number;
  trades: BacktestTrade[];
  equityCurve: Array<{ date: string; equity: number }>;
}

export interface BacktestTrade {
  symbol: string;
  side: "buy" | "sell";
  date: string;
  price: number;
  qty: number;
  value: number;
  pnl?: number;
}

// ── Community ────────────────────────────────────────────────────────────────

export interface CommunityStrategy {
  id: string;
  name: string;
  description: string;
  author: string;
  stars: number;
  clones: number;
  category: string;
  returnPct: number;
  sharpe?: number;
  symbols: string[];
  createdAt: string;
}

// ── Navigation ───────────────────────────────────────────────────────────────

export interface NavItem {
  href: string;
  label: string;
  // Using ComponentType from React is intentional — avoids next/server conflicts
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  icon: React.ComponentType<any>;
}
