// Claude / Anthropic API helpers for the Next.js frontend.
// All calls go through the Anthropic Messages API.

const ANTHROPIC_API = "https://api.anthropic.com/v1/messages";
const DEFAULT_MODEL = "claude-sonnet-4-6";

// ── Types ────────────────────────────────────────────────────────────────────

export interface ClaudeMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ClaudeUsage {
  input_tokens: number;
  output_tokens: number;
}

export interface ClaudeResponse {
  id: string;
  type: "message";
  role: "assistant";
  content: Array<{ type: "text"; text: string }>;
  model: string;
  stop_reason: string;
  usage: ClaudeUsage;
}

// ── Core caller ──────────────────────────────────────────────────────────────

export async function askClaude(
  messages: ClaudeMessage[],
  options: {
    system?: string;
    model?: string;
    maxTokens?: number;
  } = {}
): Promise<string> {
  const { system, model = DEFAULT_MODEL, maxTokens = 1024 } = options;

  const res = await fetch(ANTHROPIC_API, {
    method: "POST",
    headers: {
      "x-api-key": process.env.ANTHROPIC_API_KEY ?? "",
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      max_tokens: maxTokens,
      ...(system ? { system } : {}),
      messages,
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Claude API error ${res.status}: ${body}`);
  }

  const data: ClaudeResponse = await res.json();
  return data.content.map((c) => c.text).join("").trim();
}

// ── Domain helpers ───────────────────────────────────────────────────────────

/**
 * Ask Claude to analyze a set of portfolio metrics and return a brief insight.
 */
export async function analyzePortfolio(
  symbols: string[],
  metrics: Record<string, unknown>
): Promise<string> {
  return askClaude(
    [
      {
        role: "user",
        content: `Analyze these Mag 7 symbols: ${symbols.join(", ")}\n\nMetrics:\n${JSON.stringify(metrics, null, 2)}`,
      },
    ],
    {
      system:
        "You are a concise financial analyst assistant. Analyze the provided portfolio metrics and give actionable insights in under 150 words. Be specific and data-driven.",
      maxTokens: 512,
    }
  );
}

/**
 * Ask Claude to summarize a single stock's outlook given recent news and indicators.
 */
export async function summarizeStock(
  symbol: string,
  context: {
    price?: number;
    rsi?: number;
    newsSentiment?: number;
    newsHeadlines?: string[];
  }
): Promise<string> {
  const lines = [
    `Symbol: ${symbol}`,
    context.price != null ? `Price: $${context.price}` : null,
    context.rsi != null ? `RSI (14): ${context.rsi.toFixed(1)}` : null,
    context.newsSentiment != null
      ? `Aggregate news sentiment: ${context.newsSentiment.toFixed(2)}`
      : null,
    context.newsHeadlines?.length
      ? `Recent headlines:\n${context.newsHeadlines.slice(0, 5).map((h) => `- ${h}`).join("\n")}`
      : null,
  ]
    .filter(Boolean)
    .join("\n");

  return askClaude(
    [{ role: "user", content: lines }],
    {
      system:
        "You are a short-term equity analyst. Given the data, write a 2-3 sentence outlook for the next 1-5 trading days. Be direct and mention specific signals.",
      maxTokens: 256,
    }
  );
}
