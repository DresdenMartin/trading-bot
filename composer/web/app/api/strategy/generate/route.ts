import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";

const EXAMPLE = {
  name: "RSI Oversold Bounce",
  description: "Buy when RSI is oversold and price is above SMA",
  rules: [
    {
      id: "r1",
      conditions: {
        logic: "AND",
        conditions: [
          {
            id: "c1",
            indicator: "RSI",
            period: 14,
            operator: "<",
            compareType: "number",
            value: 30,
          },
          {
            id: "c2",
            indicator: "PRICE",
            period: 0,
            operator: ">",
            compareType: "indicator",
            compareIndicator: "SMA",
            comparePeriod: 200,
          },
        ],
      },
      thenActions: [{ id: "a1", type: "buy", asset: "SPY", pct: 100 }],
      elseActions: [{ id: "a2", type: "hold_cash", asset: "", pct: 100 }],
    },
  ],
};

export async function POST(req: Request) {
  if (!process.env.ANTHROPIC_API_KEY) {
    return NextResponse.json(
      { error: "ANTHROPIC_API_KEY not configured" },
      { status: 503 }
    );
  }

  const body = await req.json().catch(() => ({}));
  const prompt = body?.prompt?.trim();
  if (!prompt) {
    return NextResponse.json({ error: "prompt required" }, { status: 400 });
  }

  try {
    const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    const msg = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 2000,
      messages: [
        {
          role: "user",
          content: `You are a trading strategy builder. Convert this plain English strategy description into structured JSON.

Description: ${prompt}

Return ONLY valid JSON — no explanation, no markdown fences. Match this exact schema:

${JSON.stringify(EXAMPLE, null, 2)}

Schema rules:
- indicator values: RSI, SMA, EMA, MACD, BBANDS_UPPER, BBANDS_LOWER, PRICE
- operator values: >, <, >=, <=, crosses_above, crosses_below
- compareType: "number" (right side is a number) or "indicator" (right side is another indicator)
- action type values: buy, sell, hold_cash
- pct: 0–100 (percentage of portfolio)
- Use sensible periods: RSI default 14, SMA short 50, SMA long 200, EMA default 20
- IDs can be any short string`,
        },
      ],
    });

    const text =
      msg.content[0].type === "text" ? msg.content[0].text.trim() : "";

    // Strip markdown fences if model added them despite instructions
    const clean = text.replace(/^```(?:json)?\n?|\n?```$/g, "").trim();

    const parsed = JSON.parse(clean);
    return NextResponse.json({ strategy: parsed });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    // If JSON parse failed, return a helpful error
    if (message.includes("JSON")) {
      return NextResponse.json(
        { error: "Model returned invalid JSON", detail: message },
        { status: 422 }
      );
    }
    return NextResponse.json(
      { error: "Generation failed", detail: message },
      { status: 500 }
    );
  }
}
