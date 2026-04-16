import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const SYSTEM_PROMPT = `You are Rover AI, a trading strategy assistant built into the Rover platform by Quantum Focus. You help users build automated trading strategies, understand backtest results, and improve their portfolios. When a user describes a strategy, respond with both a plain English explanation AND a structured JSON strategy object they can load into the builder. Keep responses concise and focused on trading.

When returning a strategy JSON, wrap it in a markdown code block tagged with "strategy-json" like this:
\`\`\`strategy-json
{ ... }
\`\`\`

The strategy JSON must follow this exact shape:
{
  "name": "Strategy Name",
  "description": "Brief description",
  "rules": [
    {
      "id": "rule-1",
      "logicOp": "AND",
      "conditions": [
        {
          "id": "cond-1",
          "indicator": "RSI",
          "period": 14,
          "operator": "less_than",
          "valueType": "number",
          "value": 30
        }
      ],
      "thenActions": [
        { "id": "action-1", "type": "buy", "asset": "AAPL", "pct": 50 }
      ],
      "elseActions": [
        { "id": "action-2", "type": "hold_cash" }
      ]
    }
  ]
}

Valid indicators: RSI, SMA, EMA, MACD, BBANDS_UPPER, BBANDS_LOWER, PRICE
Valid operators: greater_than, less_than, crosses_above, crosses_below, equals
Valid action types: buy, sell, hold_cash`;

export async function POST(req: Request) {
  const { messages } = await req.json();

  const stream = await client.messages.stream({
    model: "claude-sonnet-4-6",
    max_tokens: 1024,
    system: SYSTEM_PROMPT,
    messages,
  });

  const encoder = new TextEncoder();
  const readable = new ReadableStream({
    async start(controller) {
      for await (const chunk of stream) {
        if (
          chunk.type === "content_block_delta" &&
          chunk.delta.type === "text_delta"
        ) {
          controller.enqueue(encoder.encode(chunk.delta.text));
        }
      }
      controller.close();
    },
  });

  return new Response(readable, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Transfer-Encoding": "chunked",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
