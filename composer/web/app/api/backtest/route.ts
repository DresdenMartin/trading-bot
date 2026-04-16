import { NextResponse } from "next/server";

const BACKTEST_URL =
  process.env.BACKTEST_URL ?? "http://localhost:8001";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const res = await fetch(`${BACKTEST_URL}/api/backtest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      // No cache — backtests are always live
      cache: "no-store",
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json(
      {
        error:
          "Backtest server unavailable. Start it with: python3 backtest_server.py",
      },
      { status: 503 }
    );
  }
}
