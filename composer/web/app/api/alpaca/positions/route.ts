import { NextResponse } from "next/server";
import { getUserAlpaca } from "@/lib/get-user-alpaca";

export async function GET() {
  const creds = await getUserAlpaca();
  if (!creds) {
    return NextResponse.json(
      { error: "Alpaca keys not configured. Go to Settings to add yours." },
      { status: 503 }
    );
  }

  try {
    const res = await fetch(`${creds.base}/v2/positions`, {
      headers: {
        "APCA-API-KEY-ID": creds.key,
        "APCA-API-SECRET-KEY": creds.secret,
      },
      cache: "no-store",
    });
    if (!res.ok) {
      const detail = await res.text();
      return NextResponse.json({ error: `Alpaca returned ${res.status}`, detail }, { status: res.status });
    }
    return NextResponse.json(await res.json());
  } catch (err) {
    return NextResponse.json({ error: "Failed to reach Alpaca", detail: String(err) }, { status: 500 });
  }
}
