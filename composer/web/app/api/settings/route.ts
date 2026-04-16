import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

const ENV_DEFAULTS = {
  alpaca_key: "",
  alpaca_secret: "",
  alpaca_paper: true,
  place_order: process.env.PLACE_ORDER === "true",
  reallocate_full: process.env.REALLOCATE_FULL === "true",
  top_allocate_count: parseInt(process.env.TOP_ALLOCATE_COUNT ?? "3"),
  invest_total_pct: parseFloat(process.env.INVEST_TOTAL_PCT ?? "0.05"),
  symbols: process.env.SYMBOLS ?? "AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA",
  extended_hours: process.env.EXTENDED_HOURS === "true",
  use_web_research: process.env.USE_WEB_RESEARCH === "true",
  openai_model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
  news_window_hours: parseInt(process.env.NEWS_WINDOW_HOURS ?? "36"),
};

async function getSupabase() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !anon) return null;
  const cookieStore = await cookies();
  return createServerClient(url, anon, {
    cookies: {
      getAll: () => cookieStore.getAll(),
      setAll: (toSet) => {
        for (const { name, value, options } of toSet) {
          cookieStore.set(name, value, options);
        }
      },
    },
  });
}

export async function GET() {
  const supabase = await getSupabase();
  if (!supabase) return NextResponse.json(ENV_DEFAULTS);

  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.json(ENV_DEFAULTS);

  const { data } = await supabase
    .from("user_settings")
    .select("*")
    .eq("user_id", user.id)
    .single();

  return NextResponse.json({ ...ENV_DEFAULTS, ...(data ?? {}) });
}

export async function POST(req: Request) {
  const body = await req.json();
  const supabase = await getSupabase();
  if (!supabase) return NextResponse.json({ error: "Supabase not configured" }, { status: 503 });

  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Not authenticated" }, { status: 401 });

  // Remove internal Supabase fields before upsert
  const { id: _id, created_at: _ca, updated_at: _ua, user_id: _uid, ...fields } = body;
  void _id; void _ca; void _ua; void _uid;

  const { data, error } = await supabase
    .from("user_settings")
    .upsert({ ...fields, user_id: user.id, updated_at: new Date().toISOString() }, {
      onConflict: "user_id",
    })
    .select()
    .single();

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ ...ENV_DEFAULTS, ...data });
}
