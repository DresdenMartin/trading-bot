import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

export async function GET(req: Request) {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!supabaseUrl || !supabaseAnonKey) {
    return NextResponse.json([]);
  }

  const cookieStore = await cookies();
  const supabase = createServerClient(supabaseUrl, supabaseAnonKey, {
    cookies: {
      getAll: () => cookieStore.getAll(),
      setAll: (toSet) => {
        for (const { name, value, options } of toSet) {
          cookieStore.set(name, value, options);
        }
      },
    },
  });

  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.json([]);

  const url = new URL(req.url);
  const strategyId = url.searchParams.get("strategy_id");

  let query = supabase
    .from("strategy_runs")
    .select("id, strategy_id, status, signals_fired, orders_placed, error_msg, created_at, strategies(name)")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false })
    .limit(100);

  if (strategyId) {
    query = query.eq("strategy_id", strategyId);
  }

  const { data, error } = await query;
  if (error) return NextResponse.json([]);
  return NextResponse.json(data ?? []);
}
