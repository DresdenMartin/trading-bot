import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";

// GET /api/community — public strategies from all users
export async function GET() {
  const supabase = await createClient();

  const { data, error } = await supabase
    .from("strategies")
    .select("id, name, description, logic_json, created_at, user_id")
    .eq("is_public", true)
    .order("created_at", { ascending: false })
    .limit(50);

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json(data ?? []);
}

// POST /api/community/clone — clone a public strategy to current user
export async function POST(req: Request) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { strategy_id } = await req.json().catch(() => ({}));
  if (!strategy_id) return NextResponse.json({ error: "strategy_id required" }, { status: 400 });

  // Fetch the public strategy
  const { data: source, error: fetchErr } = await supabase
    .from("strategies")
    .select("name, description, logic_json")
    .eq("id", strategy_id)
    .eq("is_public", true)
    .single();

  if (fetchErr || !source) {
    return NextResponse.json({ error: "Strategy not found or not public" }, { status: 404 });
  }

  // Clone it to current user
  const { data, error } = await supabase
    .from("strategies")
    .insert({
      user_id: user.id,
      name: `${source.name} (clone)`,
      description: source.description,
      logic_json: source.logic_json,
      is_public: false,
    })
    .select()
    .single();

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json(data, { status: 201 });
}
