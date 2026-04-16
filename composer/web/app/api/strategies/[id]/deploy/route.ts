import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";

// POST /api/strategies/:id/deploy — activate a strategy for live execution
export async function POST(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  // Check the strategy belongs to this user
  const { data: strategy, error: stratErr } = await supabase
    .from("strategies")
    .select("id")
    .eq("id", id)
    .eq("user_id", user.id)
    .single();

  if (stratErr || !strategy) {
    return NextResponse.json({ error: "Strategy not found" }, { status: 404 });
  }

  // Upsert active_strategies — one active entry per strategy per user
  const { data, error } = await supabase
    .from("active_strategies")
    .upsert(
      { user_id: user.id, strategy_id: id, status: "active" },
      { onConflict: "user_id,strategy_id" }
    )
    .select()
    .single();

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json(data, { status: 201 });
}

// PATCH /api/strategies/:id/deploy — pause or stop
export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { status } = await req.json().catch(() => ({}));
  if (!["active", "paused", "stopped"].includes(status)) {
    return NextResponse.json({ error: "Invalid status" }, { status: 400 });
  }

  const { data, error } = await supabase
    .from("active_strategies")
    .update({ status })
    .eq("strategy_id", id)
    .eq("user_id", user.id)
    .select()
    .single();

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json(data);
}
