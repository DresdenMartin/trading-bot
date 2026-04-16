import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";

// GET /api/strategies — list current user's strategies
export async function GET() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { data, error } = await supabase
    .from("strategies")
    .select("id, name, description, logic_json, is_public, created_at")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json(data);
}

// POST /api/strategies — create a new strategy
export async function POST(req: Request) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const body = await req.json().catch(() => ({}));
  const { name, description, logic_json, is_public } = body;

  if (!name || !logic_json) {
    return NextResponse.json({ error: "name and logic_json required" }, { status: 400 });
  }

  const { data, error } = await supabase
    .from("strategies")
    .insert({ user_id: user.id, name, description: description ?? "", logic_json, is_public: is_public ?? false })
    .select()
    .single();

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json(data, { status: 201 });
}
