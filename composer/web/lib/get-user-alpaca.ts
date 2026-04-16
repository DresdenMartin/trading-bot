import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

export interface AlpacaCreds {
  key: string;
  secret: string;
  base: string;
}

/**
 * Returns Alpaca credentials for the currently authenticated user.
 * Falls back to environment variables if the user hasn't set their own keys.
 */
export async function getUserAlpaca(): Promise<AlpacaCreds | null> {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  const envFallback: AlpacaCreds | null =
    process.env.ALPACA_KEY && process.env.ALPACA_SECRET
      ? {
          key: process.env.ALPACA_KEY,
          secret: process.env.ALPACA_SECRET,
          base:
            process.env.ALPACA_BASE_URL ?? "https://paper-api.alpaca.markets",
        }
      : null;

  if (!supabaseUrl || !supabaseAnonKey) return envFallback;

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

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return envFallback;

  const { data: settings } = await supabase
    .from("user_settings")
    .select("alpaca_key, alpaca_secret, alpaca_paper")
    .eq("user_id", user.id)
    .single();

  if (!settings?.alpaca_key || !settings?.alpaca_secret) return envFallback;

  return {
    key: settings.alpaca_key,
    secret: settings.alpaca_secret,
    base:
      settings.alpaca_paper !== false
        ? "https://paper-api.alpaca.markets"
        : "https://api.alpaca.markets",
  };
}
