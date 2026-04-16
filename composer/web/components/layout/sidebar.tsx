"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import {
  LayoutDashboard,
  TrendingUp,
  FlaskConical,
  Users,
  Settings,
  LogOut,
  X,
  BarChart2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { RoverLogo } from "@/components/ui/RoverLogo";

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/strategies", label: "Strategies", icon: TrendingUp },
  { href: "/backtest", label: "Backtest", icon: FlaskConical },
  { href: "/analytics", label: "Analytics", icon: BarChart2 },
  { href: "/community", label: "Community", icon: Users },
  { href: "/settings", label: "Settings", icon: Settings },
];

interface SidebarProps {
  mobileOpen?: boolean;
  onClose?: () => void;
}

function SidebarContent({ onClose }: { onClose?: () => void }) {
  const pathname = usePathname();
  const router = useRouter();
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const supabaseConfigured = Boolean(
    process.env.NEXT_PUBLIC_SUPABASE_URL &&
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  );

  useEffect(() => {
    if (!supabaseConfigured) return;
    import("@/lib/supabase/client").then(({ createClient }) => {
      createClient()
        .auth.getUser()
        .then(({ data }) => setUserEmail(data.user?.email ?? null));
    });
  }, [supabaseConfigured]);

  async function handleSignOut() {
    if (!supabaseConfigured) return;
    const { createClient } = await import("@/lib/supabase/client");
    await createClient().auth.signOut();
    router.push("/login");
    router.refresh();
  }

  const initials = userEmail
    ? userEmail.slice(0, 2).toUpperCase()
    : "DM";
  const displayName = userEmail ?? "Dresden Martin";

  return (
    <>
      {/* Logo */}
      <div className="h-16 flex items-center px-5 border-b border-zinc-800 shrink-0">
        <div className="flex items-center gap-2.5 flex-1">
          <RoverLogo size={28} />
          <span className="text-[15px] font-bold text-zinc-100 tracking-tight">
            Rover
          </span>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="lg:hidden p-1 text-zinc-600 hover:text-zinc-400 transition-colors rounded"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {navItems.map((item) => {
          const active =
            pathname === item.href || pathname.startsWith(item.href + "/");
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={onClose}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                active
                  ? "bg-indigo-500/10 text-indigo-400"
                  : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/60"
              )}
            >
              <item.icon
                className={cn(
                  "w-4 h-4 shrink-0",
                  active ? "text-indigo-400" : "text-zinc-500"
                )}
              />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* User */}
      <div className="px-3 pb-4 border-t border-zinc-800 pt-3 shrink-0">
        <div className="flex items-center gap-3 px-3 py-2 rounded-lg">
          <div className="w-7 h-7 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center text-xs font-semibold text-indigo-400 shrink-0">
            {initials}
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium text-zinc-200 truncate">
              {displayName}
            </p>
            <p className="text-xs text-zinc-500">Paper Trading</p>
          </div>
          {supabaseConfigured && (
            <button
              onClick={handleSignOut}
              className="text-zinc-700 hover:text-zinc-400 transition-colors"
              title="Sign out"
            >
              <LogOut className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </>
  );
}

export function Sidebar({ mobileOpen, onClose }: SidebarProps) {
  return (
    <>
      {/* Desktop sidebar — always visible on lg+ */}
      <aside className="hidden lg:flex w-60 shrink-0 flex-col bg-zinc-900 border-r border-zinc-800 h-full">
        <SidebarContent />
      </aside>

      {/* Mobile sidebar — fixed drawer */}
      <aside
        className={cn(
          "lg:hidden fixed top-0 left-0 h-full w-64 z-40 flex flex-col bg-zinc-900 border-r border-zinc-800 transition-transform duration-300",
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <SidebarContent onClose={onClose} />
      </aside>
    </>
  );
}
