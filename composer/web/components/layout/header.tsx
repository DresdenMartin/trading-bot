"use client";

import { Bell, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface HeaderProps {
  onMobileMenuClick?: () => void;
}

export function Header({ onMobileMenuClick }: HeaderProps) {
  return (
    <header className="h-16 shrink-0 border-b border-zinc-800 bg-zinc-950 flex items-center justify-between px-4 lg:px-6">
      <div className="flex items-center">
        <button
          onClick={onMobileMenuClick}
          className="lg:hidden p-1.5 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 rounded-lg transition-colors mr-2"
          aria-label="Open menu"
        >
          <Menu className="w-5 h-5" />
        </button>
      </div>
      <div className="flex items-center gap-3">
        <Badge
          variant="outline"
          className="border-amber-500/30 text-amber-400 bg-amber-500/5 text-xs px-2 py-0.5 hidden sm:flex"
        >
          Paper Mode
        </Badge>
        <Button
          variant="ghost"
          size="icon"
          className="text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 h-8 w-8"
        >
          <Bell className="w-4 h-4" />
        </Button>
        <div className="w-8 h-8 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center text-xs font-semibold text-indigo-400 cursor-pointer select-none">
          DM
        </div>
      </div>
    </header>
  );
}
