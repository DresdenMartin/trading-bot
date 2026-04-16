"use client";

import { useState } from "react";
import { Sidebar } from "./sidebar";
import { Header } from "./header";
import { QuantumFocusLogo } from "@/components/ui/QuantumFocusLogo";
import { ChatPanel } from "@/components/chat/ChatPanel";

export function MainLayoutClient({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <>
      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <Sidebar mobileOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <div className="flex flex-col flex-1 overflow-hidden min-w-0">
        <Header onMobileMenuClick={() => setSidebarOpen(true)} />
        <main className="flex-1 overflow-y-auto p-4 lg:p-6">{children}</main>
        <footer className="shrink-0 border-t border-zinc-800 px-4 lg:px-6 py-3 flex items-center gap-2">
          <QuantumFocusLogo size={20} />
          <div>
            <p className="text-xs text-zinc-500 leading-none">by Quantum Focus</p>
            <p className="text-[10px] text-zinc-600 leading-none mt-0.5">Powered by Claude</p>
          </div>
        </footer>
      </div>

      <ChatPanel />
    </>
  );
}
