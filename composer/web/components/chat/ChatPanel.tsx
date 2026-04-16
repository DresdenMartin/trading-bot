"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { MessageCircle, X, Trash2, Send, Download } from "lucide-react";
import { RoverLogo } from "@/components/ui/RoverLogo";
import { cn } from "@/lib/utils";

interface Message {
  role: "user" | "assistant";
  content: string;
  id: string;
}

const STORAGE_KEY = "rover_chat_history";
const MAX_HISTORY = 50;

const SUGGESTIONS = [
  "Build me a momentum strategy for NVDA",
  "Explain RSI and how to use it",
  "Create a simple moving average crossover strategy",
];

function extractStrategyJson(text: string): string | null {
  const match = text.match(/```strategy-json\s*([\s\S]*?)```/);
  if (!match) return null;
  try {
    JSON.parse(match[1].trim());
    return match[1].trim();
  } catch {
    return null;
  }
}

function MessageBubble({
  message,
  onLoadStrategy,
}: {
  message: Message;
  onLoadStrategy: (json: string) => void;
}) {
  const isUser = message.role === "user";
  const strategyJson = !isUser ? extractStrategyJson(message.content) : null;

  // Render content with strategy-json blocks hidden, replaced by button
  const displayText = message.content
    .replace(/```strategy-json[\s\S]*?```/g, "")
    .trim();

  return (
    <div className={cn("flex gap-2.5", isUser && "flex-row-reverse")}>
      {!isUser && (
        <div className="shrink-0 mt-0.5">
          <RoverLogo size={24} />
        </div>
      )}
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed",
          isUser
            ? "bg-indigo-500 text-white rounded-tr-sm"
            : "bg-zinc-800 text-zinc-100 rounded-tl-sm"
        )}
      >
        {displayText && <p className="whitespace-pre-wrap">{displayText}</p>}
        {strategyJson && (
          <button
            onClick={() => onLoadStrategy(strategyJson)}
            className="mt-2 flex items-center gap-1.5 text-xs bg-orange-500/20 hover:bg-orange-500/30 text-orange-300 border border-orange-500/30 rounded-lg px-2.5 py-1.5 transition-colors"
          >
            <Download className="w-3 h-3" />
            Load into Builder
          </button>
        )}
      </div>
    </div>
  );
}

export function ChatPanel() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Load history from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setMessages(JSON.parse(stored));
      }
    } catch {}
  }, []);

  // Persist history
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-MAX_HISTORY)));
    } catch {}
  }, [messages]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const genId = () => Math.random().toString(36).slice(2, 9);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || streaming) return;

      const userMsg: Message = { role: "user", content: text.trim(), id: genId() };
      const assistantId = genId();

      setMessages((prev) => [
        ...prev,
        userMsg,
        { role: "assistant", content: "", id: assistantId },
      ]);
      setInput("");
      setStreaming(true);

      const history = messages.slice(-20).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: [...history, { role: "user", content: text.trim() }],
          }),
          signal: controller.signal,
        });

        if (!res.body) throw new Error("No response body");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let accumulated = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          accumulated += chunk;

          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: accumulated } : m
            )
          );
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name !== "AbortError") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: "Sorry, something went wrong. Please try again." }
                : m
            )
          );
        }
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [messages, streaming]
  );

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  function clearHistory() {
    if (abortRef.current) abortRef.current.abort();
    setMessages([]);
    localStorage.removeItem(STORAGE_KEY);
  }

  function handleLoadStrategy(json: string) {
    try {
      const parsed = JSON.parse(json);
      localStorage.setItem("rover_pending_strategy", JSON.stringify(parsed));
      window.location.href = "/strategies";
    } catch {}
  }

  return (
    <>
      {/* Floating button */}
      <button
        onClick={() => setOpen(true)}
        className={cn(
          "fixed bottom-6 right-6 z-50 w-12 h-12 rounded-full bg-orange-500 hover:bg-orange-400 shadow-lg shadow-orange-500/30 flex items-center justify-center transition-all duration-200",
          open && "opacity-0 pointer-events-none scale-90"
        )}
        aria-label="Open Rover AI"
      >
        <MessageCircle className="w-5 h-5 text-white" />
      </button>

      {/* Backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/20 backdrop-blur-[1px]"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Panel */}
      <div
        className={cn(
          "fixed top-0 right-0 z-50 h-full w-[380px] bg-zinc-900 border-l border-zinc-800 flex flex-col shadow-2xl transition-transform duration-300",
          open ? "translate-x-0" : "translate-x-full"
        )}
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-zinc-800 shrink-0">
          <RoverLogo size={32} />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-zinc-100">Rover AI</p>
            <p className="text-xs text-zinc-500">Trading strategy assistant</p>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={clearHistory}
              className="p-1.5 text-zinc-600 hover:text-zinc-400 transition-colors rounded-lg hover:bg-zinc-800"
              title="Clear chat"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setOpen(false)}
              className="p-1.5 text-zinc-600 hover:text-zinc-400 transition-colors rounded-lg hover:bg-zinc-800"
              title="Close"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-6 pb-8">
              <RoverLogo size={64} />
              <div className="text-center">
                <p className="text-sm font-medium text-zinc-300">Hey, I&apos;m Rover AI</p>
                <p className="text-xs text-zinc-500 mt-1">
                  Describe a strategy and I&apos;ll build it for you
                </p>
              </div>
              <div className="flex flex-col gap-2 w-full">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => sendMessage(s)}
                    className="text-xs text-left px-3.5 py-2.5 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors border border-zinc-700/50"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m) => (
              <MessageBubble
                key={m.id}
                message={m}
                onLoadStrategy={handleLoadStrategy}
              />
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="shrink-0 p-4 border-t border-zinc-800">
          <div className="flex items-end gap-2 bg-zinc-800 border border-zinc-700 rounded-xl px-3 py-2.5 focus-within:border-orange-500/50 transition-colors">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about trading strategies…"
              rows={1}
              className="flex-1 bg-transparent text-sm text-zinc-200 placeholder:text-zinc-600 resize-none focus:outline-none max-h-32 leading-relaxed"
              style={{ minHeight: "20px" }}
            />
            <button
              onClick={() => sendMessage(input)}
              disabled={!input.trim() || streaming}
              className="shrink-0 w-7 h-7 rounded-lg bg-orange-500 hover:bg-orange-400 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
            >
              <Send className="w-3.5 h-3.5 text-white" />
            </button>
          </div>
          <p className="text-[10px] text-zinc-600 mt-1.5 text-center">
            Enter to send · Shift+Enter for newline
          </p>
        </div>
      </div>
    </>
  );
}
