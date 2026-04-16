"use client";

import { useState } from "react";
import {
  DndContext,
  closestCenter,
  PointerSensor,
  KeyboardSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
  GripVertical,
  Plus,
  Trash2,
  Wand2,
  Save,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type {
  Strategy,
  Rule,
  Condition,
  Action,
  LogicOp,
  Indicator,
  Operator,
  ActionType,
} from "./types";

// ── Constants ─────────────────────────────────────────────────────────────────

const INDICATORS: { value: Indicator; label: string; hasPeriod: boolean }[] = [
  { value: "PRICE", label: "Price", hasPeriod: false },
  { value: "RSI", label: "RSI", hasPeriod: true },
  { value: "SMA", label: "SMA", hasPeriod: true },
  { value: "EMA", label: "EMA", hasPeriod: true },
  { value: "MACD", label: "MACD", hasPeriod: false },
  { value: "BBANDS_UPPER", label: "BB Upper", hasPeriod: true },
  { value: "BBANDS_LOWER", label: "BB Lower", hasPeriod: true },
];

const OPERATORS: { value: Operator; label: string }[] = [
  { value: ">", label: ">" },
  { value: "<", label: "<" },
  { value: ">=", label: "≥" },
  { value: "<=", label: "≤" },
  { value: "crosses_above", label: "crosses ↑" },
  { value: "crosses_below", label: "crosses ↓" },
];

// ── Helpers ───────────────────────────────────────────────────────────────────

function uid(): string {
  return crypto.randomUUID();
}

function defaultCondition(): Condition {
  return {
    id: uid(),
    indicator: "RSI",
    period: 14,
    operator: "<",
    compareType: "number",
    value: 30,
  };
}

function defaultAction(type: ActionType = "buy"): Action {
  return { id: uid(), type, asset: "SPY", pct: 100 };
}

function defaultRule(): Rule {
  return {
    id: uid(),
    conditions: { logic: "AND", conditions: [defaultCondition()] },
    thenActions: [defaultAction("buy")],
    elseActions: [defaultAction("hold_cash")],
  };
}

// ── Primitive inputs ──────────────────────────────────────────────────────────

function Sel({
  value,
  onChange,
  options,
  className,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  className?: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={cn(
        "bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-zinc-200",
        "focus:outline-none focus:border-indigo-500 hover:border-zinc-600 transition-colors",
        "appearance-none cursor-pointer",
        className
      )}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

function NumInput({
  value,
  onChange,
  className,
}: {
  value: number;
  onChange: (v: number) => void;
  className?: string;
}) {
  return (
    <input
      type="number"
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
      className={cn(
        "bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-zinc-200",
        "focus:outline-none focus:border-indigo-500 hover:border-zinc-600 transition-colors w-16",
        className
      )}
    />
  );
}

function TickerInput({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value.toUpperCase())}
      placeholder="SPY"
      className="bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-zinc-200 w-20 focus:outline-none focus:border-indigo-500 hover:border-zinc-600 transition-colors uppercase"
    />
  );
}

// ── ConditionRow ──────────────────────────────────────────────────────────────

function ConditionRow({
  condition,
  onUpdate,
  onRemove,
  showRemove,
}: {
  condition: Condition;
  onUpdate: (c: Condition) => void;
  onRemove: () => void;
  showRemove: boolean;
}) {
  const leftMeta = INDICATORS.find((i) => i.value === condition.indicator);
  const rightMeta = INDICATORS.find(
    (i) => i.value === condition.compareIndicator
  );

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {/* Left indicator */}
      <Sel
        value={condition.indicator}
        onChange={(v) => onUpdate({ ...condition, indicator: v as Indicator })}
        options={INDICATORS.map((i) => ({ value: i.value, label: i.label }))}
        className="w-28"
      />
      {leftMeta?.hasPeriod && (
        <div className="flex items-center gap-0.5">
          <span className="text-zinc-600 text-xs">(</span>
          <NumInput
            value={condition.period}
            onChange={(v) => onUpdate({ ...condition, period: v })}
            className="w-12"
          />
          <span className="text-zinc-600 text-xs">)</span>
        </div>
      )}

      {/* Operator */}
      <Sel
        value={condition.operator}
        onChange={(v) => onUpdate({ ...condition, operator: v as Operator })}
        options={OPERATORS}
        className="w-28"
      />

      {/* Right side */}
      {condition.compareType === "number" ? (
        <>
          <NumInput
            value={condition.value}
            onChange={(v) => onUpdate({ ...condition, value: v })}
          />
          <button
            onClick={() =>
              onUpdate({
                ...condition,
                compareType: "indicator",
                compareIndicator: "SMA",
                comparePeriod: 200,
              })
            }
            className="text-[10px] text-zinc-600 hover:text-indigo-400 transition-colors px-1.5 py-1 rounded border border-zinc-700 hover:border-indigo-500"
            title="Switch to indicator comparison"
          >
            vs indicator
          </button>
        </>
      ) : (
        <>
          <Sel
            value={condition.compareIndicator ?? "SMA"}
            onChange={(v) =>
              onUpdate({ ...condition, compareIndicator: v as Indicator })
            }
            options={INDICATORS.filter((i) => i.value !== "PRICE").map(
              (i) => ({ value: i.value, label: i.label })
            )}
            className="w-24"
          />
          {rightMeta?.hasPeriod && (
            <div className="flex items-center gap-0.5">
              <span className="text-zinc-600 text-xs">(</span>
              <NumInput
                value={condition.comparePeriod ?? 200}
                onChange={(v) => onUpdate({ ...condition, comparePeriod: v })}
                className="w-12"
              />
              <span className="text-zinc-600 text-xs">)</span>
            </div>
          )}
          <button
            onClick={() => onUpdate({ ...condition, compareType: "number" })}
            className="text-[10px] text-zinc-600 hover:text-indigo-400 transition-colors px-1.5 py-1 rounded border border-zinc-700 hover:border-indigo-500"
            title="Switch to number comparison"
          >
            vs number
          </button>
        </>
      )}

      {showRemove && (
        <button
          onClick={onRemove}
          className="text-zinc-700 hover:text-red-400 transition-colors ml-auto"
        >
          <Trash2 className="w-3 h-3" />
        </button>
      )}
    </div>
  );
}

// ── ActionRow ─────────────────────────────────────────────────────────────────

function ActionRow({
  action,
  onUpdate,
  onRemove,
  showRemove,
}: {
  action: Action;
  onUpdate: (a: Action) => void;
  onRemove: () => void;
  showRemove: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      <Sel
        value={action.type}
        onChange={(v) => onUpdate({ ...action, type: v as ActionType })}
        options={[
          { value: "buy", label: "Buy" },
          { value: "sell", label: "Sell" },
          { value: "hold_cash", label: "Hold Cash" },
        ]}
        className="w-28"
      />
      {action.type !== "hold_cash" && (
        <>
          <TickerInput
            value={action.asset}
            onChange={(v) => onUpdate({ ...action, asset: v })}
          />
          <NumInput
            value={action.pct}
            onChange={(v) => onUpdate({ ...action, pct: v })}
            className="w-14"
          />
          <span className="text-xs text-zinc-500">%</span>
        </>
      )}
      {showRemove && (
        <button
          onClick={onRemove}
          className="text-zinc-700 hover:text-red-400 transition-colors ml-auto"
        >
          <Trash2 className="w-3 h-3" />
        </button>
      )}
    </div>
  );
}

// ── SortableRule ──────────────────────────────────────────────────────────────

function SortableRule({
  rule,
  ruleIndex,
  onUpdate,
  onRemove,
}: {
  rule: Rule;
  ruleIndex: number;
  onUpdate: (r: Rule) => void;
  onRemove: () => void;
}) {
  const [showElse, setShowElse] = useState(true);
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: rule.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  };

  function updateCondition(idx: number, c: Condition) {
    const conditions = [...rule.conditions.conditions];
    conditions[idx] = c;
    onUpdate({ ...rule, conditions: { ...rule.conditions, conditions } });
  }

  function removeCondition(idx: number) {
    const conditions = rule.conditions.conditions.filter((_, i) => i !== idx);
    onUpdate({ ...rule, conditions: { ...rule.conditions, conditions } });
  }

  function toggleLogic() {
    const logic: LogicOp = rule.conditions.logic === "AND" ? "OR" : "AND";
    onUpdate({ ...rule, conditions: { ...rule.conditions, logic } });
  }

  function updateThen(idx: number, a: Action) {
    const thenActions = [...rule.thenActions];
    thenActions[idx] = a;
    onUpdate({ ...rule, thenActions });
  }

  function updateElse(idx: number, a: Action) {
    const elseActions = [...rule.elseActions];
    elseActions[idx] = a;
    onUpdate({ ...rule, elseActions });
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-zinc-800 bg-zinc-800/40">
        <button
          {...attributes}
          {...listeners}
          className="text-zinc-600 hover:text-zinc-400 cursor-grab active:cursor-grabbing touch-none"
        >
          <GripVertical className="w-4 h-4" />
        </button>
        <span className="text-xs font-medium text-zinc-500">
          Rule {ruleIndex + 1}
        </span>
        <div className="flex-1" />
        <button
          onClick={onRemove}
          className="text-zinc-700 hover:text-red-400 transition-colors"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="p-4 space-y-5">
        {/* IF */}
        <div className="space-y-2">
          <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">
            IF
          </span>
          <div className="space-y-2 pl-3 border-l-2 border-indigo-500/20">
            {rule.conditions.conditions.map((c, i) => (
              <div key={c.id}>
                {i > 0 && (
                  <button
                    onClick={toggleLogic}
                    className="text-xs font-bold text-amber-400 hover:text-amber-300 transition-colors mb-2 px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20"
                  >
                    {rule.conditions.logic}
                  </button>
                )}
                <ConditionRow
                  condition={c}
                  onUpdate={(updated) => updateCondition(i, updated)}
                  onRemove={() => removeCondition(i)}
                  showRemove={rule.conditions.conditions.length > 1}
                />
              </div>
            ))}
            <button
              onClick={() => {
                const conditions = [
                  ...rule.conditions.conditions,
                  defaultCondition(),
                ];
                onUpdate({
                  ...rule,
                  conditions: { ...rule.conditions, conditions },
                });
              }}
              className="flex items-center gap-1 text-xs text-zinc-600 hover:text-zinc-400 transition-colors mt-1"
            >
              <Plus className="w-3 h-3" />
              Add condition
            </button>
          </div>
        </div>

        {/* THEN */}
        <div className="space-y-2">
          <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">
            THEN
          </span>
          <div className="space-y-2 pl-3 border-l-2 border-emerald-500/20">
            {rule.thenActions.map((a, i) => (
              <ActionRow
                key={a.id}
                action={a}
                onUpdate={(updated) => updateThen(i, updated)}
                onRemove={() => {
                  const thenActions = rule.thenActions.filter(
                    (_, idx) => idx !== i
                  );
                  onUpdate({ ...rule, thenActions });
                }}
                showRemove={rule.thenActions.length > 1}
              />
            ))}
            <button
              onClick={() =>
                onUpdate({
                  ...rule,
                  thenActions: [
                    ...rule.thenActions,
                    defaultAction("buy"),
                  ],
                })
              }
              className="flex items-center gap-1 text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
            >
              <Plus className="w-3 h-3" />
              Add action
            </button>
          </div>
        </div>

        {/* ELSE */}
        <div className="space-y-2">
          <button
            onClick={() => setShowElse(!showElse)}
            className="flex items-center gap-1 text-xs font-semibold text-zinc-600 hover:text-zinc-400 transition-colors uppercase tracking-wider"
          >
            {showElse ? (
              <ChevronDown className="w-3 h-3" />
            ) : (
              <ChevronRight className="w-3 h-3" />
            )}
            ELSE
          </button>
          {showElse && (
            <div className="space-y-2 pl-3 border-l-2 border-zinc-700/40">
              {rule.elseActions.map((a, i) => (
                <ActionRow
                  key={a.id}
                  action={a}
                  onUpdate={(updated) => updateElse(i, updated)}
                  onRemove={() => {
                    const elseActions = rule.elseActions.filter(
                      (_, idx) => idx !== i
                    );
                    onUpdate({ ...rule, elseActions });
                  }}
                  showRemove={rule.elseActions.length > 1}
                />
              ))}
              <button
                onClick={() =>
                  onUpdate({
                    ...rule,
                    elseActions: [
                      ...rule.elseActions,
                      defaultAction("hold_cash"),
                    ],
                  })
                }
                className="flex items-center gap-1 text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
              >
                <Plus className="w-3 h-3" />
                Add action
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── StrategyBuilder ───────────────────────────────────────────────────────────

export default function StrategyBuilder({
  initial,
  onSave,
}: {
  initial?: Strategy;
  onSave?: (s: Strategy) => void;
}) {
  const [strategy, setStrategy] = useState<Strategy>(
    initial ?? {
      name: "My Strategy",
      description: "",
      rules: [defaultRule()],
    }
  );
  const [aiPrompt, setAiPrompt] = useState("");
  const [generating, setGenerating] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    setStrategy((s) => {
      const oldIdx = s.rules.findIndex((r) => r.id === active.id);
      const newIdx = s.rules.findIndex((r) => r.id === over.id);
      return { ...s, rules: arrayMove(s.rules, oldIdx, newIdx) };
    });
  }

  function updateRule(id: string, rule: Rule) {
    setStrategy((s) => ({
      ...s,
      rules: s.rules.map((r) => (r.id === id ? rule : r)),
    }));
  }

  function removeRule(id: string) {
    setStrategy((s) => ({ ...s, rules: s.rules.filter((r) => r.id !== id) }));
  }

  async function generateFromAI() {
    if (!aiPrompt.trim()) return;
    setGenerating(true);
    setAiError(null);
    try {
      const res = await fetch("/api/strategy/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: aiPrompt }),
      });
      const data = await res.json();
      if (!res.ok) {
        setAiError(data.error ?? "Generation failed");
        return;
      }
      if (data.strategy) {
        const s: Strategy = {
          ...data.strategy,
          rules: (data.strategy.rules as Rule[]).map((r) => ({
            ...r,
            id: uid(),
            conditions: {
              ...r.conditions,
              conditions: r.conditions.conditions.map((c: Condition) => ({
                ...c,
                id: uid(),
              })),
            },
            thenActions: r.thenActions.map((a: Action) => ({
              ...a,
              id: uid(),
            })),
            elseActions: r.elseActions.map((a: Action) => ({
              ...a,
              id: uid(),
            })),
          })),
        };
        setStrategy(s);
        setAiPrompt("");
      }
    } catch {
      setAiError("Failed to reach generation API");
    } finally {
      setGenerating(false);
    }
  }

  return (
    <div className="space-y-4">
      {/* AI generation bar */}
      <div className="space-y-2">
        <div className="flex gap-2">
          <input
            type="text"
            value={aiPrompt}
            onChange={(e) => setAiPrompt(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && generateFromAI()}
            placeholder='Describe a strategy… e.g. "Buy SPY when RSI(14) < 30 and price is above SMA(200)"'
            className="flex-1 bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500 transition-colors"
          />
          <Button
            onClick={generateFromAI}
            disabled={generating || !aiPrompt.trim()}
            className="bg-indigo-500 hover:bg-indigo-600 text-white gap-2 text-sm shrink-0"
          >
            <Wand2 className="w-3.5 h-3.5" />
            {generating ? "Generating…" : "Generate with AI"}
          </Button>
        </div>
        {aiError && <p className="text-xs text-red-400">{aiError}</p>}
      </div>

      {/* Strategy name */}
      <input
        type="text"
        value={strategy.name}
        onChange={(e) => setStrategy({ ...strategy, name: e.target.value })}
        className="w-full bg-transparent text-xl font-bold text-zinc-100 focus:outline-none border-b border-transparent focus:border-zinc-700 pb-1 transition-colors placeholder:text-zinc-700"
        placeholder="Strategy name"
      />

      {/* Rules */}
      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragEnd={handleDragEnd}
      >
        <SortableContext
          items={strategy.rules.map((r) => r.id)}
          strategy={verticalListSortingStrategy}
        >
          <div className="space-y-3">
            {strategy.rules.map((rule, i) => (
              <SortableRule
                key={rule.id}
                rule={rule}
                ruleIndex={i}
                onUpdate={(r) => updateRule(rule.id, r)}
                onRemove={() => removeRule(rule.id)}
              />
            ))}
          </div>
        </SortableContext>
      </DndContext>

      {/* Add rule */}
      <button
        onClick={() =>
          setStrategy((s) => ({
            ...s,
            rules: [...s.rules, defaultRule()],
          }))
        }
        className="flex items-center gap-2 text-sm text-zinc-600 hover:text-zinc-400 border border-dashed border-zinc-800 hover:border-zinc-700 rounded-xl px-4 py-3 w-full justify-center transition-colors"
      >
        <Plus className="w-4 h-4" />
        Add Rule
      </button>

      {/* Save */}
      <div className="flex justify-end pt-2">
        <Button
          onClick={() => onSave?.(strategy)}
          className="bg-emerald-600 hover:bg-emerald-700 text-white gap-2 text-sm"
        >
          <Save className="w-3.5 h-3.5" />
          Save Strategy
        </Button>
      </div>
    </div>
  );
}
