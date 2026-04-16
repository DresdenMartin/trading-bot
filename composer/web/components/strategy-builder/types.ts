export type Indicator =
  | "RSI"
  | "SMA"
  | "EMA"
  | "MACD"
  | "BBANDS_UPPER"
  | "BBANDS_LOWER"
  | "PRICE";

export type Operator =
  | ">"
  | "<"
  | ">="
  | "<="
  | "crosses_above"
  | "crosses_below";

export type LogicOp = "AND" | "OR";
export type ActionType = "buy" | "sell" | "hold_cash";

export interface Condition {
  id: string;
  indicator: Indicator;
  period: number;
  operator: Operator;
  compareType: "number" | "indicator";
  value: number;
  compareIndicator?: Indicator;
  comparePeriod?: number;
}

export interface ConditionGroup {
  logic: LogicOp;
  conditions: Condition[];
}

export interface Action {
  id: string;
  type: ActionType;
  asset: string;
  pct: number;
}

export interface Rule {
  id: string;
  conditions: ConditionGroup;
  thenActions: Action[];
  elseActions: Action[];
}

export interface Strategy {
  name: string;
  description: string;
  rules: Rule[];
}
