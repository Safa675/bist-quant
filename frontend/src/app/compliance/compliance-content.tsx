"use client";

import * as React from "react";
import {
  checkActivityAnomalies,
  checkPositionLimits,
  getDefaultComplianceRules,
  runComplianceCheck,
} from "@/lib/api";
import type {
  ActivityAnomalyResult,
  ComplianceResult,
  ComplianceRule,
  ComplianceTransaction,
} from "@/lib/types";
import { PageHeader } from "@/components/shared/page-header";
import { SectionCard } from "@/components/shared/section-card";
import { ApiError } from "@/components/shared/api-error";
import { DataTable } from "@/components/shared/data-table";
import { PageScaffold, PageMain, PageSectionStack } from "@/components/shared/page-scaffold";
import { FormField, FormGrid, FormRow } from "@/components/shared/form-field";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SelectInput } from "@/components/ui/select-input";
import { Textarea } from "@/components/ui/textarea";
import type { ColumnDef } from "@tanstack/react-table";
import { Loader2, Plus, ShieldCheck, Trash2 } from "lucide-react";
import { StaggerReveal, StaggerItem } from "@/components/shared/stagger-reveal";

interface HistoryRow {
  timestamp: string;
  tx_id: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  hits: number;
  status: "PASS" | "FAIL";
}

const rulesColumns: ColumnDef<ComplianceRule, unknown>[] = [
  { accessorKey: "id", header: "Rule ID" },
  { accessorKey: "description", header: "Description" },
  { accessorKey: "field", header: "Field" },
  { accessorKey: "operator", header: "Op" },
  { accessorKey: "threshold", header: "Threshold" },
  {
    accessorKey: "severity",
    header: "Severity",
    cell: ({ getValue }) => {
      const value = getValue<string>();
      const cls = value === "critical" ? "text-[var(--bear)]" : "text-[var(--neutral)]";
      return <span className={cls}>{value}</span>;
    },
  },
];

const hitsColumns: ColumnDef<Record<string, unknown>, unknown>[] = [
  { accessorKey: "rule_id", header: "Rule" },
  { accessorKey: "message", header: "Message" },
  { accessorKey: "severity", header: "Severity" },
  { accessorKey: "observed", header: "Observed" },
  { accessorKey: "limit", header: "Limit" },
];

const historyColumns: ColumnDef<HistoryRow, unknown>[] = [
  { accessorKey: "timestamp", header: "Timestamp" },
  { accessorKey: "tx_id", header: "Transaction ID" },
  { accessorKey: "symbol", header: "Symbol" },
  { accessorKey: "side", header: "Side" },
  { accessorKey: "quantity", header: "Qty" },
  { accessorKey: "price", header: "Price" },
  { accessorKey: "hits", header: "Hits" },
  {
    accessorKey: "status",
    header: "Status",
    cell: ({ getValue }) => {
      const status = getValue<string>();
      return <span className={status === "FAIL" ? "text-[var(--bear)]" : "text-[var(--bull)]"}>{status}</span>;
    },
  },
];

function buildDefaultTransaction(symbol: string, side: "buy" | "sell", quantity: number, price: number): ComplianceTransaction {
  const now = new Date().toISOString();
  return {
    id: `TXN-${Math.random().toString(16).slice(2, 10).toUpperCase()}`,
    timestamp: now,
    user_id: "USR-001",
    order_id: `ORD-${Math.random().toString(16).slice(2, 10).toUpperCase()}`,
    symbol,
    side,
    quantity,
    price,
  };
}

export function ComplianceContent() {
  const [rules, setRules] = React.useState<ComplianceRule[]>([]);

  const [symbol, setSymbol] = React.useState("THYAO");
  const [side, setSide] = React.useState<"buy" | "sell">("buy");
  const [quantity, setQuantity] = React.useState(1000);
  const [price, setPrice] = React.useState(150);

  const [checkResult, setCheckResult] = React.useState<ComplianceResult | null>(null);
  const [history, setHistory] = React.useState<HistoryRow[]>([]);

  const [positionCsv, setPositionCsv] = React.useState("THYAO,1200000,1000000\nEREGL,450000,500000\nASELS,780000,750000");
  const [positionBreaches, setPositionBreaches] = React.useState<Array<Record<string, unknown>>>([]);

  const [activityInput, setActivityInput] = React.useState(
    "USR-001\nUSR-001\nUSR-002\nUSR-003\nUSR-003\nUSR-004\nUSR-001\nUSR-001\nUSR-001",
  );
  const [anomalyResult, setAnomalyResult] = React.useState<ActivityAnomalyResult | null>(null);

  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);

  React.useEffect(() => {
    let cancelled = false;

    const loadRules = async () => {
      try {
        const response = await getDefaultComplianceRules();
        if (!cancelled) setRules(response.rules);
      } catch {
        // no-op
      }
    };

    void loadRules();
    return () => {
      cancelled = true;
    };
  }, []);

  const runTransactionCheck = async () => {
    setError(null);
    setIsLoading(true);
    try {
      const transaction = buildDefaultTransaction(symbol, side, quantity, price);
      const result = await runComplianceCheck({ transaction, rules });
      setCheckResult(result);

      const row: HistoryRow = {
        timestamp: transaction.timestamp,
        tx_id: transaction.id,
        symbol: transaction.symbol,
        side: transaction.side,
        quantity: transaction.quantity,
        price: transaction.price,
        hits: result.hits.length,
        status: result.status,
      };
      setHistory((prev) => [row, ...prev]);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  };

  const checkLimits = async () => {
    setError(null);
    setIsLoading(true);
    try {
      const positions = positionCsv
        .trim()
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          const [sym, value, limit] = line.split(",").map((x) => x.trim());
          return {
            symbol: sym,
            value: Number(value),
            limit: Number(limit),
          };
        })
        .filter((row) => row.symbol && Number.isFinite(row.value) && Number.isFinite(row.limit));

      const response = await checkPositionLimits(positions);
      setPositionBreaches((response.breaches as Array<Record<string, unknown>>) ?? []);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  };

  const detectAnomalies = async () => {
    setError(null);
    setIsLoading(true);
    try {
      const events = activityInput
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((user_id) => ({ user_id }));
      const response = await checkActivityAnomalies(events);
      setAnomalyResult(response);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  };

  const hitIds = React.useMemo(() => new Set((checkResult?.hits ?? []).map((hit) => hit.rule_id)), [checkResult]);

  return (
    <StaggerReveal>
      <StaggerItem>
        <PageHeader title="Compliance" subtitle="Rule engine checks, anomalies, position limits, and audit history" />
      </StaggerItem>

      <PageScaffold>
        <PageMain className="lg:col-span-12 xl:col-span-12">
          <PageSectionStack>
            <ApiError error={error} />

            <StaggerItem>
            <SectionCard title={`Rule Library (${rules.length})`} noPadding>
              <DataTable columns={rulesColumns} data={rules} pageSize={10} enableExport exportFilename="compliance-rules.csv" />
            </SectionCard>
            </StaggerItem>

            <StaggerItem>
            <SectionCard title="Rule Editor">
              <div className="space-y-3">
                {rules.map((rule, idx) => (
                  <div key={rule.id} className="grid grid-cols-1 gap-2 rounded-[var(--radius-sm)] border border-[var(--border)] p-3 md:grid-cols-6">
                    <Input
                      aria-label={`Rule ID ${idx + 1}`}
                      value={rule.id}
                      onChange={(e) =>
                        setRules((prev) =>
                          prev.map((row, i) => (i === idx ? { ...row, id: e.target.value } : row)),
                        )
                      }
                    />
                    <Input
                      aria-label={`Rule Description ${idx + 1}`}
                      value={rule.description}
                      onChange={(e) =>
                        setRules((prev) =>
                          prev.map((row, i) => (i === idx ? { ...row, description: e.target.value } : row)),
                        )
                      }
                    />
                    <Input
                      aria-label={`Rule Field ${idx + 1}`}
                      value={rule.field}
                      onChange={(e) =>
                        setRules((prev) =>
                          prev.map((row, i) => (i === idx ? { ...row, field: e.target.value } : row)),
                        )
                      }
                    />
                    <SelectInput
                      aria-label={`Rule Operator ${idx + 1}`}
                      value={rule.operator}
                      onChange={(e) =>
                        setRules((prev) =>
                          prev.map((row, i) => (i === idx ? { ...row, operator: e.target.value } : row)),
                        )
                      }
                    >
                      <option value=">">&gt;</option>
                      <option value=">=">&gt;=</option>
                      <option value="<">&lt;</option>
                      <option value="<=">&lt;=</option>
                      <option value="==">==</option>
                      <option value="!=">!=</option>
                    </SelectInput>
                    <Input
                      aria-label={`Rule Threshold ${idx + 1}`}
                      type="number"
                      value={rule.threshold}
                      onChange={(e) =>
                        setRules((prev) =>
                          prev.map((row, i) => (i === idx ? { ...row, threshold: Number(e.target.value) } : row)),
                        )
                      }
                    />
                    <div className="flex items-center gap-2">
                      <SelectInput
                        aria-label={`Rule Severity ${idx + 1}`}
                        value={rule.severity}
                        onChange={(e) =>
                          setRules((prev) =>
                            prev.map((row, i) =>
                              i === idx
                                ? { ...row, severity: e.target.value as "warning" | "critical" }
                                : row,
                            ),
                          )
                        }
                      >
                        <option value="warning">warning</option>
                        <option value="critical">critical</option>
                      </SelectInput>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setRules((prev) => prev.filter((_, i) => i !== idx))}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}

                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    onClick={() =>
                      setRules((prev) => [
                        ...prev,
                        {
                          id: `RULE_${prev.length + 1}`,
                          description: "New rule",
                          field: "quantity",
                          operator: ">",
                          threshold: 0,
                          severity: "warning",
                        },
                      ])
                    }
                  >
                    <Plus className="mr-2 h-4 w-4" />
                    Add Rule
                  </Button>

                  <Button
                    variant="outline"
                    onClick={async () => {
                      const response = await getDefaultComplianceRules();
                      setRules(response.rules);
                    }}
                  >
                    Reset to Defaults
                  </Button>
                </div>
              </div>
            </SectionCard>
            </StaggerItem>

            <StaggerItem>
            <div className="grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-2">
              <SectionCard title="Transaction Check">
                <FormGrid>
                  <FormRow>
                    <FormField label="Symbol" htmlFor="compliance-symbol">
                      <Input id="compliance-symbol" value={symbol} onChange={(e) => setSymbol(e.target.value)} />
                    </FormField>
                    <FormField label="Side" htmlFor="compliance-side">
                      <SelectInput id="compliance-side" value={side} onChange={(e) => setSide(e.target.value as "buy" | "sell")}>
                        <option value="buy">buy</option>
                        <option value="sell">sell</option>
                      </SelectInput>
                    </FormField>
                  </FormRow>

                  <FormRow>
                    <FormField label="Quantity" htmlFor="compliance-quantity">
                      <Input id="compliance-quantity" type="number" value={quantity} onChange={(e) => setQuantity(Number(e.target.value))} />
                    </FormField>
                    <FormField label="Price" htmlFor="compliance-price">
                      <Input id="compliance-price" type="number" value={price} onChange={(e) => setPrice(Number(e.target.value))} />
                    </FormField>
                  </FormRow>

                  <Button data-testid="compliance-run-check" className="w-full" onClick={runTransactionCheck} disabled={isLoading}>
                    {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ShieldCheck className="mr-2 h-4 w-4" />}
                    Run Compliance Check
                  </Button>
                </FormGrid>

                {checkResult && (
                  <div className="mt-4 space-y-3">
                    <div className={`rounded-[var(--radius)] p-3 text-small font-medium ${checkResult.status === "PASS" ? "bg-[var(--bull)]/10 text-[var(--bull)]" : "bg-[var(--bear)]/10 text-[var(--bear)]"}`}>
                      {checkResult.status === "PASS" ? "PASS — All rules satisfied" : `FAIL — ${checkResult.hits.length} rule(s) triggered`}
                    </div>

                    <div className="space-y-2">
                      <p className="text-small font-semibold">Rule Checklist</p>
                      {rules.map((rule) => {
                        const triggered = hitIds.has(rule.id);
                        return (
                          <div key={rule.id} className="rounded-[var(--radius-sm)] border border-[var(--border)] p-2 text-small">
                            <span className={triggered ? "text-[var(--bear)]" : "text-[var(--bull)]"}>
                              {triggered ? "❌" : "✅"}
                            </span>{" "}
                            <span className="font-medium">{rule.id}</span> — {rule.description}
                          </div>
                        );
                      })}
                    </div>

                    {checkResult.hits.length > 0 && (
                      <SectionCard title="Triggered Rules" noPadding>
                        <DataTable columns={hitsColumns} data={checkResult.hits as unknown as Array<Record<string, unknown>>} pageSize={8} />
                      </SectionCard>
                    )}
                  </div>
                )}
              </SectionCard>

              <SectionCard title="Position Limit Monitor">
                <FormGrid>
                  <FormField label="CSV: symbol,value,limit" htmlFor="compliance-position-csv">
                    <Textarea
                      id="compliance-position-csv"
                      className="h-36 resize-y font-mono text-xs"
                      value={positionCsv}
                      onChange={(e) => setPositionCsv(e.target.value)}
                    />
                  </FormField>

                  <Button data-testid="compliance-check-limits" className="w-full" onClick={checkLimits} disabled={isLoading}>
                    {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ShieldCheck className="mr-2 h-4 w-4" />}
                    Check Position Limits
                  </Button>

                  {positionBreaches.length > 0 ? (
                    <DataTable
                      columns={[
                        { accessorKey: "symbol", header: "Symbol" },
                        { accessorKey: "value", header: "Current Value" },
                        { accessorKey: "limit", header: "Limit" },
                        {
                          accessorKey: "breach",
                          header: "Breach",
                          cell: ({ getValue }) =>
                            getValue<boolean>() ? <span className="text-[var(--bear)]">YES</span> : <span className="text-[var(--bull)]">NO</span>,
                        },
                      ]}
                      data={positionBreaches}
                      pageSize={8}
                    />
                  ) : (
                    <p className="text-small text-[var(--text-muted)]">No breaches currently loaded.</p>
                  )}
                </FormGrid>
              </SectionCard>
            </div>
            </StaggerItem>

            <StaggerItem>
            <div className="grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-2">
              <SectionCard title="User Activity Anomaly Detection">
                <FormGrid>
                  <FormField label="User IDs (one per line)" htmlFor="compliance-activity-input">
                    <Textarea
                      id="compliance-activity-input"
                      className="h-36 resize-y font-mono text-xs"
                      value={activityInput}
                      onChange={(e) => setActivityInput(e.target.value)}
                    />
                  </FormField>

                  <Button data-testid="compliance-detect-anomalies" className="w-full" onClick={detectAnomalies} disabled={isLoading}>
                    {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ShieldCheck className="mr-2 h-4 w-4" />}
                    Detect Anomalies
                  </Button>
                </FormGrid>

                {anomalyResult && (
                  <SectionCard title={`Anomalies (${anomalyResult.anomaly_count})`} noPadding>
                    <DataTable
                      columns={[
                        { accessorKey: "user_id", header: "User ID" },
                        { accessorKey: "actions_per_hour", header: "Actions/Hour" },
                        { accessorKey: "z_score", header: "Z-Score" },
                      ]}
                      data={anomalyResult.anomalies as unknown as Array<Record<string, unknown>>}
                      pageSize={8}
                    />
                  </SectionCard>
                )}
              </SectionCard>

              <SectionCard title="Run History" noPadding>
                <div className="p-[var(--space-3)]">
                  <div className="mb-2 flex justify-end">
                    <Button variant="outline" size="sm" onClick={() => setHistory([])}>
                      Clear History
                    </Button>
                  </div>
                  <DataTable columns={historyColumns} data={history} pageSize={8} enableExport exportFilename="compliance-history.csv" />
                </div>
              </SectionCard>
            </div>
            </StaggerItem>
          </PageSectionStack>
        </PageMain>
      </PageScaffold>
    </StaggerReveal>
  );
}
