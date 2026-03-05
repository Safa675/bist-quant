"use client";

import * as React from "react";
import {
  calculateCryptoSizing,
  calculateGreeks,
  calculatePipValue,
  runStressTest,
} from "@/lib/api";
import type {
  CryptoSizingInput,
  CryptoTradePlan,
  GreeksInput,
  GreeksResult,
  PipValueResult,
  StressResult,
} from "@/lib/types";
import { PageHeader } from "@/components/shared/page-header";
import { SectionCard } from "@/components/shared/section-card";
import { KpiCard } from "@/components/shared/kpi-card";
import { ApiError } from "@/components/shared/api-error";
import { DataTable } from "@/components/shared/data-table";
import { PageScaffold, PageMain, PageSectionStack } from "@/components/shared/page-scaffold";
import { FormField, FormGrid, FormRow } from "@/components/shared/form-field";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LazyBarMetricsChart as BarMetricsChart,
  LazyGreeksBarChart as GreeksBarChart,
} from "@/components/charts/lazy";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { SelectInput } from "@/components/ui/select-input";
import type { ColumnDef } from "@tanstack/react-table";
import { AlertTriangle, Bitcoin, Calculator, Loader2, Scale } from "lucide-react";
import { StaggerReveal, StaggerItem } from "@/components/shared/stagger-reveal";

const STRESS_TEMPLATES: Record<string, string> = {
  "2008 GFC": "Global Equities,-55,1.0\nCredit Spreads,500,-0.3\nUSD FX,15,-0.2",
  "2020 COVID Crash": "Equities,-34,1.0\nOil,-70,0.1\nVIX,400,-0.5",
  "Turkey 2021 Currency Crisis": "USD/TRY,60,-0.8\nBIST100,-30,1.0\nTurkish Bonds,-25,0.4",
  "+300bps Rate Shock": "Interest Rates,300,-0.25\nEquities,-15,1.0",
};

const stressColumns: ColumnDef<{ factor: string; loss_pct: number; contribution_pct: number }, unknown>[] = [
  { accessorKey: "factor", header: "Factor" },
  {
    accessorKey: "loss_pct",
    header: "Loss %",
    cell: ({ getValue }) => {
      const value = getValue<number>();
      return <span className={value < 0 ? "text-[var(--bear)]" : "text-[var(--bull)]"}>{(value * 100).toFixed(2)}%</span>;
    },
  },
  {
    accessorKey: "contribution_pct",
    header: "Contribution %",
    cell: ({ getValue }) => `${(getValue<number>() * 100).toFixed(1)}%`,
  },
];

function parseShockCsv(input: string): Array<{ factor: string; shock_pct: number; beta: number }> {
  return input
    .trim()
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [factor, shock_pct, beta] = line.split(",").map((s) => s.trim());
      return {
        factor,
        shock_pct: Number(shock_pct),
        beta: Number(beta),
      };
    })
    .filter((row) => row.factor && Number.isFinite(row.shock_pct) && Number.isFinite(row.beta));
}

function buildSmileData(
  base: GreeksInput,
  values: GreeksResult[],
): Array<{ name: string; value: number }> {
  const strikes = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2].map((ratio) => base.spot * ratio);
  return values.map((result, idx) => ({ name: strikes[idx].toFixed(0), value: result.theoretical_price }));
}

export function ProfessionalContent() {
  const [optionType, setOptionType] = React.useState<"call" | "put">("call");
  const [spot, setSpot] = React.useState(100);
  const [strike, setStrike] = React.useState(100);
  const [daysToExpiry, setDaysToExpiry] = React.useState(30);
  const [volatilityPct, setVolatilityPct] = React.useState(25);
  const [riskFreeRatePct, setRiskFreeRatePct] = React.useState(5);
  const [greeksResult, setGreeksResult] = React.useState<GreeksResult | null>(null);
  const [smileData, setSmileData] = React.useState<Array<{ name: string; value: number }>>([]);

  const [stressTemplate, setStressTemplate] = React.useState("");
  const [stressShocks, setStressShocks] = React.useState("Equity,-20,1.0\nFX,10,0.5\nRates,3,0.3");
  const [portfolioValue, setPortfolioValue] = React.useState(1_000_000);
  const [stressResult, setStressResult] = React.useState<StressResult | null>(null);

  const [pair, setPair] = React.useState("BTC/USDT");
  const [side, setSide] = React.useState<"long" | "short">("long");
  const [entryPrice, setEntryPrice] = React.useState(65_000);
  const [equity, setEquity] = React.useState(10_000);
  const [riskPct, setRiskPct] = React.useState(2);
  const [leverage, setLeverage] = React.useState(5);
  const [stopDistancePct, setStopDistancePct] = React.useState(3);
  const [cryptoPlan, setCryptoPlan] = React.useState<CryptoTradePlan | null>(null);

  const [pipPair, setPipPair] = React.useState("EURUSD");
  const [lotSize, setLotSize] = React.useState(100_000);
  const [conversionRate, setConversionRate] = React.useState(1);
  const [pipResult, setPipResult] = React.useState<PipValueResult | null>(null);

  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);

  const runGreeks = async () => {
    setError(null);
    setIsLoading(true);
    try {
      const input: GreeksInput = {
        option_type: optionType,
        spot,
        strike,
        time_years: Math.max(daysToExpiry, 1) / 365,
        volatility: volatilityPct / 100,
        risk_free_rate: riskFreeRatePct / 100,
      };

      const result = await calculateGreeks(input);
      setGreeksResult(result);

      const strikes = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2].map((ratio) => spot * ratio);
      const smile = await Promise.all(
        strikes.map((k) =>
          calculateGreeks({
            ...input,
            option_type: "call",
            strike: k,
          }),
        ),
      );
      setSmileData(buildSmileData(input, smile));
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  };

  const runStress = async () => {
    setError(null);
    setIsLoading(true);
    try {
      const shocks = parseShockCsv(stressShocks);
      if (shocks.length === 0) {
        throw new Error("Provide at least one valid shock row: factor,shock_pct,beta");
      }
      const result = await runStressTest({
        shocks,
        portfolio_value: portfolioValue,
      });
      setStressResult(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  };

  const runCrypto = async () => {
    setError(null);
    setIsLoading(true);
    try {
      const input: CryptoSizingInput = {
        pair,
        side,
        entry_price: entryPrice,
        equity,
        risk_pct: riskPct,
        leverage,
        stop_distance_pct: stopDistancePct,
      };
      const result = await calculateCryptoSizing(input);
      setCryptoPlan(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  };

  const runPip = async () => {
    setError(null);
    setIsLoading(true);
    try {
      const result = await calculatePipValue({
        pair: pipPair,
        lot_size: lotSize,
        account_conversion_rate: conversionRate,
      });
      setPipResult(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <StaggerReveal>
      <StaggerItem>
        <PageHeader title="Professional" subtitle="Greeks, stress testing, crypto sizing, and pip-value utilities" />
      </StaggerItem>

      <PageScaffold>
        <PageMain className="lg:col-span-12 xl:col-span-12">
          <PageSectionStack>
            <ApiError error={error} />

            <StaggerItem>
            <Tabs defaultValue="greeks">
              <TabsList className="w-full overflow-x-auto">
                <TabsTrigger value="greeks">Options Greeks</TabsTrigger>
                <TabsTrigger value="stress">Stress Test</TabsTrigger>
                <TabsTrigger value="crypto">Crypto Sizing</TabsTrigger>
                <TabsTrigger value="pip">Pip Value</TabsTrigger>
              </TabsList>

              <TabsContent value="greeks">
                <div className="grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-3">
                  <SectionCard title="Black-Scholes Inputs">
                    <FormGrid>
                      <FormField label="Option Type" htmlFor="professional-option-type">
                        <SelectInput id="professional-option-type" value={optionType} onChange={(e) => setOptionType(e.target.value as "call" | "put")}> 
                          <option value="call">Call</option>
                          <option value="put">Put</option>
                        </SelectInput>
                      </FormField>

                      <FormRow>
                        <Input aria-label="Spot" type="number" value={spot} onChange={(e) => setSpot(Number(e.target.value))} />
                        <Input aria-label="Strike" type="number" value={strike} onChange={(e) => setStrike(Number(e.target.value))} />
                      </FormRow>

                      <FormRow>
                        <Input aria-label="Days to Expiry" type="number" min={1} value={daysToExpiry} onChange={(e) => setDaysToExpiry(Number(e.target.value))} />
                        <Input aria-label="Implied Volatility %" type="number" min={1} max={200} value={volatilityPct} onChange={(e) => setVolatilityPct(Number(e.target.value))} />
                        <Input aria-label="Risk-free Rate %" type="number" min={-5} max={30} value={riskFreeRatePct} onChange={(e) => setRiskFreeRatePct(Number(e.target.value))} />
                      </FormRow>

                      <Button data-testid="professional-run-greeks" className="w-full" onClick={runGreeks} disabled={isLoading}>
                        {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Calculator className="mr-2 h-4 w-4" />}
                        Calculate Greeks
                      </Button>
                    </FormGrid>
                  </SectionCard>

                  <div className="space-y-[var(--section-gap)] lg:col-span-2">
                    {greeksResult && (
                      <>
                        <div className="grid grid-cols-2 gap-[var(--grid-gap)] sm:grid-cols-3">
                          <KpiCard title="Theo Price" value={greeksResult.theoretical_price} decimals={4} animate={false} />
                          <KpiCard title="Delta" value={greeksResult.delta} decimals={4} animate={false} />
                          <KpiCard title="Gamma" value={greeksResult.gamma} decimals={6} animate={false} />
                          <KpiCard title="Theta / Day" value={greeksResult.theta_per_day} decimals={4} animate={false} />
                          <KpiCard title="Vega / 1%" value={greeksResult.vega_per_1pct} decimals={4} animate={false} />
                          <KpiCard title="Rho / 1%" value={greeksResult.rho_per_1pct} decimals={4} animate={false} />
                        </div>

                        <SectionCard title="Greek Sensitivity Overview">
                          <GreeksBarChart greeks={greeksResult} height={250} />
                        </SectionCard>
                      </>
                    )}

                    {smileData.length > 0 && (
                      <SectionCard title="Implied Volatility Smile (Illustrative)">
                        <BarMetricsChart data={smileData} horizontal={false} height={280} />
                      </SectionCard>
                    )}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="stress">
                <div className="grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-3">
                  <SectionCard title="Scenario Editor">
                    <FormGrid>
                      <FormField label="Template" htmlFor="professional-stress-template">
                        <SelectInput
                          id="professional-stress-template"
                          value={stressTemplate}
                          onChange={(e) => {
                            const next = e.target.value;
                            setStressTemplate(next);
                            if (next && STRESS_TEMPLATES[next]) {
                              setStressShocks(STRESS_TEMPLATES[next]);
                            }
                          }}
                        >
                          <option value="">Custom</option>
                          {Object.keys(STRESS_TEMPLATES).map((key) => (
                            <option key={key} value={key}>
                              {key}
                            </option>
                          ))}
                        </SelectInput>
                      </FormField>

                      <FormField label="Portfolio Value" htmlFor="professional-stress-value">
                        <Input
                          id="professional-stress-value"
                          type="number"
                          value={portfolioValue}
                          onChange={(e) => setPortfolioValue(Number(e.target.value))}
                        />
                      </FormField>

                      <FormField label="CSV: factor,shock_pct,beta" htmlFor="professional-stress-shocks">
                        <Textarea
                          id="professional-stress-shocks"
                          className="h-40 resize-y font-mono text-xs"
                          value={stressShocks}
                          onChange={(e) => setStressShocks(e.target.value)}
                        />
                      </FormField>

                      <Button data-testid="professional-run-stress" className="w-full" onClick={runStress} disabled={isLoading}>
                        {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <AlertTriangle className="mr-2 h-4 w-4" />}
                        Run Stress Test
                      </Button>
                    </FormGrid>
                  </SectionCard>

                  <div className="space-y-[var(--section-gap)] lg:col-span-2">
                    {stressResult && (
                      <>
                        <div className="grid grid-cols-1 gap-[var(--grid-gap)] sm:grid-cols-2">
                          <KpiCard title="Scenario Loss %" value={stressResult.scenario_loss_pct * 100} suffix="%" decimals={2} animate={false} />
                          <KpiCard title="Scenario Loss" value={stressResult.scenario_loss_value} suffix=" USD" decimals={0} animate={false} />
                        </div>

                        <SectionCard title="Factor Contribution">
                          <BarMetricsChart
                            data={stressResult.by_factor.map((row) => ({
                              name: row.factor,
                              value: row.loss_pct * 100,
                            }))}
                            horizontal={false}
                            height={280}
                          />
                        </SectionCard>

                        <SectionCard title="Stress Detail" noPadding>
                          <DataTable columns={stressColumns} data={stressResult.by_factor} pageSize={10} />
                        </SectionCard>
                      </>
                    )}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="crypto">
                <div className="grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-3">
                  <SectionCard title="Trade Parameters">
                    <FormGrid>
                      <FormRow>
                        <Input aria-label="Pair" value={pair} onChange={(e) => setPair(e.target.value)} />
                        <SelectInput value={side} onChange={(e) => setSide(e.target.value as "long" | "short")}> 
                          <option value="long">Long</option>
                          <option value="short">Short</option>
                        </SelectInput>
                      </FormRow>

                      <Input aria-label="Entry Price" type="number" value={entryPrice} onChange={(e) => setEntryPrice(Number(e.target.value))} />

                      <FormRow>
                        <Input aria-label="Equity" type="number" value={equity} onChange={(e) => setEquity(Number(e.target.value))} />
                        <Input aria-label="Risk %" type="number" value={riskPct} onChange={(e) => setRiskPct(Number(e.target.value))} />
                      </FormRow>

                      <FormRow>
                        <Input aria-label="Leverage" type="number" value={leverage} onChange={(e) => setLeverage(Number(e.target.value))} />
                        <Input aria-label="Stop Distance %" type="number" value={stopDistancePct} onChange={(e) => setStopDistancePct(Number(e.target.value))} />
                      </FormRow>

                      <Button data-testid="professional-run-crypto" className="w-full" onClick={runCrypto} disabled={isLoading}>
                        {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Bitcoin className="mr-2 h-4 w-4" />}
                        Calculate Size
                      </Button>
                    </FormGrid>
                  </SectionCard>

                  <div className="lg:col-span-2">
                    {cryptoPlan && (
                      <SectionCard title="Trade Plan">
                        <div className="grid grid-cols-2 gap-[var(--grid-gap)] sm:grid-cols-3">
                          <KpiCard title="Notional" value={cryptoPlan.notional} suffix=" USD" decimals={2} animate={false} />
                          <KpiCard title="Margin" value={cryptoPlan.margin_required} suffix=" USD" decimals={2} animate={false} />
                          <KpiCard title="Quantity" value={cryptoPlan.quantity} decimals={6} animate={false} />
                          <KpiCard title="Liquidation" value={cryptoPlan.liquidation_price} suffix=" USD" decimals={2} animate={false} />
                          <KpiCard title="Max Loss" value={cryptoPlan.max_loss} suffix=" USD" decimals={2} animate={false} />
                          <KpiCard title="Est. Fees" value={cryptoPlan.estimated_fees} suffix=" USD" decimals={2} animate={false} />
                        </div>
                      </SectionCard>
                    )}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="pip">
                <div className="grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-3">
                  <SectionCard title="Forex Pip Inputs">
                    <FormGrid>
                      <Input aria-label="Pair" value={pipPair} onChange={(e) => setPipPair(e.target.value.toUpperCase())} />
                      <Input aria-label="Lot Size" type="number" min={1} value={lotSize} onChange={(e) => setLotSize(Number(e.target.value))} />
                      <Input aria-label="Account Conversion Rate" type="number" min={0.0001} step={0.0001} value={conversionRate} onChange={(e) => setConversionRate(Number(e.target.value))} />

                      <Button data-testid="professional-run-pip" className="w-full" onClick={runPip} disabled={isLoading}>
                        {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Scale className="mr-2 h-4 w-4" />}
                        Compute Pip Value
                      </Button>
                    </FormGrid>
                  </SectionCard>

                  <div className="lg:col-span-2">
                    {pipResult && (
                      <SectionCard title="Pip Value Result">
                        <div className="grid grid-cols-2 gap-[var(--grid-gap)] sm:grid-cols-4">
                          <KpiCard title="Pair" value={Number.NaN} animate={false} />
                          <KpiCard title="Pip Size" value={pipResult.pip_size} decimals={6} animate={false} />
                          <KpiCard title="Pip (Quote)" value={pipResult.pip_value_quote} decimals={6} animate={false} />
                          <KpiCard title="Pip (Account)" value={pipResult.pip_value_account} decimals={6} animate={false} />
                        </div>
                        <p className="mt-2 text-small text-[var(--text-muted)]">Pair: {pipResult.pair}</p>
                      </SectionCard>
                    )}
                  </div>
                </div>
              </TabsContent>
            </Tabs>
            </StaggerItem>
          </PageSectionStack>
        </PageMain>
      </PageScaffold>
    </StaggerReveal>
  );
}
