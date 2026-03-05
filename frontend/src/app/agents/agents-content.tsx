"use client";

import * as React from "react";
import type { AgentSessionEntry } from "@/lib/types";
import { PageHeader } from "@/components/shared/page-header";
import { SectionCard } from "@/components/shared/section-card";
import { PageScaffold, PageMain, PageSectionStack } from "@/components/shared/page-scaffold";
import { FormField, FormGrid, FormRow } from "@/components/shared/form-field";
import { Button } from "@/components/ui/button";
import { SelectInput } from "@/components/ui/select-input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Bot, Clock, Download, Sparkles, Trash2 } from "lucide-react";
import { StaggerReveal, StaggerItem } from "@/components/shared/stagger-reveal";

const PLANNED_AGENTS = [
  {
    name: "Market Research Agent",
    desc: "Collects macro + sector context and summarizes impact by regime state.",
    icon: Sparkles,
    tags: "LLM · Sentiment · Macro",
  },
  {
    name: "Backtesting Copilot",
    desc: "Interprets backtest outcomes and suggests practical parameter adjustments.",
    icon: Bot,
    tags: "LLM · Backtest · Tuning",
  },
  {
    name: "Risk Monitor Agent",
    desc: "Tracks risk thresholds and drafts incident-oriented diagnostics.",
    icon: Clock,
    tags: "LLM · Risk · Alerts",
  },
];

function nowIso(): string {
  return new Date().toISOString();
}

function makeId(prefix: string): string {
  const core = typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${prefix}-${core}`;
}

function downloadJson(filename: string, payload: unknown) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function AgentsContent() {
  const [selectedAgent, setSelectedAgent] = React.useState(PLANNED_AGENTS[0].name);
  const [prompt, setPrompt] = React.useState("");
  const [newestFirst, setNewestFirst] = React.useState(true);
  const [sessionLog, setSessionLog] = React.useState<AgentSessionEntry[]>([
    {
      id: "boot",
      timestamp: nowIso(),
      role: "system",
      agent: "System",
      content: "Agent framework initialized in beta stub mode. No external LLM calls are enabled.",
    },
  ]);

  const orderedLog = newestFirst ? [...sessionLog].reverse() : sessionLog;

  const sendPrompt = () => {
    const content = prompt.trim();
    if (!content) return;

    const timestamp = nowIso();
    const userEvent: AgentSessionEntry = {
      id: makeId("u"),
      timestamp,
      role: "user",
      agent: "User",
      content,
    };

    const stubReply: AgentSessionEntry = {
      id: makeId("a"),
      timestamp,
      role: "agent",
      agent: selectedAgent,
      content: `[BETA STUB] Prompt captured for ${selectedAgent}. External LLM execution is intentionally disabled in this phase. Preview: "${content.slice(0, 120)}${content.length > 120 ? "..." : ""}"`,
    };

    setSessionLog((prev) => [...prev, userEvent, stubReply]);
    setPrompt("");
  };

  return (
    <StaggerReveal>
      <StaggerItem>
        <PageHeader title="Agents" subtitle="Beta placeholder — local stub interactions only" />
      </StaggerItem>

      <PageScaffold>
        <PageMain className="lg:col-span-12 xl:col-span-12">
          <PageSectionStack>
            <StaggerItem>
            <SectionCard className="border-[var(--accent)]/30 bg-[var(--accent-dim)]/40">
              <h2 className="text-h3 text-[var(--text)]">Beta Mode</h2>
              <p className="mt-1 text-small text-[var(--text-muted)]">
                This route is intentionally a beta placeholder. Session logs are local stubs and no real LLM provider is invoked.
              </p>
            </SectionCard>
            </StaggerItem>

            <StaggerItem>
            <div className="grid grid-cols-1 gap-[var(--grid-gap)] md:grid-cols-3">
              {PLANNED_AGENTS.map((agent) => {
                const Icon = agent.icon;
                return (
                  <SectionCard key={agent.name}>
                    <div className="flex items-start gap-[var(--space-3)]">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--accent)]/10">
                        <Icon className="h-5 w-5 text-[var(--accent)]" />
                      </div>
                      <div>
                        <h3 className="text-h3 text-[var(--text)]">{agent.name}</h3>
                        <p className="mt-1 text-small leading-relaxed text-[var(--text-muted)]">{agent.desc}</p>
                        <p className="mt-2 text-micro uppercase tracking-wide text-[var(--text-faint)]">{agent.tags}</p>
                        <span className="mt-2 inline-flex rounded border border-[var(--border)] bg-[var(--surface-2)] px-2 py-0.5 text-micro text-[var(--text-muted)]">
                          Planned
                        </span>
                      </div>
                    </div>
                  </SectionCard>
                );
              })}
            </div>
            </StaggerItem>

            <StaggerItem>
            <SectionCard title="Stub Interaction">
              <FormGrid>
                <FormRow>
                  <FormField label="Target Agent" htmlFor="agents-target">
                    <SelectInput id="agents-target" value={selectedAgent} onChange={(e) => setSelectedAgent(e.target.value)}>
                      {PLANNED_AGENTS.map((agent) => (
                        <option key={agent.name} value={agent.name}>
                          {agent.name}
                        </option>
                      ))}
                    </SelectInput>
                  </FormField>
                </FormRow>

                <FormField label="Prompt" htmlFor="agents-prompt">
                  <Textarea
                    id="agents-prompt"
                    className="h-28 resize-y"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Describe the analysis task you would route to this agent..."
                  />
                </FormField>

                <div className="flex flex-wrap gap-2">
                  <Button onClick={sendPrompt}>Send Stub Prompt</Button>
                  <Button
                    variant="outline"
                    onClick={() =>
                      downloadJson(
                        `agent_session_log_${new Date().toISOString().replace(/[:.]/g, "-")}.json`,
                        sessionLog,
                      )
                    }
                  >
                    <Download className="mr-2 h-4 w-4" />
                    Export Log
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() =>
                      setSessionLog([
                        {
                          id: makeId("system"),
                          timestamp: nowIso(),
                          role: "system",
                          agent: "System",
                          content: "Session log cleared. Stub mode ready.",
                        },
                      ])
                    }
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    Clear Log
                  </Button>
                </div>
              </FormGrid>
            </SectionCard>
            </StaggerItem>

            <StaggerItem>
            <SectionCard title="Session Log">
              <label className="mb-3 flex items-center gap-2 text-small text-[var(--text-muted)]">
                <Checkbox checked={newestFirst} onChange={(e) => setNewestFirst(e.target.checked)} />
                Newest first
              </label>

              <div className="space-y-2">
                {orderedLog.map((entry) => (
                  <div key={entry.id} className="rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface-2)] p-3">
                    <p className="text-micro uppercase tracking-wide text-[var(--text-faint)]">
                      {entry.role} · {entry.agent} · {entry.timestamp}
                    </p>
                    <p className="mt-1 text-small text-[var(--text-muted)]">{entry.content}</p>
                  </div>
                ))}
              </div>
            </SectionCard>
            </StaggerItem>
          </PageSectionStack>
        </PageMain>
      </PageScaffold>
    </StaggerReveal>
  );
}
