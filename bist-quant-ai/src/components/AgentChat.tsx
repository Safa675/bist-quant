"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Target, Shield, Brain, Sparkles, Search } from "lucide-react";

interface Signal {
    name: string;
    cagr: number;
    sharpe: number;
    max_dd: number;
    ytd: number;
}

interface Props {
    holdings: Record<string, string[]>;
    signals: Signal[];
    regime: string;
}

type AgentType = "portfolio" | "risk" | "analyst" | "research";

interface Message {
    role: "user" | "agent";
    agent: AgentType;
    content: string;
    timestamp: Date;
}

type AgentHealthState = "checking" | "ok" | "error";

const AGENT_CONFIG = {
    portfolio: {
        name: "Portfolio Manager",
        icon: Target,
        color: "#10b981",
        greeting: "I manage factor allocations and explain portfolio decisions. Ask me about holdings, signals, or performance.",
    },
    risk: {
        name: "Risk Manager",
        icon: Shield,
        color: "#06b6d4",
        greeting: "I monitor drawdowns, volatility, regime shifts, and stop-losses. Ask me about risk metrics or market conditions.",
    },
    analyst: {
        name: "Market Analyst",
        icon: Brain,
        color: "#8b5cf6",
        greeting: "I analyze BIST trends, sector rotations, and macro indicators. Ask me about market drivers or sector insights.",
    },
    research: {
        name: "Research Analyst",
        icon: Search,
        color: "#f59e0b",
        greeting: "I can screen stocks, fetch financials, and run technical scans using live BIST data. Ask me to find undervalued stocks, check fundamentals, or scan for breakouts.",
    },
};

const SAMPLE_QUESTIONS: Record<AgentType, string[]> = {
    portfolio: [
        "What are our top holdings?",
        "Why is momentum underperforming?",
        "Which factors have the best YTD?",
    ],
    risk: [
        "What's our max drawdown risk?",
        "Is the regime about to change?",
        "How is vol-targeting working?",
    ],
    analyst: [
        "What's driving BIST this week?",
        "Which sectors are rotating?",
        "How does USD/TRY affect us?",
    ],
    research: [
        "Find undervalued stocks with P/E < 5",
        "Show me THYAO's financial ratios",
        "Scan XU100 for oversold stocks",
        "Compare banking sector metrics",
    ],
};

async function fetchAgentResponse(
    agent: AgentType,
    query: string,
    context: {
        regime: string;
        signals: Signal[];
        holdings: Record<string, string[]>;
    }
): Promise<string> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 25_000);
    let response: Response;
    try {
        response = await fetch(`/api/agents/${agent}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ query, context }),
            signal: controller.signal,
        });
    } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
            throw new Error("Agent API request timed out after 25s.");
        }
        throw error;
    } finally {
        clearTimeout(timeout);
    }

    let payload: { response?: unknown; error?: unknown; request_id?: unknown; mode?: unknown } | null = null;
    try {
        payload = await response.json();
    } catch {
        payload = null;
    }

    if (!response.ok) {
        const errorMessage =
            payload && typeof payload.error === "string"
                ? payload.error
                : `Agent API request failed with status ${response.status}`;
        throw new Error(errorMessage);
    }

    if (!payload || typeof payload.response !== "string" || !payload.response.trim()) {
        throw new Error("Agent API returned an empty response.");
    }

    if (typeof payload.request_id !== "string" || payload.request_id.length < 8) {
        throw new Error("Agent API returned invalid request metadata (request_id missing).");
    }

    if (payload.mode !== "llm_live") {
        throw new Error("Agent API is not in live LLM mode.");
    }

    return payload.response;
}

export default function AgentChat({ holdings, signals, regime }: Props) {
    const [activeAgent, setActiveAgent] = useState<AgentType>("portfolio");
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const [healthState, setHealthState] = useState<AgentHealthState>("checking");
    const [healthMessage, setHealthMessage] = useState<string>("");
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    useEffect(() => {
        const check = async () => {
            try {
                const res = await fetch("/api/agents/health", { cache: "no-store" });
                const payload = await res.json() as {
                    ok?: unknown;
                    mode?: unknown;
                    error?: unknown;
                };
                if (!res.ok || payload.mode !== "llm_live" || payload.ok !== true) {
                    const msg = typeof payload.error === "string" ? payload.error : "Live LLM health check failed.";
                    setHealthState("error");
                    setHealthMessage(msg);
                    return;
                }
                setHealthState("ok");
                setHealthMessage("");
            } catch (error) {
                setHealthState("error");
                setHealthMessage(error instanceof Error ? error.message : "Live LLM health check failed.");
            }
        };
        check();
    }, []);

    const handleSend = async (text?: string) => {
        const query = text || input.trim();
        if (!query) return;

        if (healthState !== "ok") {
            const stateLabel = healthState === "checking" ? "still checking LLM health" : "LLM is unavailable";
            const msg: Message = {
                role: "agent",
                agent: activeAgent,
                content: `**LLM Error**\n\nCannot send message because ${stateLabel}.\n\n${healthMessage || "Run /api/agents/health and fix Azure config first."}`,
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, msg]);
            return;
        }

        const userMsg: Message = { role: "user", agent: activeAgent, content: query, timestamp: new Date() };
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setIsTyping(true);

        try {
            const response = await fetchAgentResponse(activeAgent, query, {
                regime,
                signals,
                holdings,
            });
            const agentMsg: Message = { role: "agent", agent: activeAgent, content: response, timestamp: new Date() };
            setMessages((prev) => [...prev, agentMsg]);
        } catch (error) {
            const message = error instanceof Error ? error.message : "Unknown agent error";
            const agentMsg: Message = {
                role: "agent",
                agent: activeAgent,
                content: `**LLM Error**\n\n${message}`,
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, agentMsg]);
        } finally {
            setIsTyping(false);
        }
    };

    const config = AGENT_CONFIG[activeAgent];
    const AgentIcon = config.icon;

    return (
        <div className="glass-card" style={{ display: "flex", flexDirection: "column", height: 640, overflow: "hidden" }}>
            {/* Agent selector tabs */}
            <div style={{ display: "flex", borderBottom: "1px solid var(--border-subtle)" }}>
                {(Object.entries(AGENT_CONFIG) as [AgentType, typeof AGENT_CONFIG.portfolio][]).map(([key, cfg]) => {
                    const Icon = cfg.icon;
                    const isActive = activeAgent === key;
                    return (
                        <button
                            key={key}
                            onClick={() => setActiveAgent(key)}
                            style={{
                                flex: 1,
                                padding: "12px 8px",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                gap: 6,
                                border: "none",
                                borderBottom: isActive ? `2px solid ${cfg.color}` : "2px solid transparent",
                                background: isActive ? `${cfg.color}10` : "transparent",
                                color: isActive ? cfg.color : "var(--text-muted)",
                                fontSize: "0.75rem",
                                fontWeight: 600,
                                cursor: "pointer",
                                transition: "all var(--transition-fast)",
                            }}
                        >
                            <Icon size={14} />
                            {cfg.name.split(" ")[0]}
                        </button>
                    );
                })}
            </div>

            {/* Agent header */}
            <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border-subtle)", display: "flex", alignItems: "center", gap: 10 }}>
                <div
                    style={{
                        width: 36,
                        height: 36,
                        borderRadius: "var(--radius-sm)",
                        background: `${config.color}15`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                    }}
                >
                    <AgentIcon size={18} color={config.color} />
                </div>
                <div>
                    <div style={{ fontWeight: 700, fontSize: "0.9rem" }}>{config.name}</div>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
                        <span style={{ width: 6, height: 6, borderRadius: "50%", background: config.color, display: "inline-block" }} />
                        {healthState === "ok" ? "Online · Live LLM" : healthState === "checking" ? "Checking LLM..." : "Offline · LLM Error"}
                    </div>
                </div>
                <Sparkles size={14} color={config.color} style={{ marginLeft: "auto", opacity: 0.5 }} />
            </div>
            {healthState === "error" && (
                <div style={{ padding: "10px 16px", borderBottom: "1px solid #7f1d1d", background: "rgba(127,29,29,0.25)", color: "#fecaca", fontSize: "0.78rem", lineHeight: 1.5 }}>
                    <strong>LLM unavailable:</strong> {healthMessage || "Run /api/agents/health and fix Azure config."}
                </div>
            )}

            {/* Messages area */}
            <div className="chat-messages" style={{ flex: 1, overflowY: "auto", padding: 16, display: "flex", flexDirection: "column", gap: 8 }}>
                {/* Greeting */}
                {messages.filter(m => m.agent === activeAgent).length === 0 && (
                    <div>
                        <div className="chat-bubble chat-bubble-agent" style={{ borderColor: `${config.color}20` }}>
                            <p style={{ margin: 0, lineHeight: 1.5 }}>{config.greeting}</p>
                        </div>
                        {/* Suggested questions */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
                            {SAMPLE_QUESTIONS[activeAgent].map((q) => (
                                <button
                                    key={q}
                                    onClick={() => handleSend(q)}
                                    disabled={healthState !== "ok" || isTyping}
                                    style={{
                                        padding: "8px 12px",
                                        borderRadius: "var(--radius-sm)",
                                        border: "1px solid var(--border-subtle)",
                                        background: "transparent",
                                        color: "var(--text-secondary)",
                                        fontSize: "0.8rem",
                                        cursor: healthState === "ok" ? "pointer" : "not-allowed",
                                        opacity: healthState === "ok" ? 1 : 0.55,
                                        textAlign: "left",
                                        transition: "all var(--transition-fast)",
                                    }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.borderColor = config.color;
                                        e.currentTarget.style.color = config.color;
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.borderColor = "var(--border-subtle)";
                                        e.currentTarget.style.color = "var(--text-secondary)";
                                    }}
                                >
                                    {q}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Conversation */}
                {messages
                    .filter((m) => m.agent === activeAgent)
                    .map((msg, i) => (
                        <div key={i} className={`chat-bubble ${msg.role === "user" ? "chat-bubble-user" : "chat-bubble-agent"}`}>
                            <div
                                style={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}
                                dangerouslySetInnerHTML={{
                                    __html: msg.content
                                        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                                        .replace(/\n/g, "<br />"),
                                }}
                            />
                        </div>
                    ))}

                {/* Typing indicator */}
                {isTyping && (
                    <div className="chat-bubble chat-bubble-agent">
                        <div className="typing-indicator">
                            <span /><span /><span />
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="chat-input-wrap" style={{ display: "flex", gap: 8, padding: "12px 16px" }}>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSend()}
                    placeholder={healthState === "ok" ? `Ask ${config.name}...` : "LLM unavailable — check /api/agents/health"}
                    disabled={healthState !== "ok"}
                    style={{
                        flex: 1,
                        padding: "10px 14px",
                        borderRadius: "var(--radius-md)",
                        border: "1px solid var(--border-subtle)",
                        background: "rgba(0,0,0,0.3)",
                        color: "var(--text-primary)",
                        fontSize: "0.85rem",
                        outline: "none",
                    }}
                />
                <button
                    onClick={() => handleSend()}
                    disabled={!input.trim() || isTyping || healthState !== "ok"}
                    style={{
                        width: 40,
                        height: 40,
                        borderRadius: "var(--radius-md)",
                        border: "none",
                        background: input.trim() ? config.color : "var(--bg-tertiary)",
                        color: input.trim() ? "#fff" : "var(--text-muted)",
                        cursor: input.trim() ? "pointer" : "default",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        transition: "all var(--transition-fast)",
                    }}
                >
                    <Send size={16} />
                </button>
            </div>
        </div>
    );
}
