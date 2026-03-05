/**
 * Design tokens as TypeScript constants — used in Recharts and JS code
 * that can't read CSS custom properties directly.
 */

export const COLORS = {
    bgBase: "#0a1020",
    bgPanel: "#111a31",
    bgElevated: "#162443",
    bgHover: "#1c2d52",

    border: "#2b3c63",

    text: "#edf3ff",
    textMuted: "#b4c4e4",
    textFaint: "#8ea4cd",

    accent: "#4a9eff",
    accentDim: "rgba(74,158,255,0.18)",

    bull: "#4ed08a",
    bear: "#ff6d76",
    neutral: "#f2b157",
    info: "#4a9eff",
    purple: "#8f86ff",
    sky: "#5bc6ff",

    bullDim: "rgba(78,208,138,0.18)",
    bearDim: "rgba(255,109,118,0.18)",
    neutralDim: "rgba(242,177,87,0.18)",

    gridLine: "rgba(237,243,255,0.14)",
} as const;

export const FONTS = {
    display: "'Sora', 'Avenir Next', 'Segoe UI', sans-serif",
    sans: "'IBM Plex Sans', 'Segoe UI', sans-serif",
    mono: "'IBM Plex Mono', 'JetBrains Mono', monospace",
} as const;

export const REGIME_COLORS: Record<string, string> = {
    Bull: COLORS.bull,
    Bear: COLORS.bear,
    Sideways: COLORS.neutral,
    Unknown: COLORS.textFaint,
};

export function regimeColor(label: string): string {
    return REGIME_COLORS[label] ?? COLORS.textFaint;
}

export const CHART_MARGINS = { top: 8, right: 16, left: 0, bottom: 8 };
