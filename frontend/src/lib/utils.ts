import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Utility to merge Tailwind classes natively resolving conflicts
 */
export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

/**
 * Format a number to 2 decimal places with optional suffix
 */
export function formatNum(val: number | null | undefined, suffix = ""): string {
    if (val === null || val === undefined) return "-";
    return `${val.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}${suffix}`;
}

/**
 * Format a percentage
 */
export function formatPct(val: number | null | undefined): string {
    if (val === null || val === undefined) return "-";
    return `${val > 0 ? "+" : ""}${val.toFixed(2)}%`;
}

/**
 * Parse an ISO date to a localized short format (e.g. "Oct 12, 2023")
 */
export function formatDate(isoDate: string | null | undefined): string {
    if (!isoDate) return "-";
    try {
        const d = new Date(isoDate);
        return d.toLocaleDateString(undefined, {
            year: "numeric",
            month: "short",
            day: "numeric",
        });
    } catch {
        return isoDate;
    }
}
