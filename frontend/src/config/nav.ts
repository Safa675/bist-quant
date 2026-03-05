export interface NavItem {
    title: string;
    href: string;
    icon: string;
}

export const NAV_ITEMS: NavItem[] = [
    { title: "Dashboard", href: "/dashboard", icon: "LayoutDashboard" },
    { title: "Backtest", href: "/backtest", icon: "Activity" },
    { title: "Factor Lab", href: "/factor-lab", icon: "Beaker" },
    { title: "Signal Constr", href: "/signal-construction", icon: "GitCommit" },
    { title: "Screener", href: "/screener", icon: "Filter" },
    { title: "Analytics", href: "/analytics", icon: "LineChart" },
    { title: "Optimization", href: "/optimization", icon: "Target" },
    { title: "Professional", href: "/professional", icon: "Briefcase" },
    { title: "Compliance", href: "/compliance", icon: "ShieldCheck" },
    { title: "Agents", href: "/agents", icon: "Bot" },
];
