"use client";

import * as React from "react";
import { AlertTriangle } from "lucide-react";

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ErrorBoundary caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      return (
        <div className="flex flex-col items-center justify-center gap-3 rounded-[var(--radius-lg)] border border-[var(--bear)] bg-[var(--bear-dim)] p-8 text-center">
          <AlertTriangle className="h-8 w-8 text-[var(--bear)]" />
          <div>
            <p className="font-semibold text-[var(--text)]">Something went wrong</p>
            {this.state.error && (
              <p className="mt-1 text-xs text-[var(--text-muted)]">
                {this.state.error.message}
              </p>
            )}
          </div>
          <button
            className="text-xs text-[var(--accent)] hover:underline"
            onClick={() => this.setState({ hasError: false })}
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
