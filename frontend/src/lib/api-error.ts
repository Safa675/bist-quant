export interface ApiStructuredDetail {
  code?: string;
  detail?: string;
  hint?: string;
  errors?: unknown[];
}

interface ApiClientErrorInit {
  status: number;
  detail: string;
  code?: string;
  hint?: string;
  errors?: unknown[];
}

export class ApiClientError extends Error {
  status: number;
  code?: string;
  detail: string;
  hint?: string;
  errors?: unknown[];

  constructor({ status, detail, code, hint, errors }: ApiClientErrorInit) {
    const prefix = `API ${status}`;
    const message = hint ? `${prefix}: ${detail} Hint: ${hint}` : `${prefix}: ${detail}`;
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.detail = detail;
    this.code = code;
    this.hint = hint;
    this.errors = errors;
  }
}

export function isApiClientError(value: unknown): value is ApiClientError {
  return value instanceof ApiClientError;
}

export function parseApiClientError(status: number, payload: unknown): ApiClientError | null {
  if (!payload || typeof payload !== "object") return null;

  const root = payload as Record<string, unknown>;
  const detail = root.detail;
  if (typeof detail === "string") {
    return new ApiClientError({ status, detail });
  }

  if (detail && typeof detail === "object") {
    const row = detail as Record<string, unknown>;
    const code = typeof row.code === "string" ? row.code : undefined;
    const detailText = typeof row.detail === "string" ? row.detail : "Request failed";
    const hint = typeof row.hint === "string" ? row.hint : undefined;
    const errors = Array.isArray(row.errors) ? row.errors : undefined;
    return new ApiClientError({ status, code, detail: detailText, hint, errors });
  }

  if (typeof root.message === "string") {
    return new ApiClientError({ status, detail: root.message });
  }

  return null;
}
