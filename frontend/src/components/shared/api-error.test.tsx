import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ApiClientError } from "@/lib/api-error";
import { ApiError } from "@/components/shared/api-error";

describe("ApiError", () => {
  it("renders nothing when error is null", () => {
    const { container } = render(<ApiError error={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders plain error messages", () => {
    render(<ApiError error="Network timeout" />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("Network timeout")).toBeInTheDocument();
  });

  it("renders structured api detail, hint, and code", () => {
    const error = new ApiClientError({
      status: 422,
      code: "job_validation_error",
      detail: "Invalid request payload for job kind 'backtest'.",
      hint: "Provide a valid factor_name.",
    });

    render(<ApiError error={error} />);

    expect(screen.getByText("Invalid request payload for job kind 'backtest'.")).toBeInTheDocument();
    expect(screen.getByText("Hint: Provide a valid factor_name.")).toBeInTheDocument();
    expect(screen.getByText("Code: job_validation_error")).toBeInTheDocument();
  });
});
