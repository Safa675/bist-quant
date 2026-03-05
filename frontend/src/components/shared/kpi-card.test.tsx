import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { KpiCard } from "@/components/shared/kpi-card";

// We disable animation so the counter immediately shows the final value.
describe("KpiCard", () => {
  it("renders title and string value", () => {
    render(<KpiCard title="Total Return" value="25.50%" animate={false} />);
    expect(screen.getByText("Total Return")).toBeInTheDocument();
    expect(screen.getByText("25.50%")).toBeInTheDocument();
  });

  it("renders dash for null value", () => {
    render(<KpiCard title="Missing" value={null} animate={false} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("renders dash for undefined value", () => {
    render(<KpiCard title="Undefined" value={undefined} animate={false} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("includes suffix in string display", () => {
    render(<KpiCard title="Price" value="100" suffix=" USD" animate={false} />);
    expect(screen.getByText("100 USD")).toBeInTheDocument();
  });

  it("shows positive change badge with bull styling", () => {
    const { container } = render(
      <KpiCard title="Return" value="10" change={2.5} animate={false} />
    );
    const changeBadge = container.querySelector(".text-\\[var\\(--bull\\)\\]");
    expect(changeBadge).toBeInTheDocument();
    expect(screen.getByText("+2.50%")).toBeInTheDocument();
  });

  it("shows negative change badge with bear styling", () => {
    const { container } = render(
      <KpiCard title="Return" value="10" change={-3.2} animate={false} />
    );
    const changeBadge = container.querySelector(".text-\\[var\\(--bear\\)\\]");
    expect(changeBadge).toBeInTheDocument();
    expect(screen.getByText("-3.20%")).toBeInTheDocument();
  });

  it("has role=status and aria-label with title and value", () => {
    render(<KpiCard title="Sharpe" value="1.25" animate={false} />);
    const statusEl = screen.getByRole("status");
    expect(statusEl).toHaveAttribute("aria-label", expect.stringContaining("Sharpe"));
    expect(statusEl).toHaveAttribute("aria-label", expect.stringContaining("1.25"));
  });

  it("renders custom icon", () => {
    render(
      <KpiCard
        title="Score"
        value="99"
        animate={false}
        icon={<span data-testid="custom-icon">★</span>}
      />
    );
    expect(screen.getByTestId("custom-icon")).toBeInTheDocument();
  });
});
