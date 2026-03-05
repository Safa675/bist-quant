import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { PageHeader } from "@/components/shared/page-header";

describe("PageHeader", () => {
  it("renders title", () => {
    render(<PageHeader title="Dashboard" />);
    expect(screen.getByRole("heading", { level: 1, name: "Dashboard" })).toBeInTheDocument();
  });

  it("renders subtitle when provided", () => {
    render(<PageHeader title="Dashboard" subtitle="Overview of your portfolio" />);
    expect(screen.getByText("Overview of your portfolio")).toBeInTheDocument();
  });

  it("does not render subtitle when omitted", () => {
    const { container } = render(<PageHeader title="Dashboard" />);
    expect(container.querySelectorAll("p")).toHaveLength(0);
  });

  it("renders actions slot", () => {
    render(
      <PageHeader
        title="Dashboard"
        actions={<button data-testid="action-btn">Refresh</button>}
      />
    );
    expect(screen.getByTestId("action-btn")).toBeInTheDocument();
    expect(screen.getByText("Refresh")).toBeInTheDocument();
  });

  it("applies custom className", () => {
    const { container } = render(<PageHeader title="Test" className="custom-class" />);
    expect(container.firstChild).toHaveClass("custom-class");
  });
});
