import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SectionCard } from "@/components/shared/section-card";

describe("SectionCard", () => {
  it("renders children", () => {
    render(<SectionCard><p>Hello</p></SectionCard>);
    expect(screen.getByText("Hello")).toBeInTheDocument();
  });

  it("renders title as heading", () => {
    render(<SectionCard title="Performance"><p>Content</p></SectionCard>);
    expect(screen.getByRole("heading", { level: 2, name: "Performance" })).toBeInTheDocument();
  });

  it("renders subtitle", () => {
    render(
      <SectionCard title="Perf" subtitle="Last 30 days">
        <p>Content</p>
      </SectionCard>
    );
    expect(screen.getByText("Last 30 days")).toBeInTheDocument();
  });

  it("has role=region with aria-labelledby when title is provided", () => {
    render(<SectionCard title="Returns"><p>Data</p></SectionCard>);
    const region = screen.getByRole("region");
    expect(region).toBeInTheDocument();
    const headingId = screen.getByRole("heading", { level: 2 }).id;
    expect(region).toHaveAttribute("aria-labelledby", headingId);
  });

  it("does not set role=region when title is omitted", () => {
    const { container } = render(<SectionCard><p>Data</p></SectionCard>);
    expect(container.querySelector("[role='region']")).toBeNull();
  });

  it("renders actions slot", () => {
    render(
      <SectionCard title="Section" actions={<button data-testid="section-action">Edit</button>}>
        <p>Content</p>
      </SectionCard>
    );
    expect(screen.getByTestId("section-action")).toBeInTheDocument();
  });

  it("removes padding when noPadding is true", () => {
    const { container } = render(
      <SectionCard noPadding>
        <p>Content</p>
      </SectionCard>
    );
    // The content div should not have the p-5 class when noPadding is true
    const contentDiv = container.querySelector("div > div:last-child");
    expect(contentDiv?.className).not.toContain("p-5");
  });
});
