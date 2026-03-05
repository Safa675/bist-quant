import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  PageScaffold,
  PageSidebar,
  PageMain,
  PageSectionStack,
  PageKpiRow,
} from "@/components/shared/page-scaffold";

describe("PageScaffold", () => {
  it("renders scaffold markers and slots", () => {
    render(
      <PageScaffold>
        <PageSidebar>
          <div>Sidebar</div>
        </PageSidebar>
        <PageMain>
          <PageSectionStack>
            <PageKpiRow>
              <div>KPI</div>
            </PageKpiRow>
          </PageSectionStack>
        </PageMain>
      </PageScaffold>
    );

    expect(screen.getByText("Sidebar").closest("aside")).toHaveAttribute("data-ui-sidebar");
    expect(document.querySelector("[data-ui-kpi-row]")).toBeInTheDocument();
    expect(document.querySelector("[data-ui-scaffold]")).toBeInTheDocument();
    expect(document.querySelector("[data-ui-main]")).toBeInTheDocument();
    expect(document.querySelector("[data-ui-section-stack]")).toBeInTheDocument();
  });
});
