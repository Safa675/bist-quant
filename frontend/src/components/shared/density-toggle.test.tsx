import { describe, expect, it, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DensityProvider } from "@/components/shared/density-provider";
import { DensityToggle } from "@/components/shared/density-toggle";

describe("DensityToggle", () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute("data-density");
  });

  it("switches between compact and comfortable labels", async () => {
    const user = userEvent.setup();

    render(
      <DensityProvider>
        <DensityToggle />
      </DensityProvider>
    );

    const toggle = screen.getByRole("button", { name: /switch to compact density/i });
    expect(toggle).toHaveTextContent("Compact density");

    await user.click(toggle);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /switch to comfortable density/i })).toHaveTextContent(
        "Comfortable density"
      );
    });
  });
});
