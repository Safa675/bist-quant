import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Checkbox } from "@/components/ui/checkbox";

describe("Checkbox", () => {
  it("renders and toggles checked state", async () => {
    const user = userEvent.setup();
    render(<Checkbox aria-label="Enable" />);
    const checkbox = screen.getByLabelText("Enable");

    expect(checkbox).not.toBeChecked();
    await user.click(checkbox);
    expect(checkbox).toBeChecked();
  });
});
