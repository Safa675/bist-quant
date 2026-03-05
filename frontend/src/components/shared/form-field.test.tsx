import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { FormField, FormGrid, FormHint, FormLabel, FormRow } from "@/components/shared/form-field";
import { Input } from "@/components/ui/input";

describe("form-field primitives", () => {
  it("renders label, child control, and hint", () => {
    render(
      <FormField label="Signal" hint="Choose one" htmlFor="signal-input">
        <Input id="signal-input" />
      </FormField>
    );

    expect(screen.getByText("Signal")).toBeInTheDocument();
    expect(screen.getByText("Choose one")).toBeInTheDocument();
    expect(screen.getByRole("textbox")).toBeInTheDocument();
  });

  it("renders layout wrappers", () => {
    render(
      <FormGrid>
        <FormRow>
          <FormLabel>Label</FormLabel>
          <FormHint>Hint</FormHint>
        </FormRow>
      </FormGrid>
    );

    expect(screen.getByText("Label")).toBeInTheDocument();
    expect(screen.getByText("Hint")).toBeInTheDocument();
    expect(document.querySelector("[data-ui-form-grid]")).toBeInTheDocument();
  });
});
