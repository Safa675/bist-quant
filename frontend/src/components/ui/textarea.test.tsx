import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Textarea } from "@/components/ui/textarea";

describe("Textarea", () => {
  it("renders a textarea with passed props", () => {
    render(<Textarea aria-label="Notes" defaultValue="sample" />);
    const textarea = screen.getByLabelText("Notes");
    expect(textarea).toBeInTheDocument();
    expect(textarea).toHaveValue("sample");
  });
});
