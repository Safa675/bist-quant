import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { KeyValueList } from "@/components/shared/key-value-list";

describe("KeyValueList", () => {
  it("renders keys and formatted values", () => {
    render(
      <KeyValueList
        data={{
          name: "alpha",
          count: 5,
          params: { top_n: 10 },
        }}
      />
    );

    expect(screen.getByText("name")).toBeInTheDocument();
    expect(screen.getByText("alpha")).toBeInTheDocument();
    expect(screen.getByText("count")).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
    expect(screen.getByText("params")).toBeInTheDocument();
    expect(screen.getByText('{"top_n":10}')).toBeInTheDocument();
    expect(document.querySelector("[data-ui-key-value-list]")).toBeInTheDocument();
  });
});
