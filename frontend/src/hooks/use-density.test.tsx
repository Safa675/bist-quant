import { describe, expect, it, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { DensityProvider, useDensity } from "@/hooks/use-density";

function wrapper({ children }: { children: React.ReactNode }) {
  return <DensityProvider>{children}</DensityProvider>;
}

describe("useDensity", () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute("data-density");
  });

  it("defaults to comfortable and applies the root attribute", async () => {
    const { result } = renderHook(() => useDensity(), { wrapper });

    await waitFor(() => {
      expect(result.current.density).toBe("comfortable");
      expect(document.documentElement.getAttribute("data-density")).toBe("comfortable");
    });
  });

  it("reads persisted density from localStorage", async () => {
    window.localStorage.setItem("bq-ui-density", "compact");
    const { result } = renderHook(() => useDensity(), { wrapper });

    await waitFor(() => {
      expect(result.current.density).toBe("compact");
      expect(document.documentElement.getAttribute("data-density")).toBe("compact");
    });
  });

  it("toggles density and persists it", async () => {
    const { result } = renderHook(() => useDensity(), { wrapper });

    await waitFor(() => {
      expect(result.current.density).toBe("comfortable");
    });

    act(() => {
      result.current.toggleDensity();
    });

    await waitFor(() => {
      expect(result.current.density).toBe("compact");
      expect(window.localStorage.getItem("bq-ui-density")).toBe("compact");
      expect(document.documentElement.getAttribute("data-density")).toBe("compact");
    });
  });
});
