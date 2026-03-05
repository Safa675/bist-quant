import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import { useApi } from "@/hooks/use-api";

describe("useApi", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns initial state when fetcher is null", () => {
    const { result } = renderHook(() => useApi(null));
    expect(result.current.data).toBeUndefined();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("fetches data successfully", async () => {
    vi.useRealTimers();
    const mockData = { value: 42 };
    const fetcher = vi.fn().mockResolvedValue(mockData);

    const { result } = renderHook(() => useApi(fetcher));

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toEqual(mockData);
    expect(result.current.error).toBeNull();
    expect(fetcher).toHaveBeenCalledOnce();
  });

  it("handles fetch error", async () => {
    vi.useRealTimers();
    const fetcher = vi.fn().mockRejectedValue(new Error("Network error"));

    const { result } = renderHook(() => useApi(fetcher));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toBeUndefined();
    expect(result.current.error?.message).toBe("Network error");
  });

  it("supports skip option", () => {
    const fetcher = vi.fn().mockResolvedValue({});

    renderHook(() => useApi(fetcher, { skip: true }));

    expect(fetcher).not.toHaveBeenCalled();
  });

  it("provides initialData", () => {
    const initialData = { count: 0 };
    const { result } = renderHook(() =>
      useApi(null, { initialData }),
    );

    expect(result.current.data).toEqual(initialData);
  });

  it("refetch triggers a new fetch", async () => {
    vi.useRealTimers();
    const fetcher = vi.fn().mockResolvedValue({ v: 1 });

    const { result } = renderHook(() => useApi(fetcher));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(fetcher).toHaveBeenCalledTimes(1);

    act(() => {
      result.current.refetch();
    });

    await waitFor(() => {
      expect(fetcher).toHaveBeenCalledTimes(2);
    });
  });
});
