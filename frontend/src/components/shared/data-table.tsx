"use client";

import * as React from "react";
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
  type ColumnFiltersState,
  type RowSelectionState,
} from "@tanstack/react-table";
import { ChevronUp, ChevronDown, ChevronsUpDown, Download } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

// ---------------------------------------------------------------------------
// CSV Export helper
// ---------------------------------------------------------------------------

function escapeCsvCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  const str = String(value);
  if (str.includes(",") || str.includes('"') || str.includes("\n")) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

function exportToCsv<TData>(
  columns: ColumnDef<TData, unknown>[],
  rows: TData[],
  filename = "export.csv",
) {
  // Build headers from column definitions
  const headers = columns
    .map((col) => {
      if (typeof col.header === "string") return col.header;
      if ("accessorKey" in col && col.accessorKey) return String(col.accessorKey);
      if ("id" in col && col.id) return col.id;
      return "";
    })
    .filter(Boolean);

  const accessors = columns.map((col) => {
    if ("accessorKey" in col && col.accessorKey) return String(col.accessorKey);
    if ("accessorFn" in col && col.accessorFn) return col.accessorFn;
    return null;
  });

  const csvRows = [headers.map(escapeCsvCell).join(",")];

  for (const row of rows) {
    const cells = accessors.map((accessor) => {
      if (accessor === null) return "";
      if (typeof accessor === "function") return escapeCsvCell(accessor(row, 0));
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return escapeCsvCell((row as any)[accessor]);
    });
    csvRows.push(cells.join(","));
  }

  const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// DataTable
// ---------------------------------------------------------------------------

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[];
  data: TData[];
  searchColumn?: string;
  searchPlaceholder?: string;
  pageSize?: number;
  /** Enable row selection with checkboxes. */
  enableRowSelection?: boolean;
  /** Callback with currently selected rows. */
  onRowSelectionChange?: (selectedRows: TData[]) => void;
  /** Enable CSV export button. */
  enableExport?: boolean;
  /** Filename for CSV export (default "export.csv"). */
  exportFilename?: string;
}

export function DataTable<TData, TValue>({
  columns,
  data,
  searchColumn,
  searchPlaceholder = "Search...",
  pageSize = 20,
  enableRowSelection = false,
  onRowSelectionChange,
  enableExport = false,
  exportFilename = "export.csv",
}: DataTableProps<TData, TValue>) {
  const [sorting, setSorting] = React.useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>([]);
  const [rowSelection, setRowSelection] = React.useState<RowSelectionState>({});
  const onRowSelectionChangeRef = React.useRef(onRowSelectionChange);

  React.useEffect(() => {
    onRowSelectionChangeRef.current = onRowSelectionChange;
  }, [onRowSelectionChange]);

  // Notify parent when selection changes
  React.useEffect(() => {
    const onSelectionChange = onRowSelectionChangeRef.current;
    if (!enableRowSelection || !onSelectionChange) return;
    const selectedIndices = Object.keys(rowSelection).filter((k) => rowSelection[k]);
    const selected = selectedIndices.map((idx) => data[Number(idx)]).filter(Boolean);
    onSelectionChange(selected);
  }, [data, enableRowSelection, rowSelection]);

  // Prepend a checkbox column when selection is enabled
  const allColumns = React.useMemo(() => {
    if (!enableRowSelection) return columns;

    const checkboxCol: ColumnDef<TData, unknown> = {
      id: "_select",
      header: ({ table }) => (
        <Checkbox
          checked={table.getIsAllPageRowsSelected()}
          onChange={table.getToggleAllPageRowsSelectedHandler()}
          aria-label="Select all"
        />
      ),
      cell: ({ row }) => (
        <Checkbox
          checked={row.getIsSelected()}
          onChange={row.getToggleSelectedHandler()}
          aria-label="Select row"
        />
      ),
      enableSorting: false,
      enableColumnFilter: false,
      size: 32,
    };

    return [checkboxCol, ...columns] as ColumnDef<TData, TValue>[];
  }, [columns, enableRowSelection]);

  const table = useReactTable({
    data,
    columns: allColumns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onRowSelectionChange: enableRowSelection ? setRowSelection : undefined,
    enableRowSelection,
    initialState: { pagination: { pageSize } },
    state: {
      sorting,
      columnFilters,
      ...(enableRowSelection ? { rowSelection } : {}),
    },
  });

  const selectedCount = enableRowSelection
    ? Object.values(rowSelection).filter(Boolean).length
    : 0;

  return (
    <div className="space-y-[var(--space-3)]" data-ui-data-table>
      <div className="flex items-center justify-between gap-[var(--space-3)]">
        {searchColumn && (
          <Input
            placeholder={searchPlaceholder}
            value={(table.getColumn(searchColumn)?.getFilterValue() as string) ?? ""}
            onChange={(e) => table.getColumn(searchColumn)?.setFilterValue(e.target.value)}
            className="max-w-xs"
          />
        )}
        {!searchColumn && <div />}

        {enableExport && (
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              exportToCsv(
                columns as ColumnDef<TData, unknown>[],
                table.getFilteredRowModel().rows.map((r) => r.original),
                exportFilename,
              )
            }
          >
            <Download className="mr-1.5 h-3.5 w-3.5" />
            Export CSV
          </Button>
        )}
      </div>

      <div className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface-1)] shadow-[var(--shadow-sm)]">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((hg) => (
              <TableRow key={hg.id}>
                {hg.headers.map((header) => {
                  const canSort = header.column.getCanSort();
                  const sorted = header.column.getIsSorted();
                  return (
                    <TableHead
                      key={header.id}
                      className={cn(canSort && "cursor-pointer select-none")}
                      onClick={canSort ? header.column.getToggleSortingHandler() : undefined}
                    >
                      <div className="flex items-center gap-1">
                        {header.isPlaceholder
                          ? null
                          : flexRender(header.column.columnDef.header, header.getContext())}
                        {canSort && (
                          <span className="opacity-50">
                            {sorted === "asc" ? (
                              <ChevronUp className="h-3 w-3" />
                            ) : sorted === "desc" ? (
                              <ChevronDown className="h-3 w-3" />
                            ) : (
                              <ChevronsUpDown className="h-3 w-3" />
                            )}
                          </span>
                        )}
                      </div>
                    </TableHead>
                  );
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} data-state={row.getIsSelected() ? "selected" : undefined}>
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell
                  colSpan={allColumns.length}
                  className="h-24 text-center text-[var(--text-muted)]"
                >
                  No results.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      <div className="flex items-center justify-between text-small text-[var(--text-muted)]">
        <span>
          {enableRowSelection && selectedCount > 0
            ? `${selectedCount} of ${table.getFilteredRowModel().rows.length} selected`
            : `${table.getFilteredRowModel().rows.length} row(s)`}
        </span>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            Previous
          </Button>
          <span>
            Page {table.getState().pagination.pageIndex + 1} / {table.getPageCount()}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
}
