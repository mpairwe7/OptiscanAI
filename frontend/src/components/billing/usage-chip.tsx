"use client";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { apiGetUsage } from "@/lib/auth-api";

export function UsageChip() {
  const usage = useQuery({
    queryKey: ["billing", "usage"],
    queryFn: apiGetUsage,
    refetchInterval: 30_000,
    retry: 0,
  });

  if (!usage.data) {
    return null;
  }

  const { scans_used, scan_limit } = usage.data;
  if (scan_limit === null) {
    return (
      <Link
        href="/app/usage"
        className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full bg-teal-50 text-teal-700 hover:bg-teal-100"
      >
        Unlimited scans
      </Link>
    );
  }

  const pct = scan_limit > 0 ? (scans_used / scan_limit) * 100 : 0;
  const color =
    pct >= 100
      ? "bg-red-50 text-red-700 hover:bg-red-100"
      : pct >= 80
        ? "bg-amber-50 text-amber-700 hover:bg-amber-100"
        : "bg-teal-50 text-teal-700 hover:bg-teal-100";

  return (
    <Link
      href="/app/usage"
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full ${color}`}
      title="Click to see usage details"
    >
      <span className="font-mono">
        {scans_used} / {scan_limit}
      </span>
      <span className="hidden sm:inline">scans</span>
    </Link>
  );
}
