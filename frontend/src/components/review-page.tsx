"use client";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchPendingReviews, fetchReviewStats, resolveReview } from "@/lib/api";
import { useState } from "react";

const PRIORITY_STYLES: Record<string, string> = {
  urgent: "bg-red-100 text-red-700 border-red-200",
  high: "bg-orange-100 text-orange-700 border-orange-200",
  medium: "bg-amber-100 text-amber-700 border-amber-200",
  low: "bg-emerald-100 text-emerald-700 border-emerald-200",
};

export function ReviewPage() {
  const [filterPriority, setFilterPriority] = useState<string | undefined>(undefined);
  const queryClient = useQueryClient();

  const stats = useQuery({ queryKey: ["review-stats"], queryFn: fetchReviewStats, refetchInterval: 15_000 });
  const pending = useQuery({
    queryKey: ["pending-reviews", filterPriority],
    queryFn: () => fetchPendingReviews(filterPriority),
    refetchInterval: 10_000,
  });

  const resolve = useMutation({
    mutationFn: ({ id, decision }: { id: string; decision: "confirmed" | "rejected" }) =>
      resolveReview(id, decision),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pending-reviews"] });
      queryClient.invalidateQueries({ queryKey: ["review-stats"] });
    },
  });

  return (
    <div className="space-y-4 sm:space-y-6 animate-fade-in">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-800">Review Queue</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
          Human-in-the-loop review of AI predictions requiring clinical oversight
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4">
          <div className="text-[10px] sm:text-xs text-slate-500 mb-1">Total Reviews</div>
          <div className="text-xl sm:text-2xl font-bold text-slate-800">{stats.data?.total ?? 0}</div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4">
          <div className="text-[10px] sm:text-xs text-slate-500 mb-1">Pending</div>
          <div className="text-xl sm:text-2xl font-bold text-amber-600">{stats.data?.pending ?? 0}</div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4">
          <div className="text-[10px] sm:text-xs text-slate-500 mb-1">Resolved</div>
          <div className="text-xl sm:text-2xl font-bold text-emerald-600">{stats.data?.resolved ?? 0}</div>
        </div>
        <div className="bg-white rounded-xl border border-red-200 bg-red-50/50 p-3 sm:p-4">
          <div className="text-[10px] sm:text-xs text-slate-500 mb-1">Urgent</div>
          <div className="text-xl sm:text-2xl font-bold text-red-600">{stats.data?.by_priority?.urgent ?? 0}</div>
        </div>
      </div>

      {/* Priority Filter */}
      <div className="flex gap-1.5 sm:gap-2 overflow-x-auto pb-1">
        {[undefined, "urgent", "high", "medium", "low"].map((p) => (
          <button
            key={p ?? "all"}
            onClick={() => setFilterPriority(p)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 ${
              filterPriority === p
                ? "bg-teal-600 text-white shadow-sm"
                : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 active:bg-slate-100"
            }`}
          >
            {p ? p.charAt(0).toUpperCase() + p.slice(1) : "All"}
          </button>
        ))}
      </div>

      {/* Resolve feedback */}
      {resolve.isError && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-3 text-xs flex items-center gap-2 animate-fade-in">
          <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {resolve.error.message}
        </div>
      )}
      {resolve.isSuccess && (
        <div className="bg-emerald-50 border border-emerald-200 text-emerald-700 rounded-lg p-3 text-xs flex items-center gap-2 animate-fade-in">
          <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
          Review resolved successfully
        </div>
      )}

      {/* Pending Reviews List */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100 flex items-center justify-between">
          <h2 className="font-semibold text-sm sm:text-base text-slate-800">
            Pending Reviews ({pending.data?.total_pending ?? 0})
          </h2>
        </div>
        {pending.data?.reviews && pending.data.reviews.length > 0 ? (
          <div className="divide-y divide-slate-100">
            {pending.data.reviews.map((review) => (
              <div key={review.request_id} className="px-4 sm:px-5 py-3 sm:py-4 hover:bg-slate-50/50">
                <div className="flex items-start gap-3 sm:gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap items-center gap-2 mb-1">
                      <span className="text-xs sm:text-sm font-medium text-slate-700 font-mono">
                        {review.request_id.slice(0, 8)}
                      </span>
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium border ${
                        PRIORITY_STYLES[review.priority] ?? "bg-slate-100 text-slate-600"
                      }`}>
                        {review.priority}
                      </span>
                    </div>
                    <p className="text-[11px] sm:text-xs text-slate-500">{review.reason}</p>
                    <div className="text-[10px] text-slate-400 mt-1">
                      Created: {new Date(review.created_at).toLocaleString()}
                    </div>
                  </div>

                  {/* Action buttons - stack on mobile */}
                  <div className="flex flex-col sm:flex-row gap-1.5 sm:gap-2 shrink-0">
                    <button
                      onClick={() => resolve.mutate({ id: review.request_id, decision: "confirmed" })}
                      disabled={resolve.isPending}
                      className="px-3 py-1.5 text-xs font-medium bg-emerald-600 text-white rounded-lg
                                 hover:bg-emerald-700 active:bg-emerald-800 disabled:opacity-50 transition-colors"
                    >
                      {resolve.isPending ? "..." : "Confirm"}
                    </button>
                    <button
                      onClick={() => resolve.mutate({ id: review.request_id, decision: "rejected" })}
                      disabled={resolve.isPending}
                      className="px-3 py-1.5 text-xs font-medium bg-white border border-slate-200 text-slate-600 rounded-lg
                                 hover:bg-slate-50 active:bg-slate-100 disabled:opacity-50 transition-colors"
                    >
                      Reject
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="px-4 sm:px-5 py-10 sm:py-12 text-center">
            <svg className="mx-auto h-10 w-10 text-slate-200 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
            <p className="text-sm text-slate-400">No pending reviews</p>
            <p className="text-[11px] sm:text-xs text-slate-300 mt-1">Predictions requiring human oversight will appear here</p>
          </div>
        )}
      </div>

      {/* Info card */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-3 sm:p-4 flex items-start gap-2 sm:gap-3">
        <svg className="w-4 sm:w-5 h-4 sm:h-5 text-blue-500 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <div>
          <p className="text-xs sm:text-sm font-medium text-blue-800">Human-in-the-Loop Governance</p>
          <p className="text-[11px] sm:text-xs text-blue-700 mt-0.5">
            Predictions with low confidence, multi-disease co-occurrence, or critical referral priorities
            are automatically queued for clinical review. This satisfies EU AI Act Article 14 (Human Oversight)
            requirements for high-risk AI systems.
          </p>
        </div>
      </div>
    </div>
  );
}
