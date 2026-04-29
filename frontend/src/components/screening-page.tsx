"use client";
import { useAppStore } from "@/stores/app-store";
import { ImageUpload } from "@/components/image-upload";
import { ResultsPanel } from "@/components/results-panel";
import { ClinicalReasoning } from "@/components/clinical-reasoning";
import { ProbabilityChart } from "@/components/probability-chart";
import { ExplainabilityPanel } from "@/components/explainability-panel";
import { KnowledgeGraphPanel } from "@/components/knowledge-graph-panel";

function ScreeningSettings() {
  const { threshold, setThreshold, topK, setTopK } = useAppStore();

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-3 sm:p-4 space-y-3 sm:space-y-4">
      <h3 className="text-[10px] sm:text-xs font-semibold text-slate-500 uppercase tracking-wide">Screening Parameters</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
        <div>
          <label className="text-xs font-medium text-slate-600 flex justify-between mb-1.5">
            <span>Detection Threshold</span>
            <span className="text-teal-600 font-mono">{threshold.toFixed(2)}</span>
          </label>
          <input
            type="range" min={0.1} max={0.9} step={0.05}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full accent-teal-600 h-1.5"
            aria-label={`Detection threshold: ${threshold.toFixed(2)}`}
          />
          <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
            <span>Sensitive</span>
            <span>Specific</span>
          </div>
        </div>
        <div>
          <label className="text-xs font-medium text-slate-600 flex justify-between mb-1.5">
            <span>Top K Predictions</span>
            <span className="text-teal-600 font-mono">{topK}</span>
          </label>
          <input
            type="range" min={3} max={15} step={1}
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            className="w-full accent-teal-600 h-1.5"
            aria-label={`Top K predictions: ${topK}`}
          />
        </div>
      </div>
    </div>
  );
}

export function ScreeningPage() {
  const { result } = useAppStore();

  return (
    <div className="space-y-4 sm:space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-800">Retinal Screening</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
          Upload a fundus image for multi-disease AI analysis with clinical reasoning
        </p>
      </div>

      {/* Settings bar */}
      <ScreeningSettings />

      {/* Upload + Results row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        <div className="space-y-2">
          <h2 className="text-[10px] sm:text-xs font-semibold text-slate-500 uppercase tracking-wide">
            Fundus Image
          </h2>
          <ImageUpload />
        </div>
        <div className="space-y-2">
          <h2 className="text-[10px] sm:text-xs font-semibold text-slate-500 uppercase tracking-wide">
            Detection Results
          </h2>
          <ResultsPanel />
          {!result && (
            <div className="bg-white rounded-xl border border-slate-200 p-6 sm:p-10 text-center">
              <svg className="mx-auto h-10 sm:h-12 w-10 sm:w-12 text-slate-200 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-xs sm:text-sm text-slate-400">Upload and analyze an image to see results</p>
            </div>
          )}
        </div>
      </div>

      {/* Clinical + Probabilities (shown after prediction) */}
      {result && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
            <div className="space-y-2">
              <h2 className="text-[10px] sm:text-xs font-semibold text-slate-500 uppercase tracking-wide">
                Clinical Reasoning & Treatment
              </h2>
              <ClinicalReasoning />
            </div>
            <div className="space-y-2">
              <h2 className="text-[10px] sm:text-xs font-semibold text-slate-500 uppercase tracking-wide">
                Disease Probability Distribution
              </h2>
              <ProbabilityChart />
            </div>
          </div>

          {/* Explainability */}
          <div className="space-y-2">
            <h2 className="text-[10px] sm:text-xs font-semibold text-slate-500 uppercase tracking-wide">
              Model Explainability
            </h2>
            <ExplainabilityPanel />
          </div>

          {/* Knowledge Graph */}
          <div className="space-y-2">
            <h2 className="text-[10px] sm:text-xs font-semibold text-slate-500 uppercase tracking-wide">
              Clinical Knowledge Graph
            </h2>
            <KnowledgeGraphPanel />
          </div>
        </>
      )}
    </div>
  );
}
