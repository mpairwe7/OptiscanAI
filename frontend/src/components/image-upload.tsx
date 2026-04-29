"use client";
import { useCallback, useState } from "react";
import { useAppStore } from "@/stores/app-store";
import { usePredict } from "@/hooks/use-predict";

export function ImageUpload() {
  const { imageFile, imagePreview, setImage, clearImage } = useAppStore();
  const predict = usePredict();
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith("image/")) setImage(file);
    },
    [setImage],
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setImage(file);
  };

  return (
    <div className="space-y-3 sm:space-y-4">
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-4 sm:p-6 text-center transition-all cursor-pointer
          ${isDragging
            ? "border-teal-500 bg-teal-50/50 scale-[1.01]"
            : imagePreview
              ? "border-slate-200 bg-white"
              : "border-slate-300 bg-slate-50/50 hover:border-teal-400 hover:bg-teal-50/30 active:bg-teal-50/50"
          }`}
      >
        <label className="cursor-pointer block">
          <input
            type="file"
            accept="image/jpeg,image/png"
            onChange={handleChange}
            className="hidden"
            aria-label="Upload retinal fundus image"
          />
          {imagePreview ? (
            <div className="space-y-2 sm:space-y-3">
              <img
                src={imagePreview}
                alt="Retinal fundus preview"
                className="mx-auto max-h-48 sm:max-h-56 rounded-lg shadow-md border border-slate-100"
              />
              <div className="text-[10px] sm:text-xs text-slate-400 truncate px-2">
                {imageFile?.name} ({((imageFile?.size ?? 0) / 1024).toFixed(0)} KB)
              </div>
            </div>
          ) : (
            <div className="space-y-2 sm:space-y-3 py-3 sm:py-4">
              <div className="mx-auto w-12 sm:w-14 h-12 sm:h-14 rounded-full bg-teal-50 flex items-center justify-center">
                <svg className="h-6 sm:h-7 w-6 sm:w-7 text-teal-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 16v-8m0 0l-3 3m3-3l3 3M6.75 19.25h10.5A2.25 2.25 0 0019.5 17V7A2.25 2.25 0 0017.25 4.75H6.75A2.25 2.25 0 004.5 7v10a2.25 2.25 0 002.25 2.25z" />
                </svg>
              </div>
              <div>
                <p className="text-xs sm:text-sm font-medium text-slate-600">
                  Drop retinal fundus image here
                </p>
                <p className="text-[10px] sm:text-xs text-slate-400 mt-1">or tap to browse (JPEG, PNG up to 10MB)</p>
              </div>
            </div>
          )}
        </label>
      </div>

      {imageFile && (
        <div className="flex gap-2 sm:gap-3">
          <button
            onClick={() => predict.mutate(imageFile)}
            disabled={predict.isPending}
            className="flex-1 bg-teal-600 text-white py-2.5 sm:py-3 px-4 rounded-lg font-semibold text-sm
                       hover:bg-teal-700 active:bg-teal-800 disabled:opacity-50 transition-colors shadow-sm
                       flex items-center justify-center gap-2"
          >
            {predict.isPending ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                Analyze Image
              </>
            )}
          </button>
          <button
            onClick={clearImage}
            className="px-4 py-2.5 sm:py-3 border border-slate-300 rounded-lg text-slate-600 text-sm
                       hover:bg-slate-50 active:bg-slate-100 transition-colors"
            aria-label="Clear uploaded image"
          >
            Clear
          </button>
        </div>
      )}

      {predict.isError && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-3 text-xs sm:text-sm flex items-start gap-2 animate-fade-in">
          <svg className="w-4 h-4 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <p>{predict.error.message}</p>
            <button
              onClick={() => predict.mutate(imageFile!)}
              className="text-red-800 underline mt-1 font-medium text-xs"
            >
              Retry
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
