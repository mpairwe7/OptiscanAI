"use client";
import { useMutation } from "@tanstack/react-query";
import { predictImage, explainGradCAM, explainLIME, explainSHAP, explainIG, explainELI5, fetchAvailableMethods } from "@/lib/api";
import { useAppStore } from "@/stores/app-store";

export function usePredict() {
  const { setResult, threshold, addScanToHistory, setGradcamResult, setLimeResult, setShapResult, setIgResult, setEli5Result, setActiveXaiMethod } = useAppStore();

  return useMutation({
    mutationFn: (file: File) => predictImage(file, threshold),
    onSuccess: async (data, file) => {
      setResult(data);

      // Add to session history
      const preview = useAppStore.getState().imagePreview;
      if (preview) {
        addScanToHistory(crypto.randomUUID(), data, preview);
      }

      // Auto-trigger ALL explainability methods in parallel
      if (data.predictions.length > 0) {
        const topK = Math.min(data.predictions.length, 3);
        setActiveXaiMethod("gradcam");

        // Check which methods are available, then fire them all
        let available: Record<string, boolean> = {
          gradcam: true, lime: true, shap: true, ig: true, eli5: true,
        };
        try {
          const methods = await fetchAvailableMethods();
          available = {
            gradcam: methods?.methods?.gradcam?.available ?? true,
            lime: methods?.methods?.lime?.available ?? true,
            shap: methods?.methods?.shap?.available ?? false,
            ig: methods?.methods?.integrated_gradients?.available ?? true,
            eli5: methods?.methods?.eli5?.available ?? false,
          };
        } catch {
          // Fall back to defaults
        }

        // Fire all available methods in parallel — each is independent
        const tasks: Promise<void>[] = [];

        if (available.gradcam) {
          tasks.push(
            explainGradCAM(file, topK).then(setGradcamResult).catch(() => {}),
          );
        }
        if (available.lime) {
          tasks.push(
            explainLIME(file, topK).then(setLimeResult).catch(() => {}),
          );
        }
        if (available.shap) {
          tasks.push(
            explainSHAP(file, topK).then(setShapResult).catch(() => {}),
          );
        }
        if (available.ig) {
          tasks.push(
            explainIG(file).then(setIgResult).catch(() => {}),
          );
        }
        if (available.eli5) {
          tasks.push(
            explainELI5(file, topK).then(setEli5Result).catch(() => {}),
          );
        }

        await Promise.allSettled(tasks);
      }
    },
  });
}
