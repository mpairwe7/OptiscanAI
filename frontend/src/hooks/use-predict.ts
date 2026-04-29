"use client";
import { useMutation } from "@tanstack/react-query";
import { predictImage, explainGradCAM } from "@/lib/api";
import { useAppStore } from "@/stores/app-store";

export function usePredict() {
  const { setResult, threshold, addScanToHistory, setGradcamResult, setActiveXaiMethod } = useAppStore();

  return useMutation({
    mutationFn: (file: File) => predictImage(file, threshold),
    onSuccess: async (data, file) => {
      setResult(data);

      // Add to session history
      const preview = useAppStore.getState().imagePreview;
      if (preview) {
        addScanToHistory(crypto.randomUUID(), data, preview);
      }

      // Auto-trigger GradCAM if diseases were detected
      if (data.predictions.length > 0) {
        setActiveXaiMethod("gradcam");
        try {
          const gradcam = await explainGradCAM(file, Math.min(data.predictions.length, 3));
          setGradcamResult(gradcam);
        } catch {
          // GradCAM is optional — don't block the workflow
        }
      }
    },
  });
}
