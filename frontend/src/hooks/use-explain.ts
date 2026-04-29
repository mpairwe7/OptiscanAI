"use client";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  explainGradCAM,
  explainLIME,
  explainSHAP,
  explainIG,
  explainELI5,
  fetchAvailableMethods,
} from "@/lib/api";
import { useAppStore } from "@/stores/app-store";

export function useGradCAM() {
  const { setGradcamResult, topK } = useAppStore();
  return useMutation({
    mutationFn: (file: File) => explainGradCAM(file, Math.min(topK, 5)),
    onSuccess: (data) => setGradcamResult(data),
  });
}

export function useLIME() {
  const { setLimeResult, topK } = useAppStore();
  return useMutation({
    mutationFn: (file: File) => explainLIME(file, Math.min(topK, 3)),
    onSuccess: (data) => setLimeResult(data),
  });
}

export function useSHAP() {
  const { setShapResult, topK } = useAppStore();
  return useMutation({
    mutationFn: (file: File) => explainSHAP(file, Math.min(topK, 3)),
    onSuccess: (data) => setShapResult(data),
  });
}

export function useIG() {
  const { setIgResult } = useAppStore();
  return useMutation({
    mutationFn: (file: File) => explainIG(file),
    onSuccess: (data) => setIgResult(data),
  });
}

export function useELI5() {
  const { setEli5Result, topK } = useAppStore();
  return useMutation({
    mutationFn: (file: File) => explainELI5(file, Math.min(topK, 3)),
    onSuccess: (data) => setEli5Result(data),
  });
}

export function useAvailableMethods() {
  return useQuery({
    queryKey: ["xai-methods"],
    queryFn: fetchAvailableMethods,
    staleTime: 60_000,
    retry: 1,
  });
}
