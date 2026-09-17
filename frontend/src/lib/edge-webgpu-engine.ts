/**
 * Client-Side Edge Inference Engine & Quality Gate (WebGPU & WebAssembly).
 *
 * Runs the distilled INT8 MobileNetV3 model directly inside the user's browser,
 * enabling zero-install offline screening on low-cost smartphones and laptops
 * in rural clinics without telecommunications.
 */

export interface ImageQualityReport {
  passed: boolean;
  width: number;
  height: number;
  laplacian_variance: number;
  mean_brightness: number;
  rejection_reason?: string;
}

export interface ClientPredictionResult {
  predictions: Array<{
    code: string;
    name: string;
    probability: number;
    detected: boolean;
  }>;
  referral_priority: "ROUTINE" | "FOLLOW_UP" | "URGENT_SPECIALIST";
  conformal_set: string[];
  execution_provider: "webgpu" | "wasm" | "server_fallback";
  inference_ms: number;
}

const DISEASE_NAMES_MAP: Record<string, string> = {
  DR: "Diabetic Retinopathy",
  ARMD: "Age-Related Macular Degeneration",
  GLC: "Glaucoma",
  ODC: "Optic Disc Cupping",
  HTR: "Hypertensive Retinopathy",
  CRVO: "Central Retinal Vein Occlusion",
  BRVO: "Branch Retinal Vein Occlusion",
  MYA: "Pathological Myopia",
  DN: "Drusen",
  MH: "Macular Hole",
  CWS: "Cotton Wool Spots",
  TSLN: "Tractional Schisis-Like Network",
  ERM: "Epiretinal Membrane",
  CSR: "Central Serous Retinopathy",
  ODE: "Optic Disc Edema",
};

/**
 * Client-side Laplacian blur and quality assessment before inference.
 * Rejects severely blurred or unilluminated images immediately in the browser.
 */
export function evaluateClientQuality(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number
): ImageQualityReport {
  if (width < 100 || height < 100) {
    return {
      passed: false,
      width,
      height,
      laplacian_variance: 0,
      mean_brightness: 0,
      rejection_reason: `Image resolution too low (${width}x${height}px, minimum 100x100 required).`,
    };
  }

  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const pixelCount = width * height;

  // 1. Grayscale luminance and mean brightness
  const gray = new Float32Array(pixelCount);
  let totalLuminance = 0;

  for (let i = 0; i < data.length; i += 4) {
    // Standard perceptual luminance: 0.299R + 0.587G + 0.114B
    const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    gray[i / 4] = lum;
    totalLuminance += lum;
  }

  const meanBrightness = totalLuminance / pixelCount;
  if (meanBrightness < 20) {
    return {
      passed: false,
      width,
      height,
      laplacian_variance: 0,
      mean_brightness: meanBrightness,
      rejection_reason: "Image underexposed; pupil illumination insufficient.",
    };
  }

  // 2. Fast discrete 3x3 Laplacian variance (focus sharpness)
  let lapSum = 0;
  let lapSqSum = 0;
  let count = 0;

  // Stride of 2 for fast mobile CPU execution
  for (let y = 1; y < height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
      const center = gray[y * width + x];
      const lap =
        gray[(y - 1) * width + x] +
        gray[(y + 1) * width + x] +
        gray[y * width + (x - 1)] +
        gray[y * width + (x + 1)] -
        4 * center;

      lapSum += lap;
      lapSqSum += lap * lap;
      count++;
    }
  }

  const meanLap = lapSum / count;
  const variance = lapSqSum / count - meanLap * meanLap;

  // Minimum focus threshold for non-mydriatic fundus
  const passed = variance >= 35.0;

  return {
    passed,
    width,
    height,
    laplacian_variance: variance,
    mean_brightness: meanBrightness,
    rejection_reason: passed
      ? undefined
      : `Image out of focus (sharpness score ${variance.toFixed(1)} < 35.0 threshold). Please recapture.`,
  };
}

/**
 * Preprocess an HTMLCanvasElement into a 1x3x224x224 Float32Array matching ImageNet normalization.
 */
export function preprocessCanvasForModel(
  ctx: CanvasRenderingContext2D,
  sourceWidth: number,
  sourceHeight: number
): Float32Array {
  // Resize to 224x224 onto an offscreen canvas
  const offscreen = document.createElement("canvas");
  offscreen.width = 224;
  offscreen.height = 224;
  const offCtx = offscreen.getContext("2d");
  if (!offCtx) throw new Error("Could not acquire 2D canvas context");

  offCtx.drawImage(ctx.canvas, 0, 0, sourceWidth, sourceHeight, 0, 0, 224, 224);
  const imgData = offCtx.getImageData(0, 0, 224, 224).data;

  // NCHW shape: 1 x 3 x 224 x 224
  const floatData = new Float32Array(1 * 3 * 224 * 224);
  const channelLength = 224 * 224;

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < channelLength; i++) {
    const r = imgData[i * 4] / 255.0;
    const g = imgData[i * 4 + 1] / 255.0;
    const b = imgData[i * 4 + 2] / 255.0;

    floatData[0 * channelLength + i] = (r - mean[0]) / std[0];
    floatData[1 * channelLength + i] = (g - mean[1]) / std[1];
    floatData[2 * channelLength + i] = (b - mean[2]) / std[2];
  }

  return floatData;
}

/**
 * Map raw model output logits / probabilities to clinical triage and conformal sets.
 */
export function mapTriageOutput(
  probs: Float32Array | number[],
  classCodes: string[],
  alpha: number = 0.05
): ClientPredictionResult {
  const predictions: ClientPredictionResult["predictions"] = [];
  const conformalSet: string[] = [];

  // Default sensitivity-targeted threshold: tau = 0.15 for high recall
  const defaultThreshold = 0.15;
  // Conformal inclusion threshold dynamically scaled by significance level alpha
  const conformalThreshold = Math.max(0.05, Math.min(0.25, alpha * 2.4));
  let maxPathologyProb = 0;
  let urgentCount = 0;

  for (let i = 0; i < classCodes.length; i++) {
    const code = classCodes[i];
    const p = Math.min(1.0, Math.max(0.0, probs[i]));
    const name = DISEASE_NAMES_MAP[code] || code;
    const detected = p >= defaultThreshold;

    predictions.push({ code, name, probability: p, detected });

    if (p >= conformalThreshold) {
      conformalSet.push(code);
    }

    if (code !== "NORMAL" && p > maxPathologyProb) {
      maxPathologyProb = p;
    }
    if (["DR", "GLC", "ODC", "CRVO", "MH"].includes(code) && p >= 0.25) {
      urgentCount++;
    }
  }

  // Triage urgency
  let referral_priority: ClientPredictionResult["referral_priority"] = "ROUTINE";
  if (urgentCount > 0 || maxPathologyProb >= 0.50) {
    referral_priority = "URGENT_SPECIALIST";
  } else if (maxPathologyProb >= 0.20) {
    referral_priority = "FOLLOW_UP";
  }

  return {
    predictions: predictions.sort((a, b) => b.probability - a.probability),
    referral_priority,
    conformal_set: conformalSet,
    execution_provider: "wasm",
    inference_ms: 0,
  };
}
