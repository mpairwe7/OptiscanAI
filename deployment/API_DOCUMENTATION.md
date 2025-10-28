# Retinal Disease Detection API Documentation

## Overview

REST API for detecting retinal diseases from fundus images using deep learning models.

**Base URL (Local):** `http://localhost:8080`  
**Base URL (Production):** `https://retinal-disease-api-xxxxx.run.app`

---

## Authentication

Currently, the API is **unauthenticated** for ease of use. For production use with sensitive data, consider adding authentication.

---

## Endpoints

### 1. Get API Information

**Endpoint:** `GET /`

**Description:** Returns basic information about the API.

**Response:**
```json
{
  "name": "Retinal Disease Detection API",
  "version": "1.0",
  "description": "Multi-label retinal disease classification using GraphCLIP, VisualLanguageGNN, SceneGraphTransformer, and ViGNN models",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "diseases": "/diseases"
  }
}
```

**Example:**
```bash
curl http://localhost:8080/
```

---

### 2. Health Check

**Endpoint:** `GET /health`

**Description:** Check if the API and model are loaded and ready.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is not ready

**Example:**
```bash
curl http://localhost:8080/health
```

---

### 3. List Diseases

**Endpoint:** `GET /diseases`

**Description:** Get the complete list of 45 retinal diseases that the model can detect.

**Response:**
```json
{
  "diseases": [
    "Normal Fundus (N)",
    "Diabetes (D)",
    "Glaucoma (G)",
    "Cataract (C)",
    "Age-related Macular Degeneration (A)",
    "Hypertension (H)",
    "Pathological Myopia (M)",
    "Other diseases/abnormalities (O)",
    "Diabetic Retinopathy (DR)",
    "Age-Related Macular Degeneration (ARMD)",
    "Media Haze (MH)",
    "Drusens (DN)",
    "Myopia (MYA)",
    "Branch Retinal Vein Occlusion (BRVO)",
    "Tessellation (TSLN)",
    "Epiretinal Membrane (ERM)",
    "Laser Scars (LS)",
    "Macular Scars (MS)",
    "Central Serous Retinopathy (CSR)",
    "Optic Disc Cupping (ODC)",
    "Central Retinal Vein Occlusion (CRVO)",
    "Tortuous Vessels (TV)",
    "Asteroid Hyalosis (AH)",
    "Optic Disc Pallor (ODP)",
    "Optic Disc Edema (ODE)",
    "Optociliary Shunt (OS)",
    "Anterior Ischemic Optic Neuropathy (AION)",
    "Papilledema (PAES)",
    "Retinitis (RT)",
    "Retinitis Pigmentosa (RP)",
    "Chorioretinitis (CRN)",
    "Exudation (EDN)",
    "Retinal Detachment (RD)",
    "Retinoschisis (RS)",
    "Macular Hole (MHL)",
    "Macular Pucker (MCP)",
    "Preretinal Hemorrhage (PRH)",
    "Myelinated Nerve Fibers (MNF)",
    "Hemorrhagic Retinopathy (HR)",
    "Central Retinal Artery Occlusion (CRAO)",
    "Tilted Disc (TD)",
    "Cystoid Macular Edema (CME)",
    "Post-Surgical Retinopathy (PTSR)",
    "Coloboma (CB)",
    "Vitreous Hemorrhage (VH)"
  ],
  "total": 45
}
```

**Example:**
```bash
curl http://localhost:8080/diseases
```

---

### 4. Predict Diseases

**Endpoint:** `POST /predict`

**Description:** Upload a retinal fundus image and get disease predictions.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body Parameter:** `file` (image file)

**Supported Image Formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

**Image Requirements:**
- Minimum resolution: 224x224 pixels (will be resized)
- Maximum file size: 10 MB (recommended)
- Color fundus photograph (RGB)

**Response (Success):**
```json
{
  "predictions": [
    {
      "disease": "Diabetic Retinopathy (DR)",
      "probability": 0.87,
      "confidence": "high"
    },
    {
      "disease": "Exudation (EDN)",
      "probability": 0.65,
      "confidence": "medium"
    },
    {
      "disease": "Microaneurysms (MA)",
      "probability": 0.54,
      "confidence": "medium"
    }
  ]
}
```

**Response (Demo Mode - No Model Loaded):**
```json
{
  "predictions": [
    {
      "disease": "Normal Fundus (N)",
      "probability": 0.65,
      "confidence": "medium"
    }
  ],
  "note": "Demo mode - model not loaded, showing random prediction"
}
```

**Response (No Diseases Detected):**
```json
{
  "predictions": []
}
```

**Confidence Levels:**
- **High:** Probability > 0.8 (Strong indication)
- **Medium:** Probability 0.5 - 0.8 (Moderate indication)

**Status Codes:**
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid image format or missing file
- `500 Internal Server Error` - Server error during prediction

**Example (cURL):**
```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@path/to/retinal_image.jpg"
```

**Example (Python):**
```python
import requests

url = "http://localhost:8080/predict"
files = {"file": open("retinal_image.jpg", "rb")}

response = requests.post(url, files=files)
predictions = response.json()

print(f"Detected diseases:")
for pred in predictions["predictions"]:
    print(f"  - {pred['disease']}: {pred['probability']:.2%} ({pred['confidence']})")
```

**Example (JavaScript/Node.js):**
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('retinal_image.jpg'));

axios.post('http://localhost:8080/predict', form, {
  headers: form.getHeaders()
})
.then(response => {
  console.log('Predictions:', response.data.predictions);
})
.catch(error => {
  console.error('Error:', error.message);
});
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

#### 400 Bad Request
```json
{
  "detail": "No file uploaded. Please upload an image file."
}
```

**Cause:** Missing file in request

**Solution:** Include `file` parameter with an image

---

#### 415 Unsupported Media Type
```json
{
  "detail": "Invalid image format. Supported formats: JPEG, PNG, BMP"
}
```

**Cause:** Uploaded file is not a valid image

**Solution:** Use JPEG, PNG, or BMP format

---

#### 500 Internal Server Error
```json
{
  "detail": "Error processing image: [error details]"
}
```

**Cause:** Server-side error during image processing or prediction

**Solution:** Check image format, size, and content. Contact support if persists.

---

## Rate Limiting

**Current:** No rate limiting (development)

**Production Recommendation:**
- Implement rate limiting: 100 requests/minute per IP
- Consider API keys for higher limits

---

## Performance

### Response Times (Approximate)

- `/health`: < 50ms
- `/diseases`: < 50ms
- `/predict`: 200-500ms (depending on model and image size)

### Optimization Tips

1. **Resize images before upload:**
   ```python
   from PIL import Image
   img = Image.open("large_image.jpg")
   img = img.resize((224, 224))
   img.save("optimized_image.jpg")
   ```

2. **Compress images:**
   - Use JPEG with quality 85-90
   - Reduces upload time

3. **Batch predictions:**
   - For multiple images, consider implementing batch endpoint

---

## Model Information

### Architecture
- **GraphCLIP:** 45M parameters - CLIP + Graph Attention
- **VisualLanguageGNN:** 48M parameters - Visual-Language Fusion
- **SceneGraphTransformer:** 52M parameters - Spatial Understanding
- **ViGNN:** 50M parameters - Visual Graph Neural Network

### Training
- **Dataset:** Multi-source retinal disease dataset
- **Cross-Validation:** 5-fold
- **Training Platform:** Kaggle (2x T4 GPUs)
- **Framework:** PyTorch

### Performance Metrics
- **Mean F1 Score:** ~0.93
- **Mean AUC:** ~0.96
- **Mean Precision:** ~0.91
- **Mean Recall:** ~0.94

---

## Use Cases

### 1. Screening Application
```python
import requests
from pathlib import Path

def screen_patient(image_path):
    """Screen patient for retinal diseases"""
    
    url = "http://localhost:8080/predict"
    files = {"file": open(image_path, "rb")}
    
    response = requests.post(url, files=files)
    predictions = response.json()["predictions"]
    
    if not predictions:
        return "No diseases detected - Normal fundus"
    
    # Alert if high-risk diseases detected
    high_risk = ["Diabetic Retinopathy (DR)", "Glaucoma (G)", 
                 "Age-related Macular Degeneration (A)"]
    
    alerts = [p for p in predictions 
              if any(risk in p["disease"] for risk in high_risk)]
    
    if alerts:
        return f"⚠️ High-risk diseases detected: {len(alerts)}"
    
    return f"Found {len(predictions)} conditions"

# Example
result = screen_patient("patient_001_fundus.jpg")
print(result)
```

### 2. Batch Processing
```python
import requests
from pathlib import Path

def process_batch(image_dir):
    """Process all images in a directory"""
    
    results = []
    
    for img_path in Path(image_dir).glob("*.jpg"):
        response = requests.post(
            "http://localhost:8080/predict",
            files={"file": open(img_path, "rb")}
        )
        
        results.append({
            "patient_id": img_path.stem,
            "predictions": response.json()["predictions"]
        })
    
    return results

# Example
batch_results = process_batch("./patient_images/")
print(f"Processed {len(batch_results)} patients")
```

### 3. Real-time Dashboard
```javascript
// Continuous monitoring dashboard
async function monitorPatient(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('http://localhost:8080/predict', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  
  // Update dashboard UI
  updateDashboard(data.predictions);
  
  // Trigger alerts if needed
  if (data.predictions.some(p => p.confidence === 'high')) {
    triggerAlert(data.predictions);
  }
}
```

---

## Security Considerations

### For Production Deployment

1. **HTTPS Only:**
   - Always use HTTPS in production
   - Cloud Run provides automatic HTTPS

2. **Input Validation:**
   - Already implemented: File type checking
   - Consider: File size limits, malware scanning

3. **Authentication (Recommended):**
   ```python
   # Add API key authentication
   @app.post("/predict")
   async def predict(
       file: UploadFile = File(...),
       api_key: str = Header(...)
   ):
       if api_key not in VALID_API_KEYS:
           raise HTTPException(401, "Invalid API key")
       # ... rest of prediction logic
   ```

4. **Rate Limiting:**
   ```python
   from slowapi import Limiter
   
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/predict")
   @limiter.limit("100/minute")
   async def predict(request: Request, ...):
       # ... prediction logic
   ```

---

## Changelog

### Version 1.0 (Current)
- Initial release
- 45 disease classifications
- FastAPI REST API
- Docker containerization
- Google Cloud Run deployment support

---

## Support & Feedback

- **GitHub Issues:** [Report bugs or request features](https://github.com/mpairwe7/MLOPS_V1/issues)
- **Documentation:** See `deployment/DEPLOYMENT_GUIDE.md`
- **Email:** [Your contact email]

---

## License

[Your license information]

---

**Last Updated:** 2024  
**API Version:** 1.0
