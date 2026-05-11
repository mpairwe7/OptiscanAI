"""DICOM upload and fundus image extraction endpoint.

Accepts DICOM files, extracts fundus images, and feeds them to the
existing screening pipeline. Maps DICOM metadata to patient context
for multimodal fusion.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dicom", tags=["dicom"])


class DICOMUploadResponse(BaseModel):
    patient_id: str = ""
    patient_name: str = ""
    patient_age: str = ""
    modality: str = ""
    is_ophthalmic: bool = False
    images_extracted: int = 0
    laterality: str = ""
    study_date: str = ""
    patient_context: dict = {}


@router.post("/upload", response_model=DICOMUploadResponse)
async def upload_dicom(file: UploadFile = File(...)):
    """Upload a DICOM file and extract fundus images for screening.

    The extracted images can then be sent to `/api/v1/predict` for
    screening. Patient demographics from DICOM metadata are returned
    for multimodal fusion context.
    """
    if not file.filename or not (
        file.filename.endswith(".dcm") or file.filename.endswith(".dicom")
        or "." not in file.filename  # DICOM files often have no extension
    ):
        # Still accept — DICOM files may have any extension
        pass

    try:
        from backend.app.integrations.dicom.handler import DICOMHandler

        handler = DICOMHandler()
        content = await file.read()
        metadata, images = handler.parse(content)

        patient_context = handler.metadata_to_patient_context(metadata)

        return DICOMUploadResponse(
            patient_id=metadata.patient_id,
            patient_name=metadata.patient_name,
            patient_age=metadata.patient_age,
            modality=metadata.modality,
            is_ophthalmic=metadata.is_ophthalmic,
            images_extracted=len(images),
            laterality=metadata.laterality,
            study_date=metadata.study_date,
            patient_context=patient_context,
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=501,
            detail=f"DICOM support requires pydicom: {e}",
        )
    except Exception as e:
        logger.error("DICOM upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse DICOM: {e}")
