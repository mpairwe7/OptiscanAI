"""DICOM file parsing and fundus image extraction.

Extracts fundus photographs from DICOM studies (Ophthalmic Photography
IOD) and prepares them for the RetinalAI screening pipeline. Maps
DICOM metadata to patient context for multimodal fusion.

Supports:
  - Single-frame and multi-frame DICOM files
  - Standard Ophthalmic Photography SOP classes
  - Patient demographics extraction (age, sex)
  - Study/series metadata for audit trail
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_pydicom_available = False
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    _pydicom_available = True
except ImportError:
    pass

# DICOM SOP Class UIDs for ophthalmic photography
OPHTHALMIC_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.77.1.5.1",   # Ophthalmic Photography 8 Bit
    "1.2.840.10008.5.1.4.1.1.77.1.5.2",   # Ophthalmic Photography 16 Bit
    "1.2.840.10008.5.1.4.1.1.77.1.5.4",   # Ophthalmic Tomography Image
    "1.2.840.10008.5.1.4.1.1.7",           # Secondary Capture (fallback)
}


@dataclass
class DICOMMetadata:
    """Extracted metadata from a DICOM study."""

    patient_id: str = ""
    patient_name: str = ""
    patient_age: str = ""
    patient_sex: str = ""
    study_date: str = ""
    study_description: str = ""
    series_description: str = ""
    modality: str = ""
    sop_class_uid: str = ""
    laterality: str = ""  # L | R | B
    institution: str = ""
    referring_physician: str = ""
    accession_number: str = ""
    rows: int = 0
    columns: int = 0
    bits_stored: int = 0
    is_ophthalmic: bool = False


@dataclass
class ExtractedFundusImage:
    """A fundus image extracted from a DICOM file."""

    image_bytes: bytes  # JPEG/PNG encoded
    metadata: DICOMMetadata = field(default_factory=DICOMMetadata)
    frame_index: int = 0
    content_type: str = "image/jpeg"


class DICOMHandler:
    """Parse DICOM files and extract fundus images for screening."""

    def __init__(self):
        if not _pydicom_available:
            logger.warning(
                "pydicom not installed — DICOM handler in stub mode. "
                "Install with: pip install pydicom pillow"
            )

    def parse(self, dicom_bytes: bytes) -> tuple[DICOMMetadata, list[ExtractedFundusImage]]:
        """Parse a DICOM file and extract fundus images.

        Parameters
        ----------
        dicom_bytes : bytes
            Raw DICOM file content.

        Returns
        -------
        tuple[DICOMMetadata, list[ExtractedFundusImage]]
            Metadata and extracted fundus image(s).
        """
        if not _pydicom_available:
            raise RuntimeError("pydicom not installed")

        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
        metadata = self._extract_metadata(ds)
        images = self._extract_images(ds, metadata)

        logger.info(
            "DICOM parsed: patient=%s, modality=%s, images=%d, ophthalmic=%s",
            metadata.patient_id, metadata.modality, len(images), metadata.is_ophthalmic,
        )

        return metadata, images

    def _extract_metadata(self, ds) -> DICOMMetadata:
        """Extract clinical metadata from DICOM dataset."""
        sop_uid = str(getattr(ds, "SOPClassUID", ""))

        return DICOMMetadata(
            patient_id=str(getattr(ds, "PatientID", "")),
            patient_name=str(getattr(ds, "PatientName", "")),
            patient_age=str(getattr(ds, "PatientAge", "")),
            patient_sex=str(getattr(ds, "PatientSex", "")),
            study_date=str(getattr(ds, "StudyDate", "")),
            study_description=str(getattr(ds, "StudyDescription", "")),
            series_description=str(getattr(ds, "SeriesDescription", "")),
            modality=str(getattr(ds, "Modality", "")),
            sop_class_uid=sop_uid,
            laterality=str(getattr(ds, "Laterality", getattr(ds, "ImageLaterality", ""))),
            institution=str(getattr(ds, "InstitutionName", "")),
            referring_physician=str(getattr(ds, "ReferringPhysicianName", "")),
            accession_number=str(getattr(ds, "AccessionNumber", "")),
            rows=int(getattr(ds, "Rows", 0)),
            columns=int(getattr(ds, "Columns", 0)),
            bits_stored=int(getattr(ds, "BitsStored", 0)),
            is_ophthalmic=sop_uid in OPHTHALMIC_SOP_CLASSES,
        )

    def _extract_images(
        self, ds, metadata: DICOMMetadata
    ) -> list[ExtractedFundusImage]:
        """Extract pixel data as JPEG images."""
        images = []

        if not hasattr(ds, "PixelData"):
            logger.warning("DICOM file has no pixel data")
            return images

        try:
            from PIL import Image

            pixel_array = ds.pixel_array

            # Handle multi-frame DICOM
            if pixel_array.ndim == 4:
                # [frames, rows, cols, channels]
                for i in range(pixel_array.shape[0]):
                    frame = pixel_array[i]
                    img_bytes = self._array_to_jpeg(frame)
                    images.append(ExtractedFundusImage(
                        image_bytes=img_bytes,
                        metadata=metadata,
                        frame_index=i,
                    ))
            elif pixel_array.ndim == 3:
                # Single frame: [rows, cols, channels] or [frames, rows, cols]
                if pixel_array.shape[2] in (3, 4):
                    # RGB/RGBA
                    img_bytes = self._array_to_jpeg(pixel_array)
                    images.append(ExtractedFundusImage(
                        image_bytes=img_bytes, metadata=metadata,
                    ))
                else:
                    # Multiple grayscale frames
                    for i in range(pixel_array.shape[0]):
                        frame = pixel_array[i]
                        img_bytes = self._array_to_jpeg(
                            np.stack([frame] * 3, axis=-1)
                        )
                        images.append(ExtractedFundusImage(
                            image_bytes=img_bytes, metadata=metadata, frame_index=i,
                        ))
            elif pixel_array.ndim == 2:
                # Single grayscale frame
                rgb = np.stack([pixel_array] * 3, axis=-1)
                img_bytes = self._array_to_jpeg(rgb)
                images.append(ExtractedFundusImage(
                    image_bytes=img_bytes, metadata=metadata,
                ))

        except Exception as e:
            logger.error("Failed to extract DICOM pixel data: %s", e)

        return images

    def _array_to_jpeg(self, array: np.ndarray, quality: int = 95) -> bytes:
        """Convert numpy array to JPEG bytes."""
        from PIL import Image

        if array.dtype != np.uint8:
            if array.max() > 255:
                array = (array / array.max() * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)

        img = Image.fromarray(array)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def metadata_to_patient_context(self, metadata: DICOMMetadata) -> dict:
        """Convert DICOM metadata to patient context for multimodal fusion."""
        context = {}

        if metadata.patient_age:
            # DICOM age format: "045Y", "003M", etc.
            age_str = metadata.patient_age.strip()
            if age_str.endswith("Y"):
                try:
                    context["age"] = int(age_str[:-1])
                except ValueError:
                    pass

        if metadata.patient_sex:
            context["sex"] = metadata.patient_sex.upper()

        if metadata.laterality:
            context["eye"] = "left" if metadata.laterality == "L" else "right"

        return context
