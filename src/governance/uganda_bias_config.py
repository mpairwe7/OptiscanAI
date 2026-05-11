"""Uganda-specific bias audit configuration.

Device profiles, lighting conditions, geographic regions, and
degradation parameters for common Ugandan smartphones.
"""

from __future__ import annotations

from dataclasses import dataclass

# Common smartphones used by CHWs in Uganda
UGANDA_DEVICES = [
    "Tecno Camon 20", "Tecno Spark 10", "Tecno Pop 7",
    "Infinix Hot 30", "Infinix Note 30",
    "Samsung Galaxy A14", "Samsung Galaxy A04",
    "iPhone SE",
    "Clinical fundus camera",
]

LIGHTING_CONDITIONS = [
    "clinic_fluorescent",
    "outdoor_daylight",
    "outdoor_shade",
    "dim_indoor",
    "mixed",
]

GEOGRAPHIC_REGIONS = {
    "Kampala": ["Kampala"],
    "Central": ["Wakiso", "Mukono", "Mpigi", "Masaka", "Luwero"],
    "Eastern": ["Jinja", "Mbale", "Tororo", "Soroti", "Iganga"],
    "Western": ["Mbarara", "Fort Portal", "Kabale", "Kasese", "Hoima"],
    "Northern": ["Gulu", "Lira", "Arua", "Kitgum", "Moroto"],
}

# Flat list of all districts
ALL_DISTRICTS = [d for districts in GEOGRAPHIC_REGIONS.values() for d in districts]


@dataclass
class DeviceDegradationProfile:
    """Image degradation profile simulating a specific device."""
    device_name: str
    jpeg_quality: int  # 0-100
    max_resolution: int  # pixels on longest edge
    color_shift_r: float  # -1.0 to 1.0
    color_shift_g: float
    color_shift_b: float
    noise_std: float  # Gaussian noise std
    lens_distortion: float  # barrel distortion coefficient


DEVICE_PROFILES: dict[str, DeviceDegradationProfile] = {
    "Tecno Spark 10": DeviceDegradationProfile(
        device_name="Tecno Spark 10",
        jpeg_quality=55, max_resolution=2048,
        color_shift_r=0.02, color_shift_g=-0.01, color_shift_b=-0.03,
        noise_std=0.015, lens_distortion=0.002,
    ),
    "Tecno Camon 20": DeviceDegradationProfile(
        device_name="Tecno Camon 20",
        jpeg_quality=70, max_resolution=4096,
        color_shift_r=0.01, color_shift_g=0.0, color_shift_b=-0.01,
        noise_std=0.008, lens_distortion=0.001,
    ),
    "Infinix Hot 30": DeviceDegradationProfile(
        device_name="Infinix Hot 30",
        jpeg_quality=50, max_resolution=2048,
        color_shift_r=0.03, color_shift_g=-0.02, color_shift_b=-0.04,
        noise_std=0.020, lens_distortion=0.003,
    ),
    "Samsung Galaxy A14": DeviceDegradationProfile(
        device_name="Samsung Galaxy A14",
        jpeg_quality=65, max_resolution=3264,
        color_shift_r=0.01, color_shift_g=0.0, color_shift_b=-0.01,
        noise_std=0.010, lens_distortion=0.001,
    ),
    "Clinical fundus camera": DeviceDegradationProfile(
        device_name="Clinical fundus camera",
        jpeg_quality=90, max_resolution=4288,
        color_shift_r=0.0, color_shift_g=0.0, color_shift_b=0.0,
        noise_std=0.003, lens_distortion=0.0,
    ),
}
