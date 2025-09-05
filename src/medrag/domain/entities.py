from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SectionType(str, Enum):
    ALLERGIES = "ALLERGIES"
    MEDICATIONS = "MEDICATIONS"
    CONDITIONS = "CONDITIONS"
    CARE_PLANS = "CARE PLANS"
    REPORTS = "REPORTS"
    OBSERVATIONS = "OBSERVATIONS"
    PROCEDURES = "PROCEDURES"
    IMMUNIZATIONS = "IMMUNIZATIONS"
    ENCOUNTERS = "ENCOUNTERS"
    IMAGING_STUDIES = "IMAGING STUDIES"


class MedicalRecord(BaseModel):
    type: SectionType | str = Field(..., description="Section header name.")
    content: str = Field(..., description="All text after ':' within the section, joined by newlines.")


class Patient(BaseModel):
    name: str
    race: Optional[str] = None
    ethnicity: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = Field(None, ge=0)
    birth_date: Optional[date] = None
    marital_status: Optional[str] = None

    medical_records: list[MedicalRecord] = Field(default_factory=list)

    # Optional: normalize simple one-letter genders to uppercase
    @field_validator("gender")
    @classmethod
    def _upper_gender(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if isinstance(v, str) and len(v) <= 3 else v
