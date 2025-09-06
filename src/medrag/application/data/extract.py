import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from langchain_core.documents import Document
from loguru import logger

from medrag.domain.entities import MedicalRecord, Patient

SEP_LINE = re.compile(r"^-{5,}\s*$")
SECTION_HEADER = re.compile(r"^[A-Z][A-Z\s/()\[\]\-]+:\s*$")  # e.g., "MEDICATIONS:", "OBSERVATIONS:"
KEYVAL_LINE = re.compile(r"^([A-Za-z][A-Za-z\s]+):\s*(.*\S)\s*$")


def _strip_bars(s: str) -> str:
    # Safeguard for weird encodings/spaces
    return s.strip().rstrip()


def _split_sections(lines: List[str]) -> Dict[str, List[str]]:
    """
    Return a dict: {section_name: [lines_in_section]}
    Assumes: header block -> SEP_LINE -> sections separated by SEP_LINE
    """
    sections: Dict[str, List[str]] = {}
    i = 0
    n = len(lines)

    # Skip header & demographics until first separator
    while i < n and not SEP_LINE.match(lines[i]):
        i += 1
    # Skip the sep line itself
    while i < n and SEP_LINE.match(lines[i]):
        i += 1

    current_name = None
    buf: List[str] = []

    while i < n:
        line = lines[i]
        if SEP_LINE.match(line):
            # Flush current section (if any)
            if current_name is not None:
                sections[current_name] = buf
                buf = []
                current_name = None
            # Skip all consecutive sep lines
            while i < n and SEP_LINE.match(lines[i]):
                i += 1
            continue

        if SECTION_HEADER.match(line):
            # If we were in a section, flush it
            if current_name is not None:
                sections[current_name] = buf
                buf = []
            current_name = _strip_bars(line[:-1])  # drop trailing ':'
        else:
            if current_name is not None:
                buf.append(line.rstrip("\n"))
        i += 1

    if current_name is not None:
        sections[current_name] = buf

    return sections


def _extract_after_colon(text_lines: Iterable[str]) -> List[str]:
    """
    From a bunch of lines, collect only RHS of the *first* ':' on each line.
    Lines without ':' are ignored (per your spec).
    """
    out: List[str] = []
    for ln in text_lines:
        if ":" in ln:
            rhs = ln.split(":", 1)[1].strip()
            if rhs:
                out.append(rhs)
    return out


def parse_patient_block(lines: List[str]) -> Dict[str, Any]:
    """
    Parse the header block (above the first separator) to capture:
    name, race, ethnicity, gender, age, birth_date, marital_status
    """
    info: Dict[str, Any] = {}
    i = 0
    n = len(lines)

    # Name: first non-empty line
    while i < n and not lines[i].strip():
        i += 1
    info["name"] = lines[i].strip() if i < n else ""
    i += 1

    # Skip underline of '=' if present
    if i < n and set(lines[i].strip()) == {"="}:
        i += 1

    # Read key: value lines until first separator
    while i < n and not SEP_LINE.match(lines[i]):
        m = KEYVAL_LINE.match(lines[i])
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            val = m.group(2).strip()
            info[key] = val
        i += 1

    # Optional normalization
    # Convert age to int if possible
    if "age" in info:
        try:
            info["age"] = int(info["age"])
        except ValueError:
            pass

    return info


def parse_patient_document(
    raw: str, exclude_sections: Iterable[str] = ("IMAGING STUDIES", "ENCOUNTERS")
) -> Dict[str, Any]:
    """
    Main entry. Returns:
    {
      name: ...,
      race: ...,
      ethnicity: ...,
      gender: ...,
      age: ...,
      birth_date: ...,
      marital_status: ...,
      medical_records: [
        { "type": <SECTION NAME>, "content": [<RHS after ':' per line>, ...] },
        ...
      ]
    }
    """
    # Normalize line endings
    lines = [ln.rstrip("\r") for ln in raw.splitlines()]

    # Parse top info
    patient_info = parse_patient_block(lines)
    if not patient_info["age"] or patient_info["age"] == "DECEASED":
        return

    patient_info = Patient(**patient_info)
    # Split sections
    sections = _split_sections(lines)

    # Build medical records, excluding listed ones
    exclude_set = set(exclude_sections or [])
    medical_records = []
    for sec_name, sec_lines in sections.items():
        if sec_name in exclude_set:
            continue
        content_rhs = _extract_after_colon(sec_lines)
        medical_records.append(
            MedicalRecord(
                **{
                    "type": sec_name,
                    "content": "\n".join(content_rhs),
                }
            )
        )

    patient_info.medical_records = medical_records
    return patient_info


def convert_to_document(doc: Patient) -> List[Document]:
    """
    Convert one parsed doc into a flat list of entries.
    """
    user_info = dict(doc)
    del user_info["medical_records"]
    records = doc.medical_records

    entries: List[Document] = []

    for record in records:
        entry = Document(
            page_content=f"{record.type} \n {record.content}",
            metadata={
                **user_info,
                "type": record.type,
            },
        )
        entries.append(entry)

    return entries


def extract_documents(dir: Path):
    files = dir.glob("*.txt")
    documents = []
    patient_count = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        patient = parse_patient_document(text, exclude_sections=["IMAGING STUDIES", "ENCOUNTERS"])
        if not patient:
            continue
        patient_count += 1
        reports = convert_to_document(patient)
        documents.extend(reports)

    logger.info(f"Loaded {patient_count} patients for a total of {len(documents)} reports.")
    return documents
