# Test Results

Sample queries and their outputs from the RAG pipeline. All output files are stored in the `outputs/` folder.

## Q&A Mode Queries

### 1. Quality Improvement Programs

**Query**: "What are the requirements for quality improvement programs?"

**Output**: `outputs/what_are_the_requirements_for_quality_improvement_programs_.json`

**Summary**: Returns comprehensive answer covering QM.1, QM.6 requirements with citations to chapters QM.1, QM.6, QM.3. Confidence: medium.

---

### 2. Infection Control for Surgical Areas

**Query**: "Describe the infection control requirements for surgical areas"

**Output**: `outputs/describe_the_infection_control_requirements_for_surgical_are.json`

**Summary**: Details SS.1, IC.3, PE.3 requirements for surgical infection control. Citations include chapters SS.1, PE.3, IC.3. Confidence: medium.

---

### 3. Medication Error Handling

**Query**: "How should hospitals handle medication errors?"

**Output**: `outputs/how_should_hospitals_handle_medication_errors_.json`

**Summary**: Explains MM.6, QM.8 requirements for medication error reporting and patient safety systems. Citations: MM.6, QM.8, MM.1. Confidence: medium.

---

### 4. Staff Competency Assessment

**Query**: "What are the staff competency assessment requirements?"

**Output**: `outputs/what_are_the_staff_competency_assessment_requirements_.json`

**Summary**: Details SM.7 requirements for competency assessment and performance appraisal. Citations: SM.7, RS.2, SM.5. Confidence: medium.

---

### 5. Patient Rights and Responsibilities

**Query**: "Explain the patient rights and responsibilities outlined in the standards"

**Output**: `outputs/explain_the_patient_rights_and_responsibilities_outlined_in_.json`

**Summary**: Comprehensive overview of PR.2, PR.4, RR.2 patient rights. Citations: RR.2, PR.2, PR.4. Confidence: medium.

---

## Citation Mode Queries

### 1. Chapter QM.1

**Query**: "Show me chapter QM.1"

**Output**: `outputs/show_me_chapter_qm_1.json`

**Summary**: Returns exact text from QM.1 (Responsibility and Accountability) with full interpretive guidelines. Exact match found.

---

### 2. Chapter LS.2

**Query**: "What does chapter LS.2 say exactly?"

**Output**: `outputs/what_does_chapter_ls_2_say_exactly_.json`

**Summary**: Returns exact text from LS.2 (Potentially Infectious Blood and Products) with all SR requirements. Exact match found.

---

### 3. Chapter IC.3

**Query**: "Give me the exact text for chapter IC.3"

**Output**: `outputs/give_me_the_exact_text_for_chapter_ic_3.json`

**Summary**: Returns exact text from IC.3 (Leadership Responsibilities) for infection prevention. Exact match found.

---

### 4. Chapter PE.1

**Query**: "Cite chapter PE.1"

**Output**: `outputs/cite_chapter_pe_1.json`

**Summary**: Returns exact text from PE.1 (Facility) with comprehensive physical environment requirements. Exact match found.

---

### 5. Chapter MM.2

**Query**: "I need the verbatim language from chapter MM.2"

**Output**: `outputs/i_need_the_verbatim_language_from_chapter_mm_2.json`

**Summary**: Returns exact text from MM.2 (Formulary) with medication selection requirements. Exact match found.

---

## Mixed/Edge Case Queries

### 1. Quality Management with Citation Request

**Query**: "What does the quality management chapter say and also show me the exact text"

**Output**: `outputs/what_does_the_quality_management_chapter_say_and_also_show_m.json`

**Summary**: Triggers Q&A mode. Provides synthesized answer about QM chapter with citations to QM.6, QM.1, SM.2. Confidence: medium.

---

### 2. Hand Hygiene Search

**Query**: "Is there a chapter about hand hygiene? Show me the exact wording"

**Output**: `outputs/is_there_a_chapter_about_hand_hygiene_show_me_the_exact_word.json`

**Summary**: Triggers Q&A mode. Indicates no specific hand hygiene chapter found in retrieved context. Citations: DS.3, PE.3, IC.3. Confidence: medium.

---

### 3. Ambiguous Patient Safety Query

**Query**: "Chapters related to patient safety"

**Output**: `outputs/chapters_related_to_patient_safety.json`

**Summary**: Triggers Q&A mode. Identifies QM.8 and PR.10 as patient safety chapters with detailed explanations. Citations: QM.8, PR.10, IC.4. Confidence: medium.

---

## Test Coverage

- **Q&A Mode**: 5 queries tested ✓
- **Citation Mode**: 5 queries tested ✓
- **Mixed/Edge Cases**: 3 queries tested ✓
- **Total Queries**: 13

All queries executed successfully. Output files available in `outputs/` folder.
