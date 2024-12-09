#!/usr/bin/env python3
"""
Pathology Report Information Extraction
Extracts structured data from pathology reports and matches ICD-O codes & IHC stains.
"""

pathalogy_note = """
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Laboratory: KING HUSSEIN CANCER CENTER
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
SURGICAL PATH DIAGNOSIS
CLINICAL HISTORY:
A case of left breast cancer, S/P neoadjuvant chemotherapy.
TISSUE ORIGIN:
A-Tissue around axillary vein; biopsy.
B-Left breast; mastectomy.
C-Left axilla lymph nodes; dissection.
MACROSCOPIC DESCRIPTION:
A-The specimen is received in formalin, labeled with patient's name
and "NECROTIC TISSUE AROUND THE AXILLA". It consists of 2 pieces of
grayish tissue measuring 2X1X0.5cm. All submitted in one block.
B-Specimen received formalin fixed, labeled by patient.s name and
"LEFT BREAST"
Name of procedure: Mastectomy.
Integrity: Intact.
Side of the specimen: Left.
Size of the specimen: 19.5X16.5X5.5cm.
Size of the lesion: 14.5cm, White area, gross invasive 9.5X3.5X4.0cm.
Location of the lesion: Central.
Appearance of the lesion: Speculated lesion in fibrotic background.
Focality of lesion: Unifocal.
Nearest margin: Deep, distance: Involved.
Skeletal muscles: Absent.
Dimensions of the skin: 16.3X12.5, with firm brownish discoloration
area 3.5x2cm.
Nipple/areola: 3.5X3.5cm near the brownish discoloration area.
Scars of previous biopsy: Absent.
Representative sections were submitted in 13 cassettes and labeled as
follows: 1-5: Lesion, 6-8: Deep margin, 9: 11.5cm size, 10:
Superficial margin, 13 cm size, 11: One of the peripheral margin
(14.5cm, size), 12: Skin, 13: Nipple.
C-The specimen is received in formalin, labeled with patient's name
and "LEFT AXILLA". It consists of 3 pieces of fibrofatty tissue
measuring 6X6X3cm. Part submitted in 10 blocks (1: One lymph node,
6+7: One lymph node, 8,9: Hard area/fibrosis (3cm), 10: One lymph
node).
MICROSCOPIC DESCRIPTION:
Histologic type: Invasive lobular carcinoma.
Grade: 2.
DCIS component: Not seen.
Size of largest invasive component: 14.5cm.
Lobular carcinoma in situ (LCIS): Present.
Tumor Extension: Skin: Extensive dermal involvement. Nipple:
Extensive dermal involvement. Skeletal Muscle: Involved by tumor.
Margins:
Distance of invasive from closest margin: The deep margin is
extensively involved.
Distance of invasive from other margins: Anterior: 2 mm One of the
peripheral margins is involved (the specimen is not oriented).
Lymphovascular invasion: Present.
Perineural invasion: Present. Axillary lymph nodes: Number
examined Number involved Sentinel Lymph Node (16)
(16)
Size of largest metastatic deposit: 3cm.
Extranodal invasion: Present; extensive.
Treatment effect: In the Breast: Present. In the Lymph Nodes:
Present.
DIAGNOSIS:
A-TISSUE AROUND AXILLARY VEIN; BIOPSY:
Fibroadipose tissue involved by the tumor.
B-LEFT BREAST; MASTECTOMY:
Residual invasive lobular carcinoma; 14.5cm.
The deep margin is extensively involved by the tumor.
One of the peripheral margins is involved (the specimen is not
oriented).
Pathologic stage ypT3N3a.
C-LEFT AXILLA LYMPH NODES; DISSECTION.
Sixteen lymph nodes involved by the tumor (16/16).
**********ha **********; M.D.
********** **********r, M.D.; **********AP
American Board of Anatomic Pathology
American Board of Cytopathology
/es/ OMAR JABER, md
Signed **********UL 01, 2024@12:38:42
"""

import os
import warnings
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import asyncio
from datetime import datetime

_ = load_dotenv(find_dotenv())
warnings.filterwarnings('ignore')

class PathologyReport(BaseModel):
    """Model for structured pathology report data"""
    histologic_type: str = Field(description="Histologic type using ICD-O classification")
    malignancy: str = Field(description="Is this malignant? (Yes/No/Not_sure)")
    primary_site: str = Field(description="Primary tumor site")
    specimen_type: str = Field(description="Specimen type (e.g., biopsy, frozen section, wide local excision)")
    specimen_site: str = Field(description="Specimen anatomical location")
    
    # Optional fields
    tumor_grade: Optional[str] = Field(default=None, description="Tumor grade using Nottingham grading system")
    tumor_margins: Optional[str] = Field(default=None, description="Tumor margins (involved or uninvolved)")
    specimen_size: Optional[str] = Field(default=None, description="Specimen size in cm (e.g. 6.0 X 3.0 X 2.5cm)")
    specimen_weight: Optional[float] = Field(default=None, description="Specimen weight in grams")
    
    # Staging information
    tnm_staging: Optional[str] = Field(default=None, description="Combined TNM staging (e.g. ypT0N0(sn))")
    t_stage: Optional[str] = Field(default=None, description="T stage extracted individually (e.g. T0)")
    n_stage: Optional[str] = Field(default=None, description="N stage extracted individually (e.g. N0)")
    m_stage: Optional[str] = Field(default=None, description="M stage extracted individually (e.g. M0 if mentioned)")
    
    # Histologic details
    histologic_subtype: Optional[str] = Field(default=None, description="Histologic subtype (e.g. Invasive ductal carcinoma)")
    in_situ_component: Optional[str] = Field(default=None, description="Presence/extent of in-situ disease (e.g. DCIS, LCIS)")
    treatment_effect: Optional[str] = Field(default=None, description="Therapy effect on tumor (e.g. marked response)")
    
    # Microscopic features
    perineural_invasion: Optional[str] = Field(default=None, description="Perineural invasion (present/absent)")
    calcifications: Optional[str] = Field(default=None, description="Microcalcifications (present/absent)")
    lymphovascular_invasion: Optional[str] = Field(default=None, description="Lymphovascular invasion")
    necrosis: Optional[str] = Field(default=None, description="Necrosis percentage if reported")
    
    # ICD-O codes with similarity scores
    ICDO3_terminology: Optional[str] = Field(default=None, description="ICD-O morphology code")
    ICDO3_terminology_term: Optional[str] = Field(default=None, description="Matched morphology term")
    ICDO3_terminology_similarity: Optional[float] = Field(default=None, description="Similarity score for morphology match")
    
    ICDO3_site: Optional[str] = Field(default=None, description="ICD-O topography code")
    ICDO3_site_term: Optional[str] = Field(default=None, description="Matched site term")
    ICDO3_site_similarity: Optional[float] = Field(default=None, description="Similarity score for site match")
    
    # Immunohistochemistry
    immunohistochemistry: Optional[List[Dict[str, str]]] = Field(default=None, description="List of IHC results [{stain:..., result:...}]")
    her2_status: Optional[str] = Field(default=None, description="HER2 status (Positive/Negative/Equivocal)")
    ki67_percent: Optional[float] = Field(default=None, description="Ki-67 percentage if reported")
    
    # Lymph node information
    lymph_nodes: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of lymph node data: [{'site': '...', 'number': <int>, 'involved': <int>}]"
    )
    
    # Additional information
    others: Optional[Dict[str, str]] = Field(default=None, description="Any additional extracted key-value information")

class StainMapper:
    """Maps stain names to standardized terminology"""
    def __init__(self, stains_df: pd.DataFrame):
        self.stains_df = stains_df.rename(columns={'Stain': 'stain_name'})
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.vectors = self.vectorizer.fit_transform(self.stains_df['stain_name'].fillna(''))

    def find_closest_stain(self, stain_name: str, threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        """Find closest matching standardized stain name"""
        if not stain_name:
            return None
        query = self.vectorizer.transform([stain_name])
        sims = cosine_similarity(query, self.vectors).flatten()
        idx = sims.argmax()
        if sims[idx] < threshold:
            return None
        return {
            'standard_name': self.stains_df.iloc[idx]['stain_name'],
            'similarity': float(sims[idx])
        }

class DiagnosisMapper:
    """Maps diagnoses to ICD-O codes"""
    def __init__(self, icdo_df: pd.DataFrame, topography_df: pd.DataFrame):
        self.icdo_df = icdo_df
        self.topography_df = topography_df
        self.morph_vec = TfidfVectorizer(ngram_range=(1, 2))
        self.topo_vec = TfidfVectorizer(ngram_range=(1, 2))
        
        # Update to use 'term' instead of 'morphology'
        self.morph_vectors = self.morph_vec.fit_transform(icdo_df['term'].fillna(''))
        self.topo_vectors = self.topo_vec.fit_transform(topography_df['term'].fillna(''))

    def find_closest_diagnosis(self, desc: str, threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        """Find closest matching ICD-O morphology code"""
        if not desc:
            return None
        q = self.morph_vec.transform([desc])
        sims = cosine_similarity(q, self.morph_vectors).flatten()
        idx = sims.argmax()
        print(f"Debug - Diagnosis input: {desc}")
        print(f"Debug - Best match: {self.icdo_df.iloc[idx]['term']}")
        print(f"Debug - Similarity score: {sims[idx]}")
        return None if sims[idx] < threshold else {
            'code': self.icdo_df.iloc[idx]['ICDO3'],  # Update to use 'ICDO3' instead of 'icdo'
            'term': self.icdo_df.iloc[idx]['term'],    # Update to use 'term' instead of 'morphology'
            'similarity': float(sims[idx])
        }
    
    def find_closest_topography(self, site_description: str, threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        """Find closest matching ICD-O topography code"""
        if not site_description:
            return None
        q = self.topo_vec.transform([site_description])
        sims = cosine_similarity(q, self.topo_vectors).flatten()
        idx = sims.argmax()
        print(f"Debug - Site input: {site_description}")
        print(f"Debug - Best match: {self.topography_df.iloc[idx]['term']}")
        print(f"Debug - Similarity score: {sims[idx]}")
        return None if sims[idx] < threshold else {
            'code': self.topography_df.iloc[idx]['ICDO3'],
            'term': self.topography_df.iloc[idx]['term'],
            'similarity': float(sims[idx])
        }

@dataclass
class ExtractionDependencies:
    """Dependencies for extraction process"""
    cancer_types: List[str]
    special_items_df: pd.DataFrame
    diagnosis_mapper: DiagnosisMapper
    stain_mapper: StainMapper

# Initialize extraction agent
extraction_agent = Agent(
    "openai:gpt-4o",
    retries=3,
    deps_type=ExtractionDependencies,
    result_type=PathologyReport,
    system_prompt="""
    You are an expert pathologist specializing in structured data extraction from pathology reports.
    
    Guidelines:
    1. Extract only explicitly stated information from the report. Use None if not mentioned.
    2. histologic_type: main pathological finding.
    3. malignancy: "Yes"/"No"/"Not_sure".
    4. specimen_type: e.g., biopsy, frozen section, excision.
    5. immunohistochemistry (IHC): [{"stain": "ER","result":"+"}, ...]. For HER2 and Ki-67, if mentioned separately, map to her2_status and ki67_percent.
    6. lymph_nodes: [{"site":"Right axillary","number":3,"involved":0}] based on text.
    7. Extract t_stage, n_stage, m_stage from staging if individually mentioned or deduce from combined stage (e.g. from ypT0N0).
    8. histologic_subtype: e.g. "Invasive ductal carcinoma".
    9. in_situ_component: e.g. "DCIS present".
    10. treatment_effect: describe neoadjuvant therapy effect if stated.
    11. perineural_invasion, calcifications, lymphovascular_invasion, necrosis if stated.
    12. her2_status: Positive/Negative/Equivocal if mentioned.
    13. ki67_percent: numeric if mentioned.
    14. For extra unmapped data, put in others: {"key":"value"}.
    15. Use consistent formatting and ensure compliance with PathologyReport model.
    """
)

# Extraction agent tools
@extraction_agent.tool
async def match_stain_name(ctx: RunContext[ExtractionDependencies], stain_name: str):
    """Tool for matching stain names"""
    return ctx.deps.stain_mapper.find_closest_stain(stain_name)

@extraction_agent.tool
async def match_diagnosis_code(ctx: RunContext[ExtractionDependencies], description: str):
    """Tool for matching diagnosis codes"""
    return ctx.deps.diagnosis_mapper.find_closest_diagnosis(description)

@extraction_agent.tool
async def match_topography_code(ctx: RunContext[ExtractionDependencies], site_description: str):
    """Tool for matching topography codes"""
    return ctx.deps.diagnosis_mapper.find_closest_topography(site_description)

async def process_pathology_report(report_text: str, deps: ExtractionDependencies) -> PathologyReport:
    """Process a single pathology report and extract structured information."""
    try:
        print("Starting extraction...")
        result = await extraction_agent.run(report_text, deps=deps)
        print("Initial extraction complete")
        
        # Match diagnosis (terminology)
        if result.data.histologic_type:
            print(f"\nMatching terminology for: {result.data.histologic_type}")
            diagnosis_match = await match_diagnosis_code(
                RunContext(deps=deps, retry=0, tool_name="match_diagnosis_code"),
                result.data.histologic_type
            )
            
            if diagnosis_match and 'code' in diagnosis_match:
                print(f"Found diagnosis match: {diagnosis_match}")
                result.data.ICDO3_terminology = diagnosis_match['code']
                result.data.ICDO3_terminology_term = diagnosis_match['term']
                result.data.ICDO3_terminology_similarity = float(diagnosis_match['similarity'])
            else:
                print("No terminology match found")
        
        # Match topography (site)
        if result.data.primary_site:
            print(f"\nMatching site for: {result.data.primary_site}")
            topography_match = await match_topography_code(
                RunContext(deps=deps, retry=0, tool_name="match_topography_code"),
                result.data.primary_site
            )
            
            if topography_match and 'code' in topography_match:
                print(f"Found site match: {topography_match}")
                result.data.ICDO3_site = topography_match['code']
                result.data.ICDO3_site_term = topography_match['term']
                result.data.ICDO3_site_similarity = float(topography_match['similarity'])
            else:
                print("No site match found")

        return result.data
        
    except Exception as e:
        print(f"Detailed error in process_pathology_report: {str(e)}")
        raise

async def main():
    """Main function for testing"""
    # Example placeholder data - replace with your actual reference data
    cancer_special_items = pd.DataFrame({'Cancer': ['Breast Cancer', 'Lung Cancer']})
    icdo_df = pd.DataFrame({
        'icdo': ['8520/3', '8500/3', '8140/3'],
        'morphology': ['Lobular carcinoma', 'Ductal carcinoma', 'Adenocarcinoma']
    })
    topography_df = pd.DataFrame({
        'ICDO3': ['C50.9', 'C34.9', 'C18.9'],
        'term': ['Breast NOS', 'Lung NOS', 'Colon NOS']
    })
    stains_df = pd.DataFrame({
        'Stain': ['ER', 'PR', 'HER2', 'Ki67'],
        'category': ['Hormone', 'Hormone', 'Growth', 'Proliferation']
    })

    deps = ExtractionDependencies(
        cancer_types=list(cancer_special_items['Cancer'].unique()),
        special_items_df=cancer_special_items,
        diagnosis_mapper=DiagnosisMapper(icdo_df, topography_df),
        stain_mapper=StainMapper(stains_df)
    )

    result = await process_pathology_report(pathalogy_note, deps)
    print("\nExtracted Information:")
    print(result.dict())

if __name__ == "__main__":
    asyncio.run(main())