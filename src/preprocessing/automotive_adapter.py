"""
TrendLens - Automotive Dataset Adapter
Normalises Stanford Cars smoke-test metadata into the shared TrendLens
catalog schema (item_id, title, category, style, make, model, model_year,
file_path, tags) so FeatureExtractor, VisualClusterEngine, and
SimilarityEngine work without modification.

IMPORTANT - Forecasting:
  Automotive forecasting is DISABLED for the visual POC.
  model_year labels encode production year, NOT sales volume / popularity.
  No trend CSV is generated; the API returns a placeholder message.

Dataset Attribution:
  Stanford Cars Dataset
  Authors: Krause, J., Stark, M., Deng, J., & Fei-Fei, L. (2013)
  URL: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
  License: Non-commercial research / educational use ONLY.
  Do NOT redistribute raw images commercially.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


# ---------------------------------------------------------------------------
# Body-type vocabulary
# ---------------------------------------------------------------------------
_BODY_TYPE_MAP: Dict[str, str] = {
    "sedan": "sedan", "saloon": "sedan", "4dr": "sedan",
    "suv": "suv", "crossover": "suv", "4wd": "suv", "awd": "suv",
    "4x4": "suv", "sport utility": "suv", "utility": "suv",
    "coupe": "coupe", "2dr": "coupe", "hatchback": "coupe",
    "hatch": "coupe", "fastback": "coupe", "3dr": "coupe", "5dr": "coupe",
    "truck": "truck", "pickup": "truck", "cab": "truck",
    "convertible": "convertible", "cabriolet": "convertible",
    "roadster": "convertible", "spyder": "convertible",
    "spider": "convertible", "targa": "convertible",
    "wagon": "wagon", "estate": "wagon", "van": "wagon",
    "minivan": "wagon", "mini van": "wagon",
}


def infer_body_type(label: str) -> str:
    """Heuristically infer body-type from a Stanford Cars label string."""
    lower = label.lower()
    for token, body_type in _BODY_TYPE_MAP.items():
        if token in lower:
            return body_type
    return "other"


def parse_label_components(label: str) -> Tuple[str, str, int, str]:
    """
    Extracts (make, model, model_year, body_type) from a Stanford Cars label.
    Label format: "<Make> <Model_and_trim> <Year>"
    Last token is always the 4-digit year.
    """
    tokens = label.strip().split()
    model_year = 0
    if tokens and tokens[-1].isdigit() and len(tokens[-1]) == 4:
        model_year = int(tokens[-1])
        tokens = tokens[:-1]
    make = tokens[0] if tokens else "Unknown"
    model = " ".join(tokens[1:]) if len(tokens) > 1 else "Unknown"
    body_type = infer_body_type(label)
    return make, model, model_year, body_type


# ---------------------------------------------------------------------------
# Curated 45-image smoke-test manifest
# Files MUST exist on disk; do NOT add fabricated entries.
# ---------------------------------------------------------------------------
SMOKE_TEST_MANIFEST: List[Tuple[str, str]] = [
    # Sedan
    ("Audi S4 Sedan 2012",                  "AUTO_0001_audi_s4_sedan_2012.jpg"),
    ("BMW 3 Series Sedan 2012",             "AUTO_0002_bmw_3series_sedan_2012.jpg"),
    ("Toyota Camry Sedan 2012",             "AUTO_0003_toyota_camry_sedan_2012.jpg"),
    ("Honda Accord Sedan 2012",             "AUTO_0004_honda_accord_sedan_2012.jpg"),
    ("Mercedes-Benz C-Class Sedan 2012",    "AUTO_0005_mercedes_c_sedan_2012.jpg"),
    ("Hyundai Sonata Sedan 2012",           "AUTO_0006_hyundai_sonata_sedan_2012.jpg"),
    ("Toyota Corolla Sedan 2012",           "AUTO_0007_toyota_corolla_sedan_2012.jpg"),
    ("Rolls-Royce Phantom Sedan 2012",      "AUTO_0008_rollsroyce_phantom_sedan_2012.jpg"),
    # SUV
    ("Chevrolet TrailBlazer SS 2009",       "AUTO_0009_chevy_trailblazer_ss_2009.jpg"),
    ("Jeep Compass SUV 2012",               "AUTO_0010_jeep_compass_suv_2012.jpg"),
    ("Land Rover Range Rover SUV 2012",     "AUTO_0011_landrover_rangerover_suv_2012.jpg"),
    ("Toyota 4Runner SUV 2012",             "AUTO_0012_toyota_4runner_suv_2012.jpg"),
    ("BMW X3 SUV 2012",                     "AUTO_0013_bmw_x3_suv_2012.jpg"),
    ("Hyundai Tucson SUV 2012",             "AUTO_0014_hyundai_tucson_suv_2012.jpg"),
    ("Ford Edge SUV 2012",                  "AUTO_0015_ford_edge_suv_2012.jpg"),
    ("Jeep Grand Cherokee SUV 2012",        "AUTO_0016_jeep_grandcherokee_suv_2012.jpg"),
    # Coupe
    ("Audi A5 Coupe 2012",                  "AUTO_0017_audi_a5_coupe_2012.jpg"),
    ("BMW M3 Coupe 2012",                   "AUTO_0018_bmw_m3_coupe_2012.jpg"),
    ("Audi TT RS Coupe 2012",               "AUTO_0019_audi_ttrs_coupe_2012.jpg"),
    ("Honda Accord Coupe 2012",             "AUTO_0020_honda_accord_coupe_2012.jpg"),
    ("Dodge Challenger SRT8 2011",          "AUTO_0021_dodge_challenger_srt8_2011.jpg"),
    ("Bentley Continental GT Coupe 2012",   "AUTO_0022_bentley_cont_gt_coupe_2012.jpg"),
    ("Lamborghini Aventador Coupe 2012",    "AUTO_0023_lamborghini_aventador_2012.jpg"),
    ("Volkswagen Golf Hatchback 2012",      "AUTO_0024_vw_golf_hatchback_2012.jpg"),
    # Truck
    ("Ford F-150 Regular Cab 2012",         "AUTO_0025_ford_f150_regularcab_2012.jpg"),
    ("Chevrolet Silverado 1500 Hybrid Crew Cab 2012",
                                            "AUTO_0026_chevy_silverado_hybrid_2012.jpg"),
    ("GMC Canyon Extended Cab 2012",        "AUTO_0027_gmc_canyon_ext_2012.jpg"),
    ("Ford F-450 Super Duty Crew Cab 2012", "AUTO_0028_ford_f450_crew_2012.jpg"),
    ("Ford Ranger SuperCab 2011",           "AUTO_0029_ford_ranger_supercab_2011.jpg"),
    ("Dodge Ram Pickup 3500 Crew Cab 2010", "AUTO_0030_dodge_ram_3500_crewcab_2010.jpg"),
    ("Dodge Ram Pickup 3500 Quad Cab 2009", "AUTO_0031_dodge_ram_3500_quadcab_2009.jpg"),
    # Convertible
    ("BMW 1 Series Convertible 2012",       "AUTO_0032_bmw_1series_conv_2012.jpg"),
    ("Mercedes-Benz 300-Class Convertible 1993",
                                            "AUTO_0033_mercedes_300_conv_1993.jpg"),
    ("Ferrari California Convertible 2012", "AUTO_0034_ferrari_california_conv_2012.jpg"),
    ("Bentley Continental Supersports Conv. Convertible 2012",
                                            "AUTO_0035_bentley_ss_conv_2012.jpg"),
    ("Bugatti Veyron 16.4 Convertible 2009","AUTO_0036_bugatti_veyron_conv_2009.jpg"),
    ("smart fortwo Convertible 2012",       "AUTO_0037_smart_fortwo_conv_2012.jpg"),
    ("Spyker C8 Convertible 2009",          "AUTO_0038_spyker_c8_conv_2009.jpg"),
    # Wagon
    ("BMW 3 Series Wagon 2012",             "AUTO_0039_bmw_3series_wagon_2012.jpg"),
    ("Volvo C30 Hatchback 2012",            "AUTO_0040_volvo_c30_hatch_2012.jpg"),
    ("Chrysler Town and Country Minivan 2012",
                                            "AUTO_0041_chrysler_town_minivan_2012.jpg"),
    ("Dodge Caravan Minivan 1997",          "AUTO_0042_dodge_caravan_minivan_1997.jpg"),
    ("Nissan NV Passenger Van 2012",        "AUTO_0043_nissan_nv_van_2012.jpg"),
    ("Ram C/V Cargo Van Minivan 2012",      "AUTO_0044_ram_cv_minivan_2012.jpg"),
    ("FIAT 500 Convertible 2012",           "AUTO_0045_fiat_500_conv_2012.jpg"),
]


class AutomotiveAdapter:
    """
    Translates Stanford Cars smoke-test subset metadata into TrendLens
    catalog schema. Reuses existing pipeline without modification.
    """

    IMAGES_SUBDIR = Path("data") / "automotive" / "smoke_test" / "images"
    CATALOG_SUBDIR = Path("data") / "automotive" / "smoke_test"
    CATALOG_FILENAME = "automotive_catalog.csv"
    ATTRIBUTION_FILENAME = "ATTRIBUTION.md"

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.images_dir = self.project_root / self.IMAGES_SUBDIR
        self.catalog_dir = self.project_root / self.CATALOG_SUBDIR

    def build_catalog(
        self,
        manifest: Optional[List[Tuple[str, str]]] = None,
        skip_missing: bool = True
    ) -> pd.DataFrame:
        """
        Build catalog DataFrame from the smoke-test manifest.
        skip_missing=True silently drops entries whose image file is absent.
        skip_missing=False raises FileNotFoundError on first missing file.
        """
        if manifest is None:
            manifest = SMOKE_TEST_MANIFEST

        records: List[Dict] = []
        missing: List[str] = []

        for idx, (label, filename) in enumerate(manifest, start=1):
            rel_path = str(self.IMAGES_SUBDIR / filename)
            full_path = self.project_root / rel_path

            if not full_path.is_file():
                missing.append(filename)
                if not skip_missing:
                    raise FileNotFoundError(
                        f"Smoke-test image not found: {full_path}\n"
                        "Place Stanford Cars images in "
                        "data/automotive/smoke_test/images/"
                    )
                continue

            make, model, model_year, body_type = parse_label_components(label)
            item_id = f"AUTO_{idx:04d}"

            records.append({
                "item_id":    item_id,
                "title":      label,
                "category":   body_type,
                "style":      body_type,
                "make":       make,
                "model":      model,
                "model_year": model_year,
                "file_path":  rel_path,
                "tags": (
                    f"#{make.lower().replace(' ', '')} "
                    f"#{body_type} #cars #automotive #stanfordcars"
                ),
            })

        if missing:
            print(
                f"[AutomotiveAdapter] WARNING: {len(missing)} image(s) not found "
                f"and skipped:\n  " + "\n  ".join(missing[:10]) +
                ("  ..." if len(missing) > 10 else "")
            )

        if not records:
            if not skip_missing:
                raise RuntimeError(
                    "No valid automotive images found. Populate "
                    f"{self.images_dir} with smoke-test images first."
                )
            return pd.DataFrame(columns=[
                "item_id", "title", "category", "style", "make",
                "model", "model_year", "file_path", "tags"
            ])

        df = pd.DataFrame(records)
        print(
            f"[AutomotiveAdapter] Built catalog: {len(df)} items "
            f"from {len(manifest)} manifest entries."
        )
        return df

    def save_catalog(self, catalog_df: pd.DataFrame) -> Path:
        """Saves catalog CSV and attribution file."""
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.catalog_dir / self.CATALOG_FILENAME
        catalog_df.to_csv(csv_path, index=False)
        print(f"[AutomotiveAdapter] Saved catalog -> {csv_path}")
        self._write_attribution()
        return csv_path

    def load_catalog(self) -> pd.DataFrame:
        """Loads previously saved catalog CSV."""
        csv_path = self.catalog_dir / self.CATALOG_FILENAME
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Automotive catalog not found at {csv_path}. "
                "Run the automotive pipeline first."
            )
        return pd.read_csv(csv_path)

    def get_body_type_distribution(self, catalog_df: pd.DataFrame) -> Dict[str, int]:
        """Returns count of items per body-type category."""
        return catalog_df["category"].value_counts().to_dict()

    def _write_attribution(self) -> None:
        """Creates ATTRIBUTION.md with dataset licensing information."""
        attr_path = self.catalog_dir / self.ATTRIBUTION_FILENAME
        if attr_path.is_file():
            return
        content = (
            "# Dataset Attribution - Stanford Cars\n\n"
            "**Authors:** Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei\n"
            "**Paper:** 3D Object Representations for Fine-Grained Categorization\n"
            "*4th IEEE Workshop on 3D Representation and Recognition, ICCV 2013*\n"
            "**URL:** https://ai.stanford.edu/~jkrause/cars/car_dataset.html\n\n"
            "## License\n\n"
            "Stanford Cars is made available for **non-commercial research and "
            "educational use only**.\n"
            "- Do NOT redistribute raw images.\n"
            "- Do NOT use in commercial products.\n"
            "- Cite the original paper in any publication.\n\n"
            "## BibTeX\n\n"
            "```bibtex\n"
            "@inproceedings{KrauseStarkDengFei-Fei_3DRR2013,\n"
            "  title={3D Object Representations for Fine-Grained Categorization},\n"
            "  booktitle={4th IEEE Workshop on 3D Representation and Recognition, "
            "at ICCV 2013},\n"
            "  year={2013},\n"
            "  author={Jonathan Krause and Michael Stark and Jia Deng and Li Fei-Fei}\n"
            "}\n"
            "```\n\n"
            "## TrendLens Smoke-Test Subset\n\n"
            "This directory contains a curated 40-50-image smoke-test subset from\n"
            "the full Stanford Cars dataset for non-commercial FYP research. No\n"
            "images have been fabricated or modified from the originals.\n"
        )
        attr_path.write_text(content, encoding="utf-8")
        print(f"[AutomotiveAdapter] Wrote attribution -> {attr_path}")
