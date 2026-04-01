"""
apastats — APA 7th edition compliant statistical analyses for organizational science.

Provides:
  - Descriptive statistics and intercorrelation tables (JAP "Table 1")
  - Moderation analysis with simple slopes, Johnson-Neyman, and interaction plots
  - Mediation analysis with bootstrap confidence intervals and path diagrams
  - Effect size calculations (Cohen's d, f², R², partial η²)
  - APA-formatted export to Word (.docx), LaTeX, and CSV
"""

__version__ = "0.1.3"

from apastats.descriptives import descriptives_table
from apastats.moderation import moderation_analysis
from apastats.mediation import mediation_analysis
from apastats.effect_sizes import cohens_d, cohens_f2, r2_effect, partial_eta_squared
from apastats.export import to_docx, to_latex, to_csv
from apastats.conditional_process import conditional_process
from apastats.reliability import scale_reliability
from apastats.cfa import cfa

__all__ = [
    "descriptives_table",
    "moderation_analysis",
    "mediation_analysis",
    "conditional_process",
    "scale_reliability",
    "cfa",
    "cohens_d",
    "cohens_f2",
    "r2_effect",
    "partial_eta_squared",
    "to_docx",
    "to_latex",
    "to_csv",
]
