
from __future__ import annotations
from typing import Dict, List, TypedDict


class BackgroundConfig(TypedDict):
    plot_bg: str
    paper_bg: str
    grid: str
    text: str


Palette = List[str]


# ─────────────────────────────────────────────
# COLOR PALETTES
# ─────────────────────────────────────────────

PALETTES: Dict[str, Palette] = {
    # ── Business / Professional ──
    "corporate": [
        "#1B2A4A", "#2563EB", "#3B82F6",
        "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"
    ],
    "executive": [
        "#0F172A", "#1E40AF", "#7C3AED",
        "#DB2777", "#F97316", "#14B8A6", "#64748B"
    ],

    # ── Vibrant / Dynamic ──
    "vibrant": [
        "#FF006E", "#FB5607", "#FFBE0B",
        "#8338EC", "#3A86FF", "#06D6A0", "#118AB2"
    ],
    "neon": [
        "#00F5FF", "#FF00FF", "#39FF14",
        "#FFD700", "#FF6B6B", "#4ECDC4", "#A855F7"
    ],

    # ── Soft / Elegant ──
    "pastel": [
        "#6C5CE7", "#A29BFE", "#FD79A8",
        "#FDCB6E", "#55EFC4", "#74B9FF", "#E17055"
    ],
    "ocean": [
        "#0077B6", "#00B4D8", "#90E0EF",
        "#CAF0F8", "#023E8A", "#03045E", "#48CAE4"
    ],

    # ── Dark Mode ──
    "dark": [
        "#BB86FC", "#03DAC6", "#CF6679",
        "#FF7043", "#FFD54F", "#4FC3F7", "#81C784"
    ],
    "midnight": [
        "#60A5FA", "#34D399", "#F472B6",
        "#FBBF24", "#A78BFA", "#FB923C", "#E2E8F0"
    ],

    # ── Minimal / Neutral ──
    "monochrome": [
        "#111827", "#374151", "#6B7280",
        "#9CA3AF", "#D1D5DB", "#E5E7EB", "#F9FAFB"
    ],
    "slate": [
        "#0F172A", "#334155", "#475569",
        "#64748B", "#94A3B8", "#CBD5E1", "#F1F5F9"
    ],
}


# ─────────────────────────────────────────────
# BACKGROUND CONFIGS
# ─────────────────────────────────────────────

BACKGROUNDS: Dict[str, BackgroundConfig] = {
    "corporate": {"plot_bg": "#FAFBFC", "paper_bg": "#FFFFFF", "grid": "#E5E7EB", "text": "#1F2937"},
    "executive": {"plot_bg": "#F8FAFC", "paper_bg": "#FFFFFF", "grid": "#E2E8F0", "text": "#0F172A"},

    "vibrant":   {"plot_bg": "#FFFBF5", "paper_bg": "#FFFFFF", "grid": "#FDE8D0", "text": "#1A1A2E"},
    "neon":      {"plot_bg": "#0A0A1A", "paper_bg": "#0D0D24", "grid": "#1A1A3E", "text": "#E0E0FF"},

    "pastel":    {"plot_bg": "#FAFAFE", "paper_bg": "#FFFFFF", "grid": "#E8E5F0", "text": "#2D2D44"},
    "ocean":     {"plot_bg": "#F0F9FF", "paper_bg": "#FFFFFF", "grid": "#BAE6FD", "text": "#0C4A6E"},

    "dark":      {"plot_bg": "#121212", "paper_bg": "#1E1E1E", "grid": "#333333", "text": "#E0E0E0"},
    "midnight":  {"plot_bg": "#0F172A", "paper_bg": "#1E293B", "grid": "#334155", "text": "#E2E8F0"},

    "monochrome":{"plot_bg": "#FAFAFA", "paper_bg": "#FFFFFF", "grid": "#E5E7EB", "text": "#111827"},
    "slate":     {"plot_bg": "#F8FAFC", "paper_bg": "#FFFFFF", "grid": "#CBD5E1", "text": "#0F172A"},
}


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DEFAULT_SCHEME: str = "corporate"
SCHEME_NAMES: List[str] = list(PALETTES.keys())


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def validate_scheme(scheme: str) -> str:
    """Ensure scheme exists, fallback if invalid."""
    if scheme not in PALETTES:
        return DEFAULT_SCHEME
    return scheme


def get_palette(scheme: str = DEFAULT_SCHEME) -> Palette:
    """
    Get color palette for a scheme.

    Args:
        scheme: Name of color scheme

    Returns:
        List of hex colors
    """
    scheme = validate_scheme(scheme)
    return PALETTES[scheme]


def get_background(scheme: str = DEFAULT_SCHEME) -> BackgroundConfig:
    """
    Get background config for a scheme.

    Args:
        scheme: Name of color scheme

    Returns:
        BackgroundConfig dict
    """
    scheme = validate_scheme(scheme)
    return BACKGROUNDS[scheme]


def list_schemes() -> List[str]:
    """Return all available scheme names."""
    return SCHEME_NAMES.copy()


# ─────────────────────────────────────────────
# BONUS: GRADIENT GENERATOR
# ─────────────────────────────────────────────

def get_gradient(scheme: str = DEFAULT_SCHEME, n: int = 5) -> List[str]:
    """
    Generate a gradient subset from palette.

    Args:
        scheme: palette name
        n: number of colors needed

    Returns:
        List of colors (cycled if needed)
    """
    palette = get_palette(scheme)
    return [palette[i % len(palette)] for i in range(n)]


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    scheme = "vibrant"

    print("Available schemes:", list_schemes())
    print("Palette:", get_palette(scheme))
    print("Background:", get_background(scheme))
    print("Gradient (10):", get_gradient(scheme, 10))