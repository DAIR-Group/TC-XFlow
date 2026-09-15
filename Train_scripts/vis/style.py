STYLE = dict(
    obs_color="#000000",
    gt_color="#1F5FBF",
    pred_color="#D62728",
    ens_color="#D62728",
    ens_alpha=0.05,
    marker_size=6,
    lw_main=2.0,
    lw_thin=1.3,
    bg_color="#FFFFFF",
    land_color="#FFFFFF",
    ocean_color="#EAF3FB",
    border_color="#BBBBBB",
    grid_color="#CCCCCC",
    grid_alpha=0.5,
    error_color="#B8860B",
    title_pad=14,
    cone_50_fill="#5FA85F",
    cone_90_fill="#B08FD0",
    cone_50_alpha=0.45,
    cone_90_alpha=0.35,
    cone_edge_lw=0.0,
    text_color="#000000",
    panel_edge="#888888",
    info_box_edge="#2C4A7C",
    info_box_title_bg="#EAF0F8",
)

INTENSITY = [
    (0, 34, "TD", "#6699CC"),
    (34, 48, "TS", "#33AA33"),
    (48, 64, "TY", "#CCAA00"),
    (64, 84, "Sev.TY", "#FF8C00"),
    (84, 115, "Vis.TY", "#E03C00"),
    (115, 999, "Super TY", "#B000B0"),
]

SEED_COLORS = {
    "0": "#D62728",
    "1": "#1F77B4",
    "2": "#2CA02C",
    "3": "#9467BD",
    "4": "#FF7F0E",
    "42": "#D62728",
}
_SEED_COLOR_FALLBACK = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E", "#8C564B"]


def wind_intensity(wind_kt):
    for lo, hi, name, color in INTENSITY:
        if lo <= wind_kt < hi:
            return name, color
    return "Super TY", "#FF00FF"


def seed_color(seed_label: str, idx: int) -> str:
    return SEED_COLORS.get(str(seed_label), _SEED_COLOR_FALLBACK[idx % len(_SEED_COLOR_FALLBACK)])
