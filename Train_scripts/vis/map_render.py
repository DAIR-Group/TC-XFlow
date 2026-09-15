import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


from .style import STYLE


def make_map_ax(fig, subplot_spec, lon_range, lat_range, use_satellite_bg=True):
    if not HAS_CARTOPY:
        return _make_plain_ax(fig, subplot_spec, lon_range, lat_range)

    ax = fig.add_subplot(subplot_spec, projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())

    drew_bg = _draw_land_ocean(fig, ax, use_satellite_bg)
    if not drew_bg:
        _draw_fallback_background(fig, ax)

    _draw_coastline_and_borders(fig, ax)
    _draw_gridlines(ax)
    return ax


def _draw_land_ocean(fig, ax, use_satellite_bg):
    if not use_satellite_bg:
        return False

    for scale in ("10m", "50m"):
        try:
            land_feat = cfeature.NaturalEarthFeature(
                "physical", "land", scale, facecolor="#E8E4D8", edgecolor="none"
            )
            ocean_feat = cfeature.NaturalEarthFeature(
                "physical", "ocean", scale, facecolor="#C8DCF0", edgecolor="none"
            )
            ax.add_feature(land_feat, zorder=1)
            ax.add_feature(ocean_feat, zorder=0)
            fig.canvas.draw()

            return True
        except Exception:
            for coll in list(ax.collections):
                coll.remove()

    return False


def _draw_fallback_background(fig, ax):
    try:
        ax.add_feature(cfeature.OCEAN, facecolor=STYLE["ocean_color"], zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=STYLE["land_color"], zorder=1, alpha=0.9)
        fig.canvas.draw()
    except Exception:
        ax.set_facecolor(STYLE["ocean_color"])


def _draw_coastline_and_borders(fig, ax):
    try:
        ax.add_feature(
            cfeature.COASTLINE.with_scale("50m"), edgecolor="#4D4D4D", linewidth=0.8, zorder=2
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            edgecolor=STYLE["border_color"],
            linewidth=0.4,
            linestyle=":",
            zorder=2,
        )
        fig.canvas.draw()
    except Exception:
        ax.add_feature(cfeature.COASTLINE, edgecolor="#4D4D4D", linewidth=0.8, zorder=2)
        ax.add_feature(
            cfeature.BORDERS,
            edgecolor=STYLE["border_color"],
            linewidth=0.4,
            linestyle=":",
            zorder=2,
        )


def _draw_gridlines(ax):
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color=STYLE["grid_color"],
        alpha=STYLE["grid_alpha"],
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = dict(color=STYLE["text_color"], fontsize=7)
    gl.ylabel_style = dict(color=STYLE["text_color"], fontsize=7)


def _make_plain_ax(fig, subplot_spec, lon_range, lat_range):
    ax = fig.add_subplot(subplot_spec)
    ax.set_facecolor(STYLE["bg_color"])
    ax.set_xlim(*lon_range)
    ax.set_ylim(*lat_range)
    for lon in np.arange(np.ceil(lon_range[0] / 5) * 5, lon_range[1], 5):
        ax.axvline(lon, color=STYLE["grid_color"], alpha=STYLE["grid_alpha"], lw=0.5)
    for lat in np.arange(np.ceil(lat_range[0] / 5) * 5, lat_range[1], 5):
        ax.axhline(lat, color=STYLE["grid_color"], alpha=STYLE["grid_alpha"], lw=0.5)
    ax.set_xlabel("Longitude (°E)", color=STYLE["text_color"], fontsize=8)
    ax.set_ylabel("Latitude (°N)", color=STYLE["text_color"], fontsize=8)
    ax.tick_params(colors=STYLE["text_color"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["panel_edge"])
    return ax
