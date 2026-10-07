"""Figures of the README: the trade-off chart and the animated inspection sheet."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb
from PIL import Image

from vqgate.data import CROP_SIDE, Sample, index_category, load_display_image, load_mask
from vqgate.experiment import CONFIG_BY_NAME, CONFIGS
from vqgate.maps import anomaly_maps
from vqgate.report import GroupKey, ScoreGroup, gate_run

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
STATUS_OK = "#0ca30c"
STATUS_NOK = "#d03b3b"
HEAT_COLOR = "#eb6834"
MASK_OUTLINE = "#1560d4"
MASK_HALO = "#ffffff"
OUTLINE_WIDTH = 1.8

LINE_WIDTH = 2.0
MARKER_SIZE = 9.0
MARKER_RING = 2.0
FIGURE_DPI = 140

HERO_SIZE_INCHES = (9.0, 3.9)
HERO_DPI = 100
HERO_FRAME_MS = 2600
HEAT_FROM = 0.5
HEAT_TO = 1.0
HEAT_MAX_ALPHA = 0.85
GAUGE_SPAN = 2.0
ATTRIBUTION = "Parts: MVTec AD (Bergmann et al., CVPR 2019), CC BY-NC-SA 4.0"


def _style_axes(axes: plt.Axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=1.0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    axes.xaxis.label.set_color(TEXT_SECONDARY)
    axes.yaxis.label.set_color(TEXT_SECONDARY)


def plot_tradeoff(summary: dict, path: Path) -> None:
    """Escape rate against realised false-reject rate, and realised against promised rate.

    Every point is a threshold obtained from held-out good parts at one target rate; counts
    are pooled over categories and seeds. The marker is the 5 % target of the README table.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.3), facecolor=SURFACE)
    target = summary["target_false_reject"]
    for config, color in zip(CONFIGS, SERIES_COLORS, strict=True):
        curve = summary["tradeoff"][config.name]
        targets = np.array([point["target"] for point in curve]) * 100
        false_rejects = np.array([point["false_reject_rate"] for point in curve]) * 100
        escapes = np.array([point["escape_rate"] for point in curve]) * 100
        at_target = next(i for i, point in enumerate(curve) if point["target"] == target)
        for axes, x, y in ((left, false_rejects, escapes), (right, targets, false_rejects)):
            axes.plot(
                x,
                y,
                color=color,
                linewidth=LINE_WIDTH,
                label=config.label,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            axes.plot(
                x[at_target],
                y[at_target],
                "o",
                color=color,
                markersize=MARKER_SIZE,
                markeredgecolor=SURFACE,
                markeredgewidth=MARKER_RING,
            )
    limit = max(axes.get_xlim()[1] for axes in (left, right))
    right.plot([0, limit], [0, limit], color=TEXT_SECONDARY, linewidth=1.0, zorder=1)
    right.annotate(
        "promise kept\n(realised = target)",
        (limit * 0.62, limit * 0.62),
        xytext=(8, -26),
        textcoords="offset points",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    left.set_title("What a false-reject budget buys", loc="left", fontsize=11, color=TEXT_PRIMARY)
    left.set_xlabel("good parts refused (%)")
    left.set_ylabel("defective parts passed (%)")
    right.set_title(
        "Does the threshold keep its promise?", loc="left", fontsize=11, color=TEXT_PRIMARY
    )
    right.set_xlabel("target false-reject rate (%)")
    right.set_ylabel("good parts refused on the test set (%)")
    for axes in (left, right):
        _style_axes(axes)
        axes.set_xlim(left=0)
        axes.set_ylim(bottom=0)
    legend = left.legend(
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_PRIMARY,
        loc="upper right",
        title=f"dot = {target * 100:.0f} % target",
        title_fontsize=8,
    )
    legend.get_title().set_color(TEXT_SECONDARY)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, facecolor=SURFACE)
    plt.close(figure)


@dataclass(frozen=True)
class HeroFrame:
    """One inspected part of the animation and why it was selected."""

    role: str
    category: str
    image: str
    defective: bool
    score: float
    threshold: float

    @property
    def margin(self) -> float:
        return self.score / self.threshold

    @property
    def rejected(self) -> bool:
        return self.score > self.threshold

    @property
    def outcome(self) -> str:
        if self.defective:
            return "correct reject" if self.rejected else "ESCAPE: defect passed"
        return "FALSE REJECT: good part refused" if self.rejected else "correct accept"


def defects_outside_crop(data_root: Path, categories: list[str]) -> frozenset[tuple[str, str]]:
    """(category, image key) of defective test parts whose whole defect is cropped away.

    The network sees the central 224 px of the 256 px resized image, so a defect that lies
    entirely in the discarded border cannot be detected by any method at this input size.
    """
    return frozenset(
        (category, sample.key)
        for category in categories
        for sample in index_category(data_root, category)
        if sample.split == "test" and sample.is_defective and not load_mask(sample).any()
    )


def select_hero_frames(
    groups: dict[GroupKey, ScoreGroup],
    categories: list[str],
    config: str,
    seed: int,
    alpha: float,
    cropped_away: frozenset[tuple[str, str]] = frozenset(),
) -> list[HeroFrame]:
    """Pick five test parts by their score-to-threshold ratio, pooled over the categories.

    The rule only looks at the ratio r = score / threshold of the ``gate`` run of ``seed``:
    the median caught defect, the median accepted good part, the caught defect closest to
    the threshold, the defective part with the lowest r, and the good part with the highest r.
    Parts listed in ``cropped_away`` (defect outside the network input) are not eligible as the
    lowest-r defective part: they are escapes of the preprocessing, not of the detector.
    """
    parts: list[HeroFrame] = []
    for category in categories:
        threshold, _ = gate_run(groups, config, category, seed, alpha)
        test = groups[(config, category, seed, "gate", "test")]
        for image, score, defective in zip(test.images, test.scores, test.defective, strict=True):
            parts.append(HeroFrame("", category, image, bool(defective), float(score), threshold))
    parts.sort(key=lambda part: (part.margin, part.category, part.image))
    caught = [p for p in parts if p.defective and p.rejected]
    accepted = [p for p in parts if not p.defective and not p.rejected]
    chosen = {
        "typical defect": caught[(len(caught) - 1) // 2],
        "typical good part": accepted[(len(accepted) - 1) // 2],
        "closest call that was caught": caught[0],
        "lowest-scoring defect": next(
            p for p in parts if p.defective and (p.category, p.image) not in cropped_away
        ),
        "highest-scoring good part": next(p for p in reversed(parts) if not p.defective),
    }
    return [replace(part, role=role) for role, part in chosen.items()]


def _heat_overlay(display: np.ndarray, relative_map: np.ndarray) -> np.ndarray:
    """Grey photo with the anomaly map painted in one hue; opacity grows with the local score."""
    grey = display.mean(axis=2, keepdims=True).repeat(3, axis=2) / 255.0
    alpha = np.clip((relative_map - HEAT_FROM) / (HEAT_TO - HEAT_FROM), 0.0, 1.0) * HEAT_MAX_ALPHA
    heat = np.array(to_rgb(HEAT_COLOR)).reshape(1, 1, 3)
    return grey * (1.0 - alpha[..., None]) + heat * alpha[..., None]


def _draw_frame(
    frame: HeroFrame, sample: Sample, patch_scores: np.ndarray, label: str, position: str
) -> Image.Image:
    display = load_display_image(sample.image_path)
    smoothed = anomaly_maps(torch.from_numpy(patch_scores).unsqueeze(0), CROP_SIDE)[0].numpy()
    status = STATUS_NOK if frame.rejected else STATUS_OK
    figure = plt.figure(figsize=HERO_SIZE_INCHES, dpi=HERO_DPI, facecolor=SURFACE)
    photo = figure.add_axes((0.02, 0.12, 0.30, 0.72))
    heat = figure.add_axes((0.345, 0.12, 0.30, 0.72))
    panel = figure.add_axes((0.675, 0.12, 0.31, 0.72))
    photo.imshow(display)
    if frame.defective:
        mask = load_mask(sample)
        photo.contour(mask, levels=[0.5], colors=[MASK_HALO], linewidths=OUTLINE_WIDTH + 2.0)
        photo.contour(mask, levels=[0.5], colors=[MASK_OUTLINE], linewidths=OUTLINE_WIDTH)
    heat.imshow(_heat_overlay(display, smoothed / frame.threshold))
    truth = "defective, outline = ground truth" if frame.defective else "good"
    photo.set_title(f"part ({truth})", fontsize=9, color=TEXT_SECONDARY, loc="left")
    heat.set_title(
        "anomaly map (orange = near threshold)",
        fontsize=9,
        color=TEXT_SECONDARY,
        loc="left",
    )
    for axes in (photo, heat, panel):
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)
    panel.set_facecolor(SURFACE)
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    verdict = "NOK" if frame.rejected else "OK"
    panel.text(0.0, 0.93, verdict, fontsize=34, fontweight="bold", color=status, va="center")
    panel.text(
        0.0, 0.77, frame.outcome, fontsize=10, color=TEXT_PRIMARY, va="center", fontweight="bold"
    )
    panel.text(
        0.0, 0.70, f"selected as: {frame.role}", fontsize=8.5, color=TEXT_SECONDARY, va="center"
    )
    gauge_y, gauge_height = 0.50, 0.07
    fill = min(frame.margin / GAUGE_SPAN, 1.0)
    panel.barh(gauge_y, 1.0, height=gauge_height, color=GRID, align="center")
    panel.barh(gauge_y, fill, height=gauge_height, color=status, align="center")
    tick = 1.0 / GAUGE_SPAN
    panel.plot([tick, tick], [gauge_y - 0.075, gauge_y + 0.075], color=TEXT_PRIMARY, linewidth=2)
    panel.text(
        tick,
        gauge_y + 0.10,
        f"threshold {frame.threshold:.2f}",
        fontsize=8.5,
        color=TEXT_SECONDARY,
        ha="center",
        va="bottom",
    )
    panel.text(
        0.0,
        gauge_y - 0.10,
        f"score {frame.score:.2f}  ({frame.margin:.2f} x threshold)",
        fontsize=10,
        color=TEXT_PRIMARY,
        va="top",
    )
    defect = sample.defect.replace("_", " ")
    panel.text(
        0.0, 0.27, f"category: {frame.category.replace('_', ' ')}", fontsize=9.5, color=TEXT_PRIMARY
    )
    panel.text(0.0, 0.19, f"ground truth: {defect}", fontsize=9.5, color=TEXT_PRIMARY)
    panel.text(
        0.0,
        0.06,
        f"{label}\nthreshold from held-out good parts only",
        fontsize=8.5,
        color=TEXT_SECONDARY,
        va="center",
    )
    figure.text(
        0.02,
        0.93,
        "Visual quality gate",
        fontsize=13,
        fontweight="bold",
        color=TEXT_PRIMARY,
        va="center",
    )
    figure.text(0.985, 0.93, position, fontsize=9, color=TEXT_SECONDARY, va="center", ha="right")
    figure.text(0.02, 0.045, ATTRIBUTION, fontsize=8, color=TEXT_SECONDARY, va="center")
    figure.canvas.draw()
    pixels = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return Image.fromarray(pixels)


def render_hero(
    frames: list[HeroFrame],
    data_root: Path,
    patch_scores: dict[str, np.ndarray],
    config: str,
    path: Path,
) -> None:
    """Animated sheet: one frame per selected part, with heat map, score, threshold, verdict.

    ``patch_scores[category]`` holds the raw patch-score grids of that category's test images,
    in ``index_category`` order.
    """
    label = CONFIG_BY_NAME[config].label
    images = []
    for number, frame in enumerate(frames, start=1):
        test = [s for s in index_category(data_root, frame.category) if s.split == "test"]
        position = next(i for i, s in enumerate(test) if s.key == frame.image)
        images.append(
            _draw_frame(
                frame,
                test[position],
                patch_scores[frame.category][position],
                label,
                f"{number} / {len(frames)}",
            )
        )
    palette_source = Image.new("RGB", (images[0].width, images[0].height * len(images)))
    for row, image in enumerate(images):
        palette_source.paste(image, (0, row * image.height))
    palette = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=HERO_FRAME_MS,
        loop=0,
        optimize=False,
    )
