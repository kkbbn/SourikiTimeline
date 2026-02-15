import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Set

ASSETS_DIR = os.path.join('resources', 'assets')


class CharaTemplate:
    """A character's template image for matching, following BAAH's approach."""

    def __init__(self, name: str, pattern: np.ndarray, mask: np.ndarray):
        self.name = name
        self.pattern = pattern  # BGR, 3 channels
        self.mask = mask        # binary mask from alpha channel

    def scaled(self, scale: float) -> 'CharaTemplate':
        """Return a new CharaTemplate scaled by the given factor."""
        if scale == 1.0:
            return self
        h, w = self.pattern.shape[:2]
        nw, nh = int(w * scale), int(h * scale)
        if nw < 1 or nh < 1:
            return self
        pattern = cv2.resize(self.pattern, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(self.mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        return CharaTemplate(self.name, pattern, mask)


def load_asset_templates(assets_dir: str = ASSETS_DIR) -> List[CharaTemplate]:
    """Load all character portrait templates from the assets directory.
    Uses cv2.IMREAD_UNCHANGED to preserve alpha channel, like BAAH."""
    templates = []
    if not os.path.isdir(assets_dir):
        return templates
    for filename in sorted(os.listdir(assets_dir)):
        if not filename.endswith('.png'):
            continue
        filepath = os.path.join(assets_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        name = os.path.splitext(filename)[0]

        if len(img.shape) == 3 and img.shape[2] == 4:
            mask = img[:, :, 3].copy()
            mask[mask > 0] = 255
            pattern = img[:, :, :3]
        else:
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
            pattern = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        templates.append(CharaTemplate(name, pattern, mask))
    return templates


def match_pattern(source: np.ndarray, template: CharaTemplate,
                  threshold: float = 0.6) -> Tuple[bool, Tuple[int, int], float]:
    """
    Match a template in the source image using cv2.matchTemplate.
    Uses TM_CCOEFF_NORMED with mask, following BAAH's approach.

    Returns: (matched, (center_x, center_y), confidence)
    """
    if source is None or template.pattern is None:
        return (False, (0, 0), 0)

    sh, sw = source.shape[:2]
    th, tw = template.pattern.shape[:2]

    if sh < th or sw < tw:
        return (False, (0, 0), 0)

    result = cv2.matchTemplate(source, template.pattern, cv2.TM_CCOEFF_NORMED, mask=template.mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    center_x = max_loc[0] + tw // 2
    center_y = max_loc[1] + th // 2

    if max_val >= threshold:
        return (True, (center_x, center_y), max_val)
    return (False, (0, 0), max_val)


def find_card_matches(
    source_bgr: np.ndarray,
    templates: List[CharaTemplate],
    scale: float = 0.55,
    threshold: float = 0.6,
    card_region_x_ratio: float = 0.6,
    card_region_y_ratio: float = 0.75,
) -> List[Tuple[str, float, Tuple[int, int]]]:
    """
    Find character card matches in the bottom-right card area of a frame.

    Args:
        source_bgr: Full frame in BGR format
        templates: List of character templates
        scale: Scale factor to apply to templates before matching
        threshold: Minimum match confidence
        card_region_x_ratio: Only consider matches where x > frame_width * this ratio
        card_region_y_ratio: Only consider matches where y > frame_height * this ratio

    Returns: List of (name, confidence, (center_x, center_y)) sorted by confidence desc
    """
    h, w = source_bgr.shape[:2]

    # Crop to the bottom-right region for faster matching
    crop_y = int(h * card_region_y_ratio)
    crop_x = int(w * card_region_x_ratio)
    card_region = source_bgr[crop_y:, crop_x:]

    matches = []
    for tmpl in templates:
        scaled_tmpl = tmpl.scaled(scale)
        th, tw = scaled_tmpl.pattern.shape[:2]

        if card_region.shape[0] < th or card_region.shape[1] < tw:
            continue

        result = cv2.matchTemplate(
            card_region, scaled_tmpl.pattern, cv2.TM_CCOEFF_NORMED, mask=scaled_tmpl.mask
        )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Convert back to full-frame coordinates
        cx = crop_x + max_loc[0] + tw // 2
        cy = crop_y + max_loc[1] + th // 2

        if max_val >= threshold:
            matches.append((tmpl.name, max_val, (cx, cy)))

    matches.sort(key=lambda x: -x[1])
    return matches


def select_card_characters(
    matches: List[Tuple[str, float, Tuple[int, int]]],
    max_cards: int = 4,
    min_x_separation: int = 50,
) -> List[Tuple[str, float, Tuple[int, int]]]:
    """
    From a list of matches, select up to max_cards that are spatially separated.
    This filters out duplicate detections at the same card position.

    The matches should be sorted by confidence (highest first).
    """
    selected = []
    for name, conf, (cx, cy) in matches:
        # Check if this position is too close to an already-selected match
        too_close = False
        for _, _, (sx, sy) in selected:
            if abs(cx - sx) < min_x_separation:
                too_close = True
                break
        if not too_close:
            selected.append((name, conf, (cx, cy)))
            if len(selected) >= max_cards:
                break
    return selected


def detect_party_members(
    frames_bgr: List[np.ndarray],
    all_templates: List[CharaTemplate],
    scale: float = 0.55,
    threshold: float = 0.6,
) -> List[str]:
    """
    Analyze several frames to identify which characters are in the party.
    Returns a list of unique character names detected across the frames.
    """
    seen_names: Set[str] = set()

    for frame in frames_bgr:
        matches = find_card_matches(frame, all_templates, scale=scale, threshold=threshold)
        cards = select_card_characters(matches)
        for name, _, _ in cards:
            seen_names.add(name)

    return sorted(seen_names)
