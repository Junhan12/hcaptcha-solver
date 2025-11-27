"""
Utilities that translate inference results (bounding boxes) into real clicks
inside the live hCaptcha challenge using Selenium.
"""

from __future__ import annotations

import time
import re
from typing import Dict, Iterable, List, Optional
from collections import Counter

from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import StaleElementReferenceException

from app.utils.logger import get_logger

# Initialize logger
log = get_logger("clicker")


def _extract_detections(result: Dict):
    """
    Normalize API response variations into a flat detection list.
    Returns [] if nothing actionable exists.
    """
    if not isinstance(result, dict):
        return []
    
    detections = result.get("results", [])
    
    # Handle shape like {"message": "...", "detections": []}
    if isinstance(detections, dict):
        if "detections" in detections:
            detections = detections.get("detections", [])
        else:
            return []
    
    if not isinstance(detections, list):
        return []
    
    return detections


def _validate_detections_by_challenge_type(detections: List[Dict], challenge_type_id: Optional[str] = None) -> List[Dict]:
    """
    Apply challenge-type-specific validation rules to filter detections.
    
    Args:
        detections: List of detection dictionaries, each with 'class' and other fields
        challenge_type_id: Optional challenge type ID (e.g., 'ct-001')
    
    Returns:
        Filtered list of detections based on challenge type rules
    """
    if not detections or not challenge_type_id:
        return detections
    
    # Filter for valid detections (must have class and bbox)
    valid_detections = [
        d for d in detections 
        if isinstance(d, dict) 
        and 'class' in d 
        and 'bbox' in d 
        and isinstance(d.get('bbox'), (list, tuple))
        and len(d.get('bbox', [])) >= 4
    ]
    
    if not valid_detections:
        return []
    
    # Challenge type ct-001: "click", "two elements", "similar"
    # Rule: Only click objects that appear exactly twice (duplicates)
    if challenge_type_id == "ct-001":
        # Count occurrences of each class
        class_counts = Counter(d.get('class', '') for d in valid_detections)
        
        log.debug(f"Challenge type ct-001: Class counts = {dict(class_counts)}", indent=1)
        
        # Filter to only include detections for classes that appear exactly twice
        filtered = [
            d for d in valid_detections 
            if class_counts.get(d.get('class', ''), 0) == 2
        ]
        
        duplicate_classes = [cls for cls, count in class_counts.items() if count == 2]
        log.info(f"Classes appearing twice (to click): {duplicate_classes}", indent=1)
        log.info(f"Filtered {len(valid_detections)} detections to {len(filtered)} (only duplicates)", indent=1)
        
        # For ct-001, limit to first 2 detections (only click 2 times total)
        if len(filtered) > 2:
            log.warning(f"Limiting to first 2 detections (ct-001 click limit: {len(filtered)} -> 2)", indent=1)
            filtered = filtered[:2]
        
        return filtered
    
    # For other challenge types, return all valid detections (no filtering)
    # Future challenge types can be added here with their specific rules
    return valid_detections


def _move_and_click(
    driver: WebDriver,
    element: WebElement,
    offset_x: Optional[float] = None,
    offset_y: Optional[float] = None,
    client_x: Optional[float] = None,
    client_y: Optional[float] = None,
):
    """
    Move the physical mouse cursor (via ActionChains) to the element/offset and click.
    Falls back to dispatching DOM events if ActionChains fails.
    """
    actions = ActionChains(driver)
    try:
        if offset_x is None or offset_y is None:
            actions.move_to_element(element)
        else:
            actions.move_to_element_with_offset(element, offset_x, offset_y)
        actions.pause(1).click().pause(1).perform()
    except Exception as exc:
        print(f"  [clicker] ActionChains click failed, falling back to DOM events: {exc}")
        driver.execute_script(
            """
            const target = arguments[0];
            const point = arguments[1];
            ['mousemove','mousedown','mouseup','click'].forEach(type => {
                const evt = new MouseEvent(type, {
                    bubbles: true,
                    cancelable: true,
                    view: window,
                    clientX: point.x,
                    clientY: point.y
                });
                target.dispatchEvent(evt);
            });
            """,
            element,
            {"x": float(client_x or 0), "y": float(client_y or 0)},
        ) 


def _ensure_canvas(driver: WebDriver, element: Optional[WebElement]) -> Optional[WebElement]:
    """
    Ensure we have a fresh reference to the canvas element.
    hCaptcha re-renders the canvas after each click, which invalidates the
    original WebElement. When we detect staleness, fetch the first canvas found.
    """
    try:
        if element is not None:
            # Accessing a property forces Selenium to validate the reference.
            element.is_enabled()
            return element
    except StaleElementReferenceException:
        pass
    
    canvases = driver.find_elements(By.TAG_NAME, "canvas")
    if canvases:
        return canvases[0]
    return None


def click_canvas_from_response(
    driver: WebDriver,
    canvas_element: WebElement,
    api_result: Dict,
    confidence_threshold: float = 0.0,
    pause_seconds: float = 0.25,
    challenge_type_id: Optional[str] = None,
):
    """
    Use detections returned from /solve_hcaptcha to click positions on the canvas.
    Note: Detections are already filtered by the model's confidence and IoU thresholds
    during inference. This function clicks on ALL returned detections by default
    (confidence_threshold=0.0), unless challenge_type_id validation rules apply.
    
    Args:
        driver: Selenium WebDriver instance
        canvas_element: Canvas WebElement to click on
        api_result: API response dictionary with detections
        confidence_threshold: Minimum confidence to click (default: 0.0)
        pause_seconds: Delay between clicks (default: 0.25)
        challenge_type_id: Optional challenge type ID for validation rules (e.g., 'ct-001')
    
    Returns number of successful click attempts.
    """
    detections = _extract_detections(api_result)
    log.info(f"Extracted {len(detections)} detections from API result", indent=1)
    
    if not detections:
        log.warning(f"No detections found in API result. Result keys: {list(api_result.keys()) if isinstance(api_result, dict) else 'not a dict'}", indent=1)
        return 0
    
    # Apply challenge-type-specific validation
    if challenge_type_id:
        detections = _validate_detections_by_challenge_type(detections, challenge_type_id)
        log.info(f"After validation: {len(detections)} detections remain", indent=1)
    
    if not detections:
        log.warning("No detections remaining after validation", indent=1)
        return 0
    
    click_count = 0
    valid_detections = 0
    max_clicks = 2 if challenge_type_id == "ct-001" else len(detections)  # Limit to 2 clicks for ct-001
    
    if challenge_type_id == "ct-001":
        log.info(f"ct-001 click limit: Maximum {max_clicks} clicks allowed", indent=1)
    
    for det in detections:
        # For ct-001, stop after 2 clicks
        if challenge_type_id == "ct-001" and click_count >= max_clicks:
            log.warning(f"Reached click limit ({max_clicks}) for ct-001. Stopping.", indent=1)
            break
        
        canvas_element = _ensure_canvas(driver, canvas_element)
        if canvas_element is None:
            print("  [click_canvas] Canvas element no longer available")
            break
        
        rect = driver.execute_script(
            """
            const r = arguments[0].getBoundingClientRect();
            return {left: r.left, top: r.top, width: r.width, height: r.height};
            """,
            canvas_element,
        )
        
        intrinsic_width = float(canvas_element.get_attribute("width") or rect["width"])
        intrinsic_height = float(canvas_element.get_attribute("height") or rect["height"])
        
        if intrinsic_width == 0 or intrinsic_height == 0:
            log.error("Canvas dimensions invalid (0)", indent=1)
            break
        
        scale_x = rect["width"] / intrinsic_width
        scale_y = rect["height"] / intrinsic_height
        
        bbox = det.get("bbox", [])
        if len(bbox) < 4:
            log.warning(f"Skipping detection with invalid bbox: {bbox}", indent=1)
            continue
        
        confidence = det.get("confidence", 0.0)
        
        # Click on all detections since they're already filtered by model thresholds
        if confidence < confidence_threshold:
            log.debug(f"Skipping detection with confidence {confidence:.2f} < threshold {confidence_threshold:.2f}", indent=1)
            continue
        
        valid_detections += 1
        
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        
        client_x = rect["left"] + center_x * scale_x
        client_y = rect["top"] + center_y * scale_y
        
        # Selenium 4's W3C pointer actions treat offsets as relative to the element's center.
        # Convert canvas coordinates (origin = top-left) to center-relative offsets so that the
        # physical mouse lands on the intended tile.
        offset_x = center_x * scale_x - (rect["width"] / 2.0)
        offset_y = center_y * scale_y - (rect["height"] / 2.0)
        
        log.info(
            f"Canvas click @ ({client_x:.1f}, {client_y:.1f}) "
            f"(class={det.get('class')}, conf={confidence:.2f})",
            indent=1
        )
        
        try:
            _move_and_click(
                driver,
                canvas_element,
                offset_x=offset_x,
                offset_y=offset_y,
                client_x=client_x,
                client_y=client_y,
            )
            click_count += 1
            log.success(f"Click {click_count}/{max_clicks if challenge_type_id == 'ct-001' else valid_detections} successful", indent=1)
            time.sleep(max(pause_seconds, 0))
        except Exception as exc:
            log.error(f"Canvas click failed: {exc}", indent=1)
            import traceback
            traceback.print_exc()
    
    log.info(f"Total: {len(detections)} detections, {valid_detections} valid, {click_count} clicks performed", indent=1)
    return click_count


def _has_positive_detection(entries: Iterable[Dict], confidence_threshold: float = 0.0):
    """
    Check if there are any detections above the threshold.
    Default threshold is 0.0 to match all detections
    """
    for det in entries or []:
        if not isinstance(det, dict):
            continue
        if det.get("confidence", 0.0) >= confidence_threshold:
            return True
    return False


def click_tiles_from_batch_response(driver, tile_elements, api_result, confidence_threshold=0.0, challenge_type_id=None):
    """
    Click tiles based on batch API response results.
    
    Args:
        driver: Selenium WebDriver instance
        tile_elements: List of tile WebElements
        api_result: Batch API response dictionary
        confidence_threshold: Minimum confidence to click (default: 0.0)
        challenge_type_id: Optional challenge type ID for validation rules (e.g., 'ct-001')
    
    Returns:
        Number of successful clicks
    """
    print(f"\n{'='*60}")
    print(f"TILE CLICKING DEBUG")
    print(f"{'='*60}")
    print(f"Total tile elements available: {len(tile_elements)}")
    
    batch_results = api_result.get("results", [])
    print(f"Total batch results from API: {len(batch_results)}")
    
    if challenge_type_id:
        print(f"Challenge type: {challenge_type_id} - applying validation rules")
    
    # Filter tile_elements to only include valid tiles (same filtering as in crawler)
    # The tile_elements list might contain more divs than actual tiles
    valid_tile_elements = []
    for div in tile_elements:
        style = div.get_attribute("style") or ""
        # Check if this div has a valid image URL (same check as in crawler)
        if re.search(r'url\("(.*?)"\)', style):
            valid_tile_elements.append(div)
    
    print(f"Valid tile elements (with image URLs): {len(valid_tile_elements)}")
    
    # Determine if a sample tile was excluded (when there are 10 valid tiles, first is excluded)
    # In this case, batch_results will have 9 entries, and image_index 1 should map to valid_tile_elements[1]
    sample_offset = 0
    if len(valid_tile_elements) == 10 and len(batch_results) == 9:
        sample_offset = 1
        print("Detected 10 valid tiles with 9 batch results - first tile is sample (offset=1)")
    elif len(valid_tile_elements) == len(batch_results):
        sample_offset = 0
        print(f"Valid tiles match batch results count - no sample tile excluded")
    else:
        # Use original list if counts don't match expected patterns
        print(f"Using original {len(tile_elements)} tile elements (filtered count: {len(valid_tile_elements)}, batch: {len(batch_results)})")
        valid_tile_elements = tile_elements
        sample_offset = 0
    
    tiles_to_click = set()
    
    # For challenge type ct-001, we need to collect all detections across all tiles first
    # to count class occurrences, then determine which tiles to click
    if challenge_type_id == "ct-001":
        # Collect all detections from all tiles
        all_detections = []
        for entry in batch_results:
            detections = entry.get("results", [])
            if isinstance(detections, list):
                all_detections.extend(detections)
        
        # Apply validation to get classes that should be clicked (appear twice)
        validated_detections = _validate_detections_by_challenge_type(all_detections, challenge_type_id)
        valid_classes = set(d.get('class', '') for d in validated_detections)
        
        log.info(f"Valid classes to click (appear twice): {valid_classes}", indent=1)
        
        # For ct-001, limit to only 2 clicks total
        # Collect tiles with valid classes, but limit to first 2 tiles
        tiles_with_valid_class = []
        for entry in batch_results:
            idx = entry.get("image_index")
            tile_idx = int(idx) - 1 + sample_offset
            detections = entry.get("results", [])
            
            if isinstance(detections, list):
                # Check if this tile has any detection with a valid class (appears twice)
                has_valid_class = any(
                    d.get('class', '') in valid_classes 
                    for d in detections 
                    if isinstance(d, dict) and 'class' in d
                )
                
                log.debug(f"Tile {idx} (API image_index): {len(detections)} detections, has_valid_class={has_valid_class} -> maps to element index {tile_idx}", indent=1)
                
                if has_valid_class:
                    tiles_with_valid_class.append(tile_idx)
        
        # Limit to first 2 tiles for ct-001
        if len(tiles_with_valid_class) > 2:
            log.warning(f"Limiting to first 2 tiles (ct-001 click limit: {len(tiles_with_valid_class)} -> 2)", indent=1)
            tiles_to_click = set(tiles_with_valid_class[:2])
        else:
            tiles_to_click = set(tiles_with_valid_class)
        
        log.info(f"ct-001 click limit: Will click {len(tiles_to_click)} tile(s) (max 2)", indent=1)
    else:
        # Default behavior: click tiles with any positive detection
        for entry in batch_results:
            idx = entry.get("image_index")
            # image_index is 1-based from API, convert to 0-based index in valid_tile_elements
            # If sample_offset is 1, then image_index 1 maps to valid_tile_elements[1] (2nd tile)
            tile_idx = int(idx) - 1 + sample_offset
            detections = entry.get("results", [])
            has_positive = _has_positive_detection(detections, confidence_threshold)
            
            print(f"  Tile {idx} (API image_index): {len(detections)} detections, positive={has_positive} -> maps to element index {tile_idx}")
            
            if has_positive:
                tiles_to_click.add(tile_idx)
    
    print(f"\n✓ Tiles to click (1-based): {sorted([i+1 for i in tiles_to_click])}")
    print(f"✓ Tiles to click (0-based indices): {sorted(tiles_to_click)}")
    print(f"{'='*60}\n")
    
    # Perform clicks
    click_count = 0
    for tile_idx in sorted(tiles_to_click):
        # tile_idx is 0-based index into valid_tile_elements
        if tile_idx < 0 or tile_idx >= len(valid_tile_elements):
            print(f"  X Skipping tile {tile_idx + 1}: index {tile_idx} out of bounds (max: {len(valid_tile_elements) - 1})")
            continue
        
        print(f"[CLICK] Tile {tile_idx + 1} (element index {tile_idx})")
        try:
            tile = valid_tile_elements[tile_idx]
            _move_and_click(driver, tile)
            click_count += 1
            time.sleep(0.15)
        except Exception as e:
            print(f"  X Failed to click tile {tile_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
    
    return click_count


def perform_clicks(
    driver: WebDriver,
    mode: str,
    api_result: Dict,
    *,
    canvas_element: Optional[WebElement] = None,
    tile_elements: Optional[List[WebElement]] = None,
    confidence_threshold: float = 0.0, 
    pause_seconds: float = 1.0,
    challenge_type_id: Optional[str] = None,
) -> int:
    """
    Unified entry point used by the crawler to trigger clicks based on API results.
    Note: Detections are already filtered by the model's confidence and IoU thresholds
    during inference. This function clicks on ALL returned detections by default
    (confidence_threshold=0.0), unless challenge_type_id validation rules apply.
    
    Args:
        driver: Active Selenium WebDriver instance.
        mode: Either "canvas" or "tiles" matching the API endpoint that was used.
        api_result: JSON response returned by the inference API.
        canvas_element: Optional specific canvas element to click inside.
        tile_elements: Optional list of tile div WebElements for batch mode.
        confidence_threshold: Minimum confidence needed to click (default: 0.0 = click all returned detections).
        pause_seconds: Delay between successive canvas clicks.
        challenge_type_id: Optional challenge type ID for validation rules (e.g., 'ct-001').
    
    Returns:
        Number of clicks attempted.
    """
    mode = (mode or "").lower()
    print(f"  [perform_clicks] Mode: {mode}, confidence_threshold: {confidence_threshold}, challenge_type_id: {challenge_type_id}")
    
    if mode not in {"canvas", "tiles"}:
        raise ValueError(f"mode must be either 'canvas' or 'tiles', got: {mode}")
    
    if mode == "canvas":
        print(f"  [perform_clicks] Canvas mode selected")
        target_canvas = canvas_element
        if target_canvas is None:
            print(f"  [perform_clicks] No canvas_element provided, searching for canvases...")
            canvases = driver.find_elements(By.TAG_NAME, "canvas")
            target_canvas = canvases[0] if canvases else None
            print(f"  [perform_clicks] Found {len(canvases)} canvas(es)")
        
        if target_canvas is None:
            print("  [perform_clicks] X No canvas element available for clicking")
            return 0
        
        print(f"  [perform_clicks] Using canvas element for clicking")
        return click_canvas_from_response(
            driver,
            target_canvas,
            api_result,
            confidence_threshold=confidence_threshold,
            pause_seconds=pause_seconds,
            challenge_type_id=challenge_type_id,
        )
    
    # mode == "tiles"
    print(f"  [perform_clicks] Tiles mode selected")
    targets = tile_elements
    if targets is None:
        print(f"  [perform_clicks] No tile_elements provided, searching for tiles...")
        targets = driver.find_elements(By.XPATH, "//div[contains(@class, 'image')]")
        print(f"  [perform_clicks] Found {len(targets)} tile(s)")
    
    if not targets:
        print("  [perform_clicks] X No tile elements available for clicking")
        return 0
    
    print(f"  [perform_clicks] Using {len(targets)} tile elements for clicking")
    return click_tiles_from_batch_response(
        driver,
        targets,
        api_result,
        confidence_threshold=confidence_threshold,
        challenge_type_id=challenge_type_id,
    )