"""
Utilities that translate inference results (bounding boxes) into real clicks
inside the live hCaptcha challenge using Selenium.
"""

from __future__ import annotations

import time
import re
from typing import Dict, Iterable, List, Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import StaleElementReferenceException


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
):
    """
    Use detections returned from /solve_hcaptcha to click positions on the canvas.
    Note: Detections are already filtered by the model's confidence and IoU thresholds
    during inference. This function clicks on ALL returned detections by default
    (confidence_threshold=0.0).
    
    Returns number of successful click attempts.
    """
    detections = _extract_detections(api_result)
    print(f"  [click_canvas] Extracted {len(detections)} detections from API result")
    
    if not detections:
        print(f"  [click_canvas] No detections found in API result. Result keys: {list(api_result.keys()) if isinstance(api_result, dict) else 'not a dict'}")
        return 0
    
    click_count = 0
    valid_detections = 0
    
    for det in detections:
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
            print("  [click_canvas] Canvas dimensions invalid (0)")
            break
        
        scale_x = rect["width"] / intrinsic_width
        scale_y = rect["height"] / intrinsic_height
        
        bbox = det.get("bbox", [])
        if len(bbox) < 4:
            print(f"  [click_canvas] Skipping detection with invalid bbox: {bbox}")
            continue
        
        confidence = det.get("confidence", 0.0)
        
        # Click on all detections since they're already filtered by model thresholds
        if confidence < confidence_threshold:
            print(f"  [click_canvas] Skipping detection with confidence {confidence:.2f} < threshold {confidence_threshold:.2f}")
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
        
        print(
            f"  â†’ Canvas click @ ({client_x:.1f}, {client_y:.1f}) "
            f"(class={det.get('class')}, conf={confidence:.2f}, bbox={bbox})"
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
            print(f"  / Click {click_count}/{valid_detections} successful")
            time.sleep(max(pause_seconds, 0))
        except Exception as exc:
            print(f"  X Canvas click failed: {exc}")
            import traceback
            traceback.print_exc()
    
    print(f"  [click_canvas] Total: {len(detections)} detections, {valid_detections} valid, {click_count} clicks performed")
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


def click_tiles_from_batch_response(driver, tile_elements, api_result, confidence_threshold=0.0):
    print(f"\n{'='*60}")
    print(f"TILE CLICKING DEBUG")
    print(f"{'='*60}")
    print(f"Total tile elements available: {len(tile_elements)}")
    
    batch_results = api_result.get("results", [])
    print(f"Total batch results from API: {len(batch_results)}")
    
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
) -> int:
    """
    Unified entry point used by the crawler to trigger clicks based on API results.
    Note: Detections are already filtered by the model's confidence and IoU thresholds
    during inference. This function clicks on ALL returned detections by default
    (confidence_threshold=0.0).
    
    Args:
        driver: Active Selenium WebDriver instance.
        mode: Either "canvas" or "tiles" matching the API endpoint that was used.
        api_result: JSON response returned by the inference API.
        canvas_element: Optional specific canvas element to click inside.
        tile_elements: Optional list of tile div WebElements for batch mode.
        confidence_threshold: Minimum confidence needed to click (default: 0.0 = click all returned detections).
        pause_seconds: Delay between successive canvas clicks.
    
    Returns:
        Number of clicks attempted.
    """
    mode = (mode or "").lower()
    print(f"  [perform_clicks] Mode: {mode}, confidence_threshold: {confidence_threshold}")
    
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
    )