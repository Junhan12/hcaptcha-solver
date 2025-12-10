import requests
import os
import re
import base64
import time
from io import BytesIO

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from app.config import API_TIMEOUT
from app.utils.logger import get_logger

os.environ['WDM_LOG_LEVEL'] = '0'  # Silence webdriver_manager logs

# Initialize logger
log = get_logger("crawler")

# Import database function to check challenge_type matching
try:
    import sys
    _this_dir = os.path.dirname(__file__)
    _project_root = os.path.abspath(os.path.join(_this_dir, '..'))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from app.database import _find_challenge_type_for_question
except Exception:
    # Fallback: use API to check
    _find_challenge_type_for_question = None

def get_challenge_type_id_for_question(question):
    """
    Get challenge_type_id for a given question.
    
    Args:
        question: Question text string
    
    Returns:
        challenge_type_id string (e.g., 'ct-001') or None if not found
    """
    if not question or not _find_challenge_type_for_question:
        return None
    try:
        ct_doc = _find_challenge_type_for_question(question)
        if ct_doc:
            return ct_doc.get("challenge_type_id")
    except Exception as e:
        log.error(f"Error getting challenge_type_id: {e}")
    return None

try:
    from client.clicker import perform_clicks, extract_detections  # type: ignore
    log.success("Successfully imported perform_clicks and extract_detections from client.clicker")
except Exception as e:
    try:
        from clicker import perform_clicks, extract_detections  # type: ignore
        log.success("Successfully imported perform_clicks and extract_detections from clicker")
    except Exception as e2:
        perform_clicks = None  # type: ignore
        extract_detections = None  # type: ignore
        log.error(f"Failed to import clicker functions: {e}, {e2}")


def send_challenge_bytes(image_bytes, filename, question):
    """Send challenge image bytes to API."""
    try:
        files = {
            'image': (filename, BytesIO(image_bytes), 'image/png')
        }
        data = {'question': question}
        resp = requests.post(
            "http://localhost:5000/solve_hcaptcha", 
            files=files, 
            data=data, 
            timeout=API_TIMEOUT
        )
        return resp.json()
    except requests.exceptions.ConnectionError as e:
        log.error(f"Connection error: {e}")
        return {'error': 'connection_failed', 'message': str(e)}
    except requests.exceptions.Timeout as e:
        log.error(f"Timeout error: {e}")
        return {'error': 'timeout', 'message': str(e)}
    except Exception as e:
        return {'error': 'invalid_json', 'status_code': resp.status_code, 'text': resp.text}


def fetch_image_bytes(url):
    """Download image from URL."""
    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        if response.status_code == 200:
            return response.content
        log.warning(f"Failed to download image: {url} (status: {response.status_code})")
    except Exception as e:
        log.error(f"Error downloading image from {url}: {e}")
    return None


def send_canvas_images(driver, question):
    """Send canvas images to API for inference and perform clicking."""
    canvases = driver.find_elements(By.TAG_NAME, "canvas")
    if not canvases:
        return 0, [] 
    
    log.info(f"Found {len(canvases)} canvas element(s)")
    
    sent = 0
    accepted = []
    
    for i, canvas in enumerate(canvases, start=1):
        try:
            b64_payload = driver.execute_script(
                "return arguments[0].toDataURL('image/png').substring(22);", canvas)
            image_data = base64.b64decode(b64_payload)
            data_url = f"data:image/png;base64,{b64_payload}"
            filename = f"canvas{i}.png"
            
            # Avoid sending large data URLs; API will read and compress bytes from file payload
            result = send_challenge_bytes(image_data, filename, question)
            
            # Log inference results (crawler responsibility: log detections, counts, performance metrics, challenge ids)
            log.subsection(f"Canvas {i} Inference Results")
            if isinstance(result, dict):
                if 'error' in result:
                    log.error(f"Error: {result.get('error')}", indent=1)
                    if 'message' in result:
                        log.warning(f"Message: {result.get('message')}", indent=1)
                    log.warning("Inference NOT saved (errors are not stored)", indent=1)
                else:
                    # Use clicker's extract_detections to normalize response format
                    detections = []
                    if extract_detections:
                        detections = extract_detections(result)
                    elif 'results' in result:
                        # Fallback if extract_detections not available
                        results = result.get('results', [])
                        if isinstance(results, list):
                            detections = results
                    
                    # Log detection counts
                    if detections:
                        log.success(f"Detections: {len(detections)} object(s) found", indent=1)
                        # Log first 3 detections as sample
                        for idx, det in enumerate(detections[:3], 1):
                            if isinstance(det, dict):
                                log.info(f"{idx}. {det.get('class', 'unknown')} (conf: {det.get('confidence', 0):.2f})", indent=2)
                        if len(detections) > 3:
                            log.info(f"... and {len(detections) - 3} more", indent=2)
                    else:
                        log.warning("No detections found", indent=1)
                    
                    # Log challenge_id if present
                    if 'challenge_id' in result:
                        log.success(f"Inference saved to database (challenge_id: {result.get('challenge_id')})", indent=1)
                    
                    # Log performance metrics
                    if 'perform_time' in result:
                        log.info(f"Performance: {result['perform_time']:.3f}s", indent=1)
                    if 'model' in result and result['model']:
                        model_name = result['model'].get('model_name', 'Unknown')
                        log.info(f"Model: {model_name}", indent=1)
                
                # Call clicker to perform clicks (crawler responsibility: call perform_clicks)
                log.info("Performing automatic clicking on canvas using detections")
                if perform_clicks and isinstance(result, dict):
                    try:
                        challenge_type_id = get_challenge_type_id_for_question(question)
                        if challenge_type_id:
                            log.info(f"Challenge type ID: {challenge_type_id} - applying validation rules", indent=1)
                        
                        click_count = perform_clicks(
                            driver,
                            "canvas",
                            result,
                            canvas_element=canvas,
                            challenge_type_id=challenge_type_id,
                        )
                        if click_count > 0:
                            log.success(f"Automated canvas clicking executed: {click_count} click(s) performed", indent=1)
                        else:
                            log.warning("No clicks performed (0 detections found or no valid detections)", indent=1)
                    except Exception as e:
                        log.error(f"Automated canvas clicking failed: {e}", indent=1)
                        import traceback
                        traceback.print_exc()
                elif not perform_clicks:
                    log.error("perform_clicks function not available (import failed)", indent=1)
            
            # Provide data URL for Streamlit to render directly
            accepted.append({
                "filename": filename,
                "data_url": data_url,
                "result": result,
            })
            sent += 1
        except Exception as e:
            log.error(f"Failed to send canvas {i}: {e}")
    
    return sent, accepted


def send_nested_div_images(driver, question):
    """Extract all nested image divs containing background URLs, collect them, and send as batch to API."""
    divs = driver.find_elements(By.XPATH, "//div[contains(@class, 'image')]")
    if not divs:
        log.warning("No image divs found")
        return 0, []
    
    # Collect all image tiles first
    image_list = []
    for div in divs:
        style = div.get_attribute("style")
        match = re.search(r'url\("(.*?)"\)', style)
        if match:
            url = match.group(1)
            img_bytes = fetch_image_bytes(url)
            if img_bytes is not None:
                tile_idx = len(image_list) + 1
                filename = f"tile{tile_idx}.jpg"
                image_list.append((img_bytes, filename))
                log.debug(f"Collected Tile {tile_idx}: {filename}")
    
    if not image_list:
        log.warning("No valid image tiles collected")
        return 0, []
    
    total_tiles = len(image_list)
    sample_entry = None
    batch_tiles = image_list

    if total_tiles == 10:
        sample_entry = image_list[0]
        batch_tiles = image_list[1:]
        log.info("Detected 10 div images. Treating the first as the reference/sample image under the question")
    elif total_tiles == 9:
        log.info("Detected 9 div images. All are clickable tiles (no sample image)")
    else:
        log.info(f"Detected {total_tiles} div images. Processing all as batch")

    accepted = []

    # If a sample image exists, send it separately for contextual inference
    if sample_entry:
        sample_bytes, sample_filename = sample_entry
        log.info(f"Sending sample image {sample_filename} for inference...")
        sample_result = send_challenge_bytes(sample_bytes, sample_filename, question)
        sample_b64 = base64.b64encode(sample_bytes).decode("utf-8")
        accepted.append({
            "filename": sample_filename,
            "data_url": f"data:image/jpeg;base64,{sample_b64}",
            "result": sample_result,
        })

    result = None
    if batch_tiles:
        log.info(f"Sending {len(batch_tiles)} tile image(s) as batch...")
        def build_files():
            built = []
            for img_bytes, filename in batch_tiles:
                built.append(('images', (filename, BytesIO(img_bytes), 'image/png')))
            return built

        files = build_files()
        data = {'question': question}

        try:
            resp = requests.post(
                "http://localhost:5000/solve_hcaptcha_batch",
                files=files,
                data=data,
                timeout=API_TIMEOUT,
            )
            result = resp.json()
        except requests.exceptions.ConnectionError as e:
            log.error(f"Connection error: {e}")
            result = {'error': 'connection_failed', 'message': str(e)}
        except requests.exceptions.Timeout as e:
            log.error(f"Timeout error: {e}")
            result = {'error': 'timeout', 'message': str(e)}
        except Exception as e:
            log.error(f"Unexpected batch error: {e}")
            result = {'error': 'unexpected', 'message': str(e)}

        # Log inference results (crawler responsibility: log detections, counts, performance metrics, challenge ids)
        log.subsection("Batch Inference Results")
        if isinstance(result, dict):
            if 'error' in result:
                log.error(f"Error: {result.get('error')}", indent=1)
                if 'message' in result:
                    log.warning(f"Message: {result.get('message')}", indent=1)
                log.warning("Inference NOT saved (errors are not stored)", indent=1)
            else:
                # Use clicker's extract_detections to get all detections (flattened from batch)
                all_detections = []
                if extract_detections:
                    all_detections = extract_detections(result)
                elif 'results' in result:
                    # Fallback: manually count detections from batch format
                    all_results = result.get('results', [])
                    if isinstance(all_results, list):
                        for img_result in all_results:
                            img_results = img_result.get('results', [])
                            if isinstance(img_results, list):
                                all_detections.extend(img_results)
                
                # Log detection counts
                if all_detections:
                    log.info(f"Total detections: {len(all_detections)} object(s) across all images", indent=1)
                    # Log first 3 detections as sample
                    for idx, det in enumerate(all_detections[:3], 1):
                        if isinstance(det, dict):
                            log.info(f"{idx}. {det.get('class', 'unknown')} (conf: {det.get('confidence', 0):.2f})", indent=2)
                    if len(all_detections) > 3:
                        log.info(f"... and {len(all_detections) - 3} more", indent=2)
                else:
                    log.warning("No detections found", indent=1)
                
                # Log batch statistics
                if 'results' in result:
                    all_results = result.get('results', [])
                    if isinstance(all_results, list):
                        log.info(f"Images processed: {len(all_results)}", indent=1)
                
                # Log challenge_id if present
                if 'challenge_id' in result:
                    log.success(f"Inference records saved to database (challenge_id: {result.get('challenge_id')})", indent=1)
                
                # Log performance metrics
                if 'perform_time' in result:
                    log.info(f"Performance: {result['perform_time']:.3f}s", indent=1)
                if 'model' in result and result['model']:
                    model_name = result['model'].get('model_name', 'Unknown')
                    log.info(f"Model: {model_name}", indent=1)

        # Call clicker to perform clicks (crawler responsibility: call perform_clicks)
        log.info("Performing automatic clicking on tiles using detections")
        if perform_clicks and isinstance(result, dict):
            try:
                challenge_type_id = get_challenge_type_id_for_question(question)
                if challenge_type_id:
                    log.info(f"Challenge type ID: {challenge_type_id} - applying validation rules", indent=1)
                
                click_count = perform_clicks(
                    driver,
                    "tiles",
                    result,
                    tile_elements=divs,
                    challenge_type_id=challenge_type_id,
                )
                if click_count > 0:
                    log.success(f"Automated tile clicking executed: {click_count} click(s) performed", indent=1)
                else:
                    log.warning("No clicks performed (0 detections found or no valid detections)", indent=1)
            except Exception as e:
                log.error(f"Automated tile clicking failed: {e}", indent=1)
                import traceback
                traceback.print_exc()
        elif not perform_clicks:
            log.error("perform_clicks function not available (import failed)", indent=1)

        for img_bytes, filename in batch_tiles:
            b64_payload = base64.b64encode(img_bytes).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{b64_payload}"
            accepted.append({
                "filename": filename,
                "data_url": data_url,
                "result": result,
            })

    return total_tiles, accepted


def send_single_div_image_if_present(driver, question):
    """
    Detect when exactly one div-based tile image is present (even if a canvas exists)
    and run inference on it to capture semantic context for downstream logic.
    """
    divs = driver.find_elements(By.XPATH, "//div[contains(@class, 'image')]")
    valid_tiles = []

    for idx, div in enumerate(divs, start=1):
        style = div.get_attribute("style") or ""
        match = re.search(r'url\("(.*?)"\)', style)
        if match:
            url = match.group(1)
            img_bytes = fetch_image_bytes(url)
            if img_bytes:
                valid_tiles.append((idx, img_bytes))

    if len(valid_tiles) != 1:
        return 0, []

    tile_idx, img_bytes = valid_tiles[0]
    filename = f"tile{tile_idx}.jpg"
    log.info(f"Single div image detected (Tile {tile_idx}). Sending for inference as {filename}...")
    result = send_challenge_bytes(img_bytes, filename, question)

    data_url = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    accepted = [{
        "filename": filename,
        "data_url": data_url,
        "result": result,
    }]
    return 1, accepted


def check_question_matches_challenge_type(question):
    """Check if question matches any challenge_type in the database."""
    if not question:
        return False
    
    if _find_challenge_type_for_question:
        try:
            ct_doc = _find_challenge_type_for_question(question)
            return ct_doc is not None
        except Exception as e:
            log.error(f"Error checking challenge_type match: {e}")
            return False
    else:
        # Fallback: use API to check (less efficient)
        try:
            # Send a test request to check if question matches
            resp = requests.post(
                "http://localhost:5000/solve_hcaptcha",
                files={"image": ("test.png", BytesIO(b""), "image/png")},
                data={"question": question},
                timeout=API_TIMEOUT
            )
            if resp.ok:
                result = resp.json()
                # Check if there's a message about no match challenge type
                message = result.get("message", "")
                if "no match challenge type" in message.lower():
                    return False
                # If we get here, it likely matched (or API is not available)
                return True
        except Exception:
            pass
        return False


# FIXED: Improved refresh button clicking with better error handling
def click_refresh_button(driver, attempt_number):
    """
    Click the refresh button with multiple fallback strategies.
    Returns True if successful, False otherwise.
    """
    try:
        # Wait a bit for the page to stabilize
        time.sleep(1)
        
        # Try different possible xpaths for refresh button
        refresh_xpaths = [
            "//div[@class='refresh button']",  # Exact match as specified by user
            "//div[contains(@class, 'refresh') and contains(@class, 'button')]",  # Fallback for multiple classes
            "//div[normalize-space(@class)='refresh button']",  # Normalized whitespace
        ]
        
        for xpath in refresh_xpaths:
            try:
                # Wait for element to be present and clickable
                refresh_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                
                # Scroll element into view
                driver.execute_script("arguments[0].scrollIntoView(true);", refresh_button)
                time.sleep(0.5)
                
                # Try regular click first
                try:
                    if refresh_button.is_displayed() and refresh_button.is_enabled():
                        refresh_button.click()
                        log.success(f"Clicked refresh button (attempt {attempt_number}) using xpath: {xpath}", indent=1)
                        return True
                except Exception:
                    # If regular click fails, try JavaScript click
                    try:
                        driver.execute_script("arguments[0].click();", refresh_button)
                        log.success(f"Clicked refresh button using JavaScript (attempt {attempt_number})", indent=1)
                        return True
                    except Exception:
                        continue
            except (TimeoutException, NoSuchElementException):
                # Element not found with this xpath, try next
                continue
        
        # Try alternative method: find any element with refresh in class
        log.warning("Could not find refresh button with standard xpaths. Trying alternative methods...", indent=1)
        try:
            # Look for any element with refresh in class
            refresh_elements = driver.find_elements(By.XPATH, "//*[contains(@class, 'refresh')]")
            log.info(f"Found {len(refresh_elements)} element(s) with 'refresh' in class", indent=1)
            
            for elem in refresh_elements:
                try:
                    if elem.is_displayed():
                        # Scroll into view
                        driver.execute_script("arguments[0].scrollIntoView(true);", elem)
                        time.sleep(0.5)
                        # Try JavaScript click
                        driver.execute_script("arguments[0].click();", elem)
                        log.success("Clicked refresh element using alternative method", indent=1)
                        return True
                except Exception:
                    continue
        except Exception:
            pass
        
        log.error("Could not find or click refresh button after all attempts", indent=1)
        return False
        
    except Exception as e:
        log.error(f"Error in click_refresh_button: {e}", indent=1)
        import traceback
        traceback.print_exc()
        return False


def retrieve_question(driver, timeout=10):
    """
    Retrieve the current challenge question from the page.
    
    Args:
        driver: Selenium WebDriver instance (must be in challenge iframe)
        timeout: Maximum seconds to wait for question element (default: 10)
    
    Returns:
        str: Question text, or None if not found
    """
    try:
        # Wait for question element
        prompt_element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, "//h2[@dir='ltr']"))
        )
        time.sleep(1)  # Wait a bit for text to load
        
        try:
            span_element = prompt_element.find_element(By.TAG_NAME, "span")
            prompt_text = span_element.text.strip()
        except:
            prompt_text = prompt_element.text.strip()
        
        return prompt_text
    except Exception as e:
        log.error(f"Error retrieving question: {e}")
        return None


def check_submit_button_exists(driver, timeout=2):
    """
    Check if submit button exists without clicking.
    
    Args:
        driver: Selenium WebDriver instance
        timeout: Maximum seconds to wait for button (default: 2)
    
    Returns:
        bool: True if button exists and is visible, False otherwise
    """
    try:
        submit_button = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, "//div[@class='button-submit button']"))
        )
        return submit_button.is_displayed()
    except (TimeoutException, NoSuchElementException):
        return False


def click_submit_button(driver, wait_after_click=3):
    """
    Click the submit button to proceed to next round or verify challenge.
    
    Args:
        driver: Selenium WebDriver instance
        wait_after_click: Seconds to wait after clicking (default: 3)
    
    Returns:
        bool: True if button was clicked successfully, False otherwise
    """
    try:
        log.info("Clicking submit button...")
        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@class='button-submit button']"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", submit_button)
        time.sleep(0.5)
        submit_button.click()
        log.success(f"Submit button clicked. Waiting {wait_after_click} seconds...")
        time.sleep(wait_after_click)
        return True
    except Exception as e:
        log.error(f"Error clicking submit button: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_crawl_once(max_rounds=20):
    """
    Run single crawl of hCaptcha challenge.
    
    Args:
        max_rounds: Maximum number of rounds to process (default: 20)
    
    Returns:
        dict: Summary of crawl results
    """
    log.section("Starting Single Crawl")
    
    driver = None
    matched = False  # Track if challenge matched and inference started
    summary = {
        "question": None, 
        "sent_canvas": 0, 
        "sent_divs": 0, 
        "total_sent": 0, 
        "refresh_count": 0, 
        "crumb_count": 0
    }
    
    try:
        options = Options()
        options.add_argument("--start-maximized")
        #options.add_argument("--headless=new")
        # Show the Chrome window so you can watch the automation
        options.add_experimental_option("detach", True)
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        # Ensure the window is visible and positioned on-screen
        try:
            driver.set_window_position(0, 0)
            driver.set_window_size(1400, 900)
        except Exception:
            pass
        
        driver.get("https://accounts.hcaptcha.com/demo")
        
        # Step 1: Click checkbox
        iframe = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//iframe[@title='Widget containing checkbox for hCaptcha security challenge']"))
        )
        driver.switch_to.frame(iframe)
        checkbox = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "checkbox")))
        checkbox.click()
        driver.switch_to.default_content()
        log.success("Checkbox clicked")
        
        # Step 2: Switch to challenge iframe
        challenge_iframe = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//iframe[contains(@title, 'hCaptcha challenge')]"))
        )
        driver.switch_to.frame(challenge_iframe)
        log.success("Challenge iframe detected")
        
        # Step 3: Initialize counters and tracking
        all_accepted = []  # Store all accepted results from all rounds
        all_questions = []  # Store questions from all rounds
        summary["sent_canvas"] = 0
        summary["sent_divs"] = 0
        
        # Main processing loop: continue until submit button is not found
        # This unified approach works for both single-crumb and multi-crumb challenges
        # Question is retrieved for each round
        round_number = 0
        # max_rounds is now a parameter (default: 20) - safety limit to prevent infinite loops
        
        while round_number < max_rounds:
            round_number += 1
            log.subsection(f"Processing Round {round_number}")
            
            # Retrieve question for this round
            prompt_text = retrieve_question(driver, timeout=10)
            if not prompt_text:
                log.warning("Could not retrieve question for this round. Stopping...")
                break
            
            log.info(f"Challenge Question (Round {round_number}): {prompt_text}")
            
            # Validate question matches a challenge_type
            matched = check_question_matches_challenge_type(prompt_text)
            if not matched:
                log.warning(f"Question does not match any challenge_type. Refreshing challenge...")
                # Try to refresh the challenge
                if click_refresh_button(driver, attempt_number=round_number):
                    log.info("Challenge refreshed. Waiting for new challenge to load...", indent=1)
                    time.sleep(3)  # Wait for new challenge to load
                    continue  # Skip this round, retrieve new question in next iteration
                else:
                    log.error("Could not click refresh button. Skipping this round...", indent=1)
                    continue  # Skip this round if refresh fails
            
            log.success("Question matches a challenge_type. Proceeding with inference...")
            all_questions.append(prompt_text)
            
            # Get challenge_type_id for validation (for this round's question)
            challenge_type_id = get_challenge_type_id_for_question(prompt_text)
            
            # Small delay to allow page to stabilize
            time.sleep(2)
            
            # Perform inference + clicking (using current question)
            canvases = driver.find_elements(By.TAG_NAME, "canvas")
            if canvases:
                log.info(f"Found {len(canvases)} canvas element(s), sending...")
                sent, accepted = send_canvas_images(driver, prompt_text)
                log.success(f"Sent {sent} canvas image(s) in round {round_number}")
                summary["sent_canvas"] += sent
                all_accepted.extend(accepted)

                # Even when canvas exists, check for a single div image that may carry semantic cues
                div_sent, div_accepted = send_single_div_image_if_present(driver, prompt_text)
                if div_sent:
                    log.info(f"Captured {div_sent} div image alongside canvas in round {round_number}")
                    summary["sent_divs"] += div_sent
                    all_accepted.extend(div_accepted)
            else:
                log.info("No canvas found, scanning nested image divs...")
                sent, accepted = send_nested_div_images(driver, prompt_text)
                log.success(f"Sent {sent} nested image(s) as batch in round {round_number}")
                summary["sent_divs"] += sent
                all_accepted.extend(accepted)
            
            # Check if submit button still exists
            if check_submit_button_exists(driver, timeout=2):
                log.info("Submit button found. Clicking to continue...")
                if click_submit_button(driver, wait_after_click=3):
                    log.success("Submit button clicked. Waiting for next round...")
                    time.sleep(2)  # Additional wait for page to stabilize
                    # Continue loop - will retrieve new question in next iteration
                    continue
                else:
                    log.error("Failed to click submit button. Stopping...")
                    break
            else:
                log.success("Submit button not found. Challenge complete!")
                break
        
        if round_number >= max_rounds:
            log.warning(f"Reached maximum rounds limit ({max_rounds}). Stopping to prevent infinite loop.")
        
        # Store questions in summary (use first question as primary, or all questions)
        if all_questions:
            summary["question"] = all_questions[0]  # Primary question (first round)
            summary["all_questions"] = all_questions  # All questions from all rounds
        else:
            summary["question"] = None
            summary["all_questions"] = []
        
        # Check for crumb elements (for summary purposes)
        crumb_elements = driver.find_elements(By.XPATH, "//div[@class='Crumb']")
        crumb_count = len(crumb_elements)
        summary["crumb_count"] = crumb_count
        
        # Calculate total and store results
        summary["total_sent"] = summary["sent_canvas"] + summary["sent_divs"]
        summary["accepted"] = all_accepted
        
        # Summary of inference saving
        log.section("Crawl Summary")
        log.info(f"Question: {summary['question']}", indent=1)
        log.info(f"Crumb count: {summary['crumb_count']}", indent=1)
        log.info(f"Images sent: {summary['total_sent']}", indent=1)
        
        if summary.get('accepted'):
            successful_inferences = 0
            for item in summary['accepted']:
                result = item.get('result', {})
                if isinstance(result, dict):
                    # Check if inference was successful (not an error)
                    if 'error' not in result:
                        results = result.get('results', [])
                        # If results is a list or dict without 'error', it's successful
                        if isinstance(results, list) or (isinstance(results, dict) and 'error' not in results):
                            successful_inferences += 1
                    # For batch results, check each image result
                    elif 'results' in result:
                        all_results = result.get('results', [])
                        if isinstance(all_results, list):
                            for img_result in all_results:
                                img_results = img_result.get('results', [])
                                if isinstance(img_results, list) or (isinstance(img_results, dict) and 'error' not in img_results):
                                    successful_inferences += 1
            log.success(f"Successful inferences saved: {successful_inferences}", indent=1)
        
    except Exception as e:
        log.error(f"Something went wrong: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if driver is not None:
            # Wait 5 seconds before closing the browser to allow final verification to complete
            # This wait only happens after the submit button is no longer found (challenge complete)
            log.info("Waiting 5 seconds before closing browser (allowing final verification)...")
            time.sleep(5)
            log.info("Closing browser...")
            driver.quit()
        
        if matched:
            log.success("hCaptcha Solver successfully bypassed the challenge (inference finished)")
        else:
            log.error("hCaptcha Solver failed to bypass the challenge - no matching challenge_type found after max attempts")
    
    return summary


if __name__ == "__main__":
    result = run_crawl_once()
    log.info(f"Summary: {result}")