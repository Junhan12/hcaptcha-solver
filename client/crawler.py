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

os.environ['WDM_LOG_LEVEL'] = '0'  # Silence webdriver_manager logs

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

try:
    from client.clicker import perform_clicks  # type: ignore
    print("/ Successfully imported perform_clicks from client.clicker")
except Exception as e:
    try:
        from clicker import perform_clicks  # type: ignore
        print("/ Successfully imported perform_clicks from clicker")
    except Exception as e2:
        perform_clicks = None  # type: ignore
        print(f"X Failed to import perform_clicks: {e}, {e2}")


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
        print(f" X  Connection error: {e}")
        return {'error': 'connection_failed', 'message': str(e)}
    except requests.exceptions.Timeout as e:
        print(f" X  Timeout error: {e}")
        return {'error': 'timeout', 'message': str(e)}
    except Exception as e:
        return {'error': 'invalid_json', 'status_code': resp.status_code, 'text': resp.text}


def fetch_image_bytes(url):
    """Download image from URL."""
    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        if response.status_code == 200:
            return response.content
        print(f"Failed to download: {url}")
    except Exception as e:
        print(f"Error downloading image: {e}")
    return None


def send_canvas_images(driver, question):
    """Send canvas images to API for inference and perform clicking."""
    canvases = driver.find_elements(By.TAG_NAME, "canvas")
    if not canvases:
        return 0, [] 
    
    print(f"Found {len(canvases)} canvas elements.")
    
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
            
            # Display inference results
            print(f"\n--- Canvas {i} Inference Results ---")
            if isinstance(result, dict):
                if 'error' in result:
                    print(f"  X Error: {result['error']}")
                    print(f"  X Inference NOT saved (errors are not stored)")
                elif 'results' in result:
                    results = result.get('results', [])
                    if isinstance(results, list) and len(results) > 0:
                        print(f"  / Detections: {len(results)} objects found")
                        for idx, det in enumerate(results[:3], 1):  # Show first 3
                            if isinstance(det, dict):
                                print(f"    {idx}. {det.get('class', 'unknown')} (conf: {det.get('confidence', 0):.2f})")
                        if len(results) > 3:
                            print(f"    ... and {len(results) - 3} more")
                        print(f"  / Inference saved to database (challenge_id: {result.get('challenge_id', 'N/A')})")
                    elif isinstance(results, dict) and 'message' in results:
                        print(f"  X {results.get('message', 'No detections')}")
                        print(f"  / Inference saved to database (challenge_id: {result.get('challenge_id', 'N/A')})")
                    else:
                        print(f"  X No detections found")
                        print(f"  / Inference saved to database (challenge_id: {result.get('challenge_id', 'N/A')})")
                else:
                    print(f"  Response: {result}")
                
                # Display performance metrics
                if 'perform_time' in result:
                    print(f"  Performance: {result['perform_time']:.3f}s")
                if 'model' in result and result['model']:
                    model_name = result['model'].get('model_name', 'Unknown')
                    print(f"  Model: {model_name}")
                
                print("Performing automatic clicking on canvas using detections")
                
                # Attempt automatic clicking on canvas using detections
                if perform_clicks and isinstance(result, dict):
                    print("Inside perform_clicks function")
                    try:
                        click_count = perform_clicks(
                            driver,
                            "canvas",
                            result,
                            canvas_element=canvas,
                        )
                        if click_count > 0:
                            print(f"  / Automated canvas clicking executed: {click_count} clicks performed.")
                        else:
                            print(f"  X No clicks performed (0 detections found or no valid detections).")
                    except Exception as e:
                        print(f"  X Automated canvas clicking failed: {e}")
                        import traceback
                        traceback.print_exc()
                elif not perform_clicks:
                    print(" X  perform_clicks function not available (import failed).")
            
            # Provide data URL for Streamlit to render directly
            accepted.append({
                "filename": filename,
                "data_url": data_url,
                "result": result,
            })
            sent += 1
        except Exception as e:
            print(f"Failed to send canvas {i}: {e}")
    
    return sent, accepted


def send_nested_div_images(driver, question):
    """Extract all nested image divs containing background URLs, collect them, and send as batch to API."""
    divs = driver.find_elements(By.XPATH, "//div[contains(@class, 'image')]")
    if not divs:
        print("No image divs found.")
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
                print(f"Collected Tile {tile_idx}: {filename}")
    
    if not image_list:
        print("No valid image tiles collected.")
        return 0, []
    
    total_tiles = len(image_list)
    sample_entry = None
    batch_tiles = image_list

    if total_tiles == 10:
        sample_entry = image_list[0]
        batch_tiles = image_list[1:]
        print("Detected 10 div images. Treating the first as the reference/sample image under the question.")
    elif total_tiles == 9:
        print("Detected 9 div images. All are clickable tiles (no sample image).")
    else:
        print(f"Detected {total_tiles} div images. Processing all as batch.")

    accepted = []

    # If a sample image exists, send it separately for contextual inference
    if sample_entry:
        sample_bytes, sample_filename = sample_entry
        print(f"\nSending sample image {sample_filename} for inference...")
        sample_result = send_challenge_bytes(sample_bytes, sample_filename, question)
        sample_b64 = base64.b64encode(sample_bytes).decode("utf-8")
        accepted.append({
            "filename": sample_filename,
            "data_url": f"data:image/jpeg;base64,{sample_b64}",
            "result": sample_result,
        })

    result = None
    if batch_tiles:
        print(f"\nSending {len(batch_tiles)} tile image(s) as batch...")
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
            print(f" X  Connection error: {e}")
            result = {'error': 'connection_failed', 'message': str(e)}
        except requests.exceptions.Timeout as e:
            print(f" X  Timeout error: {e}")
            result = {'error': 'timeout', 'message': str(e)}
        except Exception as e:
            print(f"  Unexpected batch error: {e}")
            result = {'error': 'unexpected', 'message': str(e)}

        print(f"\n--- Batch Inference Results ---")
        if isinstance(result, dict):
            if 'error' in result:
                print(f"  X Error: {result['error']}")
                print(f"  X Inference NOT saved (errors are not stored)")
            elif 'results' in result:
                all_results = result.get('results', [])
                if isinstance(all_results, list):
                    total_detections = 0
                    successful_images = 0
                    error_images = 0
                    
                    for img_result in all_results:
                        img_results = img_result.get('results', [])
                        if isinstance(img_results, dict) and 'error' in img_results:
                            error_images += 1
                        elif isinstance(img_results, list):
                            successful_images += 1
                            total_detections += len(img_results)
                    
                    print(f"  Images processed: {len(all_results)}")
                    print(f"  / Successful: {successful_images} images")
                    if error_images > 0:
                        print(f"  X Errors: {error_images} images (not saved)")
                    print(f"  Total detections: {total_detections} objects")
                    
                    for img_result in all_results[:1]:
                        img_results = img_result.get('results', [])
                        if isinstance(img_results, list) and len(img_results) > 0:
                            print(f"  Sample detections from image {img_result.get('image_index', 1)}:")
                            for idx, det in enumerate(img_results[:3], 1):
                                if isinstance(det, dict):
                                    print(f"    {idx}. {det.get('class', 'unknown')} (conf: {det.get('confidence', 0):.2f})")
                    print(f"  / Inference records saved to database (challenge_id: {result.get('challenge_id', 'N/A')})")
                else:
                    print(f"  Response: {result}")
            else:
                print(f"  Response: {result}")

            if 'perform_time' in result:
                print(f"  Performance: {result['perform_time']:.3f}s")
            if 'model' in result and result['model']:
                model_name = result['model'].get('model_name', 'Unknown')
                print(f"  Model: {model_name}")

        if perform_clicks and isinstance(result, dict):
            try:
                click_count = perform_clicks(
                    driver,
                    "tiles",
                    result,
                    tile_elements=divs,
                )
                if click_count > 0:
                    print(f"  / Automated tile clicking executed: {click_count} clicks performed.")
                else:
                    print(f"  X No clicks performed (0 detections found or no valid detections).")
            except Exception as e:
                print(f"  X Automated tile clicking failed: {e}")
                import traceback
                traceback.print_exc()
        elif not perform_clicks:
            print(" X  perform_clicks function not available (import failed).")

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
    print(f"\nSingle div image detected (Tile {tile_idx}). Sending for inference as {filename}...")
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
            print(f"Error checking challenge_type match: {e}")
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
                        print(f"  / Clicked refresh button (attempt {attempt_number}) using xpath: {xpath}")
                        return True
                except Exception:
                    # If regular click fails, try JavaScript click
                    try:
                        driver.execute_script("arguments[0].click();", refresh_button)
                        print(f"  / Clicked refresh button using JavaScript (attempt {attempt_number})")
                        return True
                    except Exception:
                        continue
            except (TimeoutException, NoSuchElementException):
                # Element not found with this xpath, try next
                continue
        
        # Try alternative method: find any element with refresh in class
        print(f" X  Warning: Could not find refresh button with standard xpaths. Trying alternative methods...")
        try:
            # Look for any element with refresh in class
            refresh_elements = driver.find_elements(By.XPATH, "//*[contains(@class, 'refresh')]")
            print(f"  Found {len(refresh_elements)} elements with 'refresh' in class")
            
            for elem in refresh_elements:
                try:
                    if elem.is_displayed():
                        # Scroll into view
                        driver.execute_script("arguments[0].scrollIntoView(true);", elem)
                        time.sleep(0.5)
                        # Try JavaScript click
                        driver.execute_script("arguments[0].click();", elem)
                        print(f"  / Clicked refresh element using alternative method")
                        return True
                except Exception:
                    continue
        except Exception:
            pass
        
        print(f"  X Error: Could not find or click refresh button after all attempts.")
        return False
        
    except Exception as e:
        print(f"  X Error in click_refresh_button: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_crawl_once():
    """Run single crawl of hCaptcha challenge."""
    print(f"\n============================")
    print(f"Starting single crawl ...")
    print(f"============================")
    
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
        print("Checkbox clicked")
        
        # Step 2: Switch to challenge iframe
        challenge_iframe = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//iframe[contains(@title, 'hCaptcha challenge')]"))
        )
        driver.switch_to.frame(challenge_iframe)
        print("Challenge iframe detected")
        
        # Step 3: Retrieve Question and refresh until it matches a challenge_type
        max_refresh_attempts = 20  # Maximum number of refresh attempts
        refresh_count = 0
        prompt_text = None
        matched = False
        
        while refresh_count < max_refresh_attempts:
            try:
                # Wait for question element
                prompt_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//h2[@dir='ltr']"))
                )
                time.sleep(1)  # Wait a bit for text to load
                
                try:
                    span_element = prompt_element.find_element(By.TAG_NAME, "span")
                    prompt_text = span_element.text.strip()
                except:
                    prompt_text = prompt_element.text.strip()
                
                print(f"Challenge Question: {prompt_text}")
                
                # Check if question matches any challenge_type
                matched = check_question_matches_challenge_type(prompt_text)
                
                if matched:
                    print(f"/ Question matches a challenge_type. Proceeding with inference...")
                    break
                else:
                    print(f"X Question does not match any challenge_type. Refreshing...")
                    refresh_count += 1
                    summary["refresh_count"] = refresh_count
                    
                    # Use improved refresh button clicking function
                    if not click_refresh_button(driver, refresh_count):
                        print(f"  Could not click refresh button. Stopping refresh attempts.")
                        break
                    
                    # Wait for new challenge to load
                    print(f"  Waiting for new challenge to load...")
                    time.sleep(3)  # Increased wait time for challenge to reload
                    
            except Exception as e:
                print(f"Error retrieving question: {e}")
                break
        
        if not matched:
            print(f"\nX  Warning: Could not find a matching challenge_type after {refresh_count} refresh attempts.")
            print(f"  Last question: {prompt_text}")
            print(f"  Skipping inference for this challenge...")
            summary["question"] = prompt_text
            summary["total_sent"] = 0
            return summary
        
        if not prompt_text:
            print(f"\nX  Error: Could not retrieve question text.")
            summary["total_sent"] = 0
            return summary
        
        summary["question"] = prompt_text
        
        # Step 4: Check for crumb elements to determine if multi-crumb challenge
        time.sleep(2)
        crumb_elements = driver.find_elements(By.XPATH, "//div[@class='Crumb']")
        crumb_count = len(crumb_elements)
        summary["crumb_count"] = crumb_count
        print(f"\nFound {crumb_count} crumb element(s)")
        
        all_accepted = []  # Store all accepted results from all crumbs
        
        if crumb_count > 1:
            # Multi-crumb challenge: process each crumb separately
            print(f"Multi-crumb challenge detected ({crumb_count} crumbs). Processing each separately...")
            
            # Initialize counters for multi-crumb
            summary["sent_canvas"] = 0
            summary["sent_divs"] = 0
            
            for crumb_idx in range(1, crumb_count + 1):
                print(f"\n{'='*50}")
                print(f"Processing Crumb {crumb_idx} of {crumb_count}")
                print(f"{'='*50}")
                
                # Process current crumb
                time.sleep(2)
                
                canvases = driver.find_elements(By.TAG_NAME, "canvas")
                if canvases:
                    print(f"Found {len(canvases)} canvas elements, sending...")
                    sent, accepted = send_canvas_images(driver, prompt_text)
                    print(f"Sent {sent} canvas images for crumb {crumb_idx}.")
                    summary["sent_canvas"] += sent
                    all_accepted.extend(accepted)

                    # Even when canvas exists, check for a single div image that may carry semantic cues.
                    div_sent, div_accepted = send_single_div_image_if_present(driver, prompt_text)
                    if div_sent:
                        print(f"Captured {div_sent} div image alongside canvas for crumb {crumb_idx}.")
                        summary["sent_divs"] += div_sent
                        all_accepted.extend(div_accepted)
                else:
                    print("No canvas found â€” scanning nested image divs...")
                    sent, accepted = send_nested_div_images(driver, prompt_text)
                    print(f"Sent {sent} nested image(s) as batch for crumb {crumb_idx}.")
                    summary["sent_divs"] += sent
                    all_accepted.extend(accepted)
                
                # After processing first crumb, click submit button to proceed to next crumb
                if crumb_idx < crumb_count:
                    print(f"\nClicking submit button to proceed to crumb {crumb_idx + 1}...")
                    try:
                        submit_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, "//div[@class='button-submit button']"))
                        )
                        driver.execute_script("arguments[0].scrollIntoView(true);", submit_button)
                        time.sleep(0.5)
                        submit_button.click()
                        print(f"/ Submit button clicked. Waiting for next crumb to load...")
                        time.sleep(3)  # Wait for next crumb to load
                    except Exception as e:
                        print(f"X Error clicking submit button: {e}")
                        break
        else:
            # Single crumb challenge: process normally
            print(f"Single-crumb challenge. Proceeding with inference...")
            
            canvases = driver.find_elements(By.TAG_NAME, "canvas")
            if canvases:
                print(f"Found {len(canvases)} canvas elements, sending...")
                sent, accepted = send_canvas_images(driver, prompt_text)
                print(f"Sent {sent} canvas images.")
                summary["sent_canvas"] = sent
                all_accepted = accepted

                div_sent, div_accepted = send_single_div_image_if_present(driver, prompt_text)
                if div_sent:
                    print(f"Captured {div_sent} div image alongside canvas.")
                    summary["sent_divs"] += div_sent
                    all_accepted.extend(div_accepted)
            else:
                print("No canvas found â€” scanning nested image divs...")
                sent, accepted = send_nested_div_images(driver, prompt_text)
                print(f"Sent {sent} nested image(s) as batch.")
                summary["sent_divs"] = sent
                all_accepted = accepted
        
        summary["total_sent"] = summary["sent_canvas"] + summary["sent_divs"]
        summary["accepted"] = all_accepted
        
        # Summary of inference saving
        print(f"\n============================")
        print(f"Crawl Summary:")
        print(f"  Question: {summary['question']}")
        print(f"  Crumb count: {summary['crumb_count']}")
        print(f"  Images sent: {summary['total_sent']}")
        
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
            print(f"  / Successful inferences saved: {successful_inferences}")
        print(f"============================\n")
        
    except Exception as e:
        print("Something went wrong:", e)
        import traceback
        traceback.print_exc()
    finally:
        if driver is not None:
            # Close browser in all cases
            driver.quit()
        
        if matched:
            print(f"hCaptcha Solver successfully bypass the challenge (inference finished).\n")
        else:
            print(f"hCaptcha Solver failed to bypass the challenge due to there is no matching challenge_type found after max attempts.\n")
    
    return summary


if __name__ == "__main__":
    result = run_crawl_once()
    print("Summary:", result)