# Question Retrieval Logic - Design Document

## Current Logic

### Initial Question Retrieval:
1. Retrieve question once at the beginning (before main loop)
2. Refresh until question matches a challenge_type
3. Store question in `prompt_text`
4. Use same `prompt_text` for all rounds in the main loop

### Main Processing Loop:
```
WHILE submit button exists:
  1. Use stored prompt_text (from initial retrieval)
  2. Perform inference + clicking
  3. Click submit button
  4. Continue loop (but question is NOT retrieved again)
```

**Problem**: Question might change for each round/crumb, but we're using the same question for all rounds.

---

## New Desired Logic

### Concept:
- **Retrieve question for each round**: Question should be retrieved at the start of each round
- **Question may change**: Each round/crumb might have a different question
- **Unified approach**: Same logic for both single-crumb and multi-crumb challenges

### Logic Flow:

```
1. Initial Setup:
   - Click checkbox
   - Switch to challenge iframe
   - (Optional: Initial question validation - can be removed or kept for first round)

2. Main Processing Loop:
   WHILE submit button exists:
      a. Retrieve current question for this round
      b. Get challenge_type_id for this question
      c. Perform inference + clicking (using current question)
      d. Check if submit button still exists
      e. IF submit button EXISTS:
         - Click submit button
         - Wait for page to update
         - Continue loop (go back to step 2a - retrieve new question)
      f. IF submit button NOT FOUND:
         - Break loop (challenge complete)

3. Wait 5 seconds
4. Close web driver
```

### Process Details:

#### Step 1: Initial Setup
- Click checkbox
- Switch to challenge iframe
- (No initial question validation needed - will be done in loop)

#### Step 2: Main Processing Loop
```
WHILE submit button exists:
  1. Retrieve question for current round:
     - Wait for question element (//h2[@dir='ltr'])
     - Extract question text
     - Log question
     - (Optional: Validate question matches challenge_type - but don't refresh)
  
  2. Get challenge_type_id for current question:
     - Use get_challenge_type_id_for_question(question)
     - This will be used for validation rules in clicking
  
  3. Perform inference + clicking:
     - Detect challenge type (canvas vs div tiles)
     - Extract images
     - Send to API with current question
     - Perform clicking based on results (using challenge_type_id)
  
  4. Check if submit button exists:
     - Use check_submit_button_exists()
  
  5. IF submit button found:
     - Click submit button
     - Wait for page to update (3 seconds)
     - Wait additional time for page to stabilize (2 seconds)
     - Continue loop (will retrieve new question in next iteration)
  
  6. IF submit button NOT found:
     - Challenge complete
     - Break loop
```

---

## Implementation Details

### Helper Function: `retrieve_question(driver, timeout=10)`
```python
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
```

### Modified Main Loop Structure:
```python
# Main processing loop: continue until submit button disappears
round_number = 0
max_rounds = 20

while round_number < max_rounds:
    round_number += 1
    log.subsection(f"Processing Round {round_number}")
    
    # Retrieve question for this round
    prompt_text = retrieve_question(driver, timeout=10)
    if not prompt_text:
        log.warning("Could not retrieve question for this round. Stopping...")
        break
    
    log.info(f"Challenge Question (Round {round_number}): {prompt_text}")
    
    # Get challenge_type_id for validation
    challenge_type_id = get_challenge_type_id_for_question(prompt_text)
    
    # Small delay to allow page to stabilize
    time.sleep(2)
    
    # Perform inference + clicking (using current question)
    canvases = driver.find_elements(By.TAG_NAME, "canvas")
    if canvases:
        sent, accepted = send_canvas_images(driver, prompt_text, challenge_type_id=challenge_type_id)
        # ... rest of logic
    else:
        sent, accepted = send_nested_div_images(driver, prompt_text, challenge_type_id=challenge_type_id)
        # ... rest of logic
    
    # Check if submit button still exists
    if check_submit_button_exists(driver, timeout=2):
        log.info("Submit button found. Clicking to continue...")
        if click_submit_button(driver, wait_after_click=3):
            log.success("Submit button clicked. Waiting for next round...")
            time.sleep(2)  # Additional wait for page to stabilize
            continue  # Continue loop - will retrieve new question in next iteration
        else:
            log.error("Failed to click submit button. Stopping...")
            break
    else:
        log.success("Submit button not found. Challenge complete!")
        break
```

---

## Key Changes Required

1. **Create `retrieve_question()` helper function**:
   - Extract question retrieval logic from initial setup
   - Make it reusable for each round

2. **Remove initial question validation loop**:
   - No need to refresh until question matches before starting
   - Question validation happens naturally in the loop

3. **Move question retrieval into main loop**:
   - Retrieve question at the start of each round
   - Use current question for that round's inference

4. **Update function signatures**:
   - `send_canvas_images()` and `send_nested_div_images()` already accept `question` parameter
   - Need to ensure they also accept `challenge_type_id` parameter (or get it internally)

---

## Benefits

1. **Handles changing questions**: Each round uses its own question
2. **Works for multi-crumb**: Each crumb can have a different question
3. **More accurate inference**: Uses correct question for each round
4. **Simpler logic**: No need for initial question validation loop

---

## Edge Cases to Handle

1. **Question element not found**: Log warning and break loop
2. **Question changes during round**: Use question retrieved at start of round
3. **Empty question text**: Skip that round or break loop
4. **Question element appears slowly**: Use appropriate timeout (10 seconds)

---

## Code Changes Required

1. **Create `retrieve_question()` function**
2. **Remove initial question validation loop** (lines 611-670)
3. **Modify main loop** to retrieve question at start of each iteration
4. **Update summary tracking** to store questions from all rounds

