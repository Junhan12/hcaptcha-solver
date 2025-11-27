# Submit Button Clicking Logic - Design Document

## Current Logic

### No Crumb Challenge:
1. Perform inference + clicking
2. Click submit button once
3. Wait 5 seconds
4. Close driver

### Multi-Crumb Challenge:
1. For each crumb (1 to N):
   - Perform inference + clicking
   - If not last crumb: Click submit to proceed to next crumb
   - If last crumb: Click submit to verify
2. Wait 5 seconds
3. Close driver

**Problem**: Fixed sequence doesn't handle cases where multiple rounds are needed.

---

## New Desired Logic

### Concept:
- **Unified approach**: Same logic for both single-crumb and multi-crumb challenges
- **Loop until completion**: Continue processing until submit button disappears
- **Submit button as completion signal**: When submit button is not found, challenge is complete

### Logic Flow:

```
1. Start challenge processing
2. LOOP:
   a. Perform inference + clicking
   b. Check if submit button exists (//div[@class='button-submit button'])
   c. IF submit button EXISTS:
      - Click submit button
      - Wait for page to update (3 seconds)
      - Continue loop (go back to step 2a)
   d. IF submit button NOT FOUND:
      - Break loop (challenge complete)
3. Wait 5 seconds (allow final verification)
4. Close web driver
```

### Process Details:

#### Step 1: Initial Setup
- Extract question text
- Get challenge_type_id for validation
- Initialize counters

#### Step 2: Main Processing Loop
```
WHILE submit button exists:
  1. Detect challenge type (canvas vs div tiles)
  2. Extract images:
     - Canvas elements, OR
     - Div tiles (handle sample tile if 10 tiles found)
  3. Send to API for inference
  4. Perform clicking based on results
  5. Check for submit button:
     - Try to find: //div[@class='button-submit button']
     - Use short timeout (2-3 seconds) to avoid long waits
  6. IF submit button found:
     - Click submit button
     - Wait 3 seconds for page update
     - Continue loop
  7. IF submit button NOT found:
     - Challenge complete
     - Break loop
```

#### Step 3: Cleanup
- Wait 5 seconds (allow server-side verification)
- Close web driver

---

## Implementation Details

### Helper Function: `check_submit_button_exists(driver, timeout=2)`
```python
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
```

### Modified Main Loop Structure:
```python
# Main processing loop
round_number = 0
while True:
    round_number += 1
    log.info(f"Processing round {round_number}...")
    
    # Perform inference + clicking
    # ... (existing logic)
    
    # Check if submit button still exists
    if check_submit_button_exists(driver, timeout=2):
        log.info("Submit button found. Clicking to continue...")
        if click_submit_button(driver, wait_after_click=3):
            log.success("Submit button clicked. Waiting for next round...")
            time.sleep(2)  # Additional wait for page to stabilize
            continue  # Continue loop
        else:
            log.error("Failed to click submit button. Stopping...")
            break
    else:
        log.success("Submit button not found. Challenge complete!")
        break
```

---

## Benefits

1. **Simpler Logic**: No need to track crumb count or distinguish between single/multi-crumb
2. **More Robust**: Handles any number of rounds automatically
3. **Self-Terminating**: Loop ends naturally when challenge is complete
4. **Unified Code Path**: Same logic for all challenge types

---

## Edge Cases to Handle

1. **Submit button appears slowly**: Use appropriate timeout (2-3 seconds)
2. **Submit button click fails**: Log error and break loop
3. **Page doesn't update after click**: Wait time should handle this
4. **Infinite loop protection**: Consider max rounds limit (e.g., 10 rounds)

---

## Code Changes Required

1. **Add `check_submit_button_exists()` function**
2. **Refactor `run_crawl_once()` main loop**:
   - Remove crumb-based logic
   - Implement while loop with submit button check
   - Unify single/multi-crumb paths
3. **Update `click_submit_button()`**:
   - Keep existing implementation
   - May need to handle cases where button disappears during click

---

## Testing Considerations

1. Test with single-crumb challenges
2. Test with multi-crumb challenges (2, 3+ crumbs)
3. Test with challenges requiring multiple rounds
4. Verify driver closes only after submit button disappears
5. Verify 5-second wait happens only after loop completes

