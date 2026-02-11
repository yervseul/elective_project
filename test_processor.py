import cv2
import numpy as np
import pytest
import processor  # <--- THIS CONNECTS IT TO YOUR CODE

# ---------------------------------------------------------
# FIXTURE: Creates a "fake" image for testing
# ---------------------------------------------------------
@pytest.fixture
def blank_image():
    # A 100x100 black image (3 channels)
    return np.zeros((100, 100, 3), dtype=np.uint8)

# ---------------------------------------------------------
# TEST 1: The Cross Drawing Logic
# ---------------------------------------------------------
def test_cross_mask_draws_something():
    """
    Checks if 'processor.cross_mask' actually draws a cross.
    """
    # Ask your code to make a 100x100 mask, center at (50,50), size 20
    mask = processor.cross_mask((100, 100), 50, 50, 20)
    
    # 1. Check Dimensions: Did it return a 100x100 image?
    assert mask.shape == (100, 100)
    
    # 2. Check Content: Is there ANY white in it?
    # If sum is 0, the image is pure black (drawing failed).
    assert np.sum(mask) > 0 

# ---------------------------------------------------------
# TEST 2: The Grain/Noise Logic
# ---------------------------------------------------------
def test_grain_adds_noise(blank_image):
    """
    Checks if 'processor.add_grain' modifies the pixels.
    """
    # Apply your grain function
    noisy_img = processor.add_grain(blank_image, strength=50)
    
    # 1. Check Structure: Size should not change
    assert noisy_img.shape == blank_image.shape
    
    # 2. Check Content: Pure black (0) should now have noise (>0)
    # calculating the average pixel value
    assert np.mean(noisy_img) > 0 

# ---------------------------------------------------------
# TEST 3: The Static Lines Logic
# ---------------------------------------------------------
def test_static_lines_darken_rows(blank_image):
    """
    Checks if 'processor.add_static' darkens every 4th row.
    """
    # Create a WHITE image so we can see the dark lines
    white_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    # Run your static function
    result = processor.add_static(white_img)
    
    # Check Row 0 (Should be darkened)
    # Your code does: out[y] = out[y] * 0.92
    expected_value = int(255 * 0.92)
    
    # Allow a tiny margin of error for rounding (±2)
    assert abs(result[0, 0, 0] - expected_value) < 2
    
    # Check Row 1 (Should be untouched)
    assert result[1, 0, 0] == 255

# ---------------------------------------------------------
# TEST 4: The Integration Test (Crash Check)
# ---------------------------------------------------------
def test_apply_effect_survives_empty_image(blank_image):
    """
    Checks if the main 'apply_effect' crashes when no faces are found.
    """
    try:
        # Run the full pipeline on a black square
        result = processor.apply_effect(blank_image)
        
        # It should return a valid image
        assert result is not None
        assert result.shape == blank_image.shape
        
    except Exception as e:
        pytest.fail(f"Your main function crashed on an empty image! Error: {e}")