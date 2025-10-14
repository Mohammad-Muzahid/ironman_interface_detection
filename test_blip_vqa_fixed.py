import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import cv2
import numpy as np
import time

print("🚀 TESTING BLIP VQA - TRUE VLM (Fixed Version)")
print("=" * 50)

try:
    print("1. Loading BLIP VQA model...")
    start_time = time.time()
    
    # Load model and processor directly (not using pipeline)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    
    print(f"✅ BLIP VQA loaded in {time.time() - start_time:.1f}s")
    
    # Create test image
    print("2. Creating test image...")
    test_image = np.ones((400, 500, 3), dtype=np.uint8) * 255  # White background
    
    # Add multiple objects
    cv2.rectangle(test_image, (50, 50), (150, 150), (255, 0, 0), -1)    # Blue square
    cv2.circle(test_image, (300, 100), 50, (0, 255, 0), -1)             # Green circle
    cv2.rectangle(test_image, (200, 200), (300, 300), (0, 0, 255), -1)  # Red rectangle
    cv2.putText(test_image, "HELLO", (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    pil_image = Image.fromarray(test_image)
    
    print("   Test Image: Blue square (left), Green circle (right), Red rectangle, Text 'HELLO'")
    
    print("3. Testing TRUE VLM capabilities...")
    
    # Test questions
    test_questions = [
        "What color is the circle?",
        "What is on the left side of the image?",
        "How many colored shapes are there?",
        "What text is written in the image?",
        "What objects do you see?"
    ]
    
    for question in test_questions:
        print(f"   ❓ {question}")
        start_time = time.time()
        
        # Process inputs
        inputs = processor(pil_image, question, return_tensors="pt")
        
        # Generate answer
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        
        # Decode answer
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"   🎯 {answer}")
        print(f"   ⏱️  {time.time() - start_time:.2f}s\n")
    
    print("🎉 SUCCESS! BLIP VQA TRUE VLM IS WORKING!")
    
except Exception as e:
    print(f"❌ BLIP VQA setup failed: {e}")
    import traceback
    traceback.print_exc()