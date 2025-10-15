import torch
from PIL import Image
import argparse
import os

from model import get_efficientnet_v2 
from aug import get_val_transforms
from roi import crop_leaf_roi # We need this for preprocessing

def predict(model_path, image_path):
    """
    Loads a trained model and makes a prediction on a single image.
    """
    # --- 1. Setup ---
    # Determine the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the mapping from class index to class name
    # This must match the mapping used during training in data.py
    class_names = ["Brown Spot", "Leaf Scald", "Rice Blast", "Rice Tungro", "Sheath Blight"]

    # --- 2. Load the Trained Model ---
    model = get_efficientnet_v2(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set the model to evaluation mode
    print(f"Loaded model from {model_path}")

    # --- 3. Preprocess the Image ---
    print("Preprocessing image...")
    # For a single prediction, we can create a temporary path for the cropped image
    temp_crop_path = "temp_cropped_image.jpg"
    
    # Apply the same ROI cropping used in training
    crop_leaf_roi(image_path, temp_crop_path)
    
    # Load the cropped image
    image = Image.open(temp_crop_path).convert('RGB')
    
    # Apply the same validation transformations (resize, normalize)
    val_transforms = get_val_transforms()
    image_tensor = val_transforms(image)
    
    # Add a batch dimension (models expect a batch of images)
    # The shape changes from [C, H, W] to [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Clean up the temporary cropped image file
    os.remove(temp_crop_path)

    # --- 4. Make the Prediction ---
    print("Making prediction...")
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(image_tensor)
        
        # Convert model outputs (logits) to probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the top prediction
        top_prob, top_idx = torch.max(probabilities, 1)
        
        predicted_idx = top_idx.item()
        predicted_class = class_names[predicted_idx]
        confidence = top_prob.item()

    # --- 5. Display the Result ---
    print("\n--- Prediction Result ---")
    print(f"Predicted Disease: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify a single rice leaf image.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pt) file.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image you want to classify.')
    args = parser.parse_args()

    predict(args.model_path, args.image_path)
