
import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F

from model import UNet

# --- Configuration ---
MODEL_PATH = "kvald_unet_best.pth"  # <-- IMPORTANT: Set this to the path of your trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (256, 256)

# --- Helper Functions ---
def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Converts a frame to a tensor for the model."""
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize
    resized_frame = cv2.resize(gray_frame, IMG_SIZE)
    # Convert to tensor and normalize
    tensor = F.to_tensor(resized_frame).to(DEVICE)
    # Add batch dimension
    return tensor.unsqueeze(0)

def apply_mask_to_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Applies the mask to the frame, making the masked area red."""
    # Resize mask to frame size
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    # Create a red overlay
    red_overlay = np.zeros_like(frame, dtype=np.uint8)
    red_overlay[:] = (0, 0, 255) # BGR for red
    # Apply mask to the red overlay
    masked_red = cv2.bitwise_and(red_overlay, red_overlay, mask=mask_resized)
    # Combine with original frame
    # Use a weighted sum to make the mask translucent
    return cv2.addWeighted(frame, 1, masked_red, 0.5, 0)

# --- Main Application ---
def main():
    """Runs the main application loop."""
    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = UNet(in_channels=1, out_channels=1).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please make sure the MODEL_PATH variable in production.py is set correctly.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # 2. Initialize Camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera initialized. Press 'q' to quit.")

    # 3. Main Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Keep a copy of the original frame for display
        original_frame_display = frame.copy()

        # Preprocess the frame for the model
        input_tensor = preprocess_frame(frame)

        # Get model prediction
        with torch.no_grad():
            pred_mask = model(input_tensor)
            pred_mask = torch.sigmoid(pred_mask) # Ensure output is in [0, 1]
            pred_mask_np = pred_mask.squeeze().cpu().numpy()

        # Binarize the mask
        binary_mask = (pred_mask_np > 0.5).astype(np.uint8)

        # Apply the mask to the original frame
        masked_frame = apply_mask_to_frame(frame, binary_mask)

        # Resize frames for consistent side-by-side display
        display_size = (masked_frame.shape[1], masked_frame.shape[0])
        original_frame_display = cv2.resize(original_frame_display, display_size)


        # Create a side-by-side view
        side_by_side = np.concatenate((original_frame_display, masked_frame), axis=1)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(side_by_side, 'Input', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side_by_side, 'Masked Output', (display_size[0] + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


        # Display the resulting frame
        cv2.imshow("KVALD Production Preview", side_by_side)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Cleanup
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
