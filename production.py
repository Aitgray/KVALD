import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F

from model import UNet
from kalman_filter import MaskKalmanFilter

# --- Configuration ---
MODEL_PATH = "kvald_unet_best.pth"  # path to your trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (256, 256)  # (width, height) for cv2.resize


# --- Helper Functions ---
def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """
    Convert a BGR frame (OpenCV) to a normalized tensor for the model.
    Keep it RGB with 3 channels to match UNet(in_channels=3).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    tensor = F.to_tensor(resized).to(DEVICE)  # [C,H,W] in [0,1]
    return tensor.unsqueeze(0)                # [1,C,H,W]


def apply_mask_to_frame(frame: np.ndarray, mask_01: np.ndarray, dim_strength: float = 0.7) -> np.ndarray:
    """
    Apply a dimming mask to the frame.
    mask_01: [H,W] float32 in {0,1} (binary). 1 means "dim this pixel".
    dim_strength: 0..1, fraction of brightness removed where mask==1.
    """
    # Resize mask to frame size
    mask_resized = cv2.resize(mask_01, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_resized = mask_resized.astype(np.float32)

    # Turn into [H,W,1] so it broadcasts across color channels
    alpha = mask_resized[..., None] * dim_strength  # 0 where no glare, dim_strength where glare
    alpha = np.clip(alpha, 0.0, 1.0)

    # Dim the affected pixels
    frame_f = frame.astype(np.float32)
    out = frame_f * (1.0 - alpha)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out


# --- Main Application ---
def main():
    """Runs the main application loop."""
    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = UNet(in_channels=3, out_channels=1).to(DEVICE)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
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

    # --- Kalman + post-processing params ---
    # Threshold on SMOOTHED probabilities; start around 0.6–0.65 for your stats.
    base_thr = 0.64

    kalman = None   # will be initialized after first frame
    q_val = 0.005   # process noise (how fast state can move)
    r_val = 0.20    # measurement noise (how much we trust current frame)

    # Morphology kernels (spatial cleanup)
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        original_frame_display = frame.copy()

        # Preprocess the frame for the model
        input_tensor = preprocess_frame(frame)

        # 3. Model prediction (logits -> probabilities)
        with torch.no_grad():
            logits = model(input_tensor)          # [1,1,H,W]
            probs = torch.sigmoid(logits)[0, 0]   # [H,W] in [0,1]
            p_np = probs.cpu().numpy().astype(np.float32)

        # Initialize Kalman on first frame
        if kalman is None:
            kalman = MaskKalmanFilter(shape=p_np.shape, q_val=q_val, r_val=r_val)
            # Optional: start state at first measurement instead of all-zeros
            kalman.state = p_np.copy()

        # 4. Temporal smoothing with Kalman
        p_smooth = kalman.update(p_np)           # [H,W]
        p_smooth = np.clip(p_smooth, 0.0, 1.0)

        # Debug: print stats occasionally
        # (comment out once it looks good)
        """print(
            f"raw  min={p_np.min():.3f}, max={p_np.max():.3f}, mean={p_np.mean():.3f} | "
            f"smooth min={p_smooth.min():.3f}, max={p_smooth.max():.3f}, mean={p_smooth.mean():.3f}"
        )"""

        # 5. Threshold the SMOOTHED probabilities -> binary mask
        binary_mask = (p_smooth > base_thr).astype(np.uint8)  # [H,W] in {0,1}

        # 6. Morphology for spatial smoothing
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)

        # Also compute fraction of active pixels to see if threshold is sane
        pos_frac = binary_mask.mean()
        print(f"  active fraction: {pos_frac:.3f}")
        # If literally 0 for many frames, base_thr is probably too high for your webcam lighting.

        # 7. Apply mask as dimming
        masked_frame = apply_mask_to_frame(frame, binary_mask, dim_strength=0.7)

        # 8. Build side-by-side view with an extra debug mask view
        # (grayscale mask preview on the right)
        mask_vis = (binary_mask * 255).astype(np.uint8)
        mask_vis = cv2.resize(mask_vis, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_vis_colored = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

        top_row = np.concatenate((original_frame_display, masked_frame), axis=1)
        bottom_row = np.concatenate((mask_vis_colored, mask_vis_colored), axis=1)  # duplicate to fill width
        side_by_side = np.concatenate((top_row, bottom_row), axis=0)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(side_by_side, 'Input', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side_by_side, 'Masked Output',
                    (original_frame_display.shape[1] + 10, 30),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side_by_side, 'Binary Mask (debug)',
                    (10, original_frame_display.shape[0] + 30),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow("KVALD Production Preview", side_by_side)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. Cleanup
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
