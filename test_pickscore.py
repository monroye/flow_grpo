from flow_grpo.pickscore_scorer import PickScoreScorer
from PIL import Image
import torch
import numpy as np

def test_pickscore():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        scorer = PickScoreScorer(device=device, dtype=torch.float32)
        print("PickScoreScorer initialized successfully.")
    except Exception as e:
        print(f"Error initializing PickScoreScorer: {e}")
        return

    # Create a dummy image
    image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    prompt = "A random noise image"
    
    try:
        score = scorer([prompt], [image])
        print(f"Score for random image: {score}")
    except Exception as e:
        print(f"Error calculating score: {e}")

if __name__ == "__main__":
    test_pickscore()

