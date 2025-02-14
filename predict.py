import argparse
import torch
from torchvision import transforms
from PIL import Image
import json
import model  
from model import process_image, load_checkpoint  

def main():

    parser = argparse.ArgumentParser(description="Predict flower name from image")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top k predictions')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print("Loading checkpoint...")
    model, _ = load_checkpoint(args.checkpoint) 
    model = model.to(device)

    # Process image
    print("Processing image...")
    image = process_image(args.image_path)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Predict
    print("Making prediction...")
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        top_probs, top_indices = probabilities.topk(args.top_k)

    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    top_names = [cat_to_name[cls] for cls in top_classes]

    print("\nPredictions:")
    for i in range(args.top_k):
        print(f"{i+1}. {top_names[i]} with probability {top_probs[i]:.4f}")

if __name__ == '__main__':
    main()