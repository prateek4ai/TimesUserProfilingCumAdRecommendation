"""Command-line interface"""

import argparse
from .predictor import CTRPredictor


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Times CTR Optimizer")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--item-id", type=int, required=True)
    parser.add_argument("--model", default="outputs/best_wide_deep_model.pt")
    
    args = parser.parse_args()
    
    predictor = CTRPredictor(args.model)
    ctr = predictor.predict(args.user_id, args.item_id)
    
    print(f"Predicted CTR: {ctr:.2%}")


if __name__ == "__main__":
    main()
