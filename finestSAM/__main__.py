import argparse
from model import call_train, automatic_predictions

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="finestSAM model, allows fine-tuning (--mode train) or making predictions (--mode predict)")
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='Execution mode: train or predict')
    args, unknown = parser.parse_known_args()

    if args.mode == 'predict':
        from config import cfg_predict as cfg

        predict_parser = argparse.ArgumentParser()
        predict_parser.add_argument('--input', type=str, required=True, help='Path of the image to analyze')
        predict_parser.add_argument('--opacity', type=float, default=cfg.opacity, help='Opacity of the predicted masks when printing the image')
        predict_args = predict_parser.parse_args(unknown)
    elif args.mode == 'train':
        from config import cfg_train as cfg

        train_parser = argparse.ArgumentParser()
        train_parser.add_argument('--dataset', type=str, required=True, help='Path of the dataset to use for training')
        train_args = train_parser.parse_args(unknown)
        
        predict_args = None

    # Execute the selected mode 
    switcher = {
        "train": lambda cfg: call_train(cfg, train_args.dataset),
        "predict": lambda cfg: automatic_predictions(cfg, predict_args.input, predict_args.opacity)
    }
    switcher[args.mode](cfg)