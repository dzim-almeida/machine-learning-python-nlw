import argparse
from src.train import train_model
from src.predict import predict

def main():
    parser = argparse.ArgumentParser(description='Train or predict with LeNet on MNIST.')
    parser.add_argument('action', choices=['train', 'predict'], help='Action to perform: train or predict')
    parser.add_argument('--image', help='Path to the image for prediction')
    
    args = parser.parse_args()
    
    if args.action == 'train':
        train_model()
    elif args.action == 'predict':
        if not args.image:
            print("Please provide an image path for prediction using --image")
            return
        predict(args.image)

if __name__ == '__main__':
    main()
