import argparse

from douzero.evaluation.simulation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DurakZero Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the DurakZero checkpoint to evaluate')
    parser.add_argument('--num_games', type=int, default=100,
                        help='Number of self-play games to evaluate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for environment shuffling')
    args = parser.parse_args()

    results = evaluate(args.checkpoint, num_games=args.num_games, seed=args.seed)
    print('Evaluation results:')
    for key, value in results.items():
        print(f'{key}: {value}')
