import argparse
from tabsynfnn.tabsyn.sample_pm import diffusion_repeated_sample_pm
from tabsynfnn.prediction_interval_utils import str2bool


def main():
    parser = argparse.ArgumentParser(description="Diffusion Predictive Modeling Sampling")

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--diffusion_path', type=str, required=True)
    parser.add_argument('--sampling_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_repeats', type=int, default=3)
    parser.add_argument('--use_val', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model_filename', type=str, default='model.pt')

    args = parser.parse_args()
    
    diffusion_repeated_sample_pm(
        data_path=args.data_path,
        diffusion_path=args.diffusion_path,
        sampling_path=args.sampling_path,
        batch_size=args.batch_size,
        steps=args.steps,
        num_repeats=args.num_repeats,
        use_val=args.use_val,
        base_seed=args.seed,
        model_filename=args.model_filename,
    )

if __name__ == '__main__':
    main()
