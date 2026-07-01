import random
import argparse


def generate_random_seeds(n, seed_range=(0,100)):
    return [random.randint(*seed_range) for _ in range(n)]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate random seeds.")
    parser.add_argument("--n", type=int, default=5, help="Number of random seeds to generate")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save the generated seeds")
    return parser.parse_args()

def main():
    args = parse_arguments()

    seeds = generate_random_seeds(n=args.n)
    print("Generated seeds:", seeds)

    with open(f"{args.output_dir}/random_seeds_{args.n}.txt", "w") as f:
        for seed in seeds:
            f.write(f"{seed}\n")

if __name__ == "__main__":
    main()