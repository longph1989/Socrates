import argparse
import json

from parser import parse


def main():
    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification file')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    parse(spec)


if __name__ == '__main__':
    main()
