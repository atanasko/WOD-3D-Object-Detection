import sys
import argparse
import utils.wod_reader as wod_reader


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir')
    parser.add_argument('-c', '--context_name')
    args = parser.parse_args()

    if args.dataset_dir is None or args.context_name is None:
        parser.print_help()
        sys.exit()

    return args.dataset_dir, args.context_name


def main():
    dataset_dir, context_name = parse_arguments()


# object detector entry point
if __name__ == '__main__':
    main()
