import isaacgym
from utils.runner import Runner
from utils.runner_history_symmetry import Runner_History
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        action="store_true",
        help="Use the history runner"
    )

    args, _ = parser.parse_known_args()

    if args.history:
        sys.argv.remove("--history")
        runner = Runner_History(test=False)
    else:
        runner = Runner(test=False)

    runner.train()

