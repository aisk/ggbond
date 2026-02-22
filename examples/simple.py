import getopt
import sys

import ggbond


def main():
    # Parse command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "b:", ["backend="])
    except getopt.GetoptError as err:
        print(f"Error: {err}")
        print("Usage: python simple.py [-b|--backend cpu|metal]")
        sys.exit(1)

    backend_type = "cpu"
    for opt, arg in opts:
        if opt in ("-b", "--backend"):
            backend_type = arg.lower()

    with ggbond.Session(backend_type) as s:
        a = s.tensor([
            [2, 8],
            [5, 1],
            [4, 2],
            [8, 6]
        ])
        b = s.tensor([
            [10, 5],
            [9, 9],
            [5, 4]
        ])
        result = (a @ b).numpy()

    rows, cols = result.shape
    print(f"mul mat ({cols} x {rows}):")
    print(result)


if __name__ == "__main__":
    main()
