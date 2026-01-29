import os

"""
This file is just needed to fix the broken MVC-medium LP files from distributional MIPLIB. You don't need it other than that.
"""

def append_end_to_lp_files(directory):
    for fname in os.listdir(directory):
        if fname.lower().endswith(".lp"):
            path = os.path.join(directory, fname)
            if not os.path.isfile(path):
                continue

            with open(path, "a", encoding="utf-8") as f:
                f.write("\nEnd\n")

            print(f"Updated: {fname}")

if __name__ == "__main__":
    append_end_to_lp_files(
        "instances/indset/test"
    )
    append_end_to_lp_files(
        "instances/indset/val"
    )
    append_end_to_lp_files(
        "instances/indset/train"
    )
