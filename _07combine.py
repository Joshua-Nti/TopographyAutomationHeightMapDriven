# _07combine.py
import os

def combineGCode(in_folder, out_file):
    """
    Concatenate all .gcode files in in_folder (sorted by name)
    into a single out_file.
    """

    in_folder_abs = os.path.abspath(in_folder)
    out_file_abs  = os.path.abspath(out_file)

    print("Combining G-code from:", in_folder_abs)
    print("Target:", out_file_abs)

    files = [
        f for f in os.listdir(in_folder_abs)
        if f.lower().endswith(".gcode") and not f.startswith(".")
    ]
    files.sort()

    print("Order:", files)

    with open(out_file_abs, "w", newline="\n") as fout:
        for fname in files:
            part_path = os.path.join(in_folder_abs, fname)
            print("  +", part_path)
            with open(part_path, "r") as fin:
                for line in fin:
                    fout.write(line)

    print("Combined:", out_file_abs)
