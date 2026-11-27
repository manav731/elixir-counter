"""
Fast keyboard labelling of the crops saved with  --save-crops.

    python src/label_crops.py --crops crops --deck "hog rider,musketeer,fireball,..." 

Keys:  1-9 = card in the order of --deck     0 = background (NOT a card deployment:
       tower fire, walking troop, spell effect, UI flash, empty motion)
       SPACE = skip     u = undo last     q = quit (progress is kept)
Labelled crops are moved to  data/arena/<card name>/  (or data/arena/background/).
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np


def assign(path, label, out_root):
    """Move one crop into out_root/<label>/ . Returns the new path."""
    dest_dir = os.path.join(out_root, label)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(path))
    shutil.move(path, dest)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--deck", required=True, help="comma-separated card names, max 9")
    ap.add_argument("--out", default=os.path.join("data", "arena"))
    args = ap.parse_args()

    deck = [c.strip().lower() for c in args.deck.split(",") if c.strip()]
    if not 1 <= len(deck) <= 9:
        raise SystemExit("--deck needs 1-9 cards")
    files = sorted(glob.glob(os.path.join(args.crops, "*.png")))
    if not files:
        raise SystemExit(f"No .png files in {args.crops}. Run the tracker with --save-crops {args.crops}")

    undo, n_done = [], 0
    i = 0
    while i < len(files):
        path = files[i]
        img = cv2.resize(cv2.imread(path), (384, 384), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((384, 384 + 340, 3), np.uint8)
        canvas[:, :384] = img
        lines = [f"{i + 1}/{len(files)}   labelled: {n_done}", "model guess: " + os.path.basename(path).split("_")[1].replace("-", " "), ""]
        lines += [f"[{k + 1}] {c}" for k, c in enumerate(deck)]
        lines += ["[0] background / not a card", "[space] skip   [u] undo   [q] quit"]
        for j, t in enumerate(lines):
            cv2.putText(canvas, t, (400, 30 + 28 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("label crops", canvas)
        k = cv2.waitKey(0) & 0xFF

        if k == ord("q"):
            break
        elif k == ord(" "):
            i += 1
        elif k == ord("u") and undo:
            old_path, new_path = undo.pop()
            shutil.move(new_path, old_path)
            n_done -= 1
            i = files.index(old_path)
        elif k == ord("0"):
            undo.append((path, assign(path, "background", args.out))); n_done += 1; i += 1
        elif ord("1") <= k <= ord("9") and (k - ord("1")) < len(deck):
            undo.append((path, assign(path, deck[k - ord("1")], args.out))); n_done += 1; i += 1
    cv2.destroyAllWindows()
    print(f"Labelled {n_done} crops -> {args.out}")
    for d in sorted(os.listdir(args.out)) if os.path.isdir(args.out) else []:
        print(f"  {d:20s} {len(os.listdir(os.path.join(args.out, d)))}")


if __name__ == "__main__":
    main()
