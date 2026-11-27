"""
Real-time enemy elixir tracker for Clash Royale.

Live screen capture:
    python src/motion_detect_demo.py --region TOP,LEFT,WIDTH,HEIGHT
Recorded gameplay clip (deterministic - best for testing / making the demo):
    python src/motion_detect_demo.py --video gameplay.mp4 --start-at 3.0 --deck "card1,...,card8"

Hotkeys (window focused):  s = match just started (re-sync clock, elixir=5)
                           + / - = nudge elixir by 1 when the estimate drifts
                           q = quit
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from elixir_tracker import CardVoter, ElixirTracker, lookup_cost  # noqa: E402

# ---------------- CONFIG ----------------
CONF_THRESHOLD = 0.6
REF_WIDTH = 919               # width MIN_CONTOUR_AREA / PAD were tuned for
MIN_CONTOUR_AREA = 1500       # scaled by (width / REF_WIDTH)^2
MAX_CONTOUR_FRACTION = 0.35   # ignore "whole ROI changed" frames (flashes, transitions)
PAD = 20
ELIXIR_MARGIN = 1.0           # tolerance for estimate drift in the 'can they afford it' rule
POST_PLAY_COOLDOWN = 0.8      # s between two confirmed plays
DEFAULT_REGION = "56,1051,919,1318"   # top,left,width,height (your original values)
DEFAULT_ROI = "0.12,0.42"     # vertical band (fraction of frame height) = ENEMY half of arena
WINDOW = "Arena Elixir Tracker"
# ----------------------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="analyse a recorded clip instead of the live screen")
    ap.add_argument("--region", default=DEFAULT_REGION, help="live capture: top,left,width,height")
    ap.add_argument("--roi", default=DEFAULT_ROI,
                    help="y0,y1 fractions of the frame covering the enemy's side of the arena")
    ap.add_argument("--start-at", type=float, default=0.0,
                    help="video: second at which the battle starts (match clock = 0)")
    ap.add_argument("--fps", type=float, default=20.0, help="live capture frame rate")
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--no-mirror", action="store_true", help="disable Mirror (+1) inference")
    ap.add_argument("--model", help="path to a .pth (e.g. models/card_classifier_arena.pth)")
    ap.add_argument("--deck", help="comma-separated opponent deck (up to 8 cards); restricts predictions to it")
    ap.add_argument("--save-crops", metavar="DIR", help="save every motion crop to DIR for labelling/fine-tuning")
    ap.add_argument("--headless", action="store_true", help="no window (print events to the terminal only)")
    return ap.parse_args()


def frame_source(args):
    """Yields (bgr_frame, t_seconds, fps). Video uses frame_index/fps as the clock so
    results do not depend on how fast your PC processes the clip."""
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"Cannot open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, i / fps, fps
            i += 1
        cap.release()
    else:
        import mss
        top, left, width, height = (int(v) for v in args.region.split(","))
        monitor = {"top": top, "left": left, "width": width, "height": height}
        period = 1.0 / args.fps
        with (mss.MSS() if hasattr(mss, "MSS") else mss.mss()) as sct:
            next_tick = time.monotonic()
            while True:
                frame = np.ascontiguousarray(np.array(sct.grab(monitor))[:, :, :3])
                yield frame, time.monotonic(), args.fps
                next_tick += period                     # pace the loop to a steady frame rate
                time.sleep(max(0.0, next_tick - time.monotonic()))


def draw_overlay(frame, tracker, now, roi_px, recent, paused):
    h, w = frame.shape[:2]
    y0, y1 = roi_px
    cv2.rectangle(frame, (0, y0), (w - 1, y1), (0, 200, 255), 1)       # enemy ROI
    cv2.putText(frame, "enemy zone", (8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    e = tracker.elixir
    cv2.rectangle(frame, (20, 14), (20 + 330, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Enemy Elixir: {e:4.1f}", (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 120, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (30, 54), (330, 68), (60, 60, 60), -1)         # elixir bar
    cv2.rectangle(frame, (30, 54), (30 + int(300 * e / 10), 68), (255, 0, 200), -1)
    clock = max(0.0, now - tracker.match_start)
    cv2.putText(frame, f"{int(clock // 60)}:{int(clock % 60):02d}  {tracker.phase(now)}",
                (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    for i, line in enumerate(reversed(recent)):                          # last plays, newest first
        cv2.putText(frame, line, (30, 110 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    if paused:
        cv2.putText(frame, "waiting for battle start (--start-at)", (30, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)


def run_demo():
    args = parse_args()
    if args.model:
        os.environ["ELIXIR_MODEL"] = args.model
    from card_classifier import deck_indices, predict_card   # imported late: torch is slow to load
    allowed = deck_indices([c.strip() for c in args.deck.split(",")]) if args.deck else None
    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)
    last_save_t = -1e9
    saved = 0

    roi_frac = tuple(float(v) for v in args.roi.split(","))
    tracker = ElixirTracker(mirror=not args.no_mirror)
    voter = CardVoter(window_sec=1.2, min_votes=3)

    prev_gray = None
    recent = deque(maxlen=5)
    last_play_t = -1e9
    ignored_log = {}
    started = args.video is None or args.start_at <= 0   # live: starts immediately (press 's' to re-sync)
    first = True

    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("Running... press 'q' to quit, 's' = battle started, +/- = nudge elixir\n")

    try:
        for frame, now, fps in frame_source(args):
            h, w = frame.shape[:2]
            if first:
                tracker.reset(now if started else args.start_at)
                first = False
            if not started and now >= args.start_at:
                tracker.reset(now)
                started = True

            y0, y1 = int(h * roi_frac[0]), int(h * roi_frac[1])
            scale = w / REF_WIDTH
            min_area = MIN_CONTOUR_AREA * scale * scale
            pad = max(4, int(PAD * scale))

            if started:
                tracker.advance(now)

                # ---- motion detection, restricted to the enemy half ----
                gray = cv2.GaussianBlur(cv2.cvtColor(frame[y0:y1], cv2.COLOR_BGR2GRAY), (7, 7), 0)
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    diff = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    roi_area = gray.shape[0] * gray.shape[1]
                    contours = [c for c in contours
                                if min_area < cv2.contourArea(c) < MAX_CONTOUR_FRACTION * roi_area]

                    if contours and (now - last_play_t) > POST_PLAY_COOLDOWN:
                        c = max(contours, key=cv2.contourArea)
                        x, y, bw, bh = cv2.boundingRect(c)
                        y += y0                                         # ROI -> full-frame coords
                        cx0, cy0 = max(0, x - pad), max(0, y - pad)     # pad on ALL sides, clamped
                        cx1, cy1 = min(w, x + bw + pad), min(h, y + bh + pad)
                        crop = frame[cy0:cy1, cx0:cx1]
                        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

                        if crop.size > 0:
                            small = cv2.resize(crop, (128, 128))
                            card, conf = predict_card(small, allowed)
                            if args.save_crops and now - last_save_t > 0.4:       # <= 2.5 crops/s
                                cv2.imwrite(os.path.join(args.save_crops, f"{saved:05d}_{card.replace(' ', '-')}_{conf:.2f}.png"), small)
                                saved += 1
                                last_save_t = now
                            if card == "background":
                                voter.clear()                                      # model says: not a card deployment
                            elif conf > args.conf:
                                confirmed = voter.add(card, now)
                                if confirmed and lookup_cost(confirmed)[0] > tracker.elixir + ELIXIR_MARGIN:
                                    # a card that costs more than the enemy's elixir can't have been played
                                    if now - ignored_log.get(confirmed, -1e9) > 3.0:      # don't spam the log
                                        print(f"[{now - tracker.match_start:6.1f}s] IGNORED {confirmed}: "
                                              f"costs {lookup_cost(confirmed)[0]} but enemy has only {tracker.elixir:.1f}")
                                        ignored_log[confirmed] = now
                                    voter.clear()
                                elif confirmed:
                                    ev = tracker.play(confirmed, now)
                                    voter.clear()
                                    last_play_t = now
                                    if ev:
                                        tag = " (MIRRORED)" if ev.mirrored else ""
                                        warn = "" if ev.known else "  [cost unknown, assumed]"
                                        recent.append(f"{ev.card}{tag} -{ev.cost}")
                                        print(f"[{ev.t:6.1f}s] CARD PLAYED: {ev.card}{tag}  cost {ev.cost}  "
                                              f"elixir {ev.elixir_before:.1f} -> {ev.elixir_after:.1f}{warn}")
                prev_gray = gray

            draw_overlay(frame, tracker, now, (y0, y1), recent, paused=not started)


            if not args.headless:
                cv2.imshow(WINDOW, frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                elif k == ord("s"):
                    tracker.reset(now)
                    started = True
                    recent.clear()
                    print("Battle clock re-synced, elixir = 5")
                elif k in (ord("+"), ord("=")):
                    tracker.adjust(+1, now)
                elif k in (ord("-"), ord("_")):
                    tracker.adjust(-1, now)
    except KeyboardInterrupt:
        print("\nInterrupted - saving what we have...")

    if not args.headless:
        cv2.destroyAllWindows()
    print(f"\nFinal enemy elixir estimate: {tracker.elixir:.1f}  ({len(tracker.history)} plays tracked)")


if __name__ == "__main__":
    run_demo()
