"""A clip's video with its label drawn on as each character is signed, as a GIF.

    python -m fsboard_alignment.render <md5> <out.gif>

Reads FSboard's original video, at 10 frames per second and 180x320, and the alignment from output/.
"""
import argparse
import json

from PIL import Image, ImageDraw, ImageFont
from simple_video_utils.frames import FrameSize, read_frames_batched

from .data import OUTPUT, original

FPS, SIZE, MARGIN, LINE = 10, (180, 320), 5, 19  # LINE: pixels between lines of text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("md5")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    clip = json.load(open(f"{OUTPUT}/alignments.json"))[a.md5]
    label, width, images = "".join(c["c"] for c in clip["chars"]), SIZE[0] - 2 * MARGIN, []
    # the largest font, from 15 down to 10, at which the whole label fits on one line
    font = next(f for f in map(ImageFont.load_default, range(15, 9, -1)) if f.getlength(label) <= width or f.size == 10)
    for k, frame in enumerate(read_frames_batched(original(clip["file"]), fps=FPS, size=FrameSize(SIZE))):
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")
        shown = [c for c in clip["chars"] if c.get("start", float("inf")) <= k / FPS]
        lines = [""]
        for c in shown:  # a label too long even for the smallest font wraps at the frame's edge
            if font.getlength(lines[-1] + c["c"]) > width:
                lines.append("")
            lines[-1] += c["c"]
        top = image.height - 6 - LINE * len(lines)
        draw.rectangle((0, top - 4, image.width, image.height), fill=(0, 0, 0, 170))
        for n, line in enumerate(lines):
            draw.text((MARGIN, top + LINE * n), line, font=font, fill="white")
        if shown and k / FPS < shown[-1]["end"] + 0.2:  # the character being signed, in yellow
            x = MARGIN + draw.textlength(lines[-1][:-1], font=font)
            draw.text((x, top + LINE * (len(lines) - 1)), lines[-1][-1], font=font, fill=(255, 214, 10))
        images.append(image.quantize(64))
    images[0].save(a.out, save_all=True, append_images=images[1:], duration=1000 // FPS, loop=0)


if __name__ == "__main__":
    main()
