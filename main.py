# import matplotlib.pyplot as plt
# import numpy as np

# from palette_extraction import extract_palette_from_path
# from ga import run_ga, report_result
# from cvd_module import lab_to_srgb


# # -----------------------------------
# # SETTINGS
# # -----------------------------------
# IMAGE_PATH = "test_chart.png"      # your input image
# K_COLORS   = 6                     # number of palette colors


# # -----------------------------------
# # STEP 1: Extract palette
# # -----------------------------------
# print("\nExtracting palette from image...\n")

# original_lab, original_rgb, labels, counts = extract_palette_from_path(
#     IMAGE_PATH,
#     k=K_COLORS
# )

# print("Original RGB Palette:")
# print(np.round(original_rgb, 3))


# # -----------------------------------
# # STEP 2: Run Genetic Algorithm
# # -----------------------------------
# best_lab, history = run_ga(original_lab)


# # -----------------------------------
# # STEP 3: Convert best palette to RGB
# # -----------------------------------
# best_rgb = lab_to_srgb(best_lab)

# print("\nOptimized RGB Palette:")
# print(np.round(best_rgb, 3))


# # -----------------------------------
# # STEP 4: Print comparison report
# # -----------------------------------
# report_result(original_lab, best_lab)


# # -----------------------------------
# # STEP 5: Plot fitness graph
# # -----------------------------------
# plt.figure(figsize=(8,5))
# plt.plot(history, linewidth=2)
# plt.title("Genetic Algorithm Progress")
# plt.xlabel("Generation")
# plt.ylabel("Best Fitness")
# plt.grid(True)
# plt.show()

"""
main.py
=======
Entry point. Runs the full pipeline end to end:
 
    1. Load an image  →  extract dominant colour palette
    2. Run the GA     →  optimise palette for CVD accessibility
    3. Show results   →  before/after swatches + convergence plot
 
USAGE
-----
With an image:
    python main.py --image path/to/your/image.png
 
Without an image (uses a built-in test palette):
    python main.py
 
OPTIONS
-------
    --image     Path to input image (PNG, JPG, etc.)
    --colours   Number of palette colours to extract (default: 6)
    --cvd       CVD types to optimise for (default: all three — protan deutan tritan)
    --gens      Number of GA generations (default: 200)
    --pop       GA population size (default: 60)
    --save      Save output figures instead of just displaying them
"""
 
import argparse
import numpy as np
 
from cvd_module import srgb_to_lab, lab_to_srgb
from palette_extraction import extract_palette_from_path, print_palette
from ga import run_ga, report_result
from visualize import show_palette_comparison, show_convergence
 
 
# ─────────────────────────────────────────────────────────────
# Built-in test palette (used when no image is provided)
# A deliberately bad palette: all red-green confusion colours
# ─────────────────────────────────────────────────────────────
TEST_PALETTE_RGB = np.array([
    [0.85, 0.10, 0.10],   # red
    [0.10, 0.75, 0.10],   # green
    [0.80, 0.45, 0.05],   # orange
    [0.25, 0.60, 0.20],   # lime green
    [0.65, 0.20, 0.05],   # dark red
    [0.15, 0.50, 0.15],   # dark green
])
 
 
def main():
    # ── Parse arguments ──────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Optimise a colour palette for colour blindness accessibility"
    )
    parser.add_argument("--image",   type=str,   default=None,
                        help="Path to input image")
    parser.add_argument("--colours", type=int,   default=6,
                        help="Number of palette colours (default: 6)")
    parser.add_argument("--cvd",     type=str,   nargs="+",
                        default=["protan", "deutan", "tritan"],
                        choices=["deutan", "protan", "tritan"],
                        help="CVD types to optimise for (default: all three)")
    parser.add_argument("--gens",    type=int,   default=200,
                        help="GA generations (default: 200)")
    parser.add_argument("--pop",     type=int,   default=60,
                        help="GA population size (default: 60)")
    parser.add_argument("--save",    action="store_true",
                        help="Save figures to disk instead of displaying")
    args = parser.parse_args()
 
    print("\n" + "=" * 55)
    print("  Colour Palette Optimisation for CVD Accessibility")
    print("=" * 55)
 
    # ── Step 1: Get the palette ───────────────────────────────
    if args.image:
        print(f"\n  Loading image: {args.image}")
        palette_lab, palette_rgb, _, pixel_counts = extract_palette_from_path(
            args.image, k=args.colours
        )
        print(f"\n  Extracted {args.colours}-colour palette:")
        print_palette(palette_lab, palette_rgb, pixel_counts)
    else:
        print("\n  No image provided — using built-in test palette.")
        print("  (Run with --image path/to/image.png to use your own image)\n")
        palette_rgb = TEST_PALETTE_RGB[:args.colours]
        palette_lab = srgb_to_lab(palette_rgb)
        print(f"  Test palette ({len(palette_rgb)} colours):")
        for i, (rgb, lab) in enumerate(zip(palette_rgb, palette_lab)):
            hex_col = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            print(f"    [{i}] {hex_col}  LAB=({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f})")
 
    # ── Step 2: Run the GA ────────────────────────────────────
    print(f"\n  Optimising for: {', '.join(c.upper() for c in args.cvd)}  "
          f"({args.gens} generations, population {args.pop})")
    print("  This will take 1–3 minutes...\n")
 
    snapshot_dir = "snapshots" if args.save else None
 
    optimized_lab, history = run_ga(
        palette_lab,
        cvd_types=args.cvd,
        pop_size=args.pop,
        n_generations=args.gens,
        verbose=True,
        snapshot_dir=snapshot_dir
    )
 
    optimized_rgb = lab_to_srgb(optimized_lab)
 
    # ── Step 3: Print results ─────────────────────────────────
    report_result(palette_lab, optimized_lab, cvd_types=args.cvd)
 
    print("  Optimised palette (RGB hex):")
    for i, rgb in enumerate(optimized_rgb):
        hex_col = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        print(f"    [{i}] {hex_col}")
 
    # ── Step 4: Visualise ─────────────────────────────────────
    palette_save    = "palette_comparison.png" if args.save else None
    convergence_save = "convergence.png"       if args.save else None
 
    print("\n  Showing visualizations...")
    show_palette_comparison(
        palette_lab, optimized_lab,
        cvd_type=args.cvd,
        save_path=palette_save
    )
    show_convergence(history, save_path=convergence_save)
 
    print("  Done!\n")
 
 
if __name__ == "__main__":
    main()