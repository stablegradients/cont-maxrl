"""Allow running `python -m experiment9_imagenet_localization` for a usage hint."""

import sys


def main():
    print(
        "experiment9_imagenet_localization\n"
        "\n"
        "ImageNet localization RL experiment comparing GOS (Gap-Over-Survivors)\n"
        "against binary MaxRL, GRPO, and other baselines.\n"
        "\n"
        "Entry points (not yet implemented):\n"
        "  python -m experiment9_imagenet_localization.run      "
        "  -- train a single run with a given config\n"
        "  python -m experiment9_imagenet_localization.pilot    "
        "  -- quick smoke-test / pilot run on a small subset\n"
        "  python -m experiment9_imagenet_localization.sweep    "
        "  -- launch a hyperparameter sweep\n"
        "\n"
        "Tests:\n"
        "  python -m pytest experiment9_imagenet_localization/tests/ -v\n"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
