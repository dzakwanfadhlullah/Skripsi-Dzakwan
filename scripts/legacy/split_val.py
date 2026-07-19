"""
Deprecated helper.

Official experiment splits must follow the benchmark protocol and must not be
created by random re-splitting. Keep this script as a hard stop so the old
leaky workflow is not used again by accident.
"""


def main() -> None:
    raise SystemExit(
        "split_val.py sudah deprecated. Gunakan split resmi melalui organize_yolo_03.py "
        "dan jangan lakukan random split lagi."
    )


if __name__ == "__main__":
    main()
