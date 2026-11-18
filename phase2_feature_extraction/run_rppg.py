from visual.rppg_roi_extractor import extract_video

VIDEO_PATH = r"C:\phase2_feature_extraction\dataset\test_data\videos\s2_l_bbim3a.mov"
OPENFACE_CSV = r"C:\phase2_feature_extraction\outputs\fau\s2_l_bbim3a\s2_l_bbim3a.csv"
OUTPUT_DIR = r"C:\phase2_feature_extraction\outputs\rppg"

if __name__ == "__main__":
    result = extract_video(
        video_path=VIDEO_PATH,
        csv_path=OPENFACE_CSV,
        output_dir=OUTPUT_DIR,
        use_skin_filter=False,
        min_confidence=0.8,
        min_roi_pixels=50,
        preview=True,
        preview_stride=1,
    )
    print("\nExtraction completed:")
    for key, value in result.items():
        print(f"{key}: {value}")
