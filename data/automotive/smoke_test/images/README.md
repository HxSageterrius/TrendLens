# Automotive Smoke-Test Images

This directory must be populated with images from the **Stanford Cars** dataset
before running the automotive pipeline.

## Steps

1. Download the Stanford Cars dataset from:
   https://ai.stanford.edu/~jkrause/cars/car_dataset.html

2. Select the 45 images matching the filenames in `SMOKE_TEST_MANIFEST` inside
   `src/preprocessing/automotive_adapter.py`.

3. Copy / rename them into this directory using the exact filenames listed in
   the manifest (e.g. `AUTO_0001_audi_a5_coupe_2012.jpg`).

4. Run the pipeline:
   ```
   .\.venv\Scripts\python.exe -m src.automotive_pipeline_runner
   ```

## License Reminder

Images are from the Stanford Cars dataset (non-commercial research use only).
See `../ATTRIBUTION.md` for full attribution and license terms.
