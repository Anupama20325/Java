# 3D Behavioural Dynamics — research proof of concept

A dataset-agnostic engineering prototype for **3D behavioural sensing → temporal features/encoders → fusion interface → personal baseline deviation → screening-model interface → visual/report outputs**. It is not a clinical system, does not diagnose any condition, and does not establish clinical or longitudinal validity.

## Data contract and discovery
`Generic3DSequenceAdapter` discovers CSV files under `--dataset-root` having `participant_id, session_id, timestamp, joint_or_landmark_id, x, y, z`; `modality` and `clinical_label` are optional. It turns them into `T × J × 3`. Variable length sequences, NaNs (interpolation), resampling, normalization, and participant-safe windowing are supported. `EmotionalBodyMovementAdapter` is an alias for compatible open exports; any tendency label must not be treated as clinical diagnosis data.

No dataset is bundled. With no discovered data, the demo uses a clearly marked synthetic **software test fixture only**, reports no research metrics or predictions, and writes transparent status artifacts.

## Run
```bash
python scripts/run_demo.py
python scripts/run_full_pipeline.py --dataset-root /path/to/open_3d_csv_dataset
python -m unittest discover -s tests -v
```
The default design uses 150-frame windows / 75-frame stride. Models include optional PyTorch temporal BiGRU and experimental graph-temporal encoder (17-joint connectivity), plus documented classical baseline comparison slots. PyTorch is optional: the dependency-free pipeline and tests remain runnable without it.

## Extension points
`DAICWOZAdapter` reserves mapping of future authorized 3D face/head/AU/gaze/audio/transcript/PHQ-8 inputs to the same schema; it neither requires nor bundles DAIC-WOZ. `RGBDAdapter` is a hardware-free future boundary for RGB, depth, face/body 3D, and head pose. `FaceEncoder`, `HeadEncoder`, `BodyEncoder`, `AudioEncoder`, `TextEncoder`, and concatenation/gated fusion interfaces avoid fabricating absent modalities.

## Safety and limitations
Personalization falls back to the first 20% of valid windows as a **within-session baseline proxy; not longitudinal validation**. Never randomly split windows between participants: `participant_split` and `assert_no_leakage` enforce the constraint. Dashboard language is model-estimated screening output and behavioural deviation only. Outputs under `outputs/` are engineering artifacts, not scientific results when synthetic data is used.
