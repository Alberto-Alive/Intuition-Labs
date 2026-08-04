# Stage 7A Tau Environment Setup

This is an environment and real-pipeline smoke artifact only. No final tau-bench claim is made.

## Environment
- OS: `Windows-11-10.0.26200-SP0`
- Conda env: `stage7_tau`
- Python executable: `C:\Users\Alber\anaconda3\envs\stage7_tau\python.exe`
- tau2 package version: `1.0.0`
- tau2 commit: `0ed8c0ef0a1f6b024a0fb733e186922411874879`
- TAU2_DATA_DIR: `W:\HocusPocus\armyofbots - m4 but picpalac choose - τ-bench trajectory selection\external\tau2-bench\data`

## Commands Verified
- `tau2 --help`: `True`
- `tau2 run --help`: `True`
- `tau2 evaluate-trajs --help`: `True`
- `tau2 check-data`: `True`

## Install Commands Used
- `conda create -y -n stage7_tau python=3.12`
- `python -m pip install git+https://github.com/sierra-research/tau2-bench.git`
- `python -m pip install pytest torch datasets scikit-learn`
- `python -m pip install annotated-types python-dotenv`
- `git clone https://github.com/sierra-research/tau2-bench.git external/tau2-bench`
- `python -m pip install -e external/tau2-bench`

## Notes
- The official tau2 wheel install did not expose the bundled data at the expected env-local path, so the official repository was cloned under `external/tau2-bench`, installed editable, and `TAU2_DATA_DIR` was set to its `data` directory.
- `PYTHONIOENCODING=utf-8` is required on this Windows path because the workspace contains `τ`.
- `PYTHONUTF8=1` is required for `tau2 evaluate-trajs` on this Windows setup because tau2 reads Results JSON with Python's default file encoding.
- No API credentials were present in the environment, so the Stage 7A-real candidate pool loads existing official tau2 final-result trajectories and reruns the official evaluator on a tiny subset.
