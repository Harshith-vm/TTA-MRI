@echo off
REM ================= MRI TTA x XAI : TMI re-run (fp32, all fixes) =================
REM Reuses training checkpoints. Re-runs ONLY the evaluation (fixed) + ablations.
REM Copy this to F:\MRI\ and run it from the Anaconda Prompt.
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd /d F:\MRI

echo === [0/3] Reset old (buggy) eval results; keep checkpoints ===
call conda run -n tta_patho --no-capture-output python reset_eval.py

echo === [1/3] fp32 re-run: full grid (eval-mode fix, T3A fix, IG-primary) ===
call conda run -n tta_patho --no-capture-output python run_evaluation.py

echo === [2/3] Ablations (single seed, center 0 -- fast, ~1-2 h) ===
REM --- batch-size sensitivity (ResNet+Tent and ViT+CoTTA) ---
for %%B in (16 32 64 128) do (
  call conda run -n tta_patho --no-capture-output python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model resnet50 --method tent  --eval_bs %%B --tag bs%%B
  call conda run -n tta_patho --no-capture-output python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model vit_b16  --method cotta --eval_bs %%B --tag bs%%B
)
REM --- online vs episodic/reset ---
call conda run -n tta_patho --no-capture-output python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model resnet50 --method cotta --reset --tag episodic
call conda run -n tta_patho --no-capture-output python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model vit_b16  --method tent  --reset --tag episodic
REM --- number of adaptation steps ---
for %%S in (1 2 4 8) do (
  call conda run -n tta_patho --no-capture-output python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model resnet50 --method tent --steps %%S --tag steps%%S
  call conda run -n tta_patho --no-capture-output python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model vit_b16  --method sar  --steps %%S --tag steps%%S
)

echo === [3/3] Stats (figures/tables can also be made on your Mac) ===
call conda run -n tta_patho --no-capture-output python analysis\run_all_analysis.py

echo.
echo === DONE. Copy F:\MRI\results\ to your Mac, then run the local scripts:
echo     python make_figs.py        (figures)
echo     python local_analysis.py   (corrected tables + KL + silent-failure + thresholds)
call conda run -n tta_patho --no-capture-output python run_evaluation.py --status
pause
