#!/bin/bash
# scripts/resume.sh
# Simply re-runs evaluation — skips all completed runs automatically
echo "Resuming from last checkpoint..."
python run_evaluation.py --status
echo ""
echo "Resuming pending runs..."
python run_evaluation.py
