# setup_structure.ps1
# Run from C:\Projects\openlpr
# Creates the full repo structure and moves existing scripts into place

Write-Host "Creating directory structure..." -ForegroundColor Cyan

# Create all directories
$dirs = @(
    "configs",
    "data\raw",
    "data\processed",
    "data\augmented",
    "models\backbones",
    "models\checkpoints",
    "training",
    "evaluation",
    "mlops",
    "energy",
    "scripts",
    "dashboard",
    "docs"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Host "  created $dir" -ForegroundColor DarkGray
}

# Move scripts to their correct locations
Write-Host "`nMoving scripts..." -ForegroundColor Cyan

$moves = @{
    "prepare_dataset.py"  = "scripts\prepare_dataset.py"
    "train.py"            = "scripts\train.py"
    "evaluate.py"         = "scripts\evaluate.py"
    "mlops_iteration.py"  = "mlops\mlops_iteration.py"
}

foreach ($src in $moves.Keys) {
    $dst = $moves[$src]
    if (Test-Path $src) {
        Move-Item -Path $src -Destination $dst -Force
        Write-Host "  $src -> $dst" -ForegroundColor DarkGray
    } else {
        Write-Host "  [skip] $src not found" -ForegroundColor Yellow
    }
}

# Create placeholder files so the folders aren't empty
Write-Host "`nCreating placeholder files..." -ForegroundColor Cyan

$placeholders = @{
    "configs\base.yaml"           = "# Base training config`nbackbone: resnet50`nepochs: 50`nbatch_size: 32`nlr: 0.0001`n"
    "configs\data.yaml"           = "# Dataset config`naugmentation:`n  augment_factor: 4`n  night_mode_prob: 0.2`n  occlusion_prob: 0.15`n"
    "training\trainer.py"         = "# Core trainer logic (extracted from scripts/train.py)`n"
    "training\losses.py"          = "# Detection + CTC OCR loss functions`n"
    "training\augmentations.py"   = "# Albumentations pipeline for plate augmentation`n"
    "evaluation\benchmark.py"     = "# GPU latency benchmarking utilities`n"
    "evaluation\metrics.py"       = "# IoU, character accuracy, plate accuracy metrics`n"
    "mlops\drift_detector.py"     = "# Dataset drift detection (extracted from mlops_iteration.py)`n"
    "mlops\active_learning.py"    = "# Hard case mining + Label Studio queue`n"
    "mlops\versioning.py"         = "# DVC dataset versioning helpers`n"
    "energy\tracker.py"           = "# CodeCarbon wrapper + GPU power estimation`n"
    "energy\report.py"            = "# CO2 report generation`n"
    "dashboard\README.md"         = "# Dashboard`nThe MLOps dashboard is embedded in the main README as an interactive widget.`n"
    "docs\dataset_sources.md"     = "# Dataset Sources`nSee README.md for the full dataset registry.`n"
    "data\raw\.gitkeep"           = ""
    "data\processed\.gitkeep"     = ""
    "data\augmented\.gitkeep"     = ""
    "models\backbones\.gitkeep"   = ""
    "models\checkpoints\.gitkeep" = ""
}

foreach ($path in $placeholders.Keys) {
    if (-not (Test-Path $path)) {
        Set-Content -Path $path -Value $placeholders[$path]
        Write-Host "  created $path" -ForegroundColor DarkGray
    }
}

# Create scripts __init__ so they're importable
"" | Set-Content "scripts\__init__.py"
"" | Set-Content "training\__init__.py"
"" | Set-Content "evaluation\__init__.py"
"" | Set-Content "mlops\__init__.py"
"" | Set-Content "energy\__init__.py"

Write-Host "`nDone! Final structure:" -ForegroundColor Green
tree /F /A | Select-String -NotMatch ".venv"
