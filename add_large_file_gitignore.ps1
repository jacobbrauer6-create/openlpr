# Get-ProjectSizeAndIgnore.ps1

$limitMB = 50
$gitignorePath = ".gitignore"

Write-Host "Scanning directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host "----------------------------------------------"

# 1. Get sizes and identify large items
$items = Get-ChildItem -Path . -Recurse -ErrorAction SilentlyContinue | 
    Select-Object FullName, 
    @{Name="SizeMB"; Expression={$_.Length / 1MB}}, 
    Attributes, 
    PSIsContainer

$largeItems = @()

# Process Files
$files = $items | Where-Object { !$_.PSIsContainer }
foreach ($file in $files) {
    if ($file.SizeMB -gt $limitMB) {
        $largeItems += $file.FullName.Replace((Get-Location).Path + "\", "").Replace("\", "/")
        Write-Host "LARGE FILE: $($file.SizeMB.ToString('F2')) MB - $($file.FullName)" -ForegroundColor Yellow
    }
}

# Process Folders (Calculating total folder size)
$dirs = Get-ChildItem -Path . -Directory
foreach ($dir in $dirs) {
    $size = (Get-ChildItem $dir.FullName -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "DIR: $($size.ToString('F2')) MB - $($dir.FullName)"
    if ($size -gt $limitMB) {
        $largeItems += ($dir.Name + "/")
    }
}

# 2. Update .gitignore
if ($largeItems.Count -gt 0) {
    Write-Host "`nUpdating $gitignorePath..." -ForegroundColor Green
    $largeItems | Select-Object -Unique | Out-File -FilePath $gitignorePath -Append -Encoding utf8
    Write-Host "Added $($largeItems.Count) items to .gitignore."
} else {
    Write-Host "`nNo items over $limitMB MB found." -ForegroundColor Gray
}

Write-Host "`nDone! Review your .gitignore before your next commit." -ForegroundColor Cyan