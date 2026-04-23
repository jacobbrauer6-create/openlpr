Write-Host "Scanning for the heavy hitters in C:\Projects\openlpr..." -ForegroundColor Cyan

# This finds the top 20 largest files regardless of where they are hidden
Get-ChildItem -Path . -Recurse -File -ErrorAction SilentlyContinue | 
    Sort-Object Length -Descending | 
    Select-Object Name, 
        @{Name="Size(GB)"; Expression={$_.Length / 1GB}}, 
        @{Name="Extension"; Expression={$_.Extension}},
        FullName -First 20 | 
    Format-Table -AutoSize