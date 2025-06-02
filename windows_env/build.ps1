Set-Location ".\gym_microrts\microrts"

Remove-Item -Recurse -Force ".\build", ".\microrts.jar" -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Path ".\build" | Out-Null

Get-ChildItem -Recurse -Filter *.java -Path ".\src" | ForEach-Object { $_.FullName } | Set-Content ".\sources.txt"
cmd /c 'javac -d ".\build" -cp ".\lib\*" -sourcepath ".\src" @sources.txt'

Copy-Item -Recurse ".\lib\*" ".\build\"

Remove-Item -Force -ErrorAction SilentlyContinue ".\build\weka.jar"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ".\build\bots"

Set-Location ".\build"

Get-ChildItem -Filter *.jar | ForEach-Object {
    Write-Host "adding dependency $($_.Name)"
    jar xf $_.FullName
}

jar cvf microrts.jar *

Move-Item "microrts.jar" "..\microrts.jar" -Force

Set-Location ..
Remove-Item -Recurse -Force ".\build"
Remove-Item ".\sources.txt"