$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) { $ScriptDir = "." }

$ConfigPath = if ($args.Count -gt 0) { $args[0] } else { "$ScriptDir/../pipeline_config.json" }

Write-Host "Running config-driven extraction with $ConfigPath"
python "$ScriptDir/../run_combined_extraction.py" "$ConfigPath"
