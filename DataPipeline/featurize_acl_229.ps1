$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) { $ScriptDir = "." }
$DDIR = "$ScriptDir/../Dataset"
$DATADIR = "$DDIR/iclr_2017"
$DATASETS = @("train", "dev", "test")
$FEATDIR = "dataset"
$MAX_VOCAB = "False"
$ENCODER = "w2v"
$HAND = "True"

foreach ($DATASET in $DATASETS) {
    Write-Host "Extracting features... DATA=$DATADIR DATASET=$DATASET ENCODER=$ENCODER ALL_VOCAB=$MAX_VOCAB HAND_FEATURE=$HAND"
    $featPath = "$DATADIR/$DATASET/$FEATDIR"
    if (Test-Path $featPath) {
        Remove-Item -Recurse -Force $featPath
    }
    
    python "$ScriptDir/feature_engineering.py" `
        "$DATADIR/$DATASET/reviews/" `
        "$DATADIR/$DATASET/parsed_pdfs/" `
        "$DATADIR/$DATASET/$FEATDIR" `
        "$DATADIR/train/$FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat" `
        "$DATADIR/train/$FEATDIR/vect_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl" `
        $MAX_VOCAB $ENCODER $HAND
    Write-Host ""
}
