param(
  [string]$SpaceRepo
)
if (-not $SpaceRepo) { throw "Usage: ./scripts/push_to_hf.ps1 -SpaceRepo <user/space-name>" }

hf repo create $SpaceRepo --repo-type space --space_sdk docker --exist-ok
hf upload $SpaceRepo . --repo-type space
