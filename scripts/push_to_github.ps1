param(
  [string]$RepoUrl
)
if (-not $RepoUrl) { throw "Usage: ./scripts/push_to_github.ps1 -RepoUrl <https://github.com/user/repo.git>" }

git init
git add .
git commit -m "feat: compliance document intelligence baseline"
git branch -M main
git remote add origin $RepoUrl
git push -u origin main
