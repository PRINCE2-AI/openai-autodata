# Security Policy

## Secrets

Never commit OpenAI API keys or a populated `.env` file. Use environment variables locally and GitHub Actions secrets in CI. If a key is exposed, revoke it immediately in the OpenAI dashboard before removing it from Git history.

## Reporting a vulnerability

Please do not disclose credential leaks or exploitable vulnerabilities in a public issue. Contact the maintainer through the `PRINCE2-AI` GitHub profile or use GitHub's private vulnerability reporting feature when it is available for the repository.

Include reproduction steps, affected files, potential impact, and a suggested mitigation. Remove all real credentials and private documents from the report.
