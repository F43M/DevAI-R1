# Security

This project integrates several security checks to safeguard the code base and scraped data.

## Development Pipeline

- `bandit` scans the Python source for common vulnerabilities.
- `pip-audit` verifies installed dependencies against known CVE databases.
- `scripts/license_compliance.py` ensures scraped datasets include the required
  license information such as the **CC BY-SA** notice for Wikipedia content.

Run all checks locally with:

```bash
pre-commit run --all-files
```

## Reporting Issues

Report security issues or suspected vulnerabilities via GitHub issues.
