# Security policy

## Reporting a vulnerability

Please **don't report security vulnerabilities in public issues, discussions or pull requests.**

Report them privately through GitHub's
[private vulnerability reporting](https://github.com/pikhan/treegraphduals/security/advisories/new):
open the repository's **Security** tab and click **Report a vulnerability**. Only the maintainers
can see the report. You'll get a reply there, and the fix and advisory are coordinated in the same place.

## Supported versions

Security fixes are made for the **latest release only**. Older releases don't get fixes, and
nothing is backported, so upgrade to the latest release to receive them. The advisory for each
fix names the affected versions and the fixed version.

This policy is recorded in [ADR 0005](docs/adr/0005-security-fixes-for-latest-release-only.md).
It works alongside the project's [SPEC 0](https://scientific-python.org/specs/spec-0000/) support
policy ([ADR 0001](docs/adr/0001-spec-0-support-policy.md)): when a release drops a Python version,
users on that version keep the last release that supported it, but fixes go into new releases only.
