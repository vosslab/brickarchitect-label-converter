# News

## v26.02.0 - 2026-03-10
### Highlights
- Convert Brick Architect LBX label files into Avery 5167 PDF sheets in one
  `run` command.
- Cluster and match multi-label sheets automatically, including category labels
  drawn from LBX file names.
- Keep text and images upright, ASCII-clean, and centered with auto-shrink and
  bleed-safe scaling.
- Speed up checks with tile and page limits and a stop-before-rendering flag.

### Upgrade notes
- Install the PyMuPDF dependency before running; see [INSTALL.md](INSTALL.md).
- The CLI is now a single LBX-to-Avery pipeline; the old subcommands and
  vertical-alignment overrides are gone.
