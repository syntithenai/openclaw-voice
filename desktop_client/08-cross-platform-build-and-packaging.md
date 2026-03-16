# 08 — Cross-Platform Build and Packaging

## Target platforms

- Linux (x64/arm64 where feasible)
- Windows (x64)
- macOS (arm64 + x64 universal when possible)

## Build outputs

- Linux: AppImage (primary), optional `.deb`
- Windows: MSI (primary)
- macOS: `.dmg` (primary)

## CI/CD strategy

1. Matrix build by OS.
2. Run unit tests + smoke integration tests.
3. Sign/notarize where required.
4. Publish artifacts with checksums and release notes.

## Platform-specific notes

### Linux

- Validate tray behavior on GNOME (with extension), KDE, XFCE.
- Include icon themes for light/dark panel variants.

### Windows

- Ensure tray icon remains visible after Explorer restart.
- Verify startup registration option behavior.

### macOS

- Hardened runtime and notarization profile.
- Validate status bar icon rendering in dark/light mode.

## Distribution and updates

- Phase 1: manual download/install.
- Phase 2: auto-update channel after stability period.
