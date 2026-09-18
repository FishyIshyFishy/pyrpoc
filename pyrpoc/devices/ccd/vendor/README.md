# Andor SDK2, bundled

This folder is the primary place `sdk2.load()` looks for `atmcd64d.dll`
(`sdk2.VENDOR_DIR`, checked before any installed Andor SOLIS/SDK and before
`PYRPOC_ANDOR_DLL`). There is no config field for the DLL location any more --
put the files here once and every machine that checks out this repo can run
the camera with no per-install setup.

## What to copy in

From a machine with Andor SOLIS or the standalone Andor SDK installed
(typically `C:\Program Files\Andor SOLIS`):

1. `atmcd64d.dll` -> straight into this folder, next to this README.
2. The whole `Drivers` subfolder -> `vendor/Drivers/` here. It holds the
   detector `.ini` and firmware files `Initialize()` needs; without it you get
   `DRV_ERROR_FILELOAD` even though the DLL itself loaded fine.

Resulting layout:

```
pyrpoc/devices/ccd/vendor/
    README.md          (this file)
    atmcd64d.dll
    Drivers/
        ...             (.ini / firmware files from the Andor install)
```

Nothing else under `vendor/` needs to exist for the CCD device to work --
`sdk2.default_data_dir()` finds `Drivers/` automatically once the DLL is found
here.

## Why this is safe to do, and what it costs

Andor's SDK is not redistributable to third parties, which is why an earlier
plan for this device ruled bundling out. That restriction is about shipping
software to someone else; it does not stop checking the DLL into a private
lab repository that only runs on your own instruments. If this repository is
ever made public or shared outside the lab, pull `atmcd64d.dll` and `Drivers/`
back out first.

## If the DLL isn't here yet

`CCD.open()` raises a `CcdError` naming every path it tried, including this
folder, so a missing bundle fails with a clear message rather than silently
falling back to a system install.
