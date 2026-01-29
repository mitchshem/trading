# Raw Data Directory

This directory contains **untouched source files** from external data providers.

## Purpose

- Preserve original data files exactly as downloaded
- Maintain data provenance and traceability
- Allow for data format changes without losing source

## File Organization

- **Stooq data**: ASCII text files (e.g., `spy.us.txt`)
- Files are **NOT modified** in this directory
- All data transformations happen in `backend/data/`

## Current Files

- `spy.us.txt`: SPY daily data from Stooq (Stooq format: `<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>`)

## Notes

- Files in this directory are **read-only** from the application perspective
- All data loading for replay uses normalized CSV files from `backend/data/`
- Original files are preserved here for reference and re-processing
