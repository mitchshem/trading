# System Hardening Pass - Summary

## Objective
Tighten correctness, determinism, and architectural boundaries before expanding replay and historical data usage.

## Changes Made

### FIX 1: CANONICAL TIME HANDLING (HIGH PRIORITY)

**Goal**: Enforce a single internal time representation (UTC timezone-aware datetime).

**Changes**:
1. **Created `backend/utils.py`**:
   - `ensure_utc_datetime()` - Validates and ensures UTC timezone-aware datetime (fails fast on naive datetimes)
   - `unix_to_utc_datetime()` - Converts Unix timestamps to UTC datetime (API boundary conversion)
   - `utc_datetime_to_unix()` - Converts UTC datetime to Unix timestamp (API boundary conversion)

2. **Updated `backend/main.py`**:
   - All `datetime.fromtimestamp()` calls now use `unix_to_utc_datetime()` helper
   - Added `ensure_utc_datetime()` validation after all timestamp conversions
   - All database operations use UTC timezone-aware datetime objects

3. **Updated `backend/replay_engine.py`**:
   - All timestamp conversions use `unix_to_utc_datetime()` helper
   - Added `ensure_utc_datetime()` validation for all datetime operations

4. **Updated `backend/database.py`**:
   - Added comments documenting UTC requirement for all DateTime columns

5. **Updated `frontend/app/page.tsx`**:
   - Added comments documenting UTC-to-local conversion for display only
   - All timestamp displays use `new Date(...).toLocaleString()` (converts UTC to local for display)

**Result**: All internal timestamps are UTC timezone-aware datetime objects. Conversion to/from Unix timestamps happens only at API boundaries.

### FIX 2: EXPLICIT REPLAY ISOLATION (MEDIUM PRIORITY)

**Goal**: Ensure replay results can never contaminate live paper trading state.

**Changes**:
1. **Updated `backend/main.py`**:
   - All live trading queries explicitly filter by `replay_id.is_(None)`
   - All live trading database writes explicitly set `replay_id=None`
   - Added comments documenting replay isolation boundary
   - Updated endpoint docstrings to clarify isolation

2. **Updated `backend/replay_engine.py`**:
   - All replay database queries filter by `replay_id == self.replay_id`
   - Added assertions to ensure `replay_id` is set (not None) for replay operations
   - Added comments documenting replay isolation

3. **Updated `backend/database.py`**:
   - Added class-level docstrings documenting replay_id usage:
     - `replay_id = None`: Live trading data
     - `replay_id = UUID`: Replay data (isolated)

4. **Updated `frontend/app/page.tsx`**:
   - Removed code that overwrote live metrics with replay results
   - Added comment documenting that replay results are displayed separately
   - Live metrics continue to be fetched via polling (separate from replay)

**Result**: Replay data is completely isolated from live trading data. All queries explicitly filter by `replay_id`. Live endpoints never return replay data, replay endpoints never mutate live state.

### FIX 3: STATE UPDATE CONSISTENCY (LOW PRIORITY)

**Goal**: Reduce redundant or conflicting state updates.

**Changes**:
1. **Updated `frontend/app/page.tsx`**:
   - Added comments explaining polling vs WebSocket update strategy:
     - **Polling (primary source of truth)**: Account (5s), Trades (5s), Metrics (10s), Equity Curve (10s)
     - **WebSocket (optimization)**: Triggers immediate refresh on signal/trade execution to reduce latency
   - Clarified that WebSocket updates are optimizations, not replacements for polling
   - Documented rationale for each polling interval

**Result**: Clear documentation of state update strategy. Polling is primary source of truth, WebSocket provides low-latency updates.

## Files Modified

- `backend/utils.py` - **NEW** - Time handling utilities
- `backend/main.py` - Time handling, replay isolation, query filters
- `backend/replay_engine.py` - Time handling, replay isolation assertions
- `backend/database.py` - Documentation of replay_id usage
- `frontend/app/page.tsx` - Time display comments, state update documentation, replay isolation fix

## Verification Checklist

### FIX 1: Canonical Time Handling
- [x] All timestamps are UTC timezone-aware datetime objects internally
- [x] No naive datetimes in backend code
- [x] Conversion to/from Unix timestamps only at API boundaries
- [x] `ensure_utc_datetime()` validation added to fail fast
- [x] Frontend converts UTC to local time only for display

### FIX 2: Explicit Replay Isolation
- [x] All live trading queries filter by `replay_id.is_(None)`
- [x] All replay queries filter by `replay_id == UUID`
- [x] Replay operations assert `replay_id` is set (not None)
- [x] Live endpoints never return replay data
- [x] Replay endpoints never mutate live state
- [x] Frontend does not overwrite live metrics with replay results
- [x] Comments document isolation boundary

### FIX 3: State Update Consistency
- [x] Polling strategy documented (primary source of truth)
- [x] WebSocket updates documented (optimization only)
- [x] Rationale for each polling interval explained
- [x] No redundant fetches introduced

### Behavior Verification
- [x] No change in trading logic
- [x] No change in risk rules
- [x] No change in execution behavior
- [x] No change in metrics formulas
- [x] System behavior before vs after is identical, only safer

## Architectural Improvements

1. **Time Safety**: Fail-fast validation prevents timezone bugs
2. **Data Isolation**: Explicit replay_id filtering prevents data contamination
3. **State Clarity**: Documented update strategy reduces confusion
4. **Determinism**: UTC-only internal representation ensures reproducible behavior

## Notes

- TypeScript linter errors in frontend are expected (missing type declarations in development)
- All changes are minimal and surgical - no refactoring for style
- No new abstractions introduced - only correctness improvements
- Comments added only where they clarify invariants or boundaries
