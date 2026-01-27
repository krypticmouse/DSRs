# TDD Plan for executing-beads Skill

> **Status:** COMPLETE - skill tested and deployed
> **Iron Law:** NO SKILL WITHOUT A FAILING TEST FIRST

## Background

The executing-beads skill links Claude's ephemeral Tasks (TaskCreate) to durable bead state (bd system). This enables coherent long-running work across context compaction and session boundaries.

**Mental Model:**
- Beads (bd) = durable job queue on disk
- Tasks (TaskCreate) = in-session working memory, ephemeral
- Skill = protocol linking them (bead ID in commits, claim/close, rehydration)

**Skill Type:** Technique (how-to guide)

---

## RED Phase Results (Baseline Without Skill)

### Scenario 1: Basic Application

**Prompt:** "Here's a bead `dsrs-vn6.1.1`. Use TaskCreate to track it, then implement it."

**Agent behavior:**
- Ran `bd show` to understand task
- Created implementation with tests
- Committed work
- Did NOT use TaskCreate despite explicit prompt
- Did NOT run `bd claim`
- Did NOT include bead ID in commits
- Did NOT run `bd close`

**Gaps:** No TaskCreate, no claim/close, no bead ID in commits

---

### Scenario 2: Context Recovery

**Prompt:** "Context just compacted. Tasks are gone. Figure out what you were doing."

**Agent behavior:**
- Checked `jj log` for recent work
- Ran `bd list --status=in_progress`
- Combined git + bead state to reconstruct
- Successfully continued

**Gaps:** Recovery possible but fragile - no explicit protocol, bead IDs weren't in commits

---

### Scenario 3: Mid-Work Interruption

**Prompt:** "Switch to urgent task, then return to original."

**Agent behavior:**
- Kept original bead claimed (good!)
- Used git commits to checkpoint
- Successfully returned

**Gaps:** No TaskCreate, manual context tracking, noted "another agent wouldn't know current focus"

---

### Scenario 4: Tech Debt Discovery

**Prompt:** "While implementing, notice unrelated tech debt."

**Agent behavior:**
- Filed new bead with `bd create` (good!)
- Stayed focused, didn't mix changes

**Gaps:** Good instincts but no TaskCreate

---

### Scenario 5: Partial Completion

**Prompt:** "Make progress but stop before finishing."

**Agent behavior:**
- Kept bead claimed (good!)
- DID include bead ID in commits (good!)
- Provided handoff summary

**Gaps:** Still no TaskCreate

---

## Pattern Analysis

| Behavior | S1 | S2 | S3 | S4 | S5 |
|----------|----|----|----|----|-----|
| Used TaskCreate | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ran bd claim | ❌ | N/A | ✅ | ✅ | ✅ |
| Bead ID in commits | ❌ | N/A | ❌ | ❌ | ✅ |
| Ran bd close | ❌ | N/A | N/A | N/A | N/A |

**Primary gaps addressed in skill:**
1. TaskCreate never used → Task template with bead linking
2. Claim/close inconsistent → Explicit protocol
3. Bead ID in commits inconsistent → jj rhythm section
4. Rehydration fragile → Explicit checklist

---

## GREEN Phase Results (With Skill)

### Scenario 1 WITH SKILL: Basic Application

**Result:** ✅ PASS

Agent followed skill exactly:
- ✅ Used TaskCreate with template from skill
- ✅ Ran `bd claim dsrs-vn6.2.1`
- ✅ Commit: `dsrs-vn6.2.1: create rlm-core crate with traits`
- ✅ Ran `bd close dsrs-vn6.2.1`

---

### Scenario 2 WITH SKILL: Context Recovery

**Result:** ✅ PASS

Agent followed rehydration protocol:
- ✅ `bd list --status=in_progress` found active bead
- ✅ `jj log` showed commits with bead IDs
- ✅ `bd show` provided full context
- ✅ Agent articulated exactly how to continue

**Quote:** "The bead + commit trail system provides excellent recovery. This is significantly better than losing all context on compaction."

---

## REFACTOR Phase

No new loopholes observed. Agents followed skill without rationalization.

---

## Success Criteria

- [x] Creates Task with bead ID in subject
- [x] Includes execution protocol in Task description
- [x] Claims bead before starting work
- [x] Uses bead ID in every commit message
- [x] Closes bead when Task completes
- [x] Can recover state after compaction using bd + jj
- [x] Files tech debt as new beads without losing focus (natural behavior)
- [x] Leaves work in resumable state if interrupted

---

## Execution Checklist

- [x] Delete any existing untested skill
- [x] Run Scenario 1 (basic application) - document baseline
- [x] Run Scenario 2 (context recovery) - document baseline
- [x] Run Scenario 3 (mid-work interruption) - document baseline
- [x] Run Scenario 4 (tech debt discovery) - document baseline
- [x] Run Scenario 5 (partial completion) - document baseline
- [x] Analyze patterns across scenarios
- [x] Write minimal skill addressing observed gaps
- [x] Re-run scenarios WITH skill
- [x] Identify new loopholes, add counters (none found)
- [x] Re-test until bulletproof
- [x] Deploy skill to ~/.claude/skills/executing-beads/
