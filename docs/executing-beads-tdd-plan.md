# TDD Plan for executing-beads Skill

> **Status:** RED phase - need to run baseline tests before writing skill
> **Iron Law:** NO SKILL WITHOUT A FAILING TEST FIRST

## Background

The executing-beads skill links Claude's ephemeral Tasks (TaskCreate) to durable bead state (bd system). This enables coherent long-running work across context compaction and session boundaries.

**Mental Model:**
- Beads (bd) = durable job queue on disk
- Tasks (TaskCreate) = in-session working memory, ephemeral
- Skill = protocol linking them (bead ID in commits, claim/close, rehydration)

**Skill Type:** Technique (how-to guide)

---

## Step 0: Delete Any Untested Skill

```bash
rm -rf ~/.claude/skills/executing-beads
```

No exceptions. Don't keep as "reference."

---

## RED Phase: Baseline Testing (Without Skill)

### Scenario 1: Basic Application

**Prompt to subagent:**
> "Here's a bead `dsrs-vn6.1.1`. Use TaskCreate to track it, then implement it."

**Watch for:**
- Does agent use TaskCreate at all?
- Does agent include bead ID in Task subject/description?
- Does agent run `bd claim` before starting?
- What commit message format do they use? (bead ID included?)
- Do they run `bd close` when done?

**Document:** Exact choices, rationalizations, missing steps.

---

### Scenario 2: Context Recovery

**Prompt to subagent:**
> "Context just compacted. You were working on something but your Tasks are gone. Figure out what you were doing and continue."

**Watch for:**
- Do they check `bd list --status=in_progress`?
- Do they look at `jj log` for bead IDs in commit messages?
- Do they recreate Tasks from bead state?
- Or do they flounder / start fresh?

**Document:** Recovery strategy (or lack thereof).

---

### Scenario 3: Mid-Work Interruption

**Prompt to subagent:**
> "Stop working on the current bead. Switch to this urgent one instead: `dsrs-urgent`. Then come back to the original."

**Watch for:**
- Do they describe current jj state before switching?
- Do they leave the original bead claimed?
- Is there traceability for what was done vs pending?
- Can they resume the original work?

**Document:** State management approach.

---

### Scenario 4: Tech Debt Discovery

**Prompt to subagent:**
> "While implementing bead `dsrs-vn6.1.1`, you notice an unrelated bug in another file. Handle it appropriately."

**Watch for:**
- Do they file a new bead with `bd create`?
- Do they get distracted and fix it inline?
- Do they lose track of the original work?
- Do they mix unrelated changes in commits?

**Document:** Focus maintenance, bead hygiene.

---

### Scenario 5: Partial Completion

**Prompt to subagent:**
> "Implement bead `dsrs-vn6.1.1`. After you've made some progress but before finishing, I'll tell you to stop."
> (Interrupt mid-work)
> "Stop now. What state is everything in? How would another agent continue?"

**Watch for:**
- Is the bead still claimed (not closed prematurely)?
- Are commits descriptive with bead ID?
- Is progress observable from `jj log` + `bd show`?
- Could another agent pick this up?

**Document:** Handoff readiness.

---

## Baseline Documentation Template

For each scenario, record:

```markdown
### Scenario N: [Name]

**Prompt:** [exact prompt given]

**Agent behavior:**
- [step-by-step what they did]

**Rationalizations heard:**
- "[exact quotes]"

**Gaps identified:**
- [what they should have done but didn't]

**Information they lacked:**
- [what would have helped]
```

---

## GREEN Phase: Write Minimal Skill

After baseline testing, write skill addressing ONLY observed gaps:

| Gap Observed | Skill Section to Add |
|--------------|---------------------|
| No bead ID in commits | jj rhythm with bead ID prefix |
| No claim/close | Claim/close protocol |
| Can't recover after compaction | Rehydration section |
| Gets distracted by tech debt | Tech debt protocol |
| No Task ↔ bead linking | Task description template |

**Don't add hypothetical content.** Only what testing proved necessary.

---

## REFACTOR Phase: Close Loopholes

Run scenarios again WITH skill. Document:

| New Rationalization | Counter to Add |
|--------------------|----------------|
| "[quote]" | [explicit counter] |

Build rationalization table. Create red flags list. Re-test until bulletproof.

---

## Success Criteria

Agent successfully:
- [ ] Creates Task with bead ID in subject
- [ ] Includes execution protocol in Task description
- [ ] Claims bead before starting work
- [ ] Uses bead ID in every commit message
- [ ] Closes bead when Task completes
- [ ] Can recover state after compaction using bd + jj
- [ ] Files tech debt as new beads without losing focus
- [ ] Leaves work in resumable state if interrupted

---

## Execution Checklist

- [ ] Delete any existing untested skill
- [ ] Run Scenario 1 (basic application) - document baseline
- [ ] Run Scenario 2 (context recovery) - document baseline
- [ ] Run Scenario 3 (mid-work interruption) - document baseline
- [ ] Run Scenario 4 (tech debt discovery) - document baseline
- [ ] Run Scenario 5 (partial completion) - document baseline
- [ ] Analyze patterns across scenarios
- [ ] Write minimal skill addressing observed gaps
- [ ] Re-run scenarios WITH skill
- [ ] Identify new loopholes, add counters
- [ ] Re-test until bulletproof
- [ ] Deploy skill to ~/.claude/skills/executing-beads/
