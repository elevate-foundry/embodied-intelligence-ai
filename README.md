# Embodied Intelligence AI

Intelligence that emerges from surviving in a world with consequences.

---

## Core Principles

### 1. Closed-Loop Interaction

The agent operates in a continuous loop:

```
observe → decide → act → world changes → observe
```

All intelligence emerges from surviving and succeeding within this loop.

### 2. Substrate-Constrained Cognition

- All perception, memory, and action are encoded through a fixed substrate (e.g., 8-bit braille tokens or other discrete bodies)
- No privileged access to latent state

### 3. Persistent, Fallible Memory

The agent maintains memory across time that can:
- Degrade
- Be overwritten
- Mislead future behavior

### 4. Irreversibility

Mistakes have consequences:
- Energy depletion
- Permanent world changes
- Episode termination without free resets

---

## Architecture Overview

### Environment

- Software-only world with hidden state
- Partial observability
- Deterministic or stochastic physics
- No direct state inspection by the agent

### Body

- Fixed action vocabulary
- Quantized sensors
- Explicit limits on resolution and bandwidth

### Agent

- Transformer-based policy and/or world model
- External memory (no magical recurrence)
- Trained first to predict world dynamics, then to act

---

## Key Research Areas

| Area | Focus |
|------|-------|
| **Embodied World Models** | Predicting future observations under action constraints |
| **Substrate-First Representation** | Studying how cognition changes when vectors are not the native medium |
| **Consequence-Driven Learning** | Intelligence shaped by failure, not explanation |
| **Simulation as Embodiment** | Proving hardware is optional for grounded intelligence |

---

## Roadmap

- [ ] Braille-encoded gridworld with irreversible dynamics
- [ ] Transformer trained as next-state predictor
- [ ] Action policy trained via RL on top of world model
- [ ] Persistent memory with corruption penalties
- [ ] Evaluation via survival, task completion, and generalization

---

## What Success Looks Like

Success is **not** benchmark scores.

Success is observing behaviors that:
- Anticipate consequences
- Recover from self-caused damage
- Adapt strategies when the world changes
- Fail in ways that make sense

---

## License

MIT License
