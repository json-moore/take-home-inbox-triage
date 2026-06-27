# Engineering Manager's Log

> One page. This is where you show us how you *directed* the AI — it matters as much
> as the code. Be concrete. Bullet points are fine.

**Name: Jason Moore**

**Time spent (be honest):**

**Pre-setup/Reading requirements = 1 hr**

**Coding/Cleanup/Wrapup =** 

---

## How I broke the work down

- My approach for this project was to think about the problem that was presented and what needed to be implemented in order to solve the problem. Before touching any code, I read through the documentation provided to see what was required of me. I performed research with AI assistance on unfamiliar things, for example how to set up environments and where to store keys and how they are used on code without exposure. Then, I broke the problem into basic steps: Get emails, read them, have AI classify them, suggest an action to take that requires human approval, then execute, only if approval is given. I scanned through the code to get a brief understanding of what was happening. I then asked AI to explain each function to understand where each piece fits in the program. Then got to work using AI to assist in implementing the triage skill.



## Where I ran things in parallel

- Critical tools i used were ChatGPT for research assistance and understanding, and Cursor IDE with an Agent for actual code implementation. I used these in parrallel along with reading the requirement documentation to  ask questions to better understand the situation and mention my suggestions to see if my approaches were feasible.



## One time the AI was wrong, and how I caught it

- When implementing `get_inbox`, the Cursor agent imported `requests` and edited `requirements.txt` to add it as a dependency — even though the project already included `httpx`.
- I caught it by checking the diff before moving on and pushed back: use the existing stack (`httpx`), and only edit `triage_skill.py` unless otherwise stated.
- The agent reverted `requirements.txt` and switched the implementation to `httpx`. Small mistake, but exactly the kind of drift I was watching for.



## What I deliberately cut to fit the 2 hours

- Did not implement retry/fallback if AI provides a bad classification label
- No rate limiting for Anthropic calls



## The design decision I'm proudest of

- Separating **propose** from **execute**: the agent classifies and plans actions, but nothing hits the API until a human explicitly approves.
- `console_approver` as a stand-in for a production approval UI — it shows the full email, AI classification, and proposed action in plain English before prompting y/n.
- `test_triage.py` with two modes: `--interactive` to walk through approvals manually, and `--deny-all` as a smoke test that confirms the pipeline runs without side effects. Both write to the audit log so I could verify the gate end-to-end.
- Read-only client for fetch/classify/spam; write credentials loaded only inside `execute()` after approval.

