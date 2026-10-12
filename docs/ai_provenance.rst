AI Provenance
=============

treegraphduals is developed with AI coding agents (Claude Code). This page
publishes, for every merged pull request, how much of the change agents wrote,
how much a human wrote or edited, and how much human steering it took. The
numbers are the evidence behind the project's AI-use disclosures. The method is
recorded in :doc:`ADR 0006 <adr/0006-ai-provenance-tracking>`.

.. note::
   This page is regenerated from the latest published statistics on every docs
   build. A nightly workflow refreshes them.

What is measured
----------------

**Line authorship.** Every non-blank line a pull request adds is classified by
the commit that last wrote it:

- *AI*: an agent made the commit in a Claude Code session;
- *AI, edited by a human*: a human commit rewrote lines that an agent commit had
  added;
- *Human*: everything else.

**Human steering.** From the session transcripts: the messages the maintainer
typed to the agents (count and words), questions answered, and *active time*.
Active time counts, for each message, the time since the agent's previous output
(reading and typing), capped at 15 minutes so that breaks don't count.

**Agents.** Models, effort levels, tokens and agent working time, and the
verdicts of the adversarial ``reviewer`` agent. *Messages after the reviewer
agent approved* counts human steering that the automated review did not settle.

**Review and tests.** Reviewer rounds per pull request, formal reviews and line
comments on GitHub, which ``difficulty: max`` issues (human-owned decisions,
oracles and expected values) were closed by human-written pull requests, and the
nightly *mutation score* of the test suite on ``core`` and ``timeseries``.

The coverage rule
-----------------

Anything the transcripts don't show an agent writing counts as human. AI-written
code therefore enters the repository only through Claude Code sessions, never
pasted from another AI chat; pasted code would be counted as human. Errors
therefore mostly understate AI use. A few run the other way and count human
work as AI:

- a human edit that an agent commits along with its own;
- a human amending an agent's commit;
- a human commit whose author time falls while an agent's shell command was
  running in a session on the same pull request (for example a long test run).

Statistics
----------

.. include:: ai_stats.md
   :parser: myst_parser.sphinx_
