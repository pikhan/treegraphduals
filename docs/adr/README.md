# Design decisions

Architecture decision records (ADRs) are short notes that each record one design
decision: the context, what was decided, and what follows from it. They keep the
reasoning behind conventions, such as how tied minima are handled or what the
empty tree is, next to the code, where contributors and agents can find it.

## When to write one

Write an ADR when an issue's **Decision needed** section is settled, or when any
change fixes a mathematical convention, a public API shape, or the architecture.
Link the ADR from the issue. Code must follow accepted ADRs. Changing a decision
takes a new ADR that supersedes the old one, never an edit to the old one.

## How

1. Copy `template.md` to `NNNN-short-title.md`, using the next number.
2. Fill it in and open a PR. The ADR is accepted when the PR merges.

## Records

None yet. The PR that adds the first record also adds this listing, so every
later record appears automatically:

````md
```{toctree}
:maxdepth: 1
:glob:

0*
```
````
