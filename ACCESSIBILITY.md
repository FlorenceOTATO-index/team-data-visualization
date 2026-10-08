# Accessibility

Accessibility matters to this project because our work is meant to be read
and understood — by our classmates, our professor, and anyone who opens the
repo or the dashboard. A visualization that only works for some readers
isn't doing its job. This document covers what we commit to, what we expect
from contributors, and how to report a barrier.

## Priorities

We prioritize three outcomes:

1. **Figures and dashboard views that don't rely on color alone.** We use
   directly labeled values and sequential palettes chosen to stay readable
   for common forms of color vision deficiency.
2. **Plain-language writing** in docs, notebooks, and slide reports.
3. **Text alternatives for figures** committed under `reports/figures/`
   (a caption describing each figure's takeaway lives alongside it).

We treat WCAG 2.2 AA as an aspirational reference, not a conformance claim
— we have not done a formal accessibility evaluation.

## Contributor expectations

- Every figure committed under `reports/figures/` gets a caption stating
  its takeaway, not just its title.
- Don't encode meaning by color alone: add direct labels or a legend that
  includes values.
- Write docs and notebook markdown in plain language; expand acronyms on
  first use.
- If you add anything interactive (e.g. dashboard views), sanity-check it
  with keyboard only before opening the PR, and say so in the PR's
  Verification section.

## Reporting accessibility issues

Open a bug report (Issues → Bug report) and note that it's an accessibility
barrier — or message a maintainer directly (team table in the README) if
you'd rather not file publicly. Useful context: what you were trying to do,
the page or file, what happened versus what you expected, browser and OS,
and assistive technology if you use any. Screenshots are optional. You never
need to disclose a disability.

### Severity

We keep triage simple, consistent with our issue labels:

- **Blocker** — the task can't be completed at all (e.g. dashboard unusable
  by keyboard).
- **Major** — the task is possible but significantly harder than it should be.
- **Minor** — annoyance or polish.

Maintainers confirm or adjust severity at triage; reporters don't need to
get it right.

### How we respond

We'll acknowledge your report within a few days, post a workaround if one
exists, and update the issue as we work on it. You're welcome to verify the
fix before we close the issue.

## Ownership and maintenance

Accessibility is owned collectively by the four team members (see the README
team table). We review this statement at each project milestone
(mid-presentation, final). If team responsibilities change, ownership passes
to whoever maintains the repo.

## Supported environments

- **Tableau Public dashboard:** current Chrome, Safari, Firefox, and Edge on
  desktop, keyboard and mouse. Note: Tableau Public's built-in keyboard and
  screen-reader support is limited and outside our control.
- **Notebooks and docs:** any modern browser via GitHub rendering, or local
  Jupyter.

We have not tested screen-reader flows end to end — that's a known gap,
listed below.

## Known limitations

- Tableau Public offers limited keyboard navigation and screen-reader
  announcements for dashboard views. We can't change the platform, so we
  keep chart titles descriptive (each states its finding) to make the
  reading order meaningful.
- Some older figures in notebooks may still lack text alternatives; we are
  backfilling captions under `reports/figures/`.
- No formal assistive-technology evaluation has been done, so treat the
  commitments above as intent, not verified coverage.

## Feedback and improvements

Suggest edits to this statement via a regular issue or pull request. For
active barriers, please use the reporting process above instead so it gets
triaged.
