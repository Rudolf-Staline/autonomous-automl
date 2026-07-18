# OpenAI Build Week submission checklist

Official reference: [OpenAI Build Week rules](https://openai.devpost.com/rules).
The listed deadline is **July 21, 2026 at 5:00 PM Pacific Time**. External account,
video, repository, and submission actions below remain manual until verified by the
project owner.

## Devpost and category

- [ ] Devpost registration completed by an eligible entrant/team.
- [ ] Category selected on Devpost. Recommended: **Developer Tools**.
- [x] Project name prepared: **Autonomous AutoML**.
- [x] Tagline and short/long descriptions prepared in `docs/DEVPOST_SUBMISSION.md`.
- [ ] Devpost fields copied and reviewed against the final public repository.

## Repository and legal

- [ ] Repository made public with the final URL tested by a logged-out visitor.
- [x] MIT `LICENSE` present and package metadata consistent.
- [x] Installation and sample-data instructions present.
- [ ] Final repository URL added to Devpost.
- [ ] If the repository remains private instead, share it with the official judging
  addresses specified in the rules and verify access.

## Video and media

- [ ] Demo recorded using `docs/DEMO_SCRIPT.md`.
- [ ] Video duration is strictly under three minutes.
- [ ] English audio narration covers the working demo and actual Codex/GPT-5.6 use.
- [ ] English subtitles added or verified if used.
- [ ] Video uploaded as a publicly visible YouTube video.
- [ ] Public video URL tested while logged out and added to Devpost.
- [ ] Screenshots or project images selected and checked for secrets/personal paths.
- [ ] No unlicensed music, trademarks, or third-party media included.

## Technical evidence

- [x] Technologies used are documented.
- [x] Codex contributions and key decisions are documented without invented work.
- [x] Known product limitations are explicit.
- [ ] Any claimed GPT-5.6 use is real, evidenced, and consistent across README,
  Devpost, video, and session record. No separate session is currently verifiable
  from this repository/transcript.
- [x] Fresh-clone commands pass: `uv sync --frozen` and `uv run automl demo`.
- [x] Final release gates and timings copied into `IMPLEMENTATION_STATUS.md`.
- [ ] Trust Layer branch gates and fresh-clone demo pass after the final commit.
- [ ] `automl trust <demo-root>/leakage` displays the same status, score, gap and
  runtime values as `trust_certificate.json` and `report.html`.
- [ ] Confirm the public demo certificate states that it is self-verified and not
  an external, regulatory, or security certification.

## Required Codex session evidence

- [ ] **Manually execute `/feedback` in this primary Codex build conversation.**
- [ ] Copy the real Session ID returned by `/feedback` into the Devpost form.
- [ ] Confirm the Session ID belongs to this main build thread.

No Session ID is generated or guessed in this repository.

## Final publication

- [ ] Final public repository URL tested from a clean browser/session.
- [ ] Choose the post-Trust-Layer version/tag explicitly after merge authorization;
  do not silently move the already validated v0.1.0 tag.
- [ ] Final tag visible on the public remote and points to the audited commit.
- [ ] Devpost preview reviewed for formatting, links, category, and limitations.
- [ ] Submission completed before the deadline.
- [ ] Submission confirmation saved.
