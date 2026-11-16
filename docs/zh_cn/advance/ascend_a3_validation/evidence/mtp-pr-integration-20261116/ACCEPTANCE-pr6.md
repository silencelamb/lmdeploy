# Integrate the MTP fix into migration PR #5134

User authorization (2026-11-16): include these fixes in the migration PR and
describe the symptom, relevant commits, compiler version upgrade, observed
effects, and full investigation link in the formal PR body. This supersedes
the earlier instruction to keep the MTP fix separate from that PR.

Use a new integration worktree/branch from pr/ascend-a3-native 5f1b1b3c.
Port only circular GDN stage1 handling and the strengthened hardware state
regression from dfa3049f/d2d933da; preserve the normal stage3 path. Do not merge
the research tools, CPU18 affinity handling, raw reports or frozen IR binaries
into the migration implementation. Keep all existing original evidence.

Include an explicit optional Triton-Ascend3.2.2.post2 dependency-update profile
and bilingual instructions for an existing compatible isolated runtime. The
default Docker recipe is CANN8.5 and its complete upgrade was not tested: do not
silently turn that into a claimed validated CANN9.1/9.2/Torch combination.

Validation: run the six Q1/2/4 eager/graph CPU-oracle tests on the final PR source
with both fixed old and new compiler environments, no dlinfer. Two fresh seeded
Sage TP2/MTP3 graph/prefix trace21 runs on each profile must match all21 token
sequences and top5 scores from the previous correct single-buffer reference.
Run the relevant Ascend CPU contracts and applicable formatting checks. Do not
rerun full migration acceptance or claim fresh Docker/CANN9.2/HTTP certification.

Record new pr6 labels, full commands, source commits/diffs, dependency paths,
results and failures. After checks, fast-forward the existing PR branch and
push without force, update the PR body, and verify remote head/body. Publish
this integration evidence and updated conclusion on the fork documentation
branch. Do not merge the PR or post unrelated review/comment messages.
