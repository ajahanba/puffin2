# Related work — adversarial training does not guarantee representation invariance

Two citations, verified directly (title, venue, year, and claims checked
against the papers themselves, not secondhand summaries) per the PI's
request, supporting the standing claim in this project that adversarially
trained representations can leak to a post-hoc attacker even when an
in-training adversary reaches chance-level accuracy.

**Elazar, Y., & Goldberg, Y. (2018). "Adversarial Removal of Demographic
Attributes from Text Data." In *Proceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing (EMNLP)*, pages 11–21,
Brussels, Belgium. DOI: 10.18653/v1/D18-1002. arXiv:1808.06640.**
Trains text classifiers with an adversarial component meant to scrub
demographic attributes (age, gender, etc.) from intermediate
representations. The in-training adversary reaches chance-level accuracy
on its own held-out set, but a *separate, freshly trained* post-hoc
classifier recovers the demographic labels well above chance from the
same representations — across multiple tasks, attributes, and datasets.
The authors explicitly warn against trusting an in-training adversary's
low accuracy as evidence of invariance.

**Song, C., & Shmatikov, V. (2020). "Overlearning Reveals Sensitive
Attributes." In *International Conference on Learning Representations
(ICLR 2020)*. arXiv:1905.11742.** Shows that models trained on one
narrow objective (e.g., binary gender classification from face images)
"overlearn" and encode substantially more — including attributes never
labeled in training, such as race or individual identity. Overlearning
is intrinsic to some tasks: deliberately censoring the unwanted attribute
from the training signal does not prevent the representation from still
carrying it, and an overlearned representation can be repurposed for a
different, privacy-violating task without access to the original
training data.

**Why this matters here:** both papers establish, in a different domain
(text and vision, not mobility), that an in-training adversary's apparent
success (low accuracy, high loss) is not evidence that the information it
was trained against is actually gone from the representation — a
different, better-targeted attacker can still recover it. This is the
same failure mode `reports/03_attack_suite.md` and `reports/03b_adaptive_robustness.md`
find for Mo-PAE's `ed_soft` fairness mechanism: a naive, group-agnostic
attacker under-reports what a group-aware adaptive attacker recovers.
Per the PI's framing decision, this project does **not** claim that
naive-attacker leakage from adversarially trained representations is
itself novel — it is well established by this prior work. What `03b`/`03c`
newly show is that this known leakage is **unevenly distributed across
users**, which breaks fairness claims built on top of a naive evaluation,
specifically in the mobility-privacy setting Mo-PAE/PUFFIN targets.
