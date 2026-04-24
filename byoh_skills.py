"""Project-local skills for byoh.

Define skills here and they'll be auto-loaded by the CLI.
Activate them with --orch <name>.

    byoh --orch safety "do something risky"
    byoh --orch concise "explain quantum computing"
    byoh --orch planning safety "refactor the codebase"
"""

from byoh import Skill

SKILLS = [
    Skill(
        name="safety",
        prompt="Never execute destructive commands like rm -rf, drop tables, "
               "or delete files unless the user explicitly confirms.",
        description="Prevents destructive operations",
    ),
    Skill(
        name="concise",
        prompt="Be extremely concise. Max 2-3 sentences per response. "
               "No preamble, no filler, no repetition.",
        description="Forces brief responses",
    ),
]
