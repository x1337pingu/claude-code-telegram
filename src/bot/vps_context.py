"""VPS knowledge base — pre-loaded project context for Claude prompts.

Provides compact project knowledge so Claude knows what's on the VPS
without needing to explore each time. Context size adapts to model tier.
"""

from .router import HAIKU

# ── Full context for Opus/Sonnet (~400 tokens) ──────────────────────

_FULL_CONTEXT = (
    "[VPS Knowledge]\n"
    "Projects (~/Claude/):\n"
    "- Pingu-Intelligence/: Crypto research & marketing (Monad)\n"
    "- Pingu-Dev/: Solidity + SvelteKit dApp (Monad)\n"
    "- Astro.fun/: Crypto casino crash. Next.js 16, Hono, Prisma,\n"
    "  Solidity/Foundry. Monorepo (apps/web, apps/api,\n"
    "  packages/contracts, packages/db)\n"
    "- Recherche/: Research methodology, geopolitics, iOS MVP\n"
    "- Jessica-Production/: Film production. 11 guides, templates\n"
    "- Alimentation-et-bien-etre/: Nutrition & health content\n"
    "- kiikstart/: SaaS boilerplate. Next.js, Convex, Clerk, Stripe\n"
    "- claude-code-telegram/: This Telegram bot. Python, poetry,\n"
    "  python-telegram-bot, claude-agent-sdk\n"
    "\n"
    "Aliases: c pi, c dev, c astro, c research, c jessica, c tg, c kiik\n"
    "Services: claude-telegram.service (systemd, active)\n"
    "Server: 46.225.125.208, user: agent, Ubuntu"
)

# ── Minimal context for Haiku (~30 tokens) ───────────────────────────

_MINIMAL_CONTEXT = "[VPS: ~/Claude/ has 8+ projects. Use /opus for code tasks.]"


def get_vps_context(model: str) -> str:
    """Return VPS context appropriate for the model tier.

    Full context for Opus/Sonnet (they benefit from project awareness).
    Minimal hint for Haiku (save tokens, it handles trivial queries).
    """
    if model == HAIKU:
        return _MINIMAL_CONTEXT
    return _FULL_CONTEXT
