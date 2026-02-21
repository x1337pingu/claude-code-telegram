"""Centralized user-facing messages with casual French tone and variation.

All strings the bot sends to users go through this module. Messages are
randomized to feel natural rather than robotic.
"""

import random
from typing import Optional


class Personality:
    """User-facing messages with randomized casual French tone."""

    # ── Working / progress ───────────────────────────────────────────

    @classmethod
    def working(cls) -> str:
        return random.choice(
            [
                "Je reflechis...",
                "Ok, je me mets dessus !",
                "Laisse-moi regarder ca...",
                "Voyons voir...",
                "Je bosse dessus...",
                "2 secondes, j'y suis...",
                "J'analyse ca...",
                "C'est parti, je regarde...",
            ]
        )

    @classmethod
    def progress(
        cls,
        elapsed: str,
        tools_count: int = 0,
        current_tool: Optional[str] = None,
    ) -> str:
        parts = [f"Toujours dessus... {elapsed}"]
        if tools_count > 0:
            tool_label = "outil" if tools_count == 1 else "outils"
            parts.append(f"{tools_count} {tool_label}")
        if current_tool:
            parts.append(current_tool)
        return " · ".join(parts)

    @classmethod
    def done(cls) -> str:
        return random.choice(
            [
                "Et voila !",
                "C'est fait !",
                "Voila pour toi.",
                "Done !",
                "Hop, c'est bon.",
            ]
        )

    # ── Welcome / session ────────────────────────────────────────────

    @classmethod
    def welcome(cls, name: str) -> str:
        return random.choice(
            [
                f"Salut {name} ! Content de te voir.",
                f"Hey {name} ! Pret a coder ?",
                f"Salut {name} ! On s'y met ?",
                f"Hello {name} ! Je suis la.",
            ]
        )

    @classmethod
    def start(cls, name: str, directory: str) -> str:
        greeting = random.choice(
            [
                f"Salut {name} ! Je suis ton assistant code.",
                f"Hey {name} ! Pret a bosser.",
                f"Hello {name} ! Dis-moi ce qu'il te faut.",
            ]
        )
        return (
            f"{greeting}\n"
            f"Envoie-moi ce que tu veux — je peux lire, ecrire et executer du code.\n\n"
            f"Repertoire : <code>{directory}</code>\n"
            f"Commandes : /new (reset) · /status"
        )

    @classmethod
    def reset(cls) -> str:
        return random.choice(
            [
                "C'est reparti a zero ! Dis-moi tout.",
                "Nouvelle session. Qu'est-ce qu'on fait ?",
                "Session reset. Je t'ecoute !",
                "Ok, on repart de zero. Vas-y !",
            ]
        )

    # ── Errors ───────────────────────────────────────────────────────

    @classmethod
    def api_error(cls) -> str:
        return random.choice(
            [
                "Oups, Claude a eu un souci technique. Reessaie.",
                "Erreur cote API — ca arrive. Retente dans un instant.",
                "Le serveur Claude a eu un hoquet. Reessaie vite.",
            ]
        )

    @classmethod
    def bot_rate_limit(cls) -> str:
        return random.choice(
            [
                "Doucement ! Tu vas trop vite. Attends un instant.",
                "Ho, pas si vite ! Laisse-moi souffler.",
                "Trop de messages d'un coup. Patiente quelques secondes.",
            ]
        )

    @classmethod
    def timeout_error(cls) -> str:
        return random.choice(
            [
                "La demande a pris trop de temps. Reessaie avec un prompt plus court ?",
                "Timeout — ca a ete trop long. Retente ?",
                "Le traitement a expire. Essaie de simplifier ta demande.",
            ]
        )

    @classmethod
    def session_expired(cls) -> str:
        return random.choice(
            [
                "Session expiree. Tape /new pour repartir.",
                "La session n'existe plus. Fais /new pour recommencer.",
            ]
        )

    @classmethod
    def auth_required(cls) -> str:
        return "Acces non autorise. Contacte l'administrateur."

    @classmethod
    def security_blocked(cls) -> str:
        return (
            "Message bloque par le filtre de securite."
            " Si c'est une erreur, contacte l'admin."
        )

    @classmethod
    def generic_error(cls, detail: Optional[str] = None) -> str:
        base = "Une erreur inattendue est survenue."
        if detail:
            # Truncate to 200 chars
            truncated = detail[:200] + ("..." if len(detail) > 200 else "")
            return f"{base}\n<code>{truncated}</code>"
        return f"{base} Reessaie."

    # ── Model clarification ────────────────────────────────────────

    @classmethod
    def clarify_model(cls, message_preview: str) -> str:
        """Ask user which model to use when routing is ambiguous."""
        truncated = message_preview[:60] + ("..." if len(message_preview) > 60 else "")
        return random.choice(
            [
                f'Pour "{truncated}" — tu preferes :',
                f'Hmm, "{truncated}" — quel modele ?',
                f'"{truncated}" — rapide ou complet ?',
            ]
        )

    @classmethod
    def clarify_acknowledged(cls, model_name: str) -> str:
        """Acknowledge the user's model choice."""
        return random.choice(
            [
                f"Compris, je lance avec {model_name}.",
                f"Ok, {model_name} c'est parti !",
                f"Hop, {model_name} en route.",
            ]
        )
