"""Intelligent model router — selects Claude model based on message complexity.

Routes simple greetings/thanks to Haiku (fast, cheap), knowledge queries to
Sonnet, and agentic/code/deploy work to Opus. Supports manual override via
/opus, /sonnet, /haiku prefixes.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import structlog

logger = structlog.get_logger()

# Model IDs
OPUS = "claude-opus-4-6"
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"

DEFAULT_MODEL = SONNET  # Safe middle ground for fallback

# Display names for logging / user messages
MODEL_NAMES = {
    OPUS: "Opus",
    SONNET: "Sonnet",
    HAIKU: "Haiku",
}

# ── Override prefixes ────────────────────────────────────────────────

_OVERRIDE_PREFIXES = {
    "/opus": OPUS,
    "/sonnet": SONNET,
    "/haiku": HAIKU,
}

# ── Keyword sets for classification ─────────────────────────────────

# Opus signals: agentic, multi-file, deploy, fix/debug, VPS/server ops
_OPUS_KEYWORDS = re.compile(
    r"\b("
    r"implement|implem|architecture|archi|refactor|redesign|"
    r"multi.?file|multi.?fichier|codebase|migration|"
    r"build|construi|deploy|deploie|pipeline|ci.?cd|"
    r"create.*project|cree.*projet|scaffold|"
    r"full.*app|application.*complete|"
    r"security.*audit|pentest|"
    r"database.*schema|schema.*base|"
    r"test.*suite|e2e|end.to.end|"
    r"optimize|optimise|performance|"
    r"merge|rebase|resolve.*conflict|"
    # Fix/debug (agentic tasks)
    r"fix|debug|corrige|repare|"
    # VPS/server/deploy signals
    r"vps|server|serveur|systemd|docker|nginx|" r"deploy|deployer|redemarr|restart|"
    # File operations
    r"edit.*file|create.*file|delete.*file|"
    r"modifie.*fichier|cree.*fichier|supprime.*fichier|"
    # Run/execute
    r"run|execute|lance|exec|"
    # Git operations
    r"git|commit|push|pull|branch|"
    # Package managers
    r"install|npm|pip|poetry|pnpm|yarn|bun|"
    # Code action words
    r"write.*code|ecris.*code|"
    r"add.*function|ajoute.*fonction|"
    r"modify|modifie|change.*code|update.*code|"
    r"write.*test|ecris.*test|"
    r"review|relis|regarde.*code|check.*code|"
    # Error investigation
    r"error|erreur|bug|issue|problem|probleme" r")\b",
    re.IGNORECASE,
)

# Sonnet signals: knowledge queries, explanations, medium tasks
_SONNET_KEYWORDS = re.compile(
    r"\b("
    r"explain|explique|"
    r"analyse|analyze|"
    r"how.*work|comment.*march|"
    r"what.*does|que.*fait|"
    r"what.*is|c'est quoi|qu'est.ce que|"
    r"summarize|resume|"
    r"convert|converti|transform|"
    r"compare|difference|"
    r"config|setup|"
    # Moved from Haiku: knowledge queries
    r"translate|tradui|traduction|"
    r"calculate|calcul|combien|"
    r"list|liste|"
    r"define|definition|"
    r"who|when|where|qui|quand|ou|"
    r"meaning|signifi"
    r")\b",
    re.IGNORECASE,
)

# Haiku signals: ONLY greetings, thanks, yes/no, weather/time, bye
_HAIKU_KEYWORDS = re.compile(
    r"\b("
    r"hello|salut|bonjour|hi|hey|coucou|"
    r"thanks|merci|thank|"
    r"yes|no|oui|non|ok|d'accord|"
    r"weather|meteo|time|heure|"
    r"bye|au revoir|a plus|ciao"
    r")\b",
    re.IGNORECASE,
)

# ── Advanced signals ─────────────────────────────────────────────────

# Tool-use signal: file paths, shell commands, code markers → forces Opus
_TOOL_USE_SIGNAL = re.compile(
    r"(?:"
    r"~/|/home/|/etc/|/var/"  # Absolute paths
    r"|\.py\b|\.ts\b|\.js\b|\.tsx\b|\.jsx\b|\.go\b|\.rs\b|\.sol\b"  # Extensions
    r"|src/|tests/|packages/"  # Project paths
    r"|\bsudo\b|\bsystemctl\b|\bapt\b|\bdpkg\b"  # System commands
    r"|\bmake\b|\bpytest\b|\bforge\b"  # Build/test commands
    r"|\$\(|\$\{|&&|\|\|"  # Shell operators
    r")",
    re.IGNORECASE,
)

# Code pattern: inline code, function defs, imports → min Sonnet
_CODE_PATTERN = re.compile(
    r"(?:"
    r"```[\s\S]{10,}```"  # Fenced code blocks
    r"|`[^`]{3,}`"  # Inline code
    r"|\b(?:def |class |function |import |from .+ import|const |let |var )"
    r"|\b(?:async def |async function )"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class RouteResult:
    """Result of model routing decision."""

    model: str
    reason: str
    override: bool = False
    scores: Dict[str, int] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return MODEL_NAMES.get(self.model, self.model)


class ModelRouter:
    """Analyze messages and select the appropriate Claude model."""

    @classmethod
    def route(
        cls,
        message_text: str,
        has_attachment: bool = False,
        has_photo: bool = False,
    ) -> RouteResult:
        """Determine the best model for a given message.

        Returns RouteResult with model ID, reason, override flag, and scores.
        """
        try:
            return cls._do_route(message_text, has_attachment, has_photo)
        except Exception as e:
            logger.warning("Router error, falling back to default", error=str(e))
            return RouteResult(DEFAULT_MODEL, "fallback (router error)")

    @classmethod
    def check_override(cls, text: str) -> Tuple[Optional[str], str]:
        """Check if text starts with a model override prefix.

        Returns (model_id, cleaned_text) or (None, original_text).
        """
        stripped = text.strip()
        for prefix, model in _OVERRIDE_PREFIXES.items():
            if stripped.lower().startswith(prefix):
                rest = stripped[len(prefix) :].strip()
                if rest:
                    return model, rest
                # Bare "/opus" with no message — ignore override
                return None, text
        return None, text

    @classmethod
    def _do_route(
        cls,
        text: str,
        has_attachment: bool,
        has_photo: bool,
    ) -> RouteResult:
        """Core routing logic."""
        # File/photo attachments → always Opus (agentic tool use)
        if has_attachment:
            return RouteResult(
                OPUS,
                "file attachment (agentic)",
                scores={"opus": 99, "sonnet": 0, "haiku": 0},
            )
        if has_photo:
            return RouteResult(
                OPUS,
                "photo (needs Read tool)",
                scores={"opus": 99, "sonnet": 0, "haiku": 0},
            )

        # Score-based approach: accumulate complexity signals
        opus_score = 0
        sonnet_score = 0
        haiku_score = 0

        # ── Length signals (NO haiku bonus for short messages) ────
        length = len(text)
        if length < 200:
            pass  # No bonus — short != trivial
        elif length < 500:
            sonnet_score += 1
        else:
            opus_score += 2

        # ── Tool-use signal → forces Opus ─────────────────────────
        tool_hits = len(_TOOL_USE_SIGNAL.findall(text))
        if tool_hits > 0:
            opus_score += 3 + min(tool_hits - 1, 2)  # 3-5 points

        # ── Code pattern → min Sonnet ─────────────────────────────
        code_hits = len(_CODE_PATTERN.findall(text))
        if code_hits > 0:
            sonnet_score += 2
            if code_hits >= 2:
                opus_score += 2

        # ── Keyword matching ──────────────────────────────────────
        opus_hits = len(_OPUS_KEYWORDS.findall(text))
        sonnet_hits = len(_SONNET_KEYWORDS.findall(text))
        haiku_hits = len(_HAIKU_KEYWORDS.findall(text))

        opus_score += opus_hits * 3
        sonnet_score += sonnet_hits * 2
        haiku_score += haiku_hits * 2

        # ── Multi-line messages → more complex ────────────────────
        line_count = text.count("\n")
        if line_count > 10:
            opus_score += 2
        elif line_count > 3:
            sonnet_score += 1

        # ── File path references → agentic ────────────────────────
        path_refs = len(
            re.findall(r"(?:src/|\.py|\.ts|\.js|\.tsx|\.jsx|\.go|\.rs)\b", text)
        )
        if path_refs >= 3:
            opus_score += 3
        elif path_refs >= 1:
            opus_score += 1

        scores = {"opus": opus_score, "sonnet": sonnet_score, "haiku": haiku_score}

        # ── Decision ──────────────────────────────────────────────
        # Haiku only wins if: haiku > 0, opus == 0, sonnet <= 1
        if haiku_score > 0 and opus_score == 0 and sonnet_score <= 1:
            reason = f"simple (scores: o={opus_score} s={sonnet_score} h={haiku_score})"
            return RouteResult(HAIKU, reason, scores=scores)

        if opus_score > sonnet_score and opus_score > haiku_score:
            reason = (
                f"complex (scores: o={opus_score}" f" s={sonnet_score} h={haiku_score})"
            )
            return RouteResult(OPUS, reason, scores=scores)

        if sonnet_score > 0:
            reason = f"medium (scores: o={opus_score} s={sonnet_score} h={haiku_score})"
            return RouteResult(SONNET, reason, scores=scores)

        # No clear signal → default to Sonnet (safe middle ground)
        return RouteResult(
            DEFAULT_MODEL,
            "no clear signal, default",
            scores=scores,
        )
