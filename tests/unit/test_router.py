"""Tests for the ModelRouter — verifies routing decisions."""

from src.bot.router import HAIKU, OPUS, SONNET, ModelRouter, RouteResult


class TestHaikuRouting:
    """Haiku should only handle truly trivial messages."""

    def test_salut_routes_to_haiku(self) -> None:
        result = ModelRouter.route("Salut")
        assert result.model == HAIKU

    def test_merci_beaucoup_routes_to_haiku(self) -> None:
        result = ModelRouter.route("Merci beaucoup")
        assert result.model == HAIKU

    def test_quelle_heure_routes_to_haiku(self) -> None:
        result = ModelRouter.route("Quelle heure est-il")
        assert result.model == HAIKU

    def test_hello_routes_to_haiku(self) -> None:
        result = ModelRouter.route("Hello")
        assert result.model == HAIKU

    def test_oui_routes_to_haiku(self) -> None:
        result = ModelRouter.route("oui")
        assert result.model == HAIKU

    def test_bye_routes_to_haiku(self) -> None:
        result = ModelRouter.route("bye")
        assert result.model == HAIKU


class TestSonnetRouting:
    """Sonnet handles knowledge queries, explanations, calculations."""

    def test_explain_typescript_routes_to_sonnet(self) -> None:
        result = ModelRouter.route("Explain TypeScript")
        assert result.model == SONNET

    def test_what_is_closure_routes_to_sonnet(self) -> None:
        result = ModelRouter.route("What is a closure?")
        assert result.model == SONNET

    def test_combien_fait_routes_to_sonnet(self) -> None:
        result = ModelRouter.route("Combien fait 12 x 8 ?")
        assert result.model == SONNET

    def test_translate_routes_to_sonnet(self) -> None:
        result = ModelRouter.route("Translate this to French")
        assert result.model == SONNET

    def test_define_routes_to_sonnet(self) -> None:
        result = ModelRouter.route("Define polymorphism")
        assert result.model == SONNET

    def test_who_invented_routes_to_sonnet(self) -> None:
        result = ModelRouter.route("Who invented Python?")
        assert result.model == SONNET

    def test_list_routes_to_sonnet(self) -> None:
        result = ModelRouter.route("List the main frameworks")
        assert result.model == SONNET


class TestOpusRouting:
    """Opus handles agentic tasks, code work, deploy, fix/debug."""

    def test_fix_bug_routes_to_opus(self) -> None:
        result = ModelRouter.route("Fix bug in auth")
        assert result.model == OPUS

    def test_check_file_path_routes_to_opus(self) -> None:
        result = ModelRouter.route("Check src/bot/router.py")
        assert result.model == OPUS

    def test_deploy_routes_to_opus(self) -> None:
        result = ModelRouter.route("Deploy the new version")
        assert result.model == OPUS

    def test_regarde_code_routes_to_opus(self) -> None:
        result = ModelRouter.route("regarde le code dans Astro.fun")
        assert result.model == OPUS

    def test_debug_routes_to_opus(self) -> None:
        result = ModelRouter.route("Debug this crash")
        assert result.model == OPUS

    def test_git_push_routes_to_opus(self) -> None:
        result = ModelRouter.route("git push to main")
        assert result.model == OPUS

    def test_npm_install_routes_to_opus(self) -> None:
        result = ModelRouter.route("npm install express")
        assert result.model == OPUS

    def test_systemctl_routes_to_opus(self) -> None:
        result = ModelRouter.route("systemctl restart my-service")
        assert result.model == OPUS

    def test_docker_routes_to_opus(self) -> None:
        result = ModelRouter.route("docker compose up")
        assert result.model == OPUS

    def test_implement_routes_to_opus(self) -> None:
        result = ModelRouter.route("Implement the login page")
        assert result.model == OPUS

    def test_file_attachment_routes_to_opus(self) -> None:
        result = ModelRouter.route("review this", has_attachment=True)
        assert result.model == OPUS

    def test_photo_routes_to_opus(self) -> None:
        result = ModelRouter.route("", has_photo=True)
        assert result.model == OPUS

    def test_home_path_routes_to_opus(self) -> None:
        result = ModelRouter.route("Look at ~/Claude/Astro.fun")
        assert result.model == OPUS

    def test_corrige_routes_to_opus(self) -> None:
        result = ModelRouter.route("Corrige cette erreur")
        assert result.model == OPUS

    def test_run_tests_routes_to_opus(self) -> None:
        result = ModelRouter.route("Run the tests")
        assert result.model == OPUS

    def test_refactor_routes_to_opus(self) -> None:
        result = ModelRouter.route("Refactor the auth module")
        assert result.model == OPUS

    def test_write_test_routes_to_opus(self) -> None:
        result = ModelRouter.route("Write a test for login")
        assert result.model == OPUS


class TestOverrides:
    """Manual override prefixes."""

    def test_opus_override(self) -> None:
        model, text = ModelRouter.check_override("/opus What is Python?")
        assert model == OPUS
        assert text == "What is Python?"

    def test_sonnet_override(self) -> None:
        model, text = ModelRouter.check_override("/sonnet explain this")
        assert model == SONNET
        assert text == "explain this"

    def test_haiku_override(self) -> None:
        model, text = ModelRouter.check_override("/haiku hi")
        assert model == HAIKU
        assert text == "hi"

    def test_bare_override_ignored(self) -> None:
        model, text = ModelRouter.check_override("/opus")
        assert model is None

    def test_no_override(self) -> None:
        model, text = ModelRouter.check_override("Hello there")
        assert model is None
        assert text == "Hello there"


class TestRouteResult:
    """RouteResult dataclass."""

    def test_display_name(self) -> None:
        result = RouteResult(OPUS, "test")
        assert result.display_name == "Opus"

    def test_scores_present(self) -> None:
        result = ModelRouter.route("Fix this bug")
        assert "opus" in result.scores
        assert "sonnet" in result.scores
        assert "haiku" in result.scores

    def test_default_fallback(self) -> None:
        """No signals at all → default to Sonnet."""
        result = ModelRouter.route("")
        assert result.model == SONNET


class TestEdgeCases:
    """Edge cases and mixed signals."""

    def test_short_message_no_haiku_bonus(self) -> None:
        """Short messages without haiku keywords should NOT go to Haiku."""
        result = ModelRouter.route("Explain TypeScript")
        assert result.model != HAIKU

    def test_code_block_forces_min_sonnet(self) -> None:
        """Code blocks should at least get Sonnet."""
        result = ModelRouter.route("```python\ndef foo():\n    return 42\n```")
        assert result.model in (SONNET, OPUS)

    def test_multiline_boosts_complexity(self) -> None:
        """Many lines push toward opus/sonnet."""
        text = "\n".join([f"Line {i}" for i in range(15)])
        result = ModelRouter.route(text)
        assert result.model != HAIKU

    def test_long_text_opus(self) -> None:
        """Very long text → opus."""
        result = ModelRouter.route("x " * 300)
        assert result.model in (SONNET, OPUS)

    def test_haiku_blocked_when_opus_keywords_present(self) -> None:
        """Even with haiku keywords, opus keywords should win."""
        result = ModelRouter.route("Salut, fix this bug please merci")
        assert result.model == OPUS
