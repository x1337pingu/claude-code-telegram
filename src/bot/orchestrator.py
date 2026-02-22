"""Message orchestrator — single entry point for all Telegram updates.

Routes messages based on agentic vs classic mode. In agentic mode, provides
a minimal conversational interface (3 commands, no inline keyboards). In
classic mode, delegates to existing full-featured handlers.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional

import structlog
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..claude.exceptions import ClaudeToolValidationError
from ..config.settings import Settings
from .personality import Personality
from .progress import ProgressReporter
from .router import MODEL_NAMES, OPUS, SONNET, ModelRouter, RouteResult
from .utils.html_format import escape_html
from .vps_context import get_vps_context

logger = structlog.get_logger()

# Ambiguity threshold: if top two model scores are within this range, ask user
_CLARIFICATION_THRESHOLD = 2


class MessageOrchestrator:
    """Routes messages based on mode. Single entry point for all Telegram updates."""

    def __init__(self, settings: Settings, deps: Dict[str, Any]):
        self.settings = settings
        self.deps = deps

    def _inject_deps(self, handler: Callable) -> Callable:  # type: ignore[type-arg]
        """Wrap handler to inject dependencies into context.bot_data."""

        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            for key, value in self.deps.items():
                context.bot_data[key] = value
            context.bot_data["settings"] = self.settings
            await handler(update, context)

        return wrapped

    def register_handlers(self, app: Application) -> None:
        """Register handlers based on mode."""
        if self.settings.agentic_mode:
            self._register_agentic_handlers(app)
        else:
            self._register_classic_handlers(app)

    def _register_agentic_handlers(self, app: Application) -> None:
        """Register minimal agentic handlers: commands + text/file/photo."""
        # Commands
        for cmd, handler in [
            ("start", self.agentic_start),
            ("new", self.agentic_new),
            ("status", self.agentic_status),
            # Model override commands
            ("opus", self.agentic_model_override),
            ("sonnet", self.agentic_model_override),
            ("haiku", self.agentic_model_override),
        ]:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        # Text messages -> Claude
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(self.agentic_text),
            ),
            group=10,
        )

        # File uploads -> Claude
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(self.agentic_document)
            ),
            group=10,
        )

        # Photo uploads -> Claude
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(self.agentic_photo)),
            group=10,
        )

        # Callback handlers: cd: for project selection, clarify: for model choice
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_callback),
                pattern=r"^cd:",
            )
        )
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_clarification_callback),
                pattern=r"^clarify:",
            )
        )

        logger.info(
            "Agentic handlers registered (6 commands + text/file/photo + 2 callbacks)"
        )

    def _register_classic_handlers(self, app: Application) -> None:
        """Register full classic handler set (moved from core.py)."""
        from .handlers import callback, command, message

        handlers = [
            ("start", command.start_command),
            ("help", command.help_command),
            ("new", command.new_session),
            ("continue", command.continue_session),
            ("end", command.end_session),
            ("ls", command.list_files),
            ("cd", command.change_directory),
            ("pwd", command.print_working_directory),
            ("projects", command.show_projects),
            ("status", command.session_status),
            ("export", command.export_session),
            ("actions", command.quick_actions),
            ("git", command.git_command),
        ]

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(message.handle_text_message),
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(message.handle_document)
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(message.handle_photo)),
            group=10,
        )
        app.add_handler(
            CallbackQueryHandler(self._inject_deps(callback.handle_callback_query))
        )

        logger.info("Classic handlers registered (13 commands + full handler set)")

    async def get_bot_commands(self) -> list:  # type: ignore[type-arg]
        """Return bot commands appropriate for current mode."""
        if self.settings.agentic_mode:
            return [
                BotCommand("start", "Start the bot"),
                BotCommand("new", "Start a fresh session"),
                BotCommand("status", "Show session status"),
                BotCommand("opus", "Force Opus model"),
                BotCommand("sonnet", "Force Sonnet model"),
                BotCommand("haiku", "Force Haiku model"),
            ]
        else:
            return [
                BotCommand("start", "Start bot and show help"),
                BotCommand("help", "Show available commands"),
                BotCommand("new", "Clear context and start fresh session"),
                BotCommand("continue", "Explicitly continue last session"),
                BotCommand("end", "End current session and clear context"),
                BotCommand("ls", "List files in current directory"),
                BotCommand("cd", "Change directory (resumes project session)"),
                BotCommand("pwd", "Show current directory"),
                BotCommand("projects", "Show all projects"),
                BotCommand("status", "Show session status"),
                BotCommand("export", "Export current session"),
                BotCommand("actions", "Show quick actions"),
                BotCommand("git", "Git repository commands"),
            ]

    # --- Agentic handlers ---

    async def agentic_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Brief welcome, no buttons."""
        user = update.effective_user
        if not user:
            return
        message = update.message
        if not message:
            return
        if context.user_data is None:
            context.user_data = {}

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )

        safe_name = escape_html(user.first_name)
        await message.reply_text(
            Personality.start(safe_name, str(current_dir)),
            parse_mode="HTML",
        )

    async def agentic_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reset session, one-line confirmation."""
        message = update.message
        if not message:
            return
        if context.user_data is None:
            context.user_data = {}

        context.user_data["claude_session_id"] = None
        context.user_data["session_started"] = True

        await message.reply_text(Personality.reset())

    async def agentic_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Compact one-line status, no buttons."""
        user = update.effective_user
        if not user:
            return
        message = update.message
        if not message:
            return
        if context.user_data is None:
            context.user_data = {}

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = str(current_dir)

        session_id = context.user_data.get("claude_session_id")
        session_status = "active" if session_id else "none"

        # Cost info
        cost_str = ""
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            try:
                user_status = rate_limiter.get_user_status(user.id)
                cost_usage = user_status.get("cost_usage", {})
                current_cost = cost_usage.get("current", 0.0)
                cost_str = f" · Cost: ${current_cost:.2f}"
            except Exception:
                pass

        await message.reply_text(
            f"\U0001f4c2 {dir_display} \u00b7 Session: {session_status}{cost_str}"
        )

    async def agentic_model_override(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /opus, /sonnet, /haiku <message> commands."""
        from .router import HAIKU, OPUS, SONNET

        message = update.message
        if not message:
            return
        if context.user_data is None:
            context.user_data = {}

        text: str = message.text or ""

        cmd = text.split()[0].lower().lstrip("/")
        model_map = {"opus": OPUS, "sonnet": SONNET, "haiku": HAIKU}
        model = model_map.get(cmd, OPUS)

        # Extract the message after the command
        parts = text.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            name = MODEL_NAMES.get(model, cmd)
            await message.reply_text(
                f"Modele {name} selectionne. Ecris ton message apres la "
                f"commande, ex: /{cmd} ta question ici"
            )
            return

        # Inject into agentic_text flow with forced model
        message_text = parts[1]
        # Store override in user_data so agentic_text can pick it up
        # (PTB Message objects are immutable, so we pass text via user_data)
        context.user_data["_model_override"] = model
        context.user_data["_text_override"] = message_text
        await self.agentic_text(update, context)

    async def agentic_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Direct Claude passthrough. Simple progress. No suggestions."""
        user = update.effective_user
        if not user:
            return
        message = update.message
        if not message:
            return
        if context.user_data is None:
            context.user_data = {}

        user_id = user.id

        # Clear any stale pending clarification
        context.user_data.pop("_pending_clarification", None)

        # Use text override if set by model override command, else original
        text_override = context.user_data.pop("_text_override", None)
        message_text: str = text_override or message.text or ""

        # Check for model override (command or inline prefix)
        override_model = context.user_data.pop("_model_override", None)
        if not override_model:
            override_model, clean_text = ModelRouter.check_override(message_text)
            if override_model:
                message_text = clean_text

        # Route to appropriate model
        route = ModelRouter.route(message_text)
        if override_model:
            route.model = override_model
            route.reason = "user override"
            route.override = True

        logger.info(
            "Agentic text message",
            user_id=user_id,
            message_length=len(message_text),
            model=route.display_name,
            route_reason=route.reason,
        )

        # Check if we should ask for clarification (ambiguous routing)
        if not route.override and self._should_ask_clarification(route):
            await self._ask_model_clarification(update, context, message_text, route)
            return

        # Rate limit check
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            allowed, limit_message = await rate_limiter.check_rate_limit(user_id, 0.001)
            if not allowed:
                await message.reply_text(Personality.bot_rate_limit())
                return

        await self._execute_claude_and_respond(
            chat_id=message.chat_id,
            reply_to_message_id=message.message_id,
            message_text=message_text,
            model=route.model,
            user_id=user_id,
            context=context,
            update=update,
            route_display=route.display_name,
        )

    async def _execute_claude_and_respond(
        self,
        chat_id: int,
        reply_to_message_id: int,
        message_text: str,
        model: str,
        user_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        update: Update,
        route_display: str,
    ) -> None:
        """Core execution: send to Claude, format response, send back.

        Shared by agentic_text and clarification callback.
        """
        if context.user_data is None:
            context.user_data = {}

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        display_name = MODEL_NAMES.get(model, model)
        progress_msg = await context.bot.send_message(
            chat_id=chat_id,
            text=f"{Personality.working()} [{display_name}]",
        )

        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Progress reporter for periodic updates
        reporter = ProgressReporter(progress_msg)
        reporter.start_periodic_updates()

        # Enrich prompt with context
        enhanced_prompt = self._enhance_prompt(
            message_text, current_dir, session_id, model
        )

        success = True
        formatted_messages: List[Any] = []
        try:
            claude_response = await claude_integration.run_command(
                prompt=enhanced_prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=reporter.on_stream_update,
                model=model,
            )

            context.user_data["claude_session_id"] = claude_response.session_id

            # Track directory changes
            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            # Store interaction
            storage = context.bot_data.get("storage")
            if storage:
                try:
                    await storage.save_claude_interaction(
                        user_id=user_id,
                        session_id=claude_response.session_id,
                        prompt=message_text,
                        response=claude_response,
                        ip_address=None,
                    )
                except Exception as e:
                    logger.warning("Failed to log interaction", error=str(e))

            # Format response (no reply_markup — strip keyboards)
            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

        except ClaudeToolValidationError as e:
            success = False
            logger.error("Tool validation error", error=str(e), user_id=user_id)
            from .utils.formatting import FormattedMessage

            formatted_messages = [FormattedMessage(str(e), parse_mode="HTML")]

        except Exception as e:
            success = False
            logger.error("Claude integration failed", error=str(e), user_id=user_id)
            from .handlers.message import _format_error_message
            from .utils.formatting import FormattedMessage

            formatted_messages = [
                FormattedMessage(_format_error_message(str(e)), parse_mode="HTML")
            ]
        finally:
            await reporter.stop()

        await progress_msg.delete()

        for i, message in enumerate(formatted_messages):
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=message.text,
                    parse_mode=message.parse_mode,
                    reply_markup=None,
                    reply_to_message_id=(reply_to_message_id if i == 0 else None),
                )
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(
                    "Failed to send HTML response, retrying as plain text",
                    error=str(e),
                    message_index=i,
                )
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=message.text,
                        reply_markup=None,
                        reply_to_message_id=(reply_to_message_id if i == 0 else None),
                    )
                except Exception:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="Failed to send response. Please try again.",
                        reply_to_message_id=(reply_to_message_id if i == 0 else None),
                    )

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command="text_message",
                args=[message_text[:100]],
                success=success,
            )

    async def agentic_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process file upload -> Claude, minimal chrome."""
        user = update.effective_user
        if not user:
            return
        message = update.message
        if not message:
            return
        document = message.document
        if not document:
            return
        if context.user_data is None:
            context.user_data = {}

        user_id = user.id

        logger.info(
            "Agentic document upload",
            user_id=user_id,
            filename=document.file_name,
        )

        # Security validation
        security_validator = context.bot_data.get("security_validator")
        if security_validator:
            valid, error = security_validator.validate_filename(document.file_name)
            if not valid:
                await message.reply_text(f"File rejected: {error}")
                return

        # Size check
        max_size = 10 * 1024 * 1024
        if document.file_size and document.file_size > max_size:
            await message.reply_text(
                f"File too large ({document.file_size / 1024 / 1024:.1f}MB). Max: 10MB."
            )
            return

        progress_msg = await message.reply_text(Personality.working())

        # Try enhanced file handler, fall back to basic
        features = context.bot_data.get("features")
        file_handler = features.get_file_handler() if features else None
        prompt: Optional[str] = None

        if file_handler:
            try:
                processed_file = await file_handler.handle_document_upload(
                    document,
                    user_id,
                    message.caption or "Please review this file:",
                )
                prompt = processed_file.prompt
            except Exception:
                file_handler = None

        if not file_handler:
            file = await document.get_file()
            file_bytes = await file.download_as_bytearray()
            try:
                content = file_bytes.decode("utf-8")
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                caption = message.caption or "Please review this file:"
                prompt = (
                    f"{caption}\n\n**File:** `{document.file_name}`\n\n"
                    f"```\n{content}\n```"
                )
            except UnicodeDecodeError:
                await progress_msg.edit_text(
                    "Unsupported file format. Must be text-based (UTF-8)."
                )
                return

        # Process with Claude
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Files always route to Opus (agentic tool use)
        doc_route = ModelRouter.route(prompt or "", has_attachment=True)
        logger.info(
            "Agentic document",
            user_id=user_id,
            model=doc_route.display_name,
            route_reason=doc_route.reason,
        )

        reporter = ProgressReporter(progress_msg)
        reporter.start_periodic_updates()

        # Enrich prompt with VPS context
        enhanced_prompt = self._enhance_prompt(
            prompt or "", current_dir, session_id, doc_route.model
        )

        try:
            claude_response = await claude_integration.run_command(
                prompt=enhanced_prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=reporter.on_stream_update,
                model=doc_route.model,
            )
            context.user_data["claude_session_id"] = claude_response.session_id

            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            await progress_msg.delete()

            for i, fmt_msg in enumerate(formatted_messages):
                await message.reply_text(
                    fmt_msg.text,
                    parse_mode=fmt_msg.parse_mode,
                    reply_markup=None,
                    reply_to_message_id=(message.message_id if i == 0 else None),
                )
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(
                _format_error_message(str(e)), parse_mode="HTML"
            )
            logger.error(
                "Claude file processing failed",
                error=str(e),
                user_id=user_id,
            )
        finally:
            await reporter.stop()

    async def agentic_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process photo -> Claude by saving to disk and referencing the file."""
        import os
        import uuid as _uuid

        user = update.effective_user
        if not user:
            return
        message = update.message
        if not message:
            return
        if not message.photo:
            return
        if context.user_data is None:
            context.user_data = {}

        user_id = user.id

        # Photos always route to Opus (needs Read tool)
        photo_route = ModelRouter.route("", has_photo=True)
        logger.info(
            "Agentic photo",
            user_id=user_id,
            model=photo_route.display_name,
            route_reason=photo_route.reason,
        )

        progress_msg = await message.reply_text(Personality.working())

        try:
            # Download the highest-resolution photo
            photo = message.photo[-1]
            file = await photo.get_file()
            image_bytes = await file.download_as_bytearray()

            # Detect format
            if image_bytes[:4] == b"\x89PNG":
                ext = "png"
            elif image_bytes[:3] == b"\xff\xd8\xff":
                ext = "jpg"
            elif image_bytes[:4] == b"RIFF":
                ext = "webp"
            else:
                ext = "jpg"

            # Save to a temp file Claude can read
            images_dir = "/home/agent/claude-code-telegram/data/images"
            os.makedirs(images_dir, exist_ok=True)
            image_filename = f"{_uuid.uuid4().hex[:12]}.{ext}"
            image_path = os.path.join(images_dir, image_filename)

            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            # Build prompt that tells Claude to read the image file
            caption = message.caption or ""
            if caption:
                prompt = (
                    f'The user sent a photo with this message: "{caption}"\n\n'
                    f"The image has been saved at: {image_path}\n"
                    f"Please read this image file using your Read tool to view it, "
                    f"then analyze it and respond to the user's request."
                )
            else:
                prompt = (
                    f"The user sent a photo.\n\n"
                    f"The image has been saved at: {image_path}\n"
                    f"Please read this image file using your Read tool to view it, "
                    f"then describe and analyze what you see."
                )

            claude_integration = context.bot_data.get("claude_integration")
            if not claude_integration:
                await progress_msg.edit_text(
                    "Claude integration not available. Check configuration."
                )
                return

            current_dir = context.user_data.get(
                "current_directory", self.settings.approved_directory
            )
            session_id = context.user_data.get("claude_session_id")

            # Enrich prompt with VPS context
            enhanced_prompt = self._enhance_prompt(
                prompt, current_dir, session_id, photo_route.model
            )

            reporter = ProgressReporter(progress_msg)
            reporter.start_periodic_updates()

            try:
                claude_response = await claude_integration.run_command(
                    prompt=enhanced_prompt,
                    working_directory=current_dir,
                    user_id=user_id,
                    session_id=session_id,
                    on_stream=reporter.on_stream_update,
                    model=photo_route.model,
                )
            finally:
                await reporter.stop()

            # Cleanup temp image
            try:
                os.remove(image_path)
            except OSError:
                pass
            context.user_data["claude_session_id"] = claude_response.session_id

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            await progress_msg.delete()

            for i, fmt_msg in enumerate(formatted_messages):
                await message.reply_text(
                    fmt_msg.text,
                    parse_mode=fmt_msg.parse_mode,
                    reply_markup=None,
                    reply_to_message_id=(message.message_id if i == 0 else None),
                )
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(
                _format_error_message(str(e)), parse_mode="HTML"
            )
            logger.error(
                "Claude photo processing failed", error=str(e), user_id=user_id
            )

    # --- Prompt enrichment ---

    def _enhance_prompt(
        self,
        message_text: str,
        current_dir: Any,
        session_id: Optional[str],
        model: str = SONNET,
    ) -> str:
        """Enrich user prompt with VPS context. Original message is always preserved."""
        vps_context = get_vps_context(model)
        context_parts = [vps_context, f"[Working directory: {current_dir}]"]
        if session_id:
            context_parts.append("[Continuing existing session]")
        else:
            context_parts.append("[New session]")
        return "\n".join(context_parts) + "\n\n" + message_text

    # --- Clarification flow ---

    @staticmethod
    def _should_ask_clarification(route: RouteResult) -> bool:
        """Check if routing scores are ambiguous enough to ask the user."""
        scores = route.scores
        if not scores:
            return False

        # Get sorted scores (descending)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) < 2:
            return False

        top, second = sorted_scores[0], sorted_scores[1]

        # Both must have some signal (> 0) and be close
        if top == 0 or second == 0:
            return False

        return (top - second) <= _CLARIFICATION_THRESHOLD

    async def _ask_model_clarification(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        message_text: str,
        route: RouteResult,
    ) -> None:
        """Send inline keyboard asking user to pick a model."""
        message = update.message
        if not message:
            return
        user = update.effective_user
        if not user:
            return
        if context.user_data is None:
            context.user_data = {}

        # Determine the two candidate models from scores
        scores = route.scores
        sorted_models = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)
        top_two = sorted_models[:2]

        # Map score keys to model IDs
        model_id_map = {"opus": OPUS, "sonnet": SONNET}

        buttons = []
        for key in top_two:
            model_id = model_id_map.get(key)
            if not model_id:
                continue
            name = MODEL_NAMES.get(model_id, key)
            label = f"{name} (rapide)" if model_id == SONNET else f"{name} (complet)"
            buttons.append(
                InlineKeyboardButton(label, callback_data=f"clarify:{model_id}")
            )

        if len(buttons) < 2:
            # Fallback: just proceed with routed model
            await self._execute_claude_and_respond(
                chat_id=message.chat_id,
                reply_to_message_id=message.message_id,
                message_text=message_text,
                model=route.model,
                user_id=user.id,
                context=context,
                update=update,
                route_display=route.display_name,
            )
            return

        keyboard = InlineKeyboardMarkup([buttons])

        # Store pending message for callback
        context.user_data["_pending_clarification"] = {
            "message_text": message_text,
            "reply_to_message_id": message.message_id,
            "chat_id": message.chat_id,
        }

        await message.reply_text(
            Personality.clarify_model(message_text),
            reply_markup=keyboard,
        )

    async def _handle_clarification_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process user's model choice from clarification keyboard."""
        query = update.callback_query
        if not query:
            return
        if not query.data:
            return
        user = update.effective_user
        if not user:
            return
        if context.user_data is None:
            context.user_data = {}

        await query.answer()

        # Extract chosen model from callback data
        _, chosen_model = query.data.split(":", 1)
        chosen_name = MODEL_NAMES.get(chosen_model, chosen_model)

        # Retrieve pending message
        pending = context.user_data.pop("_pending_clarification", None)
        if not pending:
            await query.edit_message_text("Session expiree. Renvoie ton message.")
            return

        # Acknowledge choice by editing the clarification message
        await query.edit_message_text(Personality.clarify_acknowledged(chosen_name))

        # Rate limit check
        user_id = user.id
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            allowed, limit_message = await rate_limiter.check_rate_limit(user_id, 0.001)
            if not allowed:
                await context.bot.send_message(
                    chat_id=pending["chat_id"],
                    text=Personality.bot_rate_limit(),
                )
                return

        await self._execute_claude_and_respond(
            chat_id=pending["chat_id"],
            reply_to_message_id=pending["reply_to_message_id"],
            message_text=pending["message_text"],
            model=chosen_model,
            user_id=user_id,
            context=context,
            update=update,
            route_display=chosen_name,
        )

    async def _agentic_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle cd: callbacks (pattern-filtered by registration)."""
        query = update.callback_query
        if not query:
            return
        if not query.data:
            return

        await query.answer()

        data = query.data
        _, param = data.split(":", 1)

        from .handlers.callback import handle_cd_callback

        await handle_cd_callback(query, param, context)
