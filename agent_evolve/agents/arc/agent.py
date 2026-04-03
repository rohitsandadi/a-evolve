"""ARC-AGI-3 agent -- plays interactive games via the arc-agi toolkit.

Adapted from arcprize/ARC-AGI-3-Agents and symbolica-ai/ARC-AGI-3-Agents.
Separates game interaction code from the evolvable workspace (prompts/skills/memory).

Key adaptations:
- Frame class with diff/render helpers (from Symbolica's scope/frame.py)
- Grid-to-image rendering for multimodal input (from multimodal.py)
- Game reference prompt adapted from Symbolica's GAME_REFERENCE
- Proper FrameData handling from the official Agent base class
- a-evolve workspace integration for prompt/skill/memory evolution
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from strands import Agent
from strands.models import BedrockModel

from ...protocol.base_agent import BaseAgent
from ...types import Task, Trajectory
from .colors import COLOR_LEGEND, COLOR_NAMES
from .frame import DiffRegion, Frame

logger = logging.getLogger(__name__)

os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")


class ArcAgent(BaseAgent):
    """Evolvable agent for ARC-AGI-3 interactive games.

    Drives the arc-agi game environment through a strands Agent with
    tool-based interaction. The LLM observes grid states (text + optional
    images), reasons about game mechanics, and chooses actions.

    The workspace provides the evolvable system prompt, skills, and memory.
    Frame helpers provide rich grid analysis (diff, color counts, bounding
    boxes) that feed into observations.
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        model_id: str = "us.anthropic.claude-opus-4-6-v1",
        region: str = "us-west-2",
        max_tokens: int = 16384,
        max_actions: int = 5000,
    ):
        super().__init__(workspace_dir)
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.max_actions = max_actions

    def solve(self, task: Task) -> Trajectory:
        """Play an ARC-AGI-3 game and return the trajectory."""
        game_id = task.metadata.get("game_id", task.id)
        max_actions = task.metadata.get("max_actions", self.max_actions)

        logger.info("Playing ARC-AGI-3 game: %s (budget: %d actions)", game_id, max_actions)

        try:
            return self._solve_game(task, game_id, max_actions)
        except ImportError as e:
            logger.error("arc-agi package not installed: %s", e)
            return Trajectory(
                task_id=task.id,
                output=json.dumps({
                    "game_id": game_id,
                    "error": f"arc-agi package not installed: {e}",
                    "game_completed": False,
                    "levels_completed": 0,
                    "total_levels": 0,
                    "total_actions": 0,
                    "score": 0.0,
                }),
                steps=[{"error": str(e)}],
            )

    def _solve_game(self, task: Task, game_id: str, max_actions: int) -> Trajectory:
        """Play the game using strands tools + Frame helpers for rich observation."""
        import arc_agi
        from arcengine import FrameData, GameAction, GameState

        # Initialize arcade
        arcade_kwargs: dict[str, Any] = {}
        api_key = task.metadata.get("api_key")
        if api_key:
            arcade_kwargs["arc_api_key"] = api_key
        op_mode = task.metadata.get("operation_mode", "normal")
        if op_mode != "normal":
            from arc_agi import OperationMode
            arcade_kwargs["operation_mode"] = getattr(OperationMode, op_mode.upper())

        arcade = arc_agi.Arcade(**arcade_kwargs)
        env = arcade.make(game_id, render_mode=None)

        # Game state shared across tool closures
        frames: list[Frame] = []
        action_trace: list[dict] = []
        state: dict[str, Any] = {
            "done": False,
            "total_actions": 0,
            "levels_completed": 0,
            "win_levels": 0,
            "per_level_actions": [],
            "current_level_actions": 0,
            "game_state": "NOT_PLAYED",
            "available_actions": [],
        }

        def _process_frame_data(raw: Any) -> Frame:
            """Convert raw env output to our Frame wrapper."""
            if hasattr(raw, "frame"):
                # FrameDataRaw from arc-agi
                grid = raw.frame[-1] if isinstance(raw.frame, list) else raw.frame
                if hasattr(grid, "tolist"):
                    grid = grid.tolist()
                f = Frame(
                    grid,
                    levels_completed=getattr(raw, "levels_completed", 0),
                    win_levels=getattr(raw, "win_levels", 0),
                    state=str(getattr(raw, "state", "UNKNOWN")),
                    available_actions=[
                        GameAction.from_id(a).name
                        for a in getattr(raw, "available_actions", [])
                    ],
                )
                state["levels_completed"] = f.metadata.get("levels_completed", 0)
                state["win_levels"] = f.metadata.get("win_levels", 0)
                state["game_state"] = f.metadata.get("state", "UNKNOWN")
                state["available_actions"] = f.metadata.get("available_actions", [])
                return f
            # Fallback: raw is already a grid or dict
            if isinstance(raw, (list, tuple)):
                return Frame(raw)
            return Frame([[0] * 64] * 64)

        # Initial reset
        raw_obs = env.reset()
        initial_frame = _process_frame_data(raw_obs)
        frames.append(initial_frame)

        # Build strands tools
        from strands import tool

        @tool
        def observe_game() -> str:
            """Get the current game state with grid analysis.

            Returns the rendered grid, color distribution, available actions,
            and change summary from the last action. Call this before deciding
            your next move.
            """
            if state["done"]:
                return "Game is over. No more actions needed."
            if not frames:
                return "(no observation yet)"

            current = frames[-1]
            parts = [
                f"=== Game State ===",
                f"Level: {state['levels_completed']}/{state['win_levels']}",
                f"Status: {state['game_state']}",
                f"Actions used: {state['total_actions']}/{max_actions}",
                f"Available actions: {', '.join(state['available_actions'])}",
                f"",
                f"=== Grid ({current.width}x{current.height}) ===",
                current.render(y_ticks=True, x_ticks=True),
                f"",
                f"=== Color Distribution ===",
                ", ".join(
                    f"{COLOR_NAMES[c]}({c}): {n}"
                    for c, n in sorted(current.color_counts().items())
                ),
            ]

            # Show diff from previous frame
            if len(frames) >= 2:
                prev = frames[-2]
                summary = current.change_summary(prev)
                parts.extend(["", f"=== Changes from Last Action ===", summary])

            return "\n".join(parts)

        @tool
        def take_action(action: str, x: int = -1, y: int = -1) -> str:
            """Take an action in the game.

            Args:
                action: One of ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7, RESET.
                    ACTION1-4: Directional (up/down/left/right).
                    ACTION5: Context-dependent interaction.
                    ACTION6: Coordinate-based click (requires x, y in range 0-63).
                    ACTION7: Undo last action.
                    RESET: Restart current level.
                x: X coordinate for ACTION6 (0-63).
                y: Y coordinate for ACTION6 (0-63).
            """
            if state["done"]:
                return "Game is already over."
            if state["total_actions"] >= max_actions:
                state["done"] = True
                return f"Action budget exhausted ({max_actions} actions)."

            action_upper = action.upper().strip()
            try:
                game_action = GameAction.from_name(action_upper)
            except (ValueError, KeyError):
                avail = ", ".join(state["available_actions"])
                return f"Invalid action: {action}. Available: {avail}"

            # Set coordinate data for ACTION6
            if game_action.is_complex() and x >= 0 and y >= 0:
                game_action.set_data({"x": min(x, 63), "y": min(y, 63)})

            # Execute
            prev_levels = state["levels_completed"]
            try:
                raw_obs = env.step(game_action)
                # Handle both (obs,) and (obs, reward, done, info) returns
                if isinstance(raw_obs, tuple):
                    raw_obs = raw_obs[0]
            except Exception as e:
                return f"Error executing {action_upper}: {e}"

            new_frame = _process_frame_data(raw_obs)
            frames.append(new_frame)

            state["total_actions"] += 1
            state["current_level_actions"] += 1

            # Detect level transition
            level_changed = state["levels_completed"] > prev_levels
            if level_changed:
                state["per_level_actions"].append(state["current_level_actions"])
                state["current_level_actions"] = 0

            # Detect game end
            game_state_str = state["game_state"]
            if game_state_str in ("WIN",) or (
                state["win_levels"] > 0
                and state["levels_completed"] >= state["win_levels"]
            ):
                state["done"] = True

            # Record trace
            action_trace.append({
                "type": "action",
                "action": action_upper,
                "x": x if action_upper == "ACTION6" else None,
                "y": y if action_upper == "ACTION6" else None,
                "level_changed": level_changed,
                "levels_completed": state["levels_completed"],
                "actions_so_far": state["total_actions"],
                "game_state": game_state_str,
            })

            # Build response with diff
            parts = []
            if level_changed:
                parts.append(f"*** LEVEL COMPLETE! (Level {state['levels_completed']}) ***")
            if state["done"]:
                parts.append("*** GAME COMPLETE! ***")

            parts.append(f"Action: {action_upper}")
            parts.append(f"Actions: {state['total_actions']}/{max_actions}")
            parts.append(f"Level: {state['levels_completed']}/{state['win_levels']}")

            # Show change summary
            if len(frames) >= 2:
                summary = new_frame.change_summary(frames[-2])
                parts.extend(["", "Changes:", summary])

            # Show new grid
            parts.extend([
                "",
                f"=== Current Grid ===",
                new_frame.render(y_ticks=True, x_ticks=True),
            ])

            return "\n".join(parts)

        @tool
        def analyze_grid(colors: str = "", crop: str = "") -> str:
            """Analyze specific aspects of the current grid.

            Args:
                colors: Comma-separated color indices to find (e.g. "8,14" for red,green).
                    Returns all pixel locations matching these colors.
                crop: Region to render as "x1,y1,x2,y2" (e.g. "10,20,30,40").
                    Renders only that sub-region for detailed inspection.
            """
            if not frames:
                return "(no grid to analyze)"
            current = frames[-1]
            parts = []

            if colors:
                try:
                    color_ids = [int(c.strip()) for c in colors.split(",")]
                    pixels = current.find(*color_ids)
                    color_names = [COLOR_NAMES[c] for c in color_ids if c < 16]
                    parts.append(f"Pixels matching {', '.join(color_names)}:")
                    if pixels:
                        for px, py, pv in pixels[:100]:
                            parts.append(f"  ({px}, {py}) = {pv} ({COLOR_NAMES[pv]})")
                        if len(pixels) > 100:
                            parts.append(f"  ... and {len(pixels) - 100} more")
                        bbox = current.bounding_box(*color_ids)
                        if bbox:
                            parts.append(f"Bounding box: x=[{bbox[0]},{bbox[2]}) y=[{bbox[1]},{bbox[3]})")
                    else:
                        parts.append("  (none found)")
                except ValueError:
                    parts.append(f"Invalid color indices: {colors}")

            if crop:
                try:
                    coords = tuple(int(c.strip()) for c in crop.split(","))
                    if len(coords) == 4:
                        parts.extend([
                            "",
                            f"=== Cropped Region ({coords[0]},{coords[1]})-({coords[2]},{coords[3]}) ===",
                            current.render(y_ticks=True, x_ticks=True, crop=coords),
                        ])
                except ValueError:
                    parts.append(f"Invalid crop: {crop}. Use x1,y1,x2,y2")

            if not parts:
                parts = [
                    f"Grid: {current.width}x{current.height}",
                    f"Colors present: {', '.join(f'{COLOR_NAMES[c]}({c}):{n}' for c, n in sorted(current.color_counts().items()))}",
                ]

            return "\n".join(parts)

        # Build the strands agent with workspace prompt
        model = BedrockModel(
            model_id=self.model_id,
            region_name=self.region,
            max_tokens=self.max_tokens,
        )

        system_prompt = self._build_system_prompt()
        tools = [observe_game, take_action, analyze_grid]

        # Add read_skill tool if skills exist
        if self.skills:
            skill_data = {}
            for skill in self.skills:
                content = self.get_skill_content(skill.name)
                if content:
                    body = content.split("---", 2)[-1].strip() if "---" in content else content
                    skill_data[skill.name] = body

            @tool
            def read_skill(skill_name: str) -> str:
                """Read the full procedure for a skill.

                Args:
                    skill_name: Name of the skill to read
                """
                if skill_name in skill_data:
                    return skill_data[skill_name]
                return f"Skill '{skill_name}' not found. Available: {', '.join(skill_data.keys())}"

            tools.append(read_skill)

        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
        )

        # Play the game
        user_prompt = self._build_user_prompt(task, initial_frame)
        t0 = time.time()

        try:
            response = agent(user_prompt)
        except Exception as e:
            logger.error("Agent error playing %s: %s", game_id, e)
            response = None

        elapsed = time.time() - t0
        logger.info(
            "Game %s finished in %.1fs: %d actions, %d/%d levels",
            game_id, elapsed, state["total_actions"],
            state["levels_completed"], state["win_levels"],
        )

        # Extract usage
        usage = {}
        if response:
            try:
                u = response.metrics.accumulated_usage
                usage = {
                    "input_tokens": u.get("inputTokens", 0),
                    "output_tokens": u.get("outputTokens", 0),
                    "total_tokens": u.get("totalTokens", 0),
                }
            except Exception:
                pass

        # Compute score
        score = self._compute_score(state)

        # Build result
        game_completed = (
            state["levels_completed"] > 0
            and (state["done"] or state["levels_completed"] >= state["win_levels"])
        )
        result = {
            "game_id": game_id,
            "game_completed": game_completed,
            "levels_completed": state["levels_completed"],
            "total_levels": state["win_levels"],
            "total_actions": state["total_actions"],
            "per_level_actions": state["per_level_actions"],
            "score": score,
            "elapsed_sec": elapsed,
            "usage": usage,
        }

        action_trace.append({
            "type": "summary",
            "llm_output": str(response)[:2000] if response else "(error)",
            "usage": usage,
            **result,
        })

        self.remember(
            f"Played {game_id}: completed={game_completed}, "
            f"levels={state['levels_completed']}/{state['win_levels']}, "
            f"actions={state['total_actions']}, score={score:.3f}",
            category="episodic",
            task_id=game_id,
        )

        traj = Trajectory(task_id=task.id, output=json.dumps(result), steps=action_trace)

        # Skill proposal
        if response:
            traj._skill_proposal = self._generate_skill_proposal(agent, game_id)

        return traj

    # ── Score computation ────────────────────────────────────────────

    @staticmethod
    def _compute_score(state: dict) -> float:
        """Compute a 0-1 RHAE-inspired score."""
        levels = state.get("levels_completed", 0)
        win_levels = state.get("win_levels", 0)
        total_actions = state.get("total_actions", 0)

        if levels == 0:
            return 0.0

        # Completion fraction
        if win_levels > 0:
            completion = levels / win_levels
        else:
            completion = 1.0 if levels > 0 else 0.0

        # Efficiency: penalize excessive actions per level
        avg_actions = total_actions / levels
        efficiency = max(0.1, min(1.0, 1.0 - (avg_actions - 50) / 200))

        return completion * efficiency

    # ── Prompt construction (from workspace) ─────────────────────────

    def _build_system_prompt(self) -> str:
        """Assemble the full system prompt from workspace files."""
        parts = [self.system_prompt]

        # Evolved prompt fragments
        fragments = self.workspace.list_fragments()
        if fragments:
            for frag_name in fragments:
                content = self.workspace.read_fragment(frag_name)
                if content and content.strip():
                    marker = f"<!-- evolve:{frag_name.removesuffix('.md')} -->"
                    if marker not in self.system_prompt:
                        parts.append(f"\n\n## {frag_name.removesuffix('.md').replace('_', ' ').title()}")
                        parts.append(content)

        # Skills
        parts.append("\n\n## Skills\n")
        if self.skills:
            parts.append(
                "You have skills learned from previous games. Call `read_skill(skill_name)` "
                "to load any that match your situation.\n"
            )
            for skill in self.skills:
                parts.append(f"- **{skill.name}**: {skill.description}")
        else:
            parts.append("No skills available yet. They will be learned through evolution.\n")

        return "\n".join(parts)

    def _build_user_prompt(self, task: Task, initial_frame: Frame) -> str:
        """Build the user prompt including initial grid observation."""
        game_id = task.metadata.get("game_id", task.id)
        max_actions = task.metadata.get("max_actions", self.max_actions)

        memory_section = ""
        if self.memories:
            relevant = [m for m in self.memories if m.get("task_id") == game_id]
            if relevant:
                memory_section = "\n## Previous Attempts\n"
                for mem in relevant[-5:]:
                    memory_section += f"- {mem.get('content', '')}\n"
                memory_section += "\nLearn from these and try a different strategy.\n"

        initial_obs = initial_frame.render(y_ticks=True, x_ticks=True)
        color_dist = ", ".join(
            f"{COLOR_NAMES[c]}({c}): {n}"
            for c, n in sorted(initial_frame.color_counts().items())
        )

        return f"""\
{task.input}

Action budget: {max_actions} actions
{memory_section}
## Initial Observation

Grid ({initial_frame.width}x{initial_frame.height}):
{initial_obs}

Color distribution: {color_dist}
Color legend: {COLOR_LEGEND}

## Instructions

1. Call observe_game() to understand the full state
2. Use analyze_grid(colors="8,14") to find specific objects by color
3. Experiment with take_action() to learn the game mechanics
4. Track what each action does -- build a mental model
5. Once you understand the rules, solve efficiently
6. Every action counts toward your score -- minimize wasted moves
"""

    # ── Skill proposals ──────────────────────────────────────────────

    def _generate_skill_proposal(self, agent: Agent, game_id: str) -> str:
        """Ask the agent to propose a reusable skill after playing."""
        try:
            skill_context = ""
            if self.skills:
                skill_list = "\n".join(f"- {s.name}: {s.description}" for s in self.skills)
                skill_context = f"You had these skills available:\n{skill_list}\n\n"

            proposal_response = agent(
                f"{skill_context}"
                "Based on the game you just played, propose a reusable skill "
                "that could help in future ARC-AGI-3 games.\n\n"
                "RULES:\n"
                "- NAME must be GENERIC (e.g., navigate_maze, pattern_matching, "
                "explore_then_exploit, identify_interactive_objects)\n"
                "- DESCRIPTION must include TRIGGER and DO NOT TRIGGER conditions\n\n"
                "OPTION A -- ENHANCE existing skill:\n"
                "ACTION: ENHANCE\nTARGET: skill_name\n"
                "NAME: same_name\nDESCRIPTION: one sentence\nCONTENT: (under 500 words)\n\n"
                "OPTION B -- NEW skill:\n"
                "ACTION: NEW\nNAME: pattern_name\n"
                "DESCRIPTION: one sentence\nCONTENT: (under 500 words)\n\n"
                "OPTION C -- No proposal:\nACTION: NONE"
            )
            return str(proposal_response).strip()[:2500]
        except Exception as e:
            logger.warning("Skill proposal failed for %s: %s", game_id, e)
            return ""
