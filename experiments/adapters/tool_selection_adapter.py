"""
Multi-Tool Selection Test (T4)

Real-world test: an agent has 10/20/30 tool definitions in its prompt.
The correct tool is at varying positions. Tests whether CADG permutation
of tool order improves selection accuracy (F5 Position Bias).

Models pick wrong tools from position 1 or N when the correct tool is
in the middle — the same primacy/recency bias that affects constraints.
"""

from __future__ import annotations

import json
import re
from random import Random
from typing import Callable


TOOLS = [
    {"name": "search_web", "desc": "Search the internet for current information. Args: query (str)"},
    {"name": "get_weather", "desc": "Get current weather for a location. Args: city (str)"},
    {"name": "send_email", "desc": "Send an email to a recipient. Args: to (str), subject (str), body (str)"},
    {"name": "create_calendar_event", "desc": "Create a calendar event. Args: title (str), date (str), time (str)"},
    {"name": "translate_text", "desc": "Translate text to another language. Args: text (str), target_lang (str)"},
    {"name": "calculate", "desc": "Evaluate a mathematical expression. Args: expression (str)"},
    {"name": "get_stock_price", "desc": "Get current stock price. Args: ticker (str)"},
    {"name": "set_reminder", "desc": "Set a timed reminder. Args: message (str), minutes (int)"},
    {"name": "summarize_text", "desc": "Summarize a long text into key points. Args: text (str)"},
    {"name": "convert_currency", "desc": "Convert between currencies. Args: amount (float), from_currency (str), to_currency (str)"},
    {"name": "get_directions", "desc": "Get driving directions between two locations. Args: origin (str), destination (str)"},
    {"name": "book_restaurant", "desc": "Make a restaurant reservation. Args: restaurant (str), date (str), party_size (int)"},
    {"name": "check_flight_status", "desc": "Check the status of a flight. Args: flight_number (str)"},
    {"name": "play_music", "desc": "Play a song or playlist. Args: query (str)"},
    {"name": "set_alarm", "desc": "Set an alarm for a specific time. Args: time (str)"},
    {"name": "read_news", "desc": "Get top news headlines. Args: category (str, optional)"},
    {"name": "order_food", "desc": "Order food delivery. Args: restaurant (str), items (list)"},
    {"name": "find_nearby", "desc": "Find nearby places of a given type. Args: place_type (str), location (str)"},
    {"name": "manage_todo", "desc": "Add, remove, or list todo items. Args: action (str), item (str, optional)"},
    {"name": "analyze_image", "desc": "Analyze and describe an image. Args: image_url (str)"},
    {"name": "generate_code", "desc": "Generate code in a specified language. Args: description (str), language (str)"},
    {"name": "query_database", "desc": "Run a SQL query on the database. Args: query (str)"},
    {"name": "upload_file", "desc": "Upload a file to cloud storage. Args: file_path (str), destination (str)"},
    {"name": "compress_image", "desc": "Compress an image to reduce file size. Args: image_url (str), quality (int)"},
    {"name": "schedule_meeting", "desc": "Schedule a video meeting with participants. Args: title (str), participants (list), time (str)"},
    {"name": "run_diagnostic", "desc": "Run a system diagnostic check. Args: system (str)"},
    {"name": "create_invoice", "desc": "Create an invoice for a client. Args: client (str), items (list), amounts (list)"},
    {"name": "track_package", "desc": "Track a package delivery status. Args: tracking_number (str)"},
    {"name": "convert_units", "desc": "Convert between measurement units. Args: value (float), from_unit (str), to_unit (str)"},
    {"name": "backup_data", "desc": "Create a backup of specified data. Args: data_source (str), destination (str)"},
]

TASKS = [
    {"query": "What's the weather like in Tokyo right now?", "correct_tool": "get_weather"},
    {"query": "Send an email to john@example.com about the meeting tomorrow", "correct_tool": "send_email"},
    {"query": "How much is 150 euros in US dollars?", "correct_tool": "convert_currency"},
    {"query": "Find Italian restaurants near downtown Chicago", "correct_tool": "find_nearby"},
    {"query": "What's Apple's stock price today?", "correct_tool": "get_stock_price"},
    {"query": "Schedule a team meeting for next Tuesday at 3pm with Alice and Bob", "correct_tool": "schedule_meeting"},
    {"query": "Translate 'Hello, how are you?' to Japanese", "correct_tool": "translate_text"},
    {"query": "Check the status of flight AA1234", "correct_tool": "check_flight_status"},
    {"query": "Create an invoice for Client Corp: 10 widgets at $5 each", "correct_tool": "create_invoice"},
    {"query": "Track my package with tracking number 1Z999AA10123456784", "correct_tool": "track_package"},
    {"query": "Add 'Buy groceries' to my todo list", "correct_tool": "manage_todo"},
    {"query": "What's 15% tip on a $84.50 bill?", "correct_tool": "calculate"},
    {"query": "Remind me to call the dentist in 30 minutes", "correct_tool": "set_reminder"},
    {"query": "Summarize this 5-page report for me", "correct_tool": "summarize_text"},
    {"query": "Back up my project files to the cloud", "correct_tool": "backup_data"},
    {"query": "Convert 5 miles to kilometers", "correct_tool": "convert_units"},
    {"query": "Get directions from San Francisco to Los Angeles", "correct_tool": "get_directions"},
    {"query": "Book a table for 4 at Olive Garden for Friday night", "correct_tool": "book_restaurant"},
    {"query": "Run a diagnostic on the production server", "correct_tool": "run_diagnostic"},
    {"query": "Generate a Python function to sort a list of dictionaries by key", "correct_tool": "generate_code"},
]


def _check_tool_selected(output: str, correct_tool: str) -> bool:
    """Check if the correct tool name appears in the model's output."""
    output_lower = output.lower()
    correct_lower = correct_tool.lower()
    if correct_lower in output_lower:
        return True
    if correct_tool.replace("_", " ") in output_lower:
        return True
    if re.search(rf"\b{re.escape(correct_tool)}\b", output, re.IGNORECASE):
        return True
    try:
        parsed = json.loads(output)
        tool_val = parsed.get("tool", parsed.get("function", parsed.get("name", "")))
        if correct_lower == str(tool_val).lower():
            return True
    except (json.JSONDecodeError, AttributeError):
        pass
    return False


class ToolSelectionAdapter:
    """Adapter for multi-tool selection accuracy testing."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def load_tasks(
        self,
        tool_counts: list[int] | None = None,
        tasks_per_count: int = 20,
        max_tasks: int | None = None,
    ) -> list[dict]:
        if tool_counts is None:
            tool_counts = [10, 15, 20, 30]

        rng = Random(self.seed)
        tasks = []

        for n_tools in tool_counts:
            available_tools = TOOLS[:n_tools]

            for ti in range(min(tasks_per_count, len(TASKS))):
                task_def = TASKS[ti % len(TASKS)]
                correct = task_def["correct_tool"]

                # Only include tasks where the correct tool is in the available set
                if not any(t["name"] == correct for t in available_tools):
                    continue

                # Find position of correct tool
                correct_pos = next(i for i, t in enumerate(available_tools) if t["name"] == correct)

                tool_defs = "\n".join(
                    f"- **{t['name']}**: {t['desc']}" for t in available_tools
                )
                prompt = (
                    f"You have access to these {n_tools} tools:\n\n{tool_defs}\n\n"
                    f"User request: \"{task_def['query']}\"\n\n"
                    f"Which tool should you use? Respond with ONLY the tool name, nothing else."
                )

                constraints = [{
                    "id": f"tool_{correct}_{n_tools}t_{ti}",
                    "text": f"Select tool: {correct}",
                    "check_fn": lambda output, _c=correct: _check_tool_selected(output, _c),
                    "correct_tool": correct,
                    "correct_position": correct_pos,
                }]

                tasks.append({
                    "task_id": f"toolsel_{n_tools}t_{ti}",
                    "prompt": prompt,
                    "base_prompt": prompt,
                    "constraints": constraints,
                    "constraint_count": 1,
                    "n_tools": n_tools,
                    "correct_tool": correct,
                    "correct_position": correct_pos,
                })

                if max_tasks and len(tasks) >= max_tasks:
                    return tasks

        print(f"Loaded {len(tasks)} tool selection tasks ({tool_counts} tools)")
        return tasks

    def extract_constraints(self, task: dict) -> list[dict]:
        return task["constraints"]

    def score(self, output: str, task: dict) -> dict:
        correct = task["correct_tool"]
        selected_correctly = _check_tool_selected(output, correct)
        return {
            "correct": selected_correctly,
            "correct_tool": correct,
            "correct_position": task["correct_position"],
            "n_tools": task["n_tools"],
            "output_snippet": output[:100],
        }
