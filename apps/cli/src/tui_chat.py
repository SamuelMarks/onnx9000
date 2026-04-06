"""Terminal User Interface chat application."""

import sys


def start_chat_tui() -> bool:
    """Start the interactive terminal chat session.

    Returns:
        bool: True when the session finishes successfully.
    """
    print("Starting ONNX9000 TUI chat... (type 'exit' to quit)")
    try:
        while True:
            try:
                user_input = input("\nYou: ")
            except EOFError:
                break

            user_input = user_input.strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            elif not user_input:
                continue
            else:
                print(f"ONNX9000 Assistant: I received '{user_input}', but I am a simple mock.")
    except KeyboardInterrupt:
        print("\nGoodbye!")

    return True
