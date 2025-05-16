# commands/__init__.py
from commands.user_commands import setup_user_commands
from commands.admin_commands import setup_admin_commands

__all__ = ['setup_user_commands', 'setup_admin_commands']
