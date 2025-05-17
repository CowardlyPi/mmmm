# commands/__init__.py
from commands.user_commands import setup_user_commands
from commands.admin_commands import setup_admin_commands
from commands.enhanced_commands import setup_enhanced_commands

__all__ = ['setup_user_commands', 'setup_admin_commands', 'setup_enhanced_commands']
