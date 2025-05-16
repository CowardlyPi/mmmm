import re

# Read the main.py file
with open('main.py', 'r') as file:
    content = file.read()

# More robust pattern to find any OpenAI initialization
pattern = r"self\.openai_client\s*=\s*OpenAI\(.*?\)"
replacement = "self.openai_client = OpenAI(api_key=openai_api_key)"

# Apply the replacement with DOTALL flag to match across lines
modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write the modified content back to the file
with open('main.py', 'w') as file:
    file.write(modified_content)

print("Successfully patched main.py to fix OpenAI client initialization!")
