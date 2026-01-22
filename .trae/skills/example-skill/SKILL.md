---
name: example-skill
description: A simple example skill demonstrating the basic structure of Trae Skills. Use this when users want to understand how to create custom skills for Trae AI.
license: MIT
---

# Example Skill

This skill demonstrates the proper structure of Trae Skills based on the official Anthropics Skills format.

## About This Skill

This is a simple example skill that shows how to create custom skills for Trae AI. Skills are modular, self-contained packages that extend Trae's capabilities with specialized knowledge, workflows, or tool integrations.

## Usage
```
@example-skill
```

## When to Use
- When users ask about Trae Skills structure
- When users want to create their own custom skills
- When users need an example of a properly formatted SKILL.md file

## Core Structure

A proper Trae Skill consists of:

1. **YAML Frontmatter** (required)
   - `name`: Unique identifier for the skill
   - `description`: What the skill does and when to use it
   - `license`: Optional license information

2. **Markdown Body** (required)
   - Skill documentation and instructions
   - Usage examples
   - Guidelines for the AI

3. **Bundled Resources** (optional)
   - `scripts/`: Executable code for reliable tasks
   - `references/`: Documentation for AI reference
   - `assets/`: Files used in output

## Example Usage

```
User: How do I create a Trae Skill?
Trae: I can help you create a Trae Skill. Let me show you an example structure...
```

## Output Format
A detailed explanation of Trae Skills with a sample SKILL.md template that users can customize for their own needs.