# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI programming learning project. Tasks are completed using LLM APIs via OpenRouter as the gateway to access models. All code is written in Python.

## Project Structure

- Tasks are organized by season and episode: `S01E01/`, `S01E02/`, etc.
- Each folder contains standalone Python scripts for that lesson's tasks (e.g., `task_1.py`)
- Virtual environment lives in `.venv/`

## Running Code

```bash
# Activate virtual environment
source .venv/Scripts/activate   # Windows/Git Bash

# Run a task script
python S01E01/task_1.py
```

## Environment Setup

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt  # when requirements.txt exists
```

## Conventions

- Use a `.env` file in the project root for API keys (e.g., `OPENROUTER_API_KEY`) and secrets (never commit it)
- Each task script should be self-contained and runnable independently
- Language: Python only
