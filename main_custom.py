import asyncio
import json
import re
import os
import subprocess
import logging
import time
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Sequence
from operator import add
import requests

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END


# Define the state structure
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    task_description: str
    repository_context: str
    plan: str
    implementation: str
    test_results: str
    working_directory: str
    current_agent: str
    next_agent: str
    iteration: int
    max_iterations: int
    git_diff: str
    instance_id: str
    fail_tests: List[str]
    pass_tests: List[str]


class LangGraphMultiAgentSystem:
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.llm = OllamaLLM(
            model=llm_config.get("model", "gemma3:1b"),
            base_url=llm_config.get("base_url", "http://localhost:11434"),
            temperature=0.3
        )
        self.API_URL = "http://localhost:8081/task/index/"
        self.LOG_FILE = "results.log"
        self.total_tokens = 0
        self.working_directory = None
        self.checkpointer = None  # Disable checkpointer for now

        # Setup logging
        self._setup_logging()

        self.workflow = self._create_workflow()

    def _setup_logging(self):
        """Setup detailed logging for progress tracking."""
        # Create a logger
        self.logger = logging.getLogger('LangGraphAgent')
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create console handler with formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(console_handler)

    def _log_progress(self, agent: str, message: str, level: str = "info"):
        """Log progress with agent context."""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{agent.upper()}] {message}"

        if level == "info":
            self.logger.info(formatted_msg)
        elif level == "warning":
            self.logger.warning(formatted_msg)
        elif level == "error":
            self.logger.error(formatted_msg)

        # Also print to console with clear formatting
        print(f"🤖 {timestamp} [{agent.upper()}] {message}")

    def _log_llm_call(self, agent: str, prompt_preview: str):
        """Log when making LLM calls."""
        preview = prompt_preview[:100].replace('\n', ' ') + "..." if len(prompt_preview) > 100 else prompt_preview
        self._log_progress(agent, f"Calling LLM with prompt: {preview}")

    def _log_llm_response(self, agent: str, response_preview: str):
        """Log LLM response."""
        preview = response_preview[:100].replace('\n', ' ') + "..." if len(response_preview) > 100 else response_preview
        self._log_progress(agent, f"LLM response: {preview}")

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with agent nodes."""
        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("coder", self._coder_node)
        workflow.add_node("tester", self._tester_node)
        workflow.add_node("evaluator", self._evaluator_node)

        # Set entry point
        workflow.set_entry_point("planner")

        # Add edges
        workflow.add_edge("planner", "coder")
        workflow.add_edge("coder", "tester")
        workflow.add_edge("tester", "evaluator")

        # Add conditional edge from evaluator
        workflow.add_conditional_edges(
            "evaluator",
            self._should_continue,
            {
                "continue": "planner",
                "end": END
            }
        )

        return workflow.compile()  # Remove checkpointer for now

    async def _planner_node(self, state: AgentState) -> Dict[str, Any]:
        """Planner agent node - analyzes the problem and creates a strategy."""
        self._log_progress("planner", "Starting problem analysis...")

        system_prompt = """You are a Senior Software Architect and Problem Analyst.

Your responsibilities:
1. Analyze the problem statement thoroughly
2. Understand the failing tests and what they expect
3. Identify the root cause of the issue
4. Create a minimal, targeted fix strategy
5. Provide clear implementation guidance

When you receive a problem:
- Read the problem description carefully
- Analyze what the failing tests expect vs. what's currently happening
- Identify the specific files and functions that need modification
- Create a step-by-step plan that addresses ONLY the failing tests
- Avoid over-engineering - make minimal changes
- Consider edge cases and potential regressions

Always end your analysis with a clear, actionable plan for implementation."""

        user_content = f"""
Task Description: {state.get("task_description", "")}

Repository Context: {state.get("repository_context", "")}

Failing Tests: {json.dumps(state.get("fail_tests", []))}

Please analyze this problem and create a detailed implementation plan.
"""

        # Create the full prompt
        full_prompt = f"{system_prompt}\n\n{user_content}"

        self._log_llm_call("planner", full_prompt)

        # Get response from LLM (run in executor for async compatibility)
        import asyncio
        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(None, self.llm.invoke, full_prompt)
            self._log_llm_response("planner", response)
            self._log_progress("planner", "Analysis complete! Plan created.")
        except Exception as e:
            self._log_progress("planner", f"Error during LLM call: {e}", "error")
            response = f"Error during planning: {e}"

        # Update state - return only the new values
        new_messages = [AIMessage(content=response, name="planner")]

        return {
            "messages": new_messages,
            "plan": response,
            "current_agent": "planner",
            "next_agent": "coder"
        }

    async def _coder_node(self, state: AgentState) -> Dict[str, Any]:
        """Coder agent node - implements the planned solution and makes file changes."""
        self._log_progress("coder", "Starting code implementation and file modification...")

        system_prompt = """You are a Senior Software Engineer specializing in precise code implementation.

Your responsibilities:
1. Implement the plan exactly as specified
2. Make actual file modifications by writing code that reads, modifies, and saves files
3. Write clean, maintainable code that matches the project style
4. Ensure all changes are saved to disk immediately
5. Focus on minimal, targeted changes

Implementation rules:
- Always read existing code first to understand context and style
- Make the smallest possible change that fixes the issue
- Preserve all existing functionality
- Follow the project's coding conventions
- Use proper error handling where appropriate

CRITICAL: You must provide executable Python code that actually modifies files.
The code should:
1. Read the existing file content
2. Make the necessary modifications
3. Write the updated content back to the file
4. Print what changes were made

Example format:
```python
# Read the existing file
with open('path/to/file.py', 'r') as f:
    content = f.read()

# Make modifications
new_content = content.replace('old_code', 'new_code')

# Write back to file
with open('path/to/file.py', 'w') as f:
    f.write(new_content)

print("Modified path/to/file.py: replaced 'old_code' with 'new_code'")
```
"""

        user_content = f"""
Plan from Planner: {state.get("plan", "")}

Working Directory: {state.get("working_directory", "")}

Please provide executable Python code that will implement this plan by actually modifying the necessary files.
The code should read files, make changes, and write them back.

Use this format:
```python
# Your file modification code here
```
"""

        full_prompt = f"{system_prompt}\n\n{user_content}"

        self._log_llm_call("coder", full_prompt)

        import asyncio
        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(None, self.llm.invoke, full_prompt)
            self._log_llm_response("coder", response)
            self._log_progress("coder", "Code implementation generated, executing file changes...")
        except Exception as e:
            self._log_progress("coder", f"Error during LLM call: {e}", "error")
            response = f"Error during coding: {e}"

        # Extract and execute code blocks from the response
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        self._log_progress("coder", f"Found {len(code_blocks)} code blocks to execute")

        execution_results = []
        original_dir = os.getcwd()
        working_dir = state.get("working_directory", "")

        try:
            # Change to working directory if it exists
            if working_dir and os.path.exists(working_dir):
                self._log_progress("coder", f"Changing to working directory: {working_dir}")
                os.chdir(working_dir)
            else:
                self._log_progress("coder", "No working directory found", "warning")

            for i, code_block in enumerate(code_blocks):
                self._log_progress("coder", f"Executing code block {i + 1}/{len(code_blocks)}")
                try:
                    # Create a safe execution environment
                    exec_globals = {
                        'os': os,
                        'subprocess': subprocess,
                        're': re,
                        'json': json,
                        'open': open,
                        'print': print,
                        '__builtins__': __builtins__
                    }
                    exec_locals = {}

                    # Capture print output
                    import io
                    import sys
                    captured_output = io.StringIO()
                    sys.stdout = captured_output

                    exec(code_block, exec_globals, exec_locals)

                    # Restore stdout and get captured output
                    sys.stdout = sys.__stdout__
                    output = captured_output.getvalue()

                    execution_results.append(f"Code block {i + 1} executed successfully")
                    if output:
                        execution_results.append(f"Output: {output.strip()}")
                        self._log_progress("coder", f"Code block {i + 1} output: {output.strip()}")
                    else:
                        self._log_progress("coder", f"Code block {i + 1} executed successfully")

                except Exception as e:
                    error_msg = f"Error executing code block {i + 1}: {str(e)}"
                    execution_results.append(error_msg)
                    self._log_progress("coder", error_msg, "error")

            # Get git diff after execution
            self._log_progress("coder", "Checking for file changes...")
            git_diff = ""
            try:
                diff_result = subprocess.run(
                    ["git", "diff"],
                    capture_output=True,
                    text=True,
                )
                git_diff = diff_result.stdout
                if git_diff.strip():
                    self._log_progress("coder", f"Git diff shows {len(git_diff.splitlines())} lines changed")
                else:
                    self._log_progress("coder", "No file changes detected", "warning")
            except Exception as e:
                git_diff = f"Error getting git diff: {e}"
                self._log_progress("coder", f"Error getting git diff: {e}", "error")

        finally:
            # Always restore original directory
            os.chdir(original_dir)
            self._log_progress("coder", "File modification completed, directory restored")

        execution_summary = "\n".join(execution_results)
        result_content = f"Coder Implementation:\n{response}\n\nExecution Results:\n{execution_summary}\n\nGit diff:\n{git_diff}"

        new_messages = [AIMessage(content=result_content, name="coder")]

        return {
            "messages": new_messages,
            "implementation": response,
            "git_diff": git_diff,
            "current_agent": "coder",
            "next_agent": "tester"
        }

    async def _tester_node(self, state: AgentState) -> Dict[str, Any]:
        """Tester agent node - runs tests to validate the fix."""
        self._log_progress("tester", "Starting test validation...")

        system_prompt = """You are a Senior QA Engineer specializing in test execution and validation.

Your responsibilities:
1. Execute the test suite to verify fixes
2. Validate that failing tests now pass
3. Ensure existing tests continue to pass
4. Provide detailed test result analysis
5. Identify any regressions or issues

Testing approach:
- Run the specific FAIL_TO_PASS tests mentioned in the problem
- Run the PASS_TO_PASS tests to check for regressions
- Use the appropriate test framework (pytest, unittest, etc.)
- Provide clear, detailed output of test results

Provide the exact commands to run the tests. Wrap bash commands in triple backticks with 'bash' language specification.
"""

        user_content = f"""
Git diff showing changes: {state.get("git_diff", "")}

Failing tests to fix: {json.dumps(state.get("fail_tests", []))}
Tests that should continue passing: {json.dumps(state.get("pass_tests", []))}

Working directory: {state.get("working_directory", "")}

Please provide the commands to run these tests and analyze the results.
Use this format:
```bash
# Your test commands here
```
"""

        full_prompt = f"{system_prompt}\n\n{user_content}"

        self._log_llm_call("tester", full_prompt)

        import asyncio
        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(None, self.llm.invoke, full_prompt)
            self._log_llm_response("tester", response)
        except Exception as e:
            self._log_progress("tester", f"Error during LLM call: {e}", "error")
            response = f"Error during test planning: {e}"

        # Extract test commands and run them
        test_commands = re.findall(r'```bash\n(.*?)\n```', response, re.DOTALL)
        self._log_progress("tester", f"Found {len(test_commands)} test commands to execute")

        test_results = []

        working_dir = state.get("working_directory", "")
        original_dir = os.getcwd()

        try:
            if working_dir and os.path.exists(working_dir):
                self._log_progress("tester", f"Changing to working directory: {working_dir}")
                os.chdir(working_dir)

            for i, cmd in enumerate(test_commands):
                self._log_progress("tester", f"Running test command {i + 1}/{len(test_commands)}")
                try:
                    # Clean up the command (remove comments and empty lines)
                    clean_cmd = '\n'.join([line.strip() for line in cmd.split('\n')
                                           if line.strip() and not line.strip().startswith('#')])

                    if clean_cmd:
                        self._log_progress("tester", f"Executing: {clean_cmd}")
                        result = subprocess.run(
                            clean_cmd,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        test_results.append(
                            f"Command: {clean_cmd}\nReturn code: {result.returncode}\nOutput:\n{result.stdout}\nErrors:\n{result.stderr}")

                        if result.returncode == 0:
                            self._log_progress("tester", f"Command {i + 1} completed successfully")
                        else:
                            self._log_progress("tester", f"Command {i + 1} failed with return code {result.returncode}",
                                               "warning")

                except subprocess.TimeoutExpired:
                    timeout_msg = f"Command '{clean_cmd}' timed out after 60 seconds"
                    test_results.append(timeout_msg)
                    self._log_progress("tester", timeout_msg, "warning")
                except Exception as e:
                    error_msg = f"Error running command '{clean_cmd}': {str(e)}"
                    test_results.append(error_msg)
                    self._log_progress("tester", error_msg, "error")

        finally:
            os.chdir(original_dir)
            self._log_progress("tester", "Test execution completed")

        test_summary = "\n\n".join(test_results)
        result_content = f"Tester Analysis:\n{response}\n\nActual Test Results:\n{test_summary}"

        new_messages = [AIMessage(content=result_content, name="tester")]

        return {
            "messages": new_messages,
            "test_results": test_summary,
            "current_agent": "tester",
            "next_agent": "evaluator"
        }

    async def _evaluator_node(self, state: AgentState) -> Dict[str, Any]:
        """Evaluator node - determines if the solution is complete."""
        self._log_progress("evaluator", "Evaluating solution quality...")

        test_results = state.get("test_results", "")
        iteration = state.get("iteration", 0) + 1
        max_iterations = state.get("max_iterations", 5)

        # Simple evaluation: check if tests are passing
        tests_passing = ("PASSED" in test_results or "OK" in test_results or
                         "Return code: 0" in test_results)
        has_failures = ("FAILED" in test_results or "FAIL" in test_results or
                        "Error running command" in test_results)

        success = tests_passing and not has_failures

        evaluation_message = f"Iteration {iteration}/{max_iterations}: "
        if success:
            evaluation_message += "All tests are passing. Solution complete!"
            next_action = "end"
            self._log_progress("evaluator", "✅ Solution is complete! All tests passing.")
        elif iteration >= max_iterations:
            evaluation_message += f"Reached maximum iterations ({max_iterations}). Stopping."
            next_action = "end"
            self._log_progress("evaluator", f"❌ Reached maximum iterations ({max_iterations}). Stopping.", "warning")
        else:
            evaluation_message += "Tests are still failing. Need to refine the solution."
            next_action = "planner"
            self._log_progress("evaluator", f"🔄 Tests still failing. Starting iteration {iteration + 1}...")

        new_messages = [AIMessage(content=evaluation_message, name="evaluator")]

        return {
            "messages": new_messages,
            "iteration": iteration,
            "current_agent": "evaluator",
            "next_agent": next_action
        }

    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue or end the workflow."""
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 5)
        next_agent = state.get("next_agent", "end")

        if iteration >= max_iterations:
            return "end"

        if next_agent == "end":
            return "end"

        return "continue"

    def _setup_working_directory(self, repo_dir: str):
        """Set up the working directory for the agents."""
        self.working_directory = repo_dir

    def _get_repository_context(self, repo_dir: str) -> str:
        """Get basic repository context including file structure."""
        try:
            # Get basic file structure
            result = subprocess.run(
                "find . -type f -name '*.py' | head -20",
                cwd=repo_dir,
                capture_output=True,
                text=True,
                shell=True
            )
            file_list = result.stdout.strip()

            # Get recent commits
            git_log = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            recent_commits = git_log.stdout.strip()

            context = f"""
Repository Structure (Python files):
{file_list}

Recent Commits:
{recent_commits}
"""
            return context
        except Exception as e:
            return f"Error getting repository context: {e}"

    async def solve_problem(self, problem_prompt: str, instance_id: str,
                            fail_tests: List[str], pass_tests: List[str]) -> Dict[str, Any]:
        """
        Main method to solve a SWE-Bench problem using the LangGraph multi-agent system.

        Args:
            problem_prompt: The complete problem description and context
            instance_id: The instance ID for this problem
            fail_tests: List of tests that should pass after the fix
            pass_tests: List of tests that should continue passing

        Returns:
            Dictionary containing results, metrics, and metadata
        """
        try:
            # Extract repository directory from the prompt
            repo_match = re.search(r"Work in the directory: (\S+)", problem_prompt)
            if repo_match:
                repo_name = repo_match.group(1)
                repo_dir = os.path.join(os.getcwd(), repo_name)
                self._setup_working_directory(repo_dir)

            # Gather repository context
            repo_context = ""
            if self.working_directory and os.path.exists(self.working_directory):
                repo_context = self._get_repository_context(self.working_directory)

            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=problem_prompt)],
                "task_description": problem_prompt,
                "repository_context": repo_context,
                "working_directory": self.working_directory,
                "instance_id": instance_id,
                "fail_tests": fail_tests,
                "pass_tests": pass_tests,
                "iteration": 0,
                "max_iterations": 5,
                "plan": "",
                "implementation": "",
                "test_results": "",
                "git_diff": "",
                "current_agent": "planner",
                "next_agent": "coder"
            }

            print("🚀 Starting LangGraph multi-agent collaboration...")

            # Run the workflow without checkpointer config
            final_state = await self.workflow.ainvoke(initial_state)

            # Get the final git diff
            git_diff = final_state.get("git_diff", "")

            # Check if any files were modified
            files_modified = len(git_diff.strip()) > 0

            return {
                "success": True,
                "files_modified": files_modified,
                "git_diff": git_diff,
                "token_usage": self.total_tokens,
                "messages": final_state.get("messages", []),
                "working_directory": self.working_directory,
                "iterations": final_state.get("iteration", 0)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "token_usage": self.total_tokens,
                "working_directory": self.working_directory,
                "git_diff": "",
                "files_modified": False,
            }

    def reset(self):
        """Reset the agent system for a new problem."""
        self.total_tokens = 0
        self.working_directory = None

    async def handle_task(self, index: int):
        """Handle a single task by fetching, solving, and evaluating it."""
        api_url = f"{self.API_URL}{index}"
        print(f"Fetching test case {index} from {api_url}...")
        repo_dir = os.path.join("./workspaces/", f"repo_{index}")
        start_dir = os.getcwd()

        try:
            # Ensure workspaces directory exists
            os.makedirs("./workspaces", exist_ok=True)

            response = requests.get(api_url)
            if response.status_code != 200:
                raise Exception(f"Invalid response: {response.status_code}")

            testcase = response.json()
            prompt = testcase["Problem_statement"]
            git_clone = testcase["git_clone"]
            fail_tests = json.loads(testcase.get("FAIL_TO_PASS", "[]"))
            pass_tests = json.loads(testcase.get("PASS_TO_PASS", "[]"))
            instance_id = testcase["instance_id"]

            parts = git_clone.split("&&")
            clone_part = parts[0].strip()
            checkout_part = parts[-1].strip() if len(parts) > 1 else None

            repo_url = clone_part.split()[2]

            # Remove existing repo directory if it exists
            if os.path.exists(repo_dir):
                subprocess.run(["rm", "-rf", repo_dir], check=True)

            print(f"Cloning repository {repo_url} into {repo_dir}...")
            env = os.environ.copy()
            env["GIT_TERMINAL_PROMPT"] = "0"
            subprocess.run(["git", "clone", repo_url, repo_dir], check=True, env=env)

            if checkout_part:
                commit_hash = checkout_part.split()[-1]
                print(f"Checking out commit: {commit_hash}")
                subprocess.run(
                    ["git", "checkout", commit_hash], cwd=repo_dir, check=True, env=env
                )

            full_prompt = (
                f"You are a team of agents working together to fix a problem.\n"
                f"Work in the directory: repo_{index}. This is a Git repository.\n"
                f"Your goal is to fix the problem described below.\n"
                f"All code changes must be saved to the files, so they appear in `git diff`.\n"
                f"The fix will be verified by running the affected tests.\n\n"
                f"Problem description:\n"
                f"{prompt}\n\n"
                f"Make sure the fix is minimal and only touches what's necessary to resolve the failing tests."
            )

            print("Launching LangGraph agents...")
            result = await self.solve_problem(full_prompt, instance_id, fail_tests, pass_tests)

            # Call REST service for evaluation
            print(f"Calling SWE-Bench REST service with repo: {repo_dir}")
            test_payload = {
                "instance_id": instance_id,
                "repoDir": f"/repos/repo_{index}",
                "FAIL_TO_PASS": fail_tests,
                "PASS_TO_PASS": pass_tests,
            }
            res = requests.post("http://localhost:8082/test", json=test_payload)
            res.raise_for_status()
            result_raw = res.json().get("harnessOutput", "{}")
            result_json = json.loads(result_raw)

            if not result_json:
                raise ValueError("No data in harnessOutput")

            instance_id = next(iter(result_json))
            tests_status = result_json[instance_id]["tests_status"]
            fail_pass_results = tests_status["FAIL_TO_PASS"]
            fail_pass_total = len(fail_pass_results["success"]) + len(fail_pass_results["failure"])
            fail_pass_passed = len(fail_pass_results["success"])
            pass_pass_results = tests_status["PASS_TO_PASS"]
            pass_pass_total = len(pass_pass_results["success"]) + len(pass_pass_results["failure"])
            pass_pass_passed = len(pass_pass_results["success"])

            # Log results
            os.chdir(start_dir)
            with open(self.LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"\n--- TESTCASE {index} ---\n")
                log.write(f"FAIL_TO_PASS passed: {fail_pass_passed}/{fail_pass_total}\n")
                log.write(f"PASS_TO_PASS passed: {pass_pass_passed}/{pass_pass_total}\n")
                log.write(f"Iterations: {result.get('iterations', 0)}\n")
                log.write(f"Files Modified: {result.get('files_modified', False)}\n")
            print(f"Test case {index} completed and logged.")

        except Exception as e:
            os.chdir(start_dir)
            with open(self.LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"\n--- TESTCASE {index} ---\n")
                log.write(f"Error: {e}\n")
            print(f"Error in test case {index}: {e}")


async def main():
    """Main function to run the LangGraph multi-agent system."""
    config = {
        "model": "gemma3:1b",
        "base_url": "http://localhost:11434",  # Fixed: removed /v1 for Ollama
        "max_tokens": 4096,
    }

    agent = LangGraphMultiAgentSystem(llm_config=config)
    for i in range(1, 300):
        await agent.handle_task(i)
        agent.reset()  # Reset for next task


if __name__ == "__main__":
    asyncio.run(main())