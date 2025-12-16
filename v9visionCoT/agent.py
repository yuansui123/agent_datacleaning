"""
                     ┌─────────────┐
                     │   setup     │
                     │ (auto-gen   │
                     │  schema +   │
     goal,           │  base interp│
     raw_data,       └─────┬───────┘
     metadata, max_steps         │
                                 ▼
                           ┌─────────┐
                           │ planner │
                           │ - choose tools       (ToolCall[]) 
                           │ - update interp     (step-specific) 
                           └────┬────┘
                                │
                                ▼
                           ┌─────────┐
                           │executor │
                           │ - call tools.TOOL_REGISTRY[tool]
                           │   with (data, sampling_rate, **args)
                           │ - save plots → file paths
                           │ - append to execution_history
                           └────┬────┘
                                │
                                ▼
                        ┌──────────────┐
                        │image_interp  │
                        │ - vision on plots + stats
                        │ - produces Interpretation
                        │ - reasoning_history += rationale
                        └─────┬────────┘
                              │
                              ▼
                         ┌──────────┐
                         │controller│
                         │ - look at history + interp
                         │ - decide continue_loop
                         │ - enforce max_steps
                         │ - if continue_loop: step += 1
                         └───┬──────┘
               loop if       │
          continue_loop True │
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
           back to planner        goal_aggregator
                                   │
                                   ▼
                                  END
"""
import os
import json
import base64
from typing import TypedDict, List, Dict, Any, Optional, Literal

import numpy as np
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langgraph.graph import StateGraph, END
from openai import OpenAI

import tiktoken

import tools
from tools_schema import TOOLS_SCHEMAS


# ============================================================
# 0. Helpers: tool catalog + token safeguarding
# ============================================================

def build_tool_catalog_str() -> str:
    lines = []
    for t in TOOLS_SCHEMAS:
        lines.append(f"- {t['name']}: {t['description']}")
        params = t["parameters"]["properties"]
        if params:
            for pname, pinfo in params.items():
                desc = pinfo.get("description", "")
                default = pinfo.get("default", None)
                if default is not None:
                    lines.append(f"    - {pname} (default={default}): {desc}")
                else:
                    lines.append(f"    - {pname}: {desc}")
    return "\n".join(lines)


TOOLS_CATALOG_TEXT = build_tool_catalog_str()

# GPT-4o tokenizer
enc = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def enforce_token_limit(text: str, max_tokens: int, context_name: str = "") -> None:
    tok = count_tokens(text)
    if tok > max_tokens:
        raise ValueError(
            f"[TOKEN LIMIT EXCEEDED in {context_name}] "
            f"Prompt has {tok} tokens, exceeds limit of {max_tokens}. "
            f"Reduce metadata/history or increase reduction."
        )


def prompt_to_text(prompt_obj) -> str:
    return prompt_obj.to_string() if hasattr(prompt_obj, "to_string") else str(prompt_obj)

import re
def clean_json_string(s: str) -> str:
    # remove surrounding markdown fences
    s = s.strip().strip("`")

    # remove ```json and ``` blocks
    s = re.sub(r"```(?:json)?", "", s)
    s = s.replace("```", "")

    # remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    return s



# ============================================================
# 1. LLM and Vision Client Setup
# ============================================================

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or export it before running.")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    api_key=openai_api_key,
)

vision_client = OpenAI(api_key=openai_api_key)


# ============================================================
# 2. Pydantic Models
# ============================================================

class ToolCall(BaseModel):
    tool: str = Field(
        description="Name of the tool from the tool catalog to call."
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool. Do NOT include raw data or metadata; those are automatically provided."
    )
    rationale: str = Field(
        description="Why this tool and these arguments help move toward the user's goal."
    )


class GoalField(BaseModel):
    name: str = Field(
        description="Field name to appear in the final structured answer."
    )
    type: Literal["string", "number", "integer", "boolean", "enum", "object", "array"] = Field(
        description="Primitive type of the field."
    )
    description: str = Field(
        description="What this field means in the context of the user's goal."
    )
    required: bool = Field(
        default=True,
        description="Whether this field is mandatory."
    )
    allowed_values: Optional[List[str]] = Field(
        default=None,
        description="If type is 'enum', the allowed values."
    )


class GoalSchema(BaseModel):
    schema_name: str = Field(
        description="Short name for this goal's output schema, e.g. 'burst_detection'."
    )
    goal: str = Field(
        description="Original user goal this schema corresponds to."
    )
    description: str = Field(
        description="Human-readable description of what the final structured answer must contain."
    )
    fields: List[GoalField] = Field(
        description="List of fields that must appear in the final answer."
    )


class GoalOutput(BaseModel):
    schema_name: str = Field(
        description="Name of the schema used to interpret this output."
    )
    result: Dict[str, Any] = Field(
        description="The filled-in fields according to the generated goal schema."
    )
    rationale: str = Field(
        description="Short explanation describing how this result was derived."
    )


class PlannerOutput(BaseModel):
    calls: List[ToolCall] = Field(
        default_factory=list,
        description="List of plotting tool calls to execute in this step."
    )
    rationale: str = Field(
        description="High-level reasoning for why these tools were selected in this step."
    )
    interpreter_prompt_update: Optional[str] = Field(
        default=None,
        description=(
            "Optional update or refinement to the image interpreter instructions "
            "for the NEXT step. If null, keep the previous interpreter prompt."
        ),
    )


class Interpretation(BaseModel):
    findings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key findings extracted from the plots."
    )
    rationale: str = Field(
        description="Explanation of the findings from the plots."
    )


class InterpreterPrompts(BaseModel):
    image_prompt: str = Field(
        description="Instructional text for how to interpret the generated plots."
    )


class ToolExecutionItem(BaseModel):
    tool: str
    args: Dict[str, Any]
    result: Dict[str, Any]
    images: List[str] = Field(default_factory=list)


class ExecutionOutput(BaseModel):
    tools: List[ToolExecutionItem] = Field(default_factory=list)


class LoopDecision(BaseModel):
    continue_loop: bool = Field(
        description="True if we should run another planner→executor→image_interpreter cycle."
    )
    rationale: str = Field(
        description="Why this decision was made."
    )


class SetupOutput(BaseModel):
    goal_schema: GoalSchema = Field(
        description="Automatically generated schema for the final structured answer."
    )
    interpreter_prompts: InterpreterPrompts = Field(
        description="Automatically generated prompts for how to interpret plots."
    )
    rationale: str = Field(
        description="Reasoning behind the chosen schema and interpreter prompts."
    )


# ============================================================
# 3. LangGraph State
# ============================================================

class InspectionState(TypedDict, total=False):
    # static inputs
    goal: str
    raw_data: np.ndarray        # (n_channels, n_timepoints)
    metadata: Dict[str, Any]
    max_steps: int

    # configuration (generated automatically)
    goal_schema: GoalSchema
    interpreter_prompts: InterpreterPrompts

    # loop bookkeeping
    step: int
    done: bool
    planner_output: PlannerOutput
    execution_output: ExecutionOutput
    image_interp: Interpretation
    execution_history: List[Dict[str, Any]]
    reasoning_history: List[str]

    # final output
    goal_output: GoalOutput


# ============================================================
# 4. Setup Node (auto GoalSchema + base InterpreterPrompts)
# ============================================================

setup_parser = PydanticOutputParser(pydantic_object=SetupOutput)

setup_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a neural data analysis planner.

Your job in this step is ONLY to:
1) Define a structured output schema (GoalSchema) for the user's goal.
2) Define image interpretation instructions (InterpreterPrompts) for looking at neural plots.

You have access to these possible plotting tools:

{tools_catalog}

Guidelines for GoalSchema:
- Include 3–8 fields that are most relevant to the user's goal.
- Use clear, concise names and descriptions.
- Include a short 'summary' string field that gives a natural language conclusion.

Guidelines for InterpreterPrompts:
- Tell the vision model what to focus on in the plots, tied to the goal/schema.
- Mention relevant dimensions (channel differences, time windows, frequency bands, artifacts, etc.).

Respond with a JSON object matching the SetupOutput schema:
{format_instructions}
"""
    ),
    (
        "user",
        """User goal: {goal}

Metadata (JSON): {metadata}
"""
    ),
])


def setup_node(state: InspectionState) -> Dict[str, Any]:
    formatted_prompt = setup_prompt_template.format(
        goal=state["goal"],
        metadata=json.dumps(state.get("metadata", {})),
        tools_catalog=TOOLS_CATALOG_TEXT,
        format_instructions=setup_parser.get_format_instructions(),
    )

    enforce_token_limit(prompt_to_text(formatted_prompt), max_tokens=5000, context_name="setup")

    setup_out: SetupOutput = (llm | setup_parser).invoke(formatted_prompt)

    rh = state.get("reasoning_history", [])
    rh.append(f"[setup] {setup_out.rationale}")

    return {
        "goal_schema": setup_out.goal_schema,
        "interpreter_prompts": setup_out.interpreter_prompts,
        "reasoning_history": rh,
    }


# ============================================================
# 5. Planner Node (step-specific tool planning + prompt refinement)
# ============================================================

planner_parser = PydanticOutputParser(pydantic_object=PlannerOutput)

planner_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a neural data analysis planner.

You work in loops:
1) Decide which plotting / analysis tool(s) to run next.
2) The runtime executes them on the neural data and metadata.
3) Images are interpreted in a later step.
4) A controller decides whether to continue.

You ALSO may refine the image interpreter's instructions for the NEXT step.

- `interpreter_prompt_update` should be:
  - A short instruction string describing what the vision model should focus on
    in the NEXT plots (e.g., specific channels, time windows, frequency bands, or artifacts),
  - OR null if the current interpreter instructions are still appropriate.

You have access to these TOOLS (they operate on the neural data and metadata, which the runtime injects):

{tools_catalog}

IMPORTANT:
- Only choose from the listed tools.
- NEVER include raw data or metadata in tool arguments.
- Use previous execution history and interpretation to plan the next step AND to refine the interpreter prompt if helpful.
- You may choose 1–3 tools per step, but keep it minimal and targeted.

Respond with a JSON object using this schema:
{format_instructions}
"""
    ),
    (
        "user",
        """User goal: {goal}

Goal schema (JSON): {goal_schema}

Metadata (JSON): {metadata}

Current step index (0-based): {step}

Previous execution history (JSON): {history_json}

Previous image interpretation (JSON): {image_interp_json}

Current interpreter image_prompt:
{current_image_prompt}
"""
    ),
])


def planner_node(state: InspectionState) -> Dict[str, Any]:
    history = state.get("execution_history", [])
    image_interp = state.get("image_interp", None)
    image_interp_json = image_interp.json() if image_interp else "{}"

    formatted_prompt = planner_prompt_template.format(
        goal=state["goal"],
        goal_schema=state["goal_schema"].json(),
        metadata=json.dumps(state.get("metadata", {})),
        tools_catalog=TOOLS_CATALOG_TEXT,
        step=state.get("step", 0),
        history_json=json.dumps(history, default=str),
        image_interp_json=image_interp_json,
        current_image_prompt=state["interpreter_prompts"].image_prompt,
        format_instructions=planner_parser.get_format_instructions(),
    )

    enforce_token_limit(prompt_to_text(formatted_prompt), max_tokens=5000, context_name="planner")

    planner_output: PlannerOutput = (llm | planner_parser).invoke(formatted_prompt)

    rh = state.get("reasoning_history", [])
    rh.append(f"[planner step {state.get('step', 0)}] {planner_output.rationale}")

    updates: Dict[str, Any] = {
        "planner_output": planner_output,
        "reasoning_history": rh,
    }

    # Apply optional step-specific interpreter prompt refinement
    if planner_output.interpreter_prompt_update:
        current = state["interpreter_prompts"].image_prompt
        new_prompt = (
            current
            + "\n\n[Step-specific refinement]\n"
            + planner_output.interpreter_prompt_update
        )
        updates["interpreter_prompts"] = InterpreterPrompts(image_prompt=new_prompt)

    return updates


# ============================================================
# 6. Tool Executor Node
# ============================================================

def tool_executor_node(state: InspectionState) -> Dict[str, Any]:
    """
    Execute tools proposed by planner.
    Each call is to one of tools.TOOL_REGISTRY[tool_name].
    """
    planner_output = state.get("planner_output")
    if planner_output is None or not planner_output.calls:
        return {"execution_output": ExecutionOutput(tools=[])}

    calls: List[ToolCall] = planner_output.calls

    data: np.ndarray = state["raw_data"]
    sampling_rate = float(state["metadata"].get("sampling_rate", 1.0))

    exec_items: List[ToolExecutionItem] = []

    for call in calls:
        tool_name = call.tool
        args = call.args or {}

        fn = tools.TOOL_REGISTRY.get(tool_name)
        if fn is None:
            result = {
                "plot_id": None,
                "plot_type": tool_name,
                "file_path": None,
                "description": "",
                "stats_summary": "",
                "params": args,
                "errors": f"Unknown tool {tool_name}",
            }
            exec_items.append(
                ToolExecutionItem(
                    tool=tool_name,
                    args=args,
                    result=result,
                    images=[],
                )
            )
            continue

        result = fn(data=data, sampling_rate=sampling_rate, **args)
        images = [result["file_path"]] if result.get("file_path") else []

        exec_items.append(
            ToolExecutionItem(
                tool=tool_name,
                args=args,
                result=result,
                images=images,
            )
        )

    exec_out = ExecutionOutput(tools=exec_items)

    history = state.get("execution_history", [])
    history.append({
        "step": state.get("step", 0),
        "planner_rationale": state["planner_output"].rationale,
        "tools": [e.dict() for e in exec_items],
    })

    return {
        "execution_output": exec_out,
        "execution_history": history,
    }


# ============================================================
# 7. Image Interpreter Node
# ============================================================

image_interp_parser = PydanticOutputParser(pydantic_object=Interpretation)

image_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an image interpreter for neural electrophysiology analysis.

You see neural plots (time series, PSDs, rasters, spectrograms, peri-event averages, etc.).

Follow these custom instructions:

{image_prompt}

The following analysis TOOLS exist in this pipeline:

{tools_catalog}

You MUST respond using this JSON schema:
{format_instructions}
"""
    ),
    (
        "user",
        "Metadata (JSON): {metadata}\nTools used in this step (JSON): {tools_used}"
    ),
])


def image_interpreter_node(state: InspectionState) -> Dict[str, Any]:
    exec_out: ExecutionOutput = state["execution_output"]

    all_images = []
    for item in exec_out.tools:
        for img in item.images:
            if img:
                all_images.append(img)

    meta_data = state.get("metadata", {})
    prompts: InterpreterPrompts = state["interpreter_prompts"]

    if not all_images:
        interp = Interpretation(
            findings={},
            rationale="No images were generated in this step to interpret.",
        )
        return {"image_interp": interp}

    formatted_prompt = image_prompt_template.format(
        image_prompt=prompts.image_prompt,
        format_instructions=image_interp_parser.get_format_instructions(),
        tools_catalog=TOOLS_CATALOG_TEXT,
        metadata=json.dumps(meta_data),
        tools_used=json.dumps(
            [
                {
                    "tool": item.tool,
                    "args": item.args,
                    "summary": item.result.get("stats_summary", ""),
                    "description": item.result.get("description", ""),
                }
                for item in exec_out.tools
            ]
        ),
    )

    enforce_token_limit(prompt_to_text(formatted_prompt), max_tokens=5000, context_name="image_interpreter")

    user_content = [{"type": "text", "text": str(formatted_prompt)}]

    for path in all_images:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    try:
        result = vision_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_content}],
            max_tokens=1024,
        )
        response_text = result.choices[0].message.content
    except Exception as e:
        interp = Interpretation(
            findings={},
            rationale=f"Vision model failed with error: {str(e)}",
        )
        return {"image_interp": interp}

    try:
        interp: Interpretation = image_interp_parser.parse(response_text)
    except Exception as e:
        interp = Interpretation(
            findings={},
            rationale=f"Failed to parse vision model response as Interpretation JSON: {str(e)}. "
        )

    rh = state.get("reasoning_history", [])
    rh.append(f"[image_interp step {state.get('step', 0)}] {interp.rationale}")

    return {
        "image_interp": interp,
        "reasoning_history": rh,
    }


# ============================================================
# 8. Controller Node (loop + max_steps)
# ============================================================

loop_parser = PydanticOutputParser(pydantic_object=LoopDecision)

loop_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a controller for a neural data analysis agent.

The agent works in loops:
1) Planner selects tools/plots
2) Tools are executed and plots generated
3) Images are interpreted
4) You decide whether more analysis is needed

Decide if we should run ANOTHER loop or STOP and move to producing the final structured answer.

Guidelines:
- Continue if the user's goal is not yet confidently answered.
- Stop when further plots are unlikely to change the answer.
- If max_steps is reached, you MUST stop.

Return JSON with:
- continue_loop (bool)
- rationale (string)
"""
    ),
    (
        "user",
        """User goal: {goal}

Current step index (0-based): {step}

Max steps allowed: {max_steps}

Execution history (JSON): {history_json}

Latest image interpretation (JSON): {image_interp_json}
"""
    ),
])


def controller_node(state: InspectionState) -> Dict[str, Any]:
    history = state.get("execution_history", [])
    image_interp = state["image_interp"]
    step = state.get("step", 0)
    max_steps = state.get("max_steps", 5)

    formatted_prompt = loop_prompt_template.format(
        goal=state["goal"],
        step=step,
        max_steps=max_steps,
        history_json=json.dumps(history, default=str),
        image_interp_json=image_interp.json(),
        format_instructions=loop_parser.get_format_instructions(),
    )

    enforce_token_limit(prompt_to_text(formatted_prompt), max_tokens=5000, context_name="controller")

    decision: LoopDecision = (llm | loop_parser).invoke(formatted_prompt)

    # Enforce max_steps hard-stop
    if decision.continue_loop and (step + 1) >= max_steps:
        decision.continue_loop = False
        decision.rationale = (
            decision.rationale
            + f" (Overridden: reached max_steps={max_steps}, so we must stop.)"
        )

    rh = state.get("reasoning_history", [])
    rh.append(f"[controller step {step}] continue_loop={decision.continue_loop} | {decision.rationale}")

    return {
        "done": not decision.continue_loop,
        "step": step + (1 if decision.continue_loop else 0),
        "reasoning_history": rh,
    }


# ============================================================
# 9. Goal Aggregator Node (final structured answer)
# ============================================================

goal_output_parser = PydanticOutputParser(pydantic_object=GoalOutput)

goal_output_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a neural data analysis expert.

You must produce a structured answer according to the provided goal schema.
Use:
- The user's goal
- The metadata
- ALL previous image interpretations and tool execution summaries (provided as JSON)

Respond with a JSON object matching this GoalOutput schema:
{format_instructions}
"""
    ),
    (
        "user",
        """User goal: {goal}

Goal schema (JSON): {goal_schema}

Metadata (JSON): {metadata}

Full execution history (JSON): {execution_history}

Final image interpretation (JSON): {final_interp}
"""
    ),
])


def goal_aggregator_node(state: InspectionState) -> Dict[str, Any]:
    goal_schema = state["goal_schema"]
    interp = state["image_interp"]
    history = state.get("execution_history", [])

    formatted_prompt = goal_output_prompt_template.format(
        goal=state["goal"],
        goal_schema=goal_schema.json(),
        metadata=json.dumps(state.get("metadata", {})),
        execution_history=json.dumps(history, default=str),
        final_interp=interp.json(),
        format_instructions=goal_output_parser.get_format_instructions(),
    )

    enforce_token_limit(prompt_to_text(formatted_prompt), max_tokens=5000, context_name="goal_aggregator")

    goal_output: GoalOutput = (llm | goal_output_parser).invoke(formatted_prompt)
    return {
        "goal_output": goal_output,
        "done": True,
    }


# ============================================================
# 10. Build LangGraph (multi-turn loop)
# ============================================================

def build_inspection_graph():
    graph = StateGraph(
        InspectionState,
        config={"static": ["goal", "raw_data", "metadata", "max_steps"]},
    )

    graph.add_node("setup", setup_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", tool_executor_node)
    graph.add_node("image_interpreter", image_interpreter_node)
    graph.add_node("controller", controller_node)
    graph.add_node("goal_aggregator", goal_aggregator_node)

    graph.set_entry_point("setup")
    graph.add_edge("setup", "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "image_interpreter")
    graph.add_edge("image_interpreter", "controller")

    def route_after_controller(state: InspectionState) -> str:
        return "end" if state.get("done", False) else "loop"

    graph.add_conditional_edges(
        "controller",
        route_after_controller,
        {
            "loop": "planner",
            "end": "goal_aggregator",
        },
    )

    graph.add_edge("goal_aggregator", END)

    return graph.compile()


# ============================================================
# 11. User-facing convenience: run_inspection
# ============================================================

def run_inspection(
    raw_data: np.ndarray,
    metadata: Dict[str, Any],
    goal: str,
    max_steps: int = 5,
):
    """
    Run the full multi-turn chain-of-thought inspection.

    User only provides:
      - raw_data: np.ndarray (n_channels, n_timepoints)
      - metadata: dict (must include 'sampling_rate')
      - goal: str
      - max_steps: int
    """
    app = build_inspection_graph()

    init_state: InspectionState = {
        "goal": goal,
        "raw_data": raw_data,
        "metadata": metadata,
        "max_steps": max_steps,
        "step": 0,
        "done": False,
        "execution_history": [],
        "reasoning_history": [],
    }

    final_state = app.invoke(init_state)
    return final_state
