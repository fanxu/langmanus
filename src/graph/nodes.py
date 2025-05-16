import logging
import json
import logging
from copy import deepcopy
from typing import Literal
from langchain_core.messages import HumanMessage, BaseMessage

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command

from src.agents import research_agent, coder_agent, browser_agent
from src.llms.llm import get_llm_by_type
from src.config import TEAM_MEMBERS
from src.config.agents import AGENT_LLM_MAP
from src.prompts.template import apply_prompt_template
from src.tools.search import tavily_tool
from src.utils.json_utils import repair_json_output
from .types import State, Router

logger = logging.getLogger(__name__)

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"


def _prepare_response_content_string(raw_response_content: any) -> str:
    """Prepares raw response content into a single string for JSON repair."""
    if isinstance(raw_response_content, list):
        texts = []
        for item in raw_response_content:
            if isinstance(item, dict) and 'text' in item and isinstance(item.get('text'), str):
                texts.append(item['text'])
            # Optionally, handle other list item types here if needed, e.g., simple strings
            # elif isinstance(item, str):
            #     texts.append(item)
        return "\n".join(texts)
    elif isinstance(raw_response_content, str):
        return raw_response_content
    else:
        # Fallback for other types, convert to string
        return str(raw_response_content)


@tool
async def handoff_to_planner():
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return


async def research_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the researcher agent that performs research tasks."""
    logger.info("Research agent starting task")
    result = await research_agent.ainvoke(state)
    logger.info("Research agent completed task")
    raw_response_content = result["messages"][-1].content
    response_content = _prepare_response_content_string(raw_response_content)
    # 尝试修复可能的JSON输出
    response_content = repair_json_output(response_content)
    logger.debug(f"Research agent response: {response_content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name="researcher",
                )
            ]
        },
        goto="supervisor",
    )


async def code_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the coder agent that executes Python code."""
    logger.info("Code agent starting task")
    result = await coder_agent.ainvoke(state)
    logger.info("Code agent completed task")
    raw_response_content = result["messages"][-1].content
    response_content = _prepare_response_content_string(raw_response_content)
    # 尝试修复可能的JSON输出
    response_content = repair_json_output(response_content)
    logger.debug(f"Code agent response: {response_content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name="coder",
                )
            ]
        },
        goto="supervisor",
    )


async def browser_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the browser agent that performs web browsing tasks."""
    logger.info("Browser agent starting task")
    result = await browser_agent.ainvoke(state)
    logger.info("Browser agent completed task")
    raw_response_content = result["messages"][-1].content
    response_content = _prepare_response_content_string(raw_response_content)
    # 尝试修复可能的JSON输出
    response_content = repair_json_output(response_content)
    logger.debug(f"Browser agent response: {response_content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name="browser",
                )
            ]
        },
        goto="supervisor",
    )


async def supervisor_node(state: State) -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
    """Supervisor node that decides which agent should act next."""
    logger.info("Supervisor evaluating next action")
    messages = apply_prompt_template("supervisor", state)
    # preprocess messages to make supervisor execute better.
    messages = deepcopy(messages)
    for message in messages:
        if isinstance(message, BaseMessage) and message.name in TEAM_MEMBERS:
            message.content = RESPONSE_FORMAT.format(message.name, message.content)
    response = await (
        get_llm_by_type(AGENT_LLM_MAP["supervisor"])
        .with_structured_output(schema=Router, method="json_mode")
        .ainvoke(messages)
    )
    goto = response["next"]
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Supervisor response: {response}")

    if goto == "FINISH":
        goto = "__end__"
        logger.info("Workflow completed")
    else:
        logger.info(f"Supervisor delegating to: {goto}")

    return Command(goto=goto, update={"next": goto})


async def planner_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan")
    messages = apply_prompt_template("planner", state)
    # whether to enable deep thinking mode
    llm = get_llm_by_type("basic")
    if state.get("deep_thinking_mode"):
        llm = get_llm_by_type("reasoning")
    if state.get("search_before_planning"):
        searched_content = await tavily_tool.ainvoke({"query": state["messages"][-1].content})
        if isinstance(searched_content, list):
            messages = deepcopy(messages)
            messages[
                -1
            ].content += f"\n\n# Relative Search Results\n\n{json.dumps([{'title': elem['title'], 'content': elem['content']} for elem in searched_content], ensure_ascii=False)}"
        else:
            logger.error(
                f"Tavily search returned malformed response: {searched_content}"
            )
    stream = llm.astream(messages)
    full_response = ""
    async for chunk in stream:
        full_response += chunk.content
    logger.debug(f"Current state messages: {state['messages']}")
    logger.debug(f"Planner response: {full_response}")

    goto = "supervisor"
    try:
        full_response = repair_json_output(full_response)
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        goto = "__end__"

    return Command(
        update={
            "messages": [HumanMessage(content=full_response, name="planner")],
            "full_plan": full_response,
        },
        goto=goto,
    )


async def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    messages = apply_prompt_template("coordinator", state)
    response = await (
        get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        .bind_tools([handoff_to_planner])
        .ainvoke(messages)
    )
    logger.debug(f"Current state messages: {state['messages']}")

    goto = "__end__"
    if len(response.tool_calls) > 0:
        goto = "planner"

    return Command(
        goto=goto,
    )


async def reporter_node(state: State) -> Command[Literal["supervisor"]]:
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    messages = apply_prompt_template("reporter", state)
    response = await get_llm_by_type(AGENT_LLM_MAP["reporter"]).ainvoke(messages)
    logger.debug(f"Current state messages: {state['messages']}")
    raw_response_content = response.content
    response_content = _prepare_response_content_string(raw_response_content)
    # 尝试修复可能的JSON输出
    response_content = repair_json_output(response_content)
    logger.debug(f"reporter response: {response_content}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name="reporter",
                )
            ]
        },
        goto="supervisor",
    )
