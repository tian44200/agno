from __future__ import annotations

from asyncio import CancelledError, create_task
from collections import ChainMap, deque
from dataclasses import dataclass
from inspect import iscoroutinefunction
from os import getenv
from textwrap import dedent
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    overload,
)
from uuid import uuid4

from pydantic import BaseModel

from agno.culture.manager import CultureManager
from agno.db.base import AsyncBaseDb, BaseDb, SessionType, UserMemory
from agno.db.schemas.culture import CulturalKnowledge
from agno.exceptions import (
    InputCheckError,
    ModelProviderError,
    OutputCheckError,
    RunCancelledException,
    StopAgentRun,
)
from agno.guardrails import BaseGuardrail
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.types import KnowledgeFilter
from agno.media import Audio, File, Image, Video
from agno.memory import MemoryManager
from agno.models.base import Model
from agno.models.message import Message, MessageReferences
from agno.models.metrics import Metrics
from agno.models.response import ModelResponse, ModelResponseEvent, ToolExecution
from agno.reasoning.step import NextAction, ReasoningStep, ReasoningSteps
from agno.run.agent import (
    RunEvent,
    RunInput,
    RunOutput,
    RunOutputEvent,
)
from agno.run.base import RunStatus
from agno.run.cancel import (
    cancel_run as cancel_run_global,
)
from agno.run.cancel import (
    cleanup_run,
    raise_if_cancelled,
    register_run,
)
from agno.run.messages import RunMessages
from agno.run.team import TeamRunOutputEvent
from agno.session import AgentSession, SessionSummaryManager, TeamSession, WorkflowSession
from agno.session.summary import SessionSummary
from agno.tools import Toolkit
from agno.tools.function import Function
from agno.utils.agent import (
    aget_chat_history_util,
    aget_last_run_output_util,
    aget_run_output_util,
    aget_session_metrics_util,
    aget_session_name_util,
    aget_session_state_util,
    aset_session_name_util,
    aupdate_session_state_util,
    await_for_background_tasks,
    await_for_background_tasks_stream,
    collect_joint_audios,
    collect_joint_files,
    collect_joint_images,
    collect_joint_videos,
    get_chat_history_util,
    get_last_run_output_util,
    get_run_output_util,
    get_session_metrics_util,
    get_session_name_util,
    get_session_state_util,
    scrub_history_messages_from_run_output,
    scrub_media_from_run_output,
    scrub_tool_results_from_run_output,
    set_session_name_util,
    update_session_state_util,
    wait_for_background_tasks,
    wait_for_background_tasks_stream,
)
from agno.utils.common import is_typed_dict, validate_typed_dict
from agno.utils.events import (
    create_parser_model_response_completed_event,
    create_parser_model_response_started_event,
    create_post_hook_completed_event,
    create_post_hook_started_event,
    create_pre_hook_completed_event,
    create_pre_hook_started_event,
    create_reasoning_completed_event,
    create_reasoning_started_event,
    create_reasoning_step_event,
    create_run_cancelled_event,
    create_run_completed_event,
    create_run_content_completed_event,
    create_run_continued_event,
    create_run_error_event,
    create_run_output_content_event,
    create_run_paused_event,
    create_run_started_event,
    create_session_summary_completed_event,
    create_session_summary_started_event,
    create_tool_call_completed_event,
    create_tool_call_started_event,
    handle_event,
)
from agno.utils.hooks import filter_hook_args, normalize_hooks
from agno.utils.knowledge import get_agentic_or_user_search_filters
from agno.utils.log import (
    log_debug,
    log_error,
    log_exception,
    log_info,
    log_warning,
    set_log_level_to_debug,
    set_log_level_to_info,
)
from agno.utils.merge_dict import merge_dictionaries
from agno.utils.message import filter_tool_calls, get_text_from_message
from agno.utils.print_response.agent import (
    aprint_response,
    aprint_response_stream,
    print_response,
    print_response_stream,
)
from agno.utils.prompts import get_json_output_prompt, get_response_model_format_prompt
from agno.utils.reasoning import (
    add_reasoning_metrics_to_metadata,
    add_reasoning_step_to_metadata,
    append_to_reasoning_content,
    update_run_output_with_reasoning,
)
from agno.utils.response import (
    async_generator_wrapper,
    generator_wrapper,
    get_paused_content,
)
from agno.utils.safe_formatter import SafeFormatter
from agno.utils.string import generate_id_from_name, parse_response_model_str
from agno.utils.timer import Timer


@dataclass(init=False)
class Agent:
    # --- Agent settings ---
    # Model for this Agent
    model: Optional[Model] = None
    # Agent name
    name: Optional[str] = None
    # Agent UUID (autogenerated if not set)
    id: Optional[str] = None

    # --- User settings ---
    # Default user_id to use for this agent
    user_id: Optional[str] = None

    # --- Session settings ---
    # Default session_id to use for this agent (autogenerated if not set)
    session_id: Optional[str] = None
    # Default session state (stored in the database to persist across runs)
    session_state: Optional[Dict[str, Any]] = None
    # Set to True to add the session_state to the context
    add_session_state_to_context: bool = False
    # Set to True to give the agent tools to update the session_state dynamically
    enable_agentic_state: bool = False
    # Set to True to overwrite the stored session_state with the session_state provided in the run. Default behaviour merges the current session state with the session state in the db
    overwrite_db_session_state: bool = False
    # If True, cache the current Agent session in memory for faster access
    cache_session: bool = False

    search_session_history: Optional[bool] = False
    num_history_sessions: Optional[int] = None
    # If True, the agent creates/updates session summaries at the end of runs
    enable_session_summaries: bool = False
    # If True, the agent adds session summaries to the context
    add_session_summary_to_context: Optional[bool] = None
    # Session summary manager
    session_summary_manager: Optional[SessionSummaryManager] = None

    # --- Agent Dependencies ---
    # Dependencies available for tools and prompt functions
    dependencies: Optional[Dict[str, Any]] = None
    # If True, add the dependencies to the user prompt
    add_dependencies_to_context: bool = False

    # --- Agent Memory ---
    # Memory manager to use for this agent
    memory_manager: Optional[MemoryManager] = None
    # Enable the agent to manage memories of the user
    enable_agentic_memory: bool = False
    # If True, the agent creates/updates user memories at the end of runs
    enable_user_memories: bool = False
    # If True, the agent adds a reference to the user memories in the response
    add_memories_to_context: Optional[bool] = None

    # --- Database ---
    # Database to use for this agent
    db: Optional[Union[BaseDb, AsyncBaseDb]] = None

    # --- Agent History ---
    # add_history_to_context=true adds messages from the chat history to the messages list sent to the Model.
    add_history_to_context: bool = False
    # Number of historical runs to include in the messages
    num_history_runs: Optional[int] = None
    # Number of historical messages to include in the messages list sent to the Model.
    num_history_messages: Optional[int] = None
    # Maximum number of tool calls to include from history (None = no limit)
    max_tool_calls_from_history: Optional[int] = None

    # --- Knowledge ---
    knowledge: Optional[Knowledge] = None
    # Enable RAG by adding references from Knowledge to the user prompt.
    # Add knowledge_filters to the Agent class attributes
    knowledge_filters: Optional[Dict[str, Any]] = None
    # Let the agent choose the knowledge filters
    enable_agentic_knowledge_filters: Optional[bool] = False
    add_knowledge_to_context: bool = False
    # Retrieval function to get references
    # This function, if provided, is used instead of the default search_knowledge function
    # Signature:
    # def knowledge_retriever(agent: Agent, query: str, num_documents: Optional[int], **kwargs) -> Optional[list[dict]]:
    #     ...
    knowledge_retriever: Optional[Callable[..., Optional[List[Union[Dict, str]]]]] = None
    references_format: Literal["json", "yaml"] = "json"

    # --- Agent Tools ---
    # A list of tools provided to the Model.
    # Tools are functions the model may generate JSON inputs for.
    tools: Optional[List[Union[Toolkit, Callable, Function, Dict]]] = None

    # Maximum number of tool calls allowed.
    tool_call_limit: Optional[int] = None
    # Controls which (if any) tool is called by the model.
    # "none" means the model will not call a tool and instead generates a message.
    # "auto" means the model can pick between generating a message or calling a tool.
    # Specifying a particular function via {"type: "function", "function": {"name": "my_function"}}
    #   forces the model to call that tool.
    # "none" is the default when no tools are present. "auto" is the default if tools are present.
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # A function that acts as middleware and is called around tool calls.
    tool_hooks: Optional[List[Callable]] = None

    # --- Agent Hooks ---
    # Functions called right after agent-session is loaded, before processing starts
    pre_hooks: Optional[Union[List[Callable[..., Any]], List[BaseGuardrail]]] = None
    # Functions called after output is generated but before the response is returned
    post_hooks: Optional[Union[List[Callable[..., Any]], List[BaseGuardrail]]] = None

    # --- Agent Reasoning ---
    # Enable reasoning by working through the problem step by step.
    reasoning: bool = False
    reasoning_model: Optional[Model] = None
    reasoning_agent: Optional[Agent] = None
    reasoning_min_steps: int = 1
    reasoning_max_steps: int = 10

    # --- Default tools ---
    # Add a tool that allows the Model to read the chat history.
    read_chat_history: bool = False
    # Add a tool that allows the Model to search the knowledge base (aka Agentic RAG)
    # Added only if knowledge is provided.
    search_knowledge: bool = True
    # Call limit for search_knowledge tool per run. None means no limit.
    search_knowledge_call_limit: Optional[int] = None
    # Add a tool that allows the Agent to update Knowledge.
    update_knowledge: bool = False
    # Add a tool that allows the Model to get the tool call history.
    read_tool_call_history: bool = False
    # If False, media (images, videos, audio, files) is only available to tools and not sent to the LLM
    send_media_to_model: bool = True
    # If True, store media in run output
    store_media: bool = True
    # If True, store tool results in run output
    store_tool_messages: bool = True
    # If True, store history messages in run output
    store_history_messages: bool = True

    # --- System message settings ---
    # Provide the system message as a string or function
    system_message: Optional[Union[str, Callable, Message]] = None
    # Role for the system message
    system_message_role: str = "system"
    # Set to False to skip context building
    build_context: bool = True

    # --- Settings for building the default system message ---
    # A description of the Agent that is added to the start of the system message.
    description: Optional[str] = None
    # List of instructions for the agent.
    instructions: Optional[Union[str, List[str], Callable]] = None
    # Provide the expected output from the Agent.
    expected_output: Optional[str] = None
    # Additional context added to the end of the system message.
    additional_context: Optional[str] = None
    # If markdown=true, add instructions to format the output using markdown
    markdown: bool = False
    # If True, add the agent name to the instructions
    add_name_to_context: bool = False
    # If True, add the current datetime to the instructions to give the agent a sense of time
    # This allows for relative times like "tomorrow" to be used in the prompt
    add_datetime_to_context: bool = False
    # If True, add the current location to the instructions to give the agent a sense of place
    # This allows for location-aware responses and local context
    add_location_to_context: bool = False
    # Allows for custom timezone for datetime instructions following the TZ Database format (e.g. "Etc/UTC")
    timezone_identifier: Optional[str] = None
    # If True, resolve session_state, dependencies, and metadata in the user and system messages
    resolve_in_context: bool = True

    # --- Extra Messages ---
    # A list of extra messages added after the system message and before the user message.
    # Use these for few-shot learning or to provide additional context to the Model.
    # Note: these are not retained in memory, they are added directly to the messages sent to the model.
    additional_input: Optional[List[Union[str, Dict, BaseModel, Message]]] = None
    # --- User message settings ---
    # Role for the user message
    user_message_role: str = "user"
    # Set to False to skip building the user context
    build_user_context: bool = True

    # --- Agent Response Settings ---
    # Number of retries to attempt
    retries: int = 0
    # Delay between retries (in seconds)
    delay_between_retries: int = 1
    # Exponential backoff: if True, the delay between retries is doubled each time
    exponential_backoff: bool = False

    # --- Agent Response Model Settings ---
    # Provide an input schema to validate the input
    input_schema: Optional[Type[BaseModel]] = None
    # Provide a response model to get the response as a Pydantic model
    output_schema: Optional[Type[BaseModel]] = None
    # Provide a secondary model to parse the response from the primary model
    parser_model: Optional[Model] = None
    # Provide a prompt for the parser model
    parser_model_prompt: Optional[str] = None
    # Provide an output model to structure the response from the main model
    output_model: Optional[Model] = None
    # Provide a prompt for the output model
    output_model_prompt: Optional[str] = None
    # If True, the response from the Model is converted into the output_schema
    # Otherwise, the response is returned as a JSON string
    parse_response: bool = True
    # Use model enforced structured_outputs if supported (e.g. OpenAIChat)
    structured_outputs: Optional[bool] = None
    # If `output_schema` is set, sets the response mode of the model, i.e. if the model should explicitly respond with a JSON object instead of a Pydantic model
    use_json_mode: bool = False
    # Save the response to a file
    save_response_to_file: Optional[str] = None

    # --- Agent Streaming ---
    # Stream the response from the Agent
    stream: Optional[bool] = None
    # Stream the intermediate steps from the Agent
    stream_events: Optional[bool] = None
    # [Deprecated] Stream the intermediate steps from the Agent
    stream_intermediate_steps: Optional[bool] = None

    # Persist the events on the run response
    store_events: bool = False
    events_to_skip: Optional[List[RunEvent]] = None

    # --- If this Agent is part of a team ---
    # If this Agent is part of a team, this is the role of the agent in the team
    role: Optional[str] = None
    # Optional team ID. Indicates this agent is part of a team.
    team_id: Optional[str] = None

    # --- If this Agent is part of a workflow ---
    # Optional workflow ID. Indicates this agent is part of a workflow.
    workflow_id: Optional[str] = None

    # Metadata stored with this agent
    metadata: Optional[Dict[str, Any]] = None

    # --- Experimental Features ---
    # --- Agent Culture ---
    # Culture manager to use for this agent
    culture_manager: Optional[CultureManager] = None
    # Enable the agent to manage cultural knowledge
    enable_agentic_culture: bool = False
    # Update cultural knowledge after every run
    update_cultural_knowledge: bool = False
    # If True, the agent adds cultural knowledge in the response
    add_culture_to_context: Optional[bool] = None

    # --- Debug ---
    # Enable debug logs
    debug_mode: bool = False
    debug_level: Literal[1, 2] = 1

    # --- Telemetry ---
    # telemetry=True logs minimal telemetry for analytics
    # This helps us improve the Agent and provide better support
    telemetry: bool = True

    def __init__(
        self,
        *,
        model: Optional[Model] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
        introduction: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        add_session_state_to_context: bool = False,
        overwrite_db_session_state: bool = False,
        enable_agentic_state: bool = False,
        cache_session: bool = False,
        search_session_history: Optional[bool] = False,
        num_history_sessions: Optional[int] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        add_dependencies_to_context: bool = False,
        db: Optional[Union[BaseDb, AsyncBaseDb]] = None,
        memory_manager: Optional[MemoryManager] = None,
        enable_agentic_memory: bool = False,
        enable_user_memories: bool = False,
        add_memories_to_context: Optional[bool] = None,
        enable_session_summaries: bool = False,
        add_session_summary_to_context: Optional[bool] = None,
        session_summary_manager: Optional[SessionSummaryManager] = None,
        add_history_to_context: bool = False,
        num_history_runs: Optional[int] = None,
        num_history_messages: Optional[int] = None,
        max_tool_calls_from_history: Optional[int] = None,
        store_media: bool = True,
        store_tool_messages: bool = True,
        store_history_messages: bool = True,
        knowledge: Optional[Knowledge] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        enable_agentic_knowledge_filters: Optional[bool] = None,
        add_knowledge_to_context: bool = False,
        knowledge_retriever: Optional[Callable[..., Optional[List[Union[Dict, str]]]]] = None,
        references_format: Literal["json", "yaml"] = "json",
        metadata: Optional[Dict[str, Any]] = None,
        tools: Optional[Sequence[Union[Toolkit, Callable, Function, Dict]]] = None,
        tool_call_limit: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_hooks: Optional[List[Callable]] = None,
        pre_hooks: Optional[Union[List[Callable[..., Any]], List[BaseGuardrail]]] = None,
        post_hooks: Optional[Union[List[Callable[..., Any]], List[BaseGuardrail]]] = None,
        reasoning: bool = False,
        reasoning_model: Optional[Model] = None,
        reasoning_agent: Optional[Agent] = None,
        reasoning_min_steps: int = 1,
        reasoning_max_steps: int = 10,
        read_chat_history: bool = False,
        search_knowledge: bool = True,
        search_knowledge_call_limit: Optional[int] = None,
        update_knowledge: bool = False,
        read_tool_call_history: bool = False,
        send_media_to_model: bool = True,
        system_message: Optional[Union[str, Callable, Message]] = None,
        system_message_role: str = "system",
        build_context: bool = True,
        description: Optional[str] = None,
        instructions: Optional[Union[str, List[str], Callable]] = None,
        expected_output: Optional[str] = None,
        additional_context: Optional[str] = None,
        markdown: bool = False,
        add_name_to_context: bool = False,
        add_datetime_to_context: bool = False,
        add_location_to_context: bool = False,
        timezone_identifier: Optional[str] = None,
        resolve_in_context: bool = True,
        additional_input: Optional[List[Union[str, Dict, BaseModel, Message]]] = None,
        user_message_role: str = "user",
        build_user_context: bool = True,
        retries: int = 0,
        delay_between_retries: int = 1,
        exponential_backoff: bool = False,
        parser_model: Optional[Model] = None,
        parser_model_prompt: Optional[str] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        parse_response: bool = True,
        output_model: Optional[Model] = None,
        output_model_prompt: Optional[str] = None,
        structured_outputs: Optional[bool] = None,
        use_json_mode: bool = False,
        save_response_to_file: Optional[str] = None,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        store_events: bool = False,
        events_to_skip: Optional[List[RunEvent]] = None,
        role: Optional[str] = None,
        culture_manager: Optional[CultureManager] = None,
        enable_agentic_culture: bool = False,
        update_cultural_knowledge: bool = False,
        add_culture_to_context: Optional[bool] = None,
        debug_mode: bool = False,
        debug_level: Literal[1, 2] = 1,
        telemetry: bool = True,
    ):
        self.model = model
        self.name = name
        self.id = id
        self.introduction = introduction
        self.user_id = user_id

        self.session_id = session_id
        self.session_state = session_state
        self.overwrite_db_session_state = overwrite_db_session_state
        self.enable_agentic_state = enable_agentic_state
        self.cache_session = cache_session

        self.search_session_history = search_session_history
        self.num_history_sessions = num_history_sessions

        self.dependencies = dependencies
        self.add_dependencies_to_context = add_dependencies_to_context
        self.add_session_state_to_context = add_session_state_to_context

        self.db = db

        self.memory_manager = memory_manager
        self.enable_agentic_memory = enable_agentic_memory
        self.enable_user_memories = enable_user_memories
        self.add_memories_to_context = add_memories_to_context

        self.session_summary_manager = session_summary_manager
        self.enable_session_summaries = enable_session_summaries
        self.add_session_summary_to_context = add_session_summary_to_context

        self.add_history_to_context = add_history_to_context
        self.num_history_runs = num_history_runs
        self.num_history_messages = num_history_messages
        if self.num_history_messages is not None and self.num_history_runs is not None:
            log_warning(
                "num_history_messages and num_history_runs cannot be set at the same time. Using num_history_runs."
            )
            self.num_history_messages = None
        if self.num_history_messages is None and self.num_history_runs is None:
            self.num_history_runs = 3

        self.max_tool_calls_from_history = max_tool_calls_from_history

        self.store_media = store_media
        self.store_tool_messages = store_tool_messages
        self.store_history_messages = store_history_messages

        self.knowledge = knowledge
        self.knowledge_filters = knowledge_filters
        self.enable_agentic_knowledge_filters = enable_agentic_knowledge_filters
        self.add_knowledge_to_context = add_knowledge_to_context
        self.knowledge_retriever = knowledge_retriever
        self.references_format = references_format

        self.metadata = metadata

        self.tools = list(tools) if tools else []
        self.tool_call_limit = tool_call_limit
        self.tool_choice = tool_choice
        self.tool_hooks = tool_hooks

        # Initialize hooks with backward compatibility
        self.pre_hooks = pre_hooks
        self.post_hooks = post_hooks

        self.reasoning = reasoning
        self.reasoning_model = reasoning_model
        self.reasoning_agent = reasoning_agent
        self.reasoning_min_steps = reasoning_min_steps
        self.reasoning_max_steps = reasoning_max_steps

        self.read_chat_history = read_chat_history
        self.search_knowledge = search_knowledge
        self.search_knowledge_call_limit = search_knowledge_call_limit
        self.update_knowledge = update_knowledge
        self.read_tool_call_history = read_tool_call_history
        self.send_media_to_model = send_media_to_model
        self.system_message = system_message
        self.system_message_role = system_message_role
        self.build_context = build_context

        self.description = description
        self.instructions = instructions
        self.expected_output = expected_output
        self.additional_context = additional_context
        self.markdown = markdown
        self.add_name_to_context = add_name_to_context
        self.add_datetime_to_context = add_datetime_to_context
        self.add_location_to_context = add_location_to_context
        self.timezone_identifier = timezone_identifier
        self.resolve_in_context = resolve_in_context
        self.additional_input = additional_input
        self.user_message_role = user_message_role
        self.build_user_context = build_user_context

        self.retries = retries
        self.delay_between_retries = delay_between_retries
        self.exponential_backoff = exponential_backoff
        self.parser_model = parser_model
        self.parser_model_prompt = parser_model_prompt
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.parse_response = parse_response
        self.output_model = output_model
        self.output_model_prompt = output_model_prompt

        self.structured_outputs = structured_outputs

        self.use_json_mode = use_json_mode
        self.save_response_to_file = save_response_to_file

        self.stream = stream
        self.stream_events = stream_events or stream_intermediate_steps

        self.store_events = store_events
        self.role = role
        # By default, we skip the run response content event
        self.events_to_skip = events_to_skip
        if self.events_to_skip is None:
            self.events_to_skip = [RunEvent.run_content]

        self.culture_manager = culture_manager
        self.enable_agentic_culture = enable_agentic_culture
        self.update_cultural_knowledge = update_cultural_knowledge
        self.add_culture_to_context = add_culture_to_context

        self.debug_mode = debug_mode
        if debug_level not in [1, 2]:
            log_warning(f"Invalid debug level: {debug_level}. Setting to 1.")
            debug_level = 1
        self.debug_level = debug_level
        self.telemetry = telemetry

        # If we are caching the agent session
        self._cached_session: Optional[AgentSession] = None

        self._tool_instructions: Optional[List[str]] = None

        self._formatter: Optional[SafeFormatter] = None

        self._hooks_normalised = False

        self._mcp_tools_initialized_on_run: List[Any] = []

        # Lazy-initialized shared thread pool executor for background tasks (memory, cultural knowledge, etc.)
        self._background_executor: Optional[Any] = None

    @property
    def background_executor(self) -> Any:
        """Lazy initialization of shared thread pool executor for background tasks.

        Handles both memory creation and cultural knowledge updates concurrently.
        Initialized only on first use (runtime, not instantiation) and reused across runs.
        """
        if self._background_executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._background_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="agno-bg")
        return self._background_executor

    @property
    def should_parse_structured_output(self) -> bool:
        return self.output_schema is not None and self.parse_response and self.parser_model is None

    @property
    def cached_session(self) -> Optional[AgentSession]:
        return self._cached_session

    def set_id(self) -> None:
        if self.id is None:
            self.id = generate_id_from_name(self.name)

    def _set_debug(self, debug_mode: Optional[bool] = None) -> None:
        # If the default debug mode is set, or passed on run, or via environment variable, set the debug mode to True
        if self.debug_mode or debug_mode or getenv("AGNO_DEBUG", "false").lower() == "true":
            set_log_level_to_debug(level=self.debug_level)
        else:
            set_log_level_to_info()

    def _set_telemetry(self) -> None:
        """Override telemetry settings based on environment variables."""

        telemetry_env = getenv("AGNO_TELEMETRY")
        if telemetry_env is not None:
            self.telemetry = telemetry_env.lower() == "true"

    def _set_default_model(self) -> None:
        # Use the default Model (OpenAIChat) if no model is provided
        if self.model is None:
            try:
                from agno.models.openai import OpenAIChat
            except ModuleNotFoundError as e:
                log_exception(e)
                log_error(
                    "Agno agents use `openai` as the default model provider. "
                    "Please provide a `model` or install `openai`."
                )
                exit(1)

            log_info("Setting default model to OpenAI Chat")
            self.model = OpenAIChat(id="gpt-4o")

    def _validate_input(
        self, input: Union[str, List, Dict, Message, BaseModel]
    ) -> Union[str, List, Dict, Message, BaseModel]:
        """Parse and validate input against input_schema if provided, otherwise return input as-is"""
        if self.input_schema is None:
            return input  # Return input unchanged if no schema is set

        # Handle Message objects - extract content
        if isinstance(input, Message):
            input = input.content  # type: ignore

        # If input is a string, convert it to a dict
        if isinstance(input, str):
            import json

            try:
                input = json.loads(input)
            except Exception as e:
                raise ValueError(f"Failed to parse input. Is it a valid JSON string?: {e}")

        # Case 1: Message is already a BaseModel instance
        if isinstance(input, BaseModel):
            if isinstance(input, self.input_schema):
                try:
                    return input
                except Exception as e:
                    raise ValueError(f"BaseModel validation failed: {str(e)}")
            else:
                # Different BaseModel types
                raise ValueError(f"Expected {self.input_schema.__name__} but got {type(input).__name__}")

        # Case 2: Message is a dict
        elif isinstance(input, dict):
            try:
                # Check if the schema is a TypedDict
                if is_typed_dict(self.input_schema):
                    validated_dict = validate_typed_dict(input, self.input_schema)
                    return validated_dict
                else:
                    validated_model = self.input_schema(**input)
                    return validated_model
            except Exception as e:
                raise ValueError(f"Failed to parse dict into {self.input_schema.__name__}: {str(e)}")

        # Case 3: Other types not supported for structured input
        else:
            raise ValueError(
                f"Cannot validate {type(input)} against input_schema. Expected dict or {self.input_schema.__name__} instance."
            )

    def _set_culture_manager(self) -> None:
        if self.db is None:
            log_warning("Database not provided. Cultural knowledge will not be stored.")

        if self.culture_manager is None:
            self.culture_manager = CultureManager(model=self.model, db=self.db)
        else:
            if self.culture_manager.model is None:
                self.culture_manager.model = self.model
            if self.culture_manager.db is None:
                self.culture_manager.db = self.db

        if self.add_culture_to_context is None:
            self.add_culture_to_context = (
                self.enable_agentic_culture or self.update_cultural_knowledge or self.culture_manager is not None
            )

    def _set_memory_manager(self) -> None:
        if self.db is None:
            log_warning("Database not provided. Memories will not be stored.")

        if self.memory_manager is None:
            self.memory_manager = MemoryManager(model=self.model, db=self.db)
        else:
            if self.memory_manager.model is None:
                self.memory_manager.model = self.model
            if self.memory_manager.db is None:
                self.memory_manager.db = self.db

        if self.add_memories_to_context is None:
            self.add_memories_to_context = (
                self.enable_user_memories or self.enable_agentic_memory or self.memory_manager is not None
            )

    def _set_session_summary_manager(self) -> None:
        if self.enable_session_summaries and self.session_summary_manager is None:
            self.session_summary_manager = SessionSummaryManager(model=self.model)

        if self.session_summary_manager is not None:
            if self.session_summary_manager.model is None:
                self.session_summary_manager.model = self.model

        if self.add_session_summary_to_context is None:
            self.add_session_summary_to_context = (
                self.enable_session_summaries or self.session_summary_manager is not None
            )

    def _has_async_db(self) -> bool:
        """Return True if the db the agent is equipped with is an Async implementation"""
        return self.db is not None and isinstance(self.db, AsyncBaseDb)

    def initialize_agent(self, debug_mode: Optional[bool] = None) -> None:
        self._set_default_model()
        self._set_debug(debug_mode=debug_mode)
        self.set_id()
        if self.enable_user_memories or self.enable_agentic_memory or self.memory_manager is not None:
            self._set_memory_manager()
        if (
            self.add_culture_to_context
            or self.update_cultural_knowledge
            or self.enable_agentic_culture
            or self.culture_manager is not None
        ):
            self._set_culture_manager()
        if self.enable_session_summaries or self.session_summary_manager is not None:
            self._set_session_summary_manager()

        log_debug(f"Agent ID: {self.id}", center=True)

        if self._formatter is None:
            self._formatter = SafeFormatter()

    def add_tool(self, tool: Union[Toolkit, Callable, Function, Dict]):
        if not self.tools:
            self.tools = []
        self.tools.append(tool)

    def set_tools(self, tools: Sequence[Union[Toolkit, Callable, Function, Dict]]):
        self.tools = list(tools) if tools else []

    async def _connect_mcp_tools(self) -> None:
        """Connect the MCP tools to the agent."""
        if self.tools:
            for tool in self.tools:
                if tool.__class__.__name__ in ["MCPTools", "MultiMCPTools"] and not tool.initialized:  # type: ignore
                    # Connect the MCP server
                    await tool.connect()  # type: ignore
                    self._mcp_tools_initialized_on_run.append(tool)

    async def _disconnect_mcp_tools(self) -> None:
        """Disconnect the MCP tools from the agent."""
        for tool in self._mcp_tools_initialized_on_run:
            await tool.close()
        self._mcp_tools_initialized_on_run = []

    def _initialize_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """Initialize the session for the agent."""

        if session_id is None:
            if self.session_id:
                session_id = self.session_id
            else:
                session_id = str(uuid4())
                # We make the session_id sticky to the agent instance if no session_id is provided
                self.session_id = session_id

        log_debug(f"Session ID: {session_id}", center=True)

        # Use the default user_id when necessary
        if user_id is None or user_id == "":
            user_id = self.user_id

        return session_id, user_id

    def _initialize_session_state(
        self,
        session_state: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Initialize the session state for the agent."""
        if user_id:
            session_state["current_user_id"] = user_id
        if session_id is not None:
            session_state["current_session_id"] = session_id
        if run_id is not None:
            session_state["current_run_id"] = run_id
        return session_state

    def _run(
        self,
        run_response: RunOutput,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> RunOutput:
        """Run the Agent and return the RunOutput.

        Steps:
        1. Execute pre-hooks
        2. Determine tools for model
        3. Prepare run messages
        4. Start memory creation in background thread
        5. Reason about the task if reasoning is enabled
        6. Generate a response from the Model (includes running function calls)
        7. Update the RunOutput with the model response
        8. Store media if enabled
        9. Convert the response to the structured format if needed
        10. Execute post-hooks
        11. Wait for background memory creation and cultural knowledge creation
        12. Create session summary
        13. Cleanup and store the run response and session
        """

        # Register run for cancellation tracking
        register_run(run_response.run_id)  # type: ignore

        # 1. Execute pre-hooks
        run_input = cast(RunInput, run_response.input)
        self.model = cast(Model, self.model)
        if self.pre_hooks is not None:
            # Can modify the run input
            pre_hook_iterator = self._execute_pre_hooks(
                hooks=self.pre_hooks,  # type: ignore
                run_response=run_response,
                run_input=run_input,
                session_state=session_state,
                dependencies=dependencies,
                metadata=metadata,
                session=session,
                user_id=user_id,
                debug_mode=debug_mode,
                **kwargs,
            )
            # Consume the generator without yielding
            deque(pre_hook_iterator, maxlen=0)

        # 2. Determine tools for model
        processed_tools = self.get_tools(
            run_response=run_response,
            session=session,
            user_id=user_id,
            knowledge_filters=knowledge_filters,
        )
        _tools = self._determine_tools_for_model(
            model=self.model,
            processed_tools=processed_tools,
            run_response=run_response,
            session=session,
            session_state=session_state,
            dependencies=dependencies,
        )

        # 3. Prepare run messages
        run_messages: RunMessages = self._get_run_messages(
            run_response=run_response,
            input=run_input.input_content,
            session=session,
            session_state=session_state,
            user_id=user_id,
            audio=run_input.audios,
            images=run_input.images,
            videos=run_input.videos,
            files=run_input.files,
            knowledge_filters=knowledge_filters,
            add_history_to_context=add_history_to_context,
            dependencies=dependencies,
            add_dependencies_to_context=add_dependencies_to_context,
            add_session_state_to_context=add_session_state_to_context,
            metadata=metadata,
            tools=_tools,
            **kwargs,
        )
        if len(run_messages.messages) == 0:
            log_error("No messages to be sent to the model.")

        log_debug(f"Agent Run Start: {run_response.run_id}", center=True)

        # Start memory creation on a separate thread (runs concurrently with the main execution loop)
        memory_future = None
        # 4. Start memory creation in background thread if memory manager is enabled and agentic memory is disabled
        if run_messages.user_message is not None and self.memory_manager is not None and not self.enable_agentic_memory:
            log_debug("Starting memory creation in background thread.")
            memory_future = self.background_executor.submit(
                self._make_memories, run_messages=run_messages, user_id=user_id
            )

        # Start cultural knowledge creation on a separate thread (runs concurrently with the main execution loop)
        cultural_knowledge_future = None
        if (
            run_messages.user_message is not None
            and self.culture_manager is not None
            and self.update_cultural_knowledge
        ):
            log_debug("Starting cultural knowledge creation in background thread.")
            cultural_knowledge_future = self.background_executor.submit(
                self._make_cultural_knowledge, run_messages=run_messages
            )

        try:
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 5. Reason about the task
            self._handle_reasoning(run_response=run_response, run_messages=run_messages)

            # Check for cancellation before model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 6. Generate a response from the Model (includes running function calls)
            self.model = cast(Model, self.model)
            model_response: ModelResponse = self.model.response(
                messages=run_messages.messages,
                tools=_tools,
                tool_choice=self.tool_choice,
                tool_call_limit=self.tool_call_limit,
                per_tool_call_limits=self._build_per_tool_call_limits(_tools) if _tools else None,
                response_format=response_format,
                run_response=run_response,
                send_media_to_model=self.send_media_to_model,
            )

            # Check for cancellation after model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # If an output model is provided, generate output using the output model
            self._generate_response_with_output_model(model_response, run_messages)

            # If a parser model is provided, structure the response separately
            self._parse_response_with_parser_model(model_response, run_messages)

            # 7. Update the RunOutput with the model response
            self._update_run_response(
                model_response=model_response, run_response=run_response, run_messages=run_messages
            )

            # We should break out of the run function
            if any(tool_call.is_paused for tool_call in run_response.tools or []):
                wait_for_background_tasks(
                    memory_future=memory_future, cultural_knowledge_future=cultural_knowledge_future
                )

                return self._handle_agent_run_paused(run_response=run_response, session=session, user_id=user_id)

            # 8. Store media if enabled
            if self.store_media:
                self._store_media(run_response, model_response)

            # 9. Convert the response to the structured format if needed
            self._convert_response_to_structured_format(run_response)

            # 10. Execute post-hooks after output is generated but before response is returned
            if self.post_hooks is not None:
                post_hook_iterator = self._execute_post_hooks(
                    hooks=self.post_hooks,  # type: ignore
                    run_output=run_response,
                    session=session,
                    user_id=user_id,
                    session_state=session_state,
                    dependencies=dependencies,
                    metadata=metadata,
                    debug_mode=debug_mode,
                    **kwargs,
                )
                deque(post_hook_iterator, maxlen=0)

            # Check for cancellation
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 11. Wait for background memory creation and cultural knowledge creation
            wait_for_background_tasks(memory_future=memory_future, cultural_knowledge_future=cultural_knowledge_future)

            # 12. Create session summary
            if self.session_summary_manager is not None:
                # Upsert the RunOutput to Agent Session before creating the session summary
                session.upsert_run(run=run_response)
                try:
                    self.session_summary_manager.create_session_summary(session=session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")

            run_response.status = RunStatus.completed

            # 13. Cleanup and store the run response and session
            self._cleanup_and_store(run_response=run_response, session=session, user_id=user_id)

            # Log Agent Telemetry
            self._log_agent_telemetry(session_id=session.session_id, run_id=run_response.run_id)

            log_debug(f"Agent Run End: {run_response.run_id}", center=True, symbol="*")

            return run_response
        except RunCancelledException as e:
            # Handle run cancellation
            log_info(f"Run {run_response.run_id} was cancelled")
            run_response.content = str(e)
            run_response.status = RunStatus.cancelled

            # Cleanup and store the run response and session
            self._cleanup_and_store(run_response=run_response, session=session, user_id=user_id)

            return run_response
        finally:
            # Always clean up the run tracking
            cleanup_run(run_response.run_id)  # type: ignore

    def _run_stream(
        self,
        run_response: RunOutput,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        stream_events: bool = False,
        yield_run_response: bool = False,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[Union[RunOutputEvent, RunOutput]]:
        """Run the Agent and yield the RunOutput.

        Steps:
        1. Execute pre-hooks
        2. Determine tools for model
        3. Prepare run messages
        4. Start memory creation in background thread
        5. Reason about the task if reasoning is enabled
        6. Process model response
        7. Parse response with parser model if provided
        8. Wait for background memory creation and cultural knowledge creation
        9. Create session summary
        10. Cleanup and store the run response and session
        """

        # Register run for cancellation tracking
        register_run(run_response.run_id)  # type: ignore

        # 1. Execute pre-hooks
        run_input = cast(RunInput, run_response.input)
        self.model = cast(Model, self.model)
        if self.pre_hooks is not None:
            # Can modify the run input
            pre_hook_iterator = self._execute_pre_hooks(
                hooks=self.pre_hooks,  # type: ignore
                run_response=run_response,
                run_input=run_input,
                session_state=session_state,
                dependencies=dependencies,
                metadata=metadata,
                session=session,
                user_id=user_id,
                debug_mode=debug_mode,
                **kwargs,
            )
            for event in pre_hook_iterator:
                yield event

        # 2. Determine tools for model
        processed_tools = self.get_tools(
            run_response=run_response,
            session=session,
            user_id=user_id,
            knowledge_filters=knowledge_filters,
        )
        _tools = self._determine_tools_for_model(
            model=self.model,
            processed_tools=processed_tools,
            run_response=run_response,
            session=session,
            session_state=session_state,
            dependencies=dependencies,
        )

        # 3. Prepare run messages
        run_messages: RunMessages = self._get_run_messages(
            run_response=run_response,
            input=run_input.input_content,
            session=session,
            session_state=session_state,
            user_id=user_id,
            audio=run_input.audios,
            images=run_input.images,
            videos=run_input.videos,
            files=run_input.files,
            knowledge_filters=knowledge_filters,
            add_history_to_context=add_history_to_context,
            dependencies=dependencies,
            add_dependencies_to_context=add_dependencies_to_context,
            add_session_state_to_context=add_session_state_to_context,
            metadata=metadata,
            tools=_tools,
            **kwargs,
        )
        if len(run_messages.messages) == 0:
            log_error("No messages to be sent to the model.")

        log_debug(f"Agent Run Start: {run_response.run_id}", center=True)

        # Start memory creation on a separate thread (runs concurrently with the main execution loop)
        memory_future = None
        # 4. Start memory creation in background thread if memory manager is enabled and agentic memory is disabled
        if run_messages.user_message is not None and self.memory_manager is not None and not self.enable_agentic_memory:
            log_debug("Starting memory creation in background thread.")
            memory_future = self.background_executor.submit(
                self._make_memories, run_messages=run_messages, user_id=user_id
            )

        # Start cultural knowledge creation on a separate thread (runs concurrently with the main execution loop)
        cultural_knowledge_future = None
        if (
            run_messages.user_message is not None
            and self.culture_manager is not None
            and self.update_cultural_knowledge
        ):
            log_debug("Starting cultural knowledge creation in background thread.")
            cultural_knowledge_future = self.background_executor.submit(
                self._make_cultural_knowledge, run_messages=run_messages
            )

        try:
            # Start the Run by yielding a RunStarted event
            if stream_events:
                yield handle_event(  # type: ignore
                    create_run_started_event(run_response),
                    run_response,
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

            # 5. Reason about the task if reasoning is enabled
            yield from self._handle_reasoning_stream(
                run_response=run_response,
                run_messages=run_messages,
                stream_events=stream_events,
            )

            # Check for cancellation before model processing
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 6. Process model response
            if self.output_model is None:
                for event in self._handle_model_response_stream(
                    session=session,
                    run_response=run_response,
                    run_messages=run_messages,
                    tools=_tools,
                    response_format=response_format,
                    stream_events=stream_events,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    yield event
            else:
                from agno.run.agent import (
                    IntermediateRunContentEvent,
                    RunContentEvent,
                )  # type: ignore

                for event in self._handle_model_response_stream(
                    session=session,
                    run_response=run_response,
                    run_messages=run_messages,
                    tools=_tools,
                    response_format=response_format,
                    stream_events=stream_events,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    if isinstance(event, RunContentEvent):
                        if stream_events:
                            yield IntermediateRunContentEvent(
                                content=event.content,
                                content_type=event.content_type,
                            )
                    else:
                        yield event

                # If an output model is provided, generate output using the output model
                for event in self._generate_response_with_output_model_stream(
                    session=session,
                    run_response=run_response,
                    run_messages=run_messages,
                    stream_events=stream_events,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    yield event

            # Check for cancellation after model processing
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 7. Parse response with parser model if provided
            yield from self._parse_response_with_parser_model_stream(
                session=session, run_response=run_response, stream_events=stream_events
            )

            # We should break out of the run function
            if any(tool_call.is_paused for tool_call in run_response.tools or []):
                yield from wait_for_background_tasks_stream(
                    memory_future=memory_future,
                    cultural_knowledge_future=cultural_knowledge_future,
                    stream_events=stream_events,
                    run_response=run_response,
                    events_to_skip=self.events_to_skip,
                    store_events=self.store_events,
                )

                # Handle the paused run
                yield from self._handle_agent_run_paused_stream(
                    run_response=run_response, session=session, user_id=user_id
                )
                return

            # Yield RunContentCompletedEvent
            if stream_events:
                yield handle_event(  # type: ignore
                    create_run_content_completed_event(from_run_response=run_response),
                    run_response,
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

            # Execute post-hooks after output is generated but before response is returned
            if self.post_hooks is not None:
                yield from self._execute_post_hooks(
                    hooks=self.post_hooks,  # type: ignore
                    run_output=run_response,
                    session_state=session_state,
                    dependencies=dependencies,
                    metadata=metadata,
                    session=session,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    **kwargs,
                )

            # 8. Wait for background memory creation and cultural knowledge creation
            yield from wait_for_background_tasks_stream(
                memory_future=memory_future,
                cultural_knowledge_future=cultural_knowledge_future,
                stream_events=stream_events,
                run_response=run_response,
            )

            # 9. Create session summary
            if self.session_summary_manager is not None:
                # Upsert the RunOutput to Agent Session before creating the session summary
                session.upsert_run(run=run_response)

                if stream_events:
                    yield handle_event(  # type: ignore
                        create_session_summary_started_event(from_run_response=run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )
                try:
                    self.session_summary_manager.create_session_summary(session=session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")
                if stream_events:
                    yield handle_event(  # type: ignore
                        create_session_summary_completed_event(
                            from_run_response=run_response, session_summary=session.summary
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )

            # Create the run completed event
            completed_event = handle_event(  # type: ignore
                create_run_completed_event(from_run_response=run_response),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

            # Set the run status to completed
            run_response.status = RunStatus.completed

            # 10. Cleanup and store the run response and session
            self._cleanup_and_store(run_response=run_response, session=session, user_id=user_id)

            if stream_events:
                yield completed_event  # type: ignore

            if yield_run_response:
                yield run_response

            # Log Agent Telemetry
            self._log_agent_telemetry(session_id=session.session_id, run_id=run_response.run_id)

            log_debug(f"Agent Run End: {run_response.run_id}", center=True, symbol="*")

        except RunCancelledException as e:
            # Handle run cancellation during streaming
            log_info(f"Run {run_response.run_id} was cancelled during streaming")
            run_response.status = RunStatus.cancelled
            # Don't overwrite content - preserve any partial content that was streamed
            # Only set content if it's empty
            if not run_response.content:
                run_response.content = str(e)

            # Yield the cancellation event
            yield handle_event(  # type: ignore
                create_run_cancelled_event(from_run_response=run_response, reason=str(e)),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

            # Cleanup and store the run response and session
            self._cleanup_and_store(run_response=run_response, session=session, user_id=user_id)
        finally:
            # Always clean up the run tracking
            cleanup_run(run_response.run_id)  # type: ignore

    @overload
    def run(
        self,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        *,
        stream: Literal[False] = False,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> RunOutput: ...

    @overload
    def run(
        self,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        *,
        stream: Literal[True] = True,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        yield_run_response: bool = False,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[Union[RunOutputEvent, RunOutput]]: ...

    def run(
        self,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        *,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        yield_run_response: bool = False,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[RunOutput, Iterator[Union[RunOutputEvent, RunOutput]]]:
        """Run the Agent and return the response."""
        if self._has_async_db():
            raise RuntimeError(
                "`run` method is not supported with an async database. Please use `arun` method instead."
            )

        if (add_history_to_context or self.add_history_to_context) and not self.db and not self.team_id:
            log_warning(
                "add_history_to_context is True, but no database has been assigned to the agent. History will not be added to the context."
            )

        # Create a run_id for this specific run
        run_id = str(uuid4())

        # Validate input against input_schema if provided
        validated_input = self._validate_input(input)

        # Normalise hook & guardails
        if not self._hooks_normalised:
            if self.pre_hooks:
                self.pre_hooks = normalize_hooks(self.pre_hooks)
            if self.post_hooks:
                self.post_hooks = normalize_hooks(self.post_hooks)
            self._hooks_normalised = True

        session_id, user_id = self._initialize_session(session_id=session_id, user_id=user_id)

        # Initialize the Agent
        self.initialize_agent(debug_mode=debug_mode)

        image_artifacts, video_artifacts, audio_artifacts, file_artifacts = self._validate_media_object_id(
            images=images, videos=videos, audios=audio, files=files
        )

        # Create RunInput to capture the original user input
        run_input = RunInput(
            input_content=validated_input,
            images=image_artifacts,
            videos=video_artifacts,
            audios=audio_artifacts,
            files=file_artifacts,
        )

        # Read existing session from database
        agent_session = self._read_or_create_session(session_id=session_id, user_id=user_id)
        self._update_metadata(session=agent_session)

        # Initialize session state
        session_state = self._initialize_session_state(
            session_state=session_state or {}, user_id=user_id, session_id=session_id, run_id=run_id
        )
        # Update session state from DB
        session_state = self._load_session_state(session=agent_session, session_state=session_state)
        # Determine runtime dependencies
        run_dependencies = dependencies if dependencies is not None else self.dependencies

        # Resolve dependencies
        if run_dependencies is not None:
            self._resolve_run_dependencies(dependencies=run_dependencies)

        add_dependencies = (
            add_dependencies_to_context if add_dependencies_to_context is not None else self.add_dependencies_to_context
        )
        add_session_state = (
            add_session_state_to_context
            if add_session_state_to_context is not None
            else self.add_session_state_to_context
        )
        add_history = add_history_to_context if add_history_to_context is not None else self.add_history_to_context

        # Initialize Knowledge Filters
        effective_filters = knowledge_filters

        # When filters are passed manually
        if self.knowledge_filters or knowledge_filters:
            effective_filters = self._get_effective_filters(knowledge_filters)

        # Use stream override value when necessary
        if stream is None:
            stream = False if self.stream is None else self.stream

        # Considering both stream_events and stream_intermediate_steps (deprecated)
        stream_events = stream_events or stream_intermediate_steps

        # Can't stream events if streaming is disabled
        if stream is False:
            stream_events = False

        if stream_events is None:
            stream_events = False if self.stream_events is None else self.stream_events

        self.stream = self.stream or stream
        self.stream_events = self.stream_events or stream_events

        # Prepare arguments for the model
        response_format = self._get_response_format() if self.parser_model is None else None
        self.model = cast(Model, self.model)

        # Merge agent metadata with run metadata
        if self.metadata is not None:
            if metadata is None:
                metadata = self.metadata
            else:
                merge_dictionaries(metadata, self.metadata)

        # Create a new run_response for this attempt
        run_response = RunOutput(
            run_id=run_id,
            session_id=session_id,
            agent_id=self.id,
            user_id=user_id,
            agent_name=self.name,
            metadata=metadata,
            input=run_input,
        )

        run_response.model = self.model.id if self.model is not None else None
        run_response.model_provider = self.model.provider if self.model is not None else None

        # Start the run metrics timer, to calculate the run duration
        run_response.metrics = Metrics()
        run_response.metrics.start_timer()

        # If no retries are set, use the agent's default retries
        retries = retries if retries is not None else self.retries

        last_exception = None
        num_attempts = retries + 1

        for attempt in range(num_attempts):
            try:
                if stream:
                    response_iterator = self._run_stream(
                        run_response=run_response,
                        session=agent_session,
                        session_state=session_state,
                        user_id=user_id,
                        knowledge_filters=effective_filters,
                        add_history_to_context=add_history,
                        add_dependencies_to_context=add_dependencies,
                        add_session_state_to_context=add_session_state,
                        metadata=metadata,
                        dependencies=run_dependencies,
                        response_format=response_format,
                        stream_events=stream_events,
                        yield_run_response=yield_run_response,
                        debug_mode=debug_mode,
                        **kwargs,
                    )
                    return response_iterator
                else:
                    response = self._run(
                        run_response=run_response,
                        session=agent_session,
                        session_state=session_state,
                        user_id=user_id,
                        knowledge_filters=effective_filters,
                        add_history_to_context=add_history,
                        add_dependencies_to_context=add_dependencies,
                        add_session_state_to_context=add_session_state,
                        metadata=metadata,
                        dependencies=run_dependencies,
                        response_format=response_format,
                        debug_mode=debug_mode,
                        **kwargs,
                    )
                    return response
            except (InputCheckError, OutputCheckError) as e:
                log_error(f"Validation failed: {str(e)} | Check: {e.check_trigger}")
                raise e
            except ModelProviderError as e:
                log_warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")
                if isinstance(e, StopAgentRun):
                    raise e
                last_exception = e
                if attempt < num_attempts - 1:  # Don't sleep on the last attempt
                    if self.exponential_backoff:
                        delay = 2**attempt * self.delay_between_retries
                    else:
                        delay = self.delay_between_retries
                    import time

                    time.sleep(delay)
            except KeyboardInterrupt:
                run_response.content = "Operation cancelled by user"
                run_response.status = RunStatus.cancelled

                if stream:
                    return generator_wrapper(  # type: ignore
                        create_run_cancelled_event(
                            from_run_response=run_response,
                            reason="Operation cancelled by user",
                        )
                    )
                else:
                    return run_response

        # If we get here, all retries failed
        if last_exception is not None:
            log_error(
                f"Failed after {num_attempts} attempts. Last error using {last_exception.model_name}({last_exception.model_id})"
            )
            if stream:
                return generator_wrapper(create_run_error_event(run_response, error=str(last_exception)))  # type: ignore

            raise last_exception
        else:
            if stream:
                return generator_wrapper(create_run_error_event(run_response, error=str(last_exception)))  # type: ignore
            raise Exception(f"Failed after {num_attempts} attempts.")

    async def _arun(
        self,
        run_response: RunOutput,
        session_id: str,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> RunOutput:
        """Run the Agent and return the RunOutput.

        Steps:
        1. Read or create session
        2. Update metadata and session state
        3. Resolve dependencies
        4. Execute pre-hooks
        5. Determine tools for model
        6. Prepare run messages
        7. Start memory creation in background task
        8. Reason about the task if reasoning is enabled
        9. Generate a response from the Model (includes running function calls)
        10. Update the RunOutput with the model response
        11. Convert response to structured format
        12. Store media if enabled
        13. Execute post-hooks
        14. Wait for background memory creation
        15. Create session summary
        16. Cleanup and store (scrub, stop timer, save to file, add to session, calculate metrics, save session)
        """
        log_debug(f"Agent Run Start: {run_response.run_id}", center=True)

        # Register run for cancellation tracking
        register_run(run_response.run_id)  # type: ignore

        # 1. Read or create session. Reads from the database if provided.
        agent_session = await self._aread_or_create_session(session_id=session_id, user_id=user_id)

        # 2. Update metadata and session state
        self._update_metadata(session=agent_session)
        # Initialize session state
        session_state = self._initialize_session_state(
            session_state=session_state or {}, user_id=user_id, session_id=session_id, run_id=run_response.run_id
        )
        # Update session state from DB
        if session_state is not None:
            session_state = self._load_session_state(session=agent_session, session_state=session_state)

        # 3. Resolve dependencies
        if dependencies is not None:
            await self._aresolve_run_dependencies(dependencies=dependencies)

        # 4. Execute pre-hooks
        run_input = cast(RunInput, run_response.input)
        self.model = cast(Model, self.model)
        if self.pre_hooks is not None:
            # Can modify the run input
            pre_hook_iterator = self._aexecute_pre_hooks(
                hooks=self.pre_hooks,  # type: ignore
                run_response=run_response,
                run_input=run_input,
                session_state=session_state,
                dependencies=dependencies,
                metadata=metadata,
                session=agent_session,
                user_id=user_id,
                debug_mode=debug_mode,
                **kwargs,
            )
            # Consume the async iterator without yielding
            async for _ in pre_hook_iterator:
                pass

        # 5. Determine tools for model
        self.model = cast(Model, self.model)
        processed_tools = await self.aget_tools(
            run_response=run_response,
            session=agent_session,
            user_id=user_id,
            knowledge_filters=knowledge_filters,
        )
        _tools = self._determine_tools_for_model(
            model=self.model,
            processed_tools=processed_tools,
            run_response=run_response,
            session=agent_session,
            session_state=session_state,
            dependencies=dependencies,
        )

        # 6. Prepare run messages
        run_messages: RunMessages = await self._aget_run_messages(
            run_response=run_response,
            input=run_input.input_content,
            session=agent_session,
            session_state=session_state,
            user_id=user_id,
            audio=run_input.audios,
            images=run_input.images,
            videos=run_input.videos,
            files=run_input.files,
            knowledge_filters=knowledge_filters,
            add_history_to_context=add_history_to_context,
            dependencies=dependencies,
            add_dependencies_to_context=add_dependencies_to_context,
            add_session_state_to_context=add_session_state_to_context,
            metadata=metadata,
            tools=_tools,
            **kwargs,
        )
        if len(run_messages.messages) == 0:
            log_error("No messages to be sent to the model.")

        # 7. Start memory creation as a background task (runs concurrently with the main execution)
        memory_task = None
        if run_messages.user_message is not None and self.memory_manager is not None and not self.enable_agentic_memory:
            log_debug("Starting memory creation in background task.")
            memory_task = create_task(self._amake_memories(run_messages=run_messages, user_id=user_id))

        # Start cultural knowledge creation on a separate thread (runs concurrently with the main execution loop)
        cultural_knowledge_task = None
        if (
            run_messages.user_message is not None
            and self.culture_manager is not None
            and self.update_cultural_knowledge
        ):
            log_debug("Starting cultural knowledge creation in background thread.")
            cultural_knowledge_task = create_task(self._acreate_cultural_knowledge(run_messages=run_messages))

        try:
            # Check for cancellation before model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 8. Reason about the task if reasoning is enabled
            await self._ahandle_reasoning(run_response=run_response, run_messages=run_messages)

            # Check for cancellation before model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 9. Generate a response from the Model (includes running function calls)
            model_response: ModelResponse = await self.model.aresponse(
                messages=run_messages.messages,
                tools=_tools,
                tool_choice=self.tool_choice,
                tool_call_limit=self.tool_call_limit,
                per_tool_call_limits=self._build_per_tool_call_limits(_tools) if _tools else None,
                response_format=response_format,
                send_media_to_model=self.send_media_to_model,
            )

            # Check for cancellation after model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # If an output model is provided, generate output using the output model
            await self._agenerate_response_with_output_model(model_response=model_response, run_messages=run_messages)

            # If a parser model is provided, structure the response separately
            await self._aparse_response_with_parser_model(model_response=model_response, run_messages=run_messages)

            # 10. Update the RunOutput with the model response
            self._update_run_response(
                model_response=model_response,
                run_response=run_response,
                run_messages=run_messages,
            )

            # We should break out of the run function
            if any(tool_call.is_paused for tool_call in run_response.tools or []):
                await await_for_background_tasks(
                    memory_task=memory_task, cultural_knowledge_task=cultural_knowledge_task
                )
                return await self._ahandle_agent_run_paused(
                    run_response=run_response, session=agent_session, user_id=user_id
                )

            # 11. Convert the response to the structured format if needed
            self._convert_response_to_structured_format(run_response)

            # 12. Store media if enabled
            if self.store_media:
                self._store_media(run_response, model_response)

            # 13. Execute post-hooks (after output is generated but before response is returned)
            if self.post_hooks is not None:
                async for _ in self._aexecute_post_hooks(
                    hooks=self.post_hooks,  # type: ignore
                    run_output=run_response,
                    session_state=session_state,
                    dependencies=dependencies,
                    metadata=metadata,
                    session=agent_session,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    **kwargs,
                ):
                    pass

            # Check for cancellation
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 14. Wait for background memory creation
            await await_for_background_tasks(memory_task=memory_task, cultural_knowledge_task=cultural_knowledge_task)

            # 15. Create session summary
            if self.session_summary_manager is not None:
                # Upsert the RunOutput to Agent Session before creating the session summary
                agent_session.upsert_run(run=run_response)
                try:
                    await self.session_summary_manager.acreate_session_summary(session=agent_session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")

            run_response.status = RunStatus.completed

            # 16. Cleanup and store the run response and session
            await self._acleanup_and_store(run_response=run_response, session=agent_session, user_id=user_id)

            # Log Agent Telemetry
            await self._alog_agent_telemetry(session_id=agent_session.session_id, run_id=run_response.run_id)

            log_debug(f"Agent Run End: {run_response.run_id}", center=True, symbol="*")

            return run_response

        except RunCancelledException as e:
            # Handle run cancellation
            log_info(f"Run {run_response.run_id} was cancelled")
            run_response.content = str(e)
            run_response.status = RunStatus.cancelled

            # Cleanup and store the run response and session
            await self._acleanup_and_store(run_response=run_response, session=agent_session, user_id=user_id)

            return run_response

        finally:
            # Always disconnect MCP tools
            await self._disconnect_mcp_tools()

            # Cancel the memory task if it's still running
            if memory_task is not None and not memory_task.done():
                memory_task.cancel()
                try:
                    await memory_task
                except CancelledError:
                    pass
            # Cancel the cultural knowledge task if it's still running
            if cultural_knowledge_task is not None and not cultural_knowledge_task.done():
                cultural_knowledge_task.cancel()
                try:
                    await cultural_knowledge_task
                except CancelledError:
                    pass
            # Always clean up the run tracking
            cleanup_run(run_response.run_id)  # type: ignore

    async def _arun_stream(
        self,
        run_response: RunOutput,
        session_id: str,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        stream_events: bool = False,
        yield_run_response: Optional[bool] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[RunOutputEvent, RunOutput]]:
        """Run the Agent and yield the RunOutput.

        Steps:
        1. Read or create session
        2. Update metadata and session state
        3. Resolve dependencies
        4. Execute pre-hooks
        5. Determine tools for model
        6. Prepare run messages
        7. Start memory creation in background task
        8. Reason about the task if reasoning is enabled
        9. Generate a response from the Model (includes running function calls)
        10. Parse response with parser model if provided
        11. Wait for background memory creation
        12. Create session summary
        13. Cleanup and store (scrub, stop timer, save to file, add to session, calculate metrics, save session)
        """
        log_debug(f"Agent Run Start: {run_response.run_id}", center=True)

        # Start the Run by yielding a RunStarted event
        if stream_events:
            yield handle_event(  # type: ignore
                create_run_started_event(run_response),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

        # 1. Read or create session. Reads from the database if provided.
        agent_session = await self._aread_or_create_session(session_id=session_id, user_id=user_id)

        # 2. Update metadata and session state
        self._update_metadata(session=agent_session)
        # Initialize session state
        session_state = self._initialize_session_state(
            session_state=session_state or {}, user_id=user_id, session_id=session_id, run_id=run_response.run_id
        )
        # Update session state from DB
        if session_state is not None:
            session_state = self._load_session_state(session=agent_session, session_state=session_state)

        # 3. Resolve dependencies
        if dependencies is not None:
            await self._aresolve_run_dependencies(dependencies=dependencies)

        # 4. Execute pre-hooks
        run_input = cast(RunInput, run_response.input)
        self.model = cast(Model, self.model)
        if self.pre_hooks is not None:
            # Can modify the run input
            pre_hook_iterator = self._aexecute_pre_hooks(
                hooks=self.pre_hooks,  # type: ignore
                run_response=run_response,
                run_input=run_input,
                session=agent_session,
                session_state=session_state,
                dependencies=dependencies,
                metadata=metadata,
                user_id=user_id,
                debug_mode=debug_mode,
                **kwargs,
            )
            async for event in pre_hook_iterator:
                yield event

        # 5. Determine tools for model
        self.model = cast(Model, self.model)
        processed_tools = await self.aget_tools(
            run_response=run_response,
            session=agent_session,
            user_id=user_id,
            knowledge_filters=knowledge_filters,
        )
        _tools = self._determine_tools_for_model(
            model=self.model,
            processed_tools=processed_tools,
            run_response=run_response,
            session=agent_session,
            session_state=session_state,
            dependencies=dependencies,
        )

        # 6. Prepare run messages
        run_messages: RunMessages = await self._aget_run_messages(
            run_response=run_response,
            input=run_input.input_content,
            session=agent_session,
            session_state=session_state,
            user_id=user_id,
            audio=run_input.audios,
            images=run_input.images,
            videos=run_input.videos,
            files=run_input.files,
            knowledge_filters=knowledge_filters,
            add_history_to_context=add_history_to_context,
            dependencies=dependencies,
            add_dependencies_to_context=add_dependencies_to_context,
            add_session_state_to_context=add_session_state_to_context,
            metadata=metadata,
            tools=_tools,
            **kwargs,
        )
        if len(run_messages.messages) == 0:
            log_error("No messages to be sent to the model.")

        # 7. Start memory creation as a background task (runs concurrently with the main execution)
        memory_task = None
        if run_messages.user_message is not None and self.memory_manager is not None and not self.enable_agentic_memory:
            log_debug("Starting memory creation in background task.")
            memory_task = create_task(self._amake_memories(run_messages=run_messages, user_id=user_id))

        # Start cultural knowledge creation on a separate thread (runs concurrently with the main execution loop)
        cultural_knowledge_task = None
        if (
            run_messages.user_message is not None
            and self.culture_manager is not None
            and self.update_cultural_knowledge
        ):
            log_debug("Starting cultural knowledge creation in background task.")
            cultural_knowledge_task = create_task(self._acreate_cultural_knowledge(run_messages=run_messages))

        # Register run for cancellation tracking
        register_run(run_response.run_id)  # type: ignore

        try:
            # 8. Reason about the task if reasoning is enabled
            async for item in self._ahandle_reasoning_stream(
                run_response=run_response,
                run_messages=run_messages,
                stream_events=stream_events,
            ):
                raise_if_cancelled(run_response.run_id)  # type: ignore
                yield item

            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 9. Generate a response from the Model
            if self.output_model is None:
                async for event in self._ahandle_model_response_stream(
                    session=agent_session,
                    run_response=run_response,
                    run_messages=run_messages,
                    tools=_tools,
                    response_format=response_format,
                    stream_events=stream_events,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    yield event
            else:
                from agno.run.agent import (
                    IntermediateRunContentEvent,
                    RunContentEvent,
                )  # type: ignore

                async for event in self._ahandle_model_response_stream(
                    session=agent_session,
                    run_response=run_response,
                    run_messages=run_messages,
                    tools=_tools,
                    response_format=response_format,
                    stream_events=stream_events,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    if isinstance(event, RunContentEvent):
                        if stream_events:
                            yield IntermediateRunContentEvent(
                                content=event.content,
                                content_type=event.content_type,
                            )
                    else:
                        yield event

                # If an output model is provided, generate output using the output model
                async for event in self._agenerate_response_with_output_model_stream(
                    session=agent_session,
                    run_response=run_response,
                    run_messages=run_messages,
                    stream_events=stream_events,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    yield event

            # Check for cancellation after model processing
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 10. Parse response with parser model if provided
            async for event in self._aparse_response_with_parser_model_stream(
                session=agent_session, run_response=run_response, stream_events=stream_events
            ):
                yield event

            if stream_events:
                yield handle_event(  # type: ignore
                    create_run_content_completed_event(from_run_response=run_response),
                    run_response,
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

            # Break out of the run function if a tool call is paused
            if any(tool_call.is_paused for tool_call in run_response.tools or []):
                async for item in await_for_background_tasks_stream(
                    memory_task=memory_task,
                    cultural_knowledge_task=cultural_knowledge_task,
                    stream_events=stream_events,
                    run_response=run_response,
                ):
                    yield item

                async for item in self._ahandle_agent_run_paused_stream(
                    run_response=run_response, session=agent_session, user_id=user_id
                ):
                    yield item
                return

            # Execute post-hooks (after output is generated but before response is returned)
            if self.post_hooks is not None:
                async for event in self._aexecute_post_hooks(
                    hooks=self.post_hooks,  # type: ignore
                    run_output=run_response,
                    session_state=session_state,
                    dependencies=dependencies,
                    metadata=metadata,
                    session=agent_session,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    **kwargs,
                ):
                    yield event

            # 11. Wait for background memory creation
            async for item in await_for_background_tasks_stream(
                memory_task=memory_task,
                cultural_knowledge_task=cultural_knowledge_task,
                stream_events=stream_events,
                run_response=run_response,
                events_to_skip=self.events_to_skip,
                store_events=self.store_events,
            ):
                yield item

            # 12. Create session summary
            if self.session_summary_manager is not None:
                # Upsert the RunOutput to Agent Session before creating the session summary
                agent_session.upsert_run(run=run_response)

                if stream_events:
                    yield handle_event(  # type: ignore
                        create_session_summary_started_event(from_run_response=run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )
                try:
                    await self.session_summary_manager.acreate_session_summary(session=agent_session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")
                if stream_events:
                    yield handle_event(  # type: ignore
                        create_session_summary_completed_event(
                            from_run_response=run_response, session_summary=agent_session.summary
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )

            # Create the run completed event
            completed_event = handle_event(
                create_run_completed_event(from_run_response=run_response),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

            # Set the run status to completed
            run_response.status = RunStatus.completed

            # 13. Cleanup and store the run response and session
            await self._acleanup_and_store(run_response=run_response, session=agent_session, user_id=user_id)

            if stream_events:
                yield completed_event  # type: ignore

            if yield_run_response:
                yield run_response

            # Log Agent Telemetry
            await self._alog_agent_telemetry(session_id=agent_session.session_id, run_id=run_response.run_id)

            log_debug(f"Agent Run End: {run_response.run_id}", center=True, symbol="*")

        except RunCancelledException as e:
            # Handle run cancellation during async streaming
            log_info(f"Run {run_response.run_id} was cancelled during async streaming")
            run_response.status = RunStatus.cancelled
            # Don't overwrite content - preserve any partial content that was streamed
            # Only set content if it's empty
            if not run_response.content:
                run_response.content = str(e)

            # Yield the cancellation event
            yield handle_event(  # type: ignore
                create_run_cancelled_event(from_run_response=run_response, reason=str(e)),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

            # Cleanup and store the run response and session
            await self._acleanup_and_store(run_response=run_response, session=agent_session, user_id=user_id)
        finally:
            # Always disconnect MCP tools
            await self._disconnect_mcp_tools()

            # Cancel the memory task if it's still running
            if memory_task is not None and not memory_task.done():
                memory_task.cancel()
                try:
                    await memory_task
                except CancelledError:
                    pass

            if cultural_knowledge_task is not None and not cultural_knowledge_task.done():
                cultural_knowledge_task.cancel()
                try:
                    await cultural_knowledge_task
                except CancelledError:
                    pass

            # Always clean up the run tracking
            cleanup_run(run_response.run_id)  # type: ignore

    @overload
    async def arun(
        self,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        *,
        stream: Literal[False] = False,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> RunOutput: ...

    @overload
    def arun(
        self,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        *,
        stream: Literal[True] = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        yield_run_response: Optional[bool] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[RunOutputEvent, RunOutput]]: ...

    def arun(  # type: ignore
        self,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        *,
        stream: Optional[bool] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        yield_run_response: Optional[bool] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[RunOutput, AsyncIterator[RunOutputEvent]]:
        """Async Run the Agent and return the response."""

        if (add_history_to_context or self.add_history_to_context) and not self.db and not self.team_id:
            log_warning(
                "add_history_to_context is True, but no database has been assigned to the agent. History will not be added to the context."
            )

        # Create a run_id for this specific run
        run_id = str(uuid4())

        # 2. Validate input against input_schema if provided
        validated_input = self._validate_input(input)

        # Normalise hook & guardails
        if not self._hooks_normalised:
            if self.pre_hooks:
                self.pre_hooks = normalize_hooks(self.pre_hooks, async_mode=True)
            if self.post_hooks:
                self.post_hooks = normalize_hooks(self.post_hooks, async_mode=True)
            self._hooks_normalised = True

        # Initialize session
        session_id, user_id = self._initialize_session(session_id=session_id, user_id=user_id)

        # Initialize the Agent
        self.initialize_agent(debug_mode=debug_mode)

        image_artifacts, video_artifacts, audio_artifacts, file_artifacts = self._validate_media_object_id(
            images=images, videos=videos, audios=audio, files=files
        )

        # Resolve variables
        run_dependencies = dependencies if dependencies is not None else self.dependencies
        add_dependencies = (
            add_dependencies_to_context if add_dependencies_to_context is not None else self.add_dependencies_to_context
        )
        add_session_state = (
            add_session_state_to_context
            if add_session_state_to_context is not None
            else self.add_session_state_to_context
        )
        add_history = add_history_to_context if add_history_to_context is not None else self.add_history_to_context

        # Create RunInput to capture the original user input
        run_input = RunInput(
            input_content=validated_input,
            images=image_artifacts,
            videos=video_artifacts,
            audios=audio_artifacts,
            files=files,
        )

        # Use stream override value when necessary
        if stream is None:
            stream = False if self.stream is None else self.stream

        # Considering both stream_events and stream_intermediate_steps (deprecated)
        stream_events = stream_events or stream_intermediate_steps

        # Can't stream events if streaming is disabled
        if stream is False:
            stream_events = False

        if stream_events is None:
            stream_events = False if self.stream_events is None else self.stream_events

        self.stream = self.stream or stream
        self.stream_events = self.stream_events or stream_events

        # Prepare arguments for the model
        response_format = self._get_response_format() if self.parser_model is None else None
        self.model = cast(Model, self.model)

        # Get knowledge filters
        effective_filters = knowledge_filters
        if self.knowledge_filters or knowledge_filters:
            effective_filters = self._get_effective_filters(knowledge_filters)

        # Merge agent metadata with run metadata
        if self.metadata is not None:
            if metadata is None:
                metadata = self.metadata
            else:
                merge_dictionaries(metadata, self.metadata)

        # If no retries are set, use the agent's default retries
        retries = retries if retries is not None else self.retries

        # Create a new run_response for this attempt
        run_response = RunOutput(
            run_id=run_id,
            session_id=session_id,
            agent_id=self.id,
            user_id=user_id,
            agent_name=self.name,
            metadata=metadata,
            input=run_input,
        )

        run_response.model = self.model.id if self.model is not None else None
        run_response.model_provider = self.model.provider if self.model is not None else None

        # Start the run metrics timer, to calculate the run duration
        run_response.metrics = Metrics()
        run_response.metrics.start_timer()

        last_exception = None
        num_attempts = retries + 1

        for attempt in range(num_attempts):
            try:
                # Pass the new run_response to _arun
                if stream:
                    return self._arun_stream(  # type: ignore
                        run_response=run_response,
                        user_id=user_id,
                        response_format=response_format,
                        stream_events=stream_events,
                        yield_run_response=yield_run_response,
                        dependencies=run_dependencies,
                        session_id=session_id,
                        session_state=session_state,
                        knowledge_filters=effective_filters,
                        add_history_to_context=add_history,
                        add_dependencies_to_context=add_dependencies,
                        add_session_state_to_context=add_session_state,
                        metadata=metadata,
                        debug_mode=debug_mode,
                        **kwargs,
                    )  # type: ignore[assignment]
                else:
                    return self._arun(  # type: ignore
                        run_response=run_response,
                        user_id=user_id,
                        response_format=response_format,
                        dependencies=run_dependencies,
                        session_id=session_id,
                        session_state=session_state,
                        knowledge_filters=effective_filters,
                        add_history_to_context=add_history,
                        add_dependencies_to_context=add_dependencies,
                        add_session_state_to_context=add_session_state,
                        metadata=metadata,
                        debug_mode=debug_mode,
                        **kwargs,
                    )

            except (InputCheckError, OutputCheckError) as e:
                log_error(f"Validation failed: {str(e)} | Check trigger: {e.check_trigger}")
                raise e
            except ModelProviderError as e:
                log_warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")
                if isinstance(e, StopAgentRun):
                    raise e
                last_exception = e
                if attempt < num_attempts - 1:  # Don't sleep on the last attempt
                    if self.exponential_backoff:
                        delay = 2**attempt * self.delay_between_retries
                    else:
                        delay = self.delay_between_retries
                    import time

                    time.sleep(delay)
            except KeyboardInterrupt:
                run_response.content = "Operation cancelled by user"
                run_response.status = RunStatus.cancelled

                if stream:
                    return async_generator_wrapper(  # type: ignore
                        create_run_cancelled_event(
                            from_run_response=run_response,
                            reason="Operation cancelled by user",
                        )
                    )
                else:
                    return run_response

        # If we get here, all retries failed
        if last_exception is not None:
            log_error(
                f"Failed after {num_attempts} attempts. Last error using {last_exception.model_name}({last_exception.model_id})"
            )

            if stream:
                return async_generator_wrapper(create_run_error_event(run_response, error=str(last_exception)))  # type: ignore
            raise last_exception
        else:
            if stream:
                return async_generator_wrapper(create_run_error_event(run_response, error=str(last_exception)))  # type: ignore
            raise Exception(f"Failed after {num_attempts} attempts.")

    @overload
    def continue_run(
        self,
        run_response: Optional[RunOutput] = None,
        *,
        run_id: Optional[str] = None,
        updated_tools: Optional[List[ToolExecution]] = None,
        stream: Literal[False] = False,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
    ) -> RunOutput: ...

    @overload
    def continue_run(
        self,
        run_response: Optional[RunOutput] = None,
        *,
        run_id: Optional[str] = None,
        updated_tools: Optional[List[ToolExecution]] = None,
        stream: Literal[True] = True,
        stream_events: Optional[bool] = False,
        stream_intermediate_steps: Optional[bool] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
    ) -> Iterator[RunOutputEvent]: ...

    def continue_run(
        self,
        run_response: Optional[RunOutput] = None,
        *,
        run_id: Optional[str] = None,
        updated_tools: Optional[List[ToolExecution]] = None,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = False,
        stream_intermediate_steps: Optional[bool] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs,
    ) -> Union[RunOutput, Iterator[RunOutputEvent]]:
        """Continue a previous run.

        Args:
            run_response: The run response to continue.
            run_id: The run id to continue. Alternative to passing run_response.
            updated_tools: The updated tools to use for the run. Required to be used with `run_id`.
            stream: Whether to stream the response.
            stream_events: Whether to stream all events.
            user_id: The user id to continue the run for.
            session_id: The session id to continue the run for.
            retries: The number of retries to continue the run for.
            knowledge_filters: The knowledge filters to use for the run.
            dependencies: The dependencies to use for the run.
            metadata: The metadata to use for the run.
            debug_mode: Whether to enable debug mode.
            (deprecated) stream_intermediate_steps: Whether to stream all steps.
        """
        if run_response is None and run_id is None:
            raise ValueError("Either run_response or run_id must be provided.")

        if run_response is None and (run_id is not None and (session_id is None and self.session_id is None)):
            raise ValueError("Session ID is required to continue a run from a run_id.")

        if self._has_async_db():
            raise Exception("continue_run() is not supported with an async DB. Please use acontinue_arun() instead.")

        session_id = run_response.session_id if run_response else session_id

        session_id, user_id = self._initialize_session(
            session_id=session_id,
            user_id=user_id,
        )
        # Initialize the Agent
        self.initialize_agent(debug_mode=debug_mode)

        # Read existing session from storage
        agent_session = self._read_or_create_session(session_id=session_id, user_id=user_id)
        self._update_metadata(session=agent_session)

        # Initialize session state
        session_state = self._initialize_session_state(
            session_state={}, user_id=user_id, session_id=session_id, run_id=run_id
        )
        # Update session state from DB
        session_state = self._load_session_state(session=agent_session, session_state=session_state)

        run_dependencies = dependencies if dependencies is not None else self.dependencies

        # Resolve dependencies
        if run_dependencies is not None:
            self._resolve_run_dependencies(dependencies=run_dependencies)

        effective_filters = knowledge_filters

        # When filters are passed manually
        if self.knowledge_filters or knowledge_filters:
            effective_filters = self._get_effective_filters(knowledge_filters)

        # Merge agent metadata with run metadata
        if self.metadata is not None:
            if metadata is None:
                metadata = self.metadata
            else:
                merge_dictionaries(metadata, self.metadata)

        # If no retries are set, use the agent's default retries
        retries = retries if retries is not None else self.retries

        # Use stream override value when necessary
        if stream is None:
            stream = False if self.stream is None else self.stream

        # Considering both stream_events and stream_intermediate_steps (deprecated)
        stream_events = stream_events or stream_intermediate_steps

        # Can't stream events if streaming is disabled
        if stream is False:
            stream_events = False

        if stream_events is None:
            stream_events = False if self.stream_events is None else self.stream_events

        # Can't stream events if streaming is disabled
        if stream is False:
            stream_events = False

        self.stream = self.stream or stream
        self.stream_events = self.stream_events or stream_events

        # Run can be continued from previous run response or from passed run_response context
        if run_response is not None:
            # The run is continued from a provided run_response. This contains the updated tools.
            input = run_response.messages or []
        elif run_id is not None:
            # The run is continued from a run_id. This requires the updated tools to be passed.
            if updated_tools is None:
                raise ValueError("Updated tools are required to continue a run from a run_id.")

            runs = agent_session.runs
            run_response = next((r for r in runs if r.run_id == run_id), None)  # type: ignore
            if run_response is None:
                raise RuntimeError(f"No runs found for run ID {run_id}")
            run_response.tools = updated_tools
            input = run_response.messages or []
        else:
            raise ValueError("Either run_response or run_id must be provided.")

        # Prepare arguments for the model
        self._set_default_model()
        response_format = self._get_response_format()
        self.model = cast(Model, self.model)

        processed_tools = self.get_tools(
            run_response=run_response,
            session=agent_session,
            user_id=user_id,
            knowledge_filters=effective_filters,
        )
        _tools = self._determine_tools_for_model(
            model=self.model,
            processed_tools=processed_tools,
            run_response=run_response,
            session=agent_session,
            session_state=session_state,
            dependencies=run_dependencies,
        )

        last_exception = None
        num_attempts = retries + 1
        for attempt in range(num_attempts):
            run_response = cast(RunOutput, run_response)

            log_debug(f"Agent Run Start: {run_response.run_id}", center=True)

            # Prepare run messages
            run_messages = self._get_continue_run_messages(
                input=input,
            )

            # Reset the run state
            run_response.status = RunStatus.running

            try:
                if stream:
                    response_iterator = self._continue_run_stream(
                        run_response=run_response,
                        run_messages=run_messages,
                        tools=_tools,
                        user_id=user_id,
                        session=agent_session,
                        session_state=session_state,
                        dependencies=run_dependencies,
                        metadata=metadata,
                        response_format=response_format,
                        stream_events=stream_events,
                        debug_mode=debug_mode,
                        **kwargs,
                    )
                    return response_iterator
                else:
                    response = self._continue_run(
                        run_response=run_response,
                        run_messages=run_messages,
                        tools=_tools,
                        user_id=user_id,
                        session=agent_session,
                        session_state=session_state,
                        dependencies=run_dependencies,
                        metadata=metadata,
                        response_format=response_format,
                        debug_mode=debug_mode,
                        **kwargs,
                    )
                    return response
            except ModelProviderError as e:
                log_warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")
                if isinstance(e, StopAgentRun):
                    raise e
                last_exception = e
                if attempt < num_attempts - 1:  # Don't sleep on the last attempt
                    if self.exponential_backoff:
                        delay = 2**attempt * self.delay_between_retries
                    else:
                        delay = self.delay_between_retries
                    import time

                    time.sleep(delay)
            except KeyboardInterrupt:
                if stream:
                    return generator_wrapper(  # type: ignore
                        create_run_cancelled_event(run_response, "Operation cancelled by user")
                    )
                else:
                    run_response.content = "Operation cancelled by user"
                    run_response.status = RunStatus.cancelled
                    return run_response

        # If we get here, all retries failed
        if last_exception is not None:
            log_error(
                f"Failed after {num_attempts} attempts. Last error using {last_exception.model_name}({last_exception.model_id})"
            )

            if stream:
                return generator_wrapper(create_run_error_event(run_response, error=str(last_exception)))  # type: ignore
            raise last_exception
        else:
            if stream:
                return generator_wrapper(create_run_error_event(run_response, error=str(last_exception)))  # type: ignore
            raise Exception(f"Failed after {num_attempts} attempts.")

    def _continue_run(
        self,
        run_response: RunOutput,
        run_messages: RunMessages,
        session: AgentSession,
        tools: List[Union[Function, dict]],
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs,
    ) -> RunOutput:
        """Continue a previous run.

        Steps:
        1. Handle any updated tools
        2. Generate a response from the Model
        3. Update the RunOutput with the model response
        4. Convert response to structured format
        5. Store media if enabled
        6. Execute post-hooks
        7. Create session summary
        8. Cleanup and store (scrub, stop timer, save to file, add to session, calculate metrics, save session)
        """
        # Register run for cancellation tracking
        register_run(run_response.run_id)  # type: ignore

        self.model = cast(Model, self.model)

        # 1. Handle the updated tools
        self._handle_tool_call_updates(run_response=run_response, run_messages=run_messages, tools=tools)

        try:
            # Check for cancellation before model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 2. Generate a response from the Model (includes running function calls)
            self.model = cast(Model, self.model)
            model_response: ModelResponse = self.model.response(
                messages=run_messages.messages,
                response_format=response_format,
                tools=tools,
                tool_choice=self.tool_choice,
                tool_call_limit=self.tool_call_limit,
                per_tool_call_limits=self._build_per_tool_call_limits(tools) if tools else None,
            )

            # Check for cancellation after model processing
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 3. Update the RunOutput with the model response
            self._update_run_response(
                model_response=model_response, run_response=run_response, run_messages=run_messages
            )

            # We should break out of the run function
            if any(tool_call.is_paused for tool_call in run_response.tools or []):
                return self._handle_agent_run_paused(run_response=run_response, session=session, user_id=user_id)

            # 4. Convert the response to the structured format if needed
            self._convert_response_to_structured_format(run_response)

            # 5. Store media if enabled
            if self.store_media:
                self._store_media(run_response, model_response)

            # 6. Execute post-hooks
            if self.post_hooks is not None:
                post_hook_iterator = self._execute_post_hooks(
                    hooks=self.post_hooks,  # type: ignore
                    run_output=run_response,
                    session=session,
                    user_id=user_id,
                    session_state=session_state,
                    dependencies=dependencies,
                    metadata=metadata,
                    debug_mode=debug_mode,
                    **kwargs,
                )
                deque(post_hook_iterator, maxlen=0)
            # Check for cancellation
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 7. Create session summary
            if self.session_summary_manager is not None:
                # Upsert the RunOutput to Agent Session before creating the session summary
                session.upsert_run(run=run_response)

                try:
                    self.session_summary_manager.create_session_summary(session=session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")

            # Set the run status to completed
            run_response.status = RunStatus.completed

            # 8. Cleanup and store the run response and session
            self._cleanup_and_store(run_response=run_response, session=session, user_id=user_id)

            # Log Agent Telemetry
            self._log_agent_telemetry(session_id=session.session_id, run_id=run_response.run_id)

            return run_response
        except RunCancelledException as e:
            # Handle run cancellation during async streaming
            log_info(f"Run {run_response.run_id} was cancelled")
            run_response.status = RunStatus.cancelled
            run_response.content = str(e)

            # Cleanup and store the run response and session
            self._cleanup_and_store(run_response=run_response, session=session, user_id=user_id)

            return run_response
        finally:
            # Always clean up the run tracking
            cleanup_run(run_response.run_id)  # type: ignore

    def _continue_run_stream(
        self,
        run_response: RunOutput,
        run_messages: RunMessages,
        session: AgentSession,
        tools: List[Union[Function, dict]],
        session_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        stream_events: bool = False,
        dependencies: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs,
    ) -> Iterator[RunOutputEvent]:
        """Continue a previous run.

        Steps:
        1. Resolve dependencies
        2. Handle any updated tools
        3. Process model response
        4. Execute post-hooks
        5. Create session summary
        6. Cleanup and store the run response and session
        """

        # 1. Resolve dependencies
        if dependencies is not None:
            self._resolve_run_dependencies(dependencies=dependencies)

        # Start the Run by yielding a RunContinued event
        if stream_events:
            yield handle_event(  # type: ignore
                create_run_continued_event(run_response),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

        # 2. Handle the updated tools
        yield from self._handle_tool_call_updates_stream(
            run_response=run_response, run_messages=run_messages, tools=tools, stream_events=stream_events
        )

        try:
            # 3. Process model response
            for event in self._handle_model_response_stream(
                session=session,
                run_response=run_response,
                run_messages=run_messages,
                tools=tools,
                response_format=response_format,
                stream_events=stream_events,
            ):
                yield event

            # Parse response with parser model if provided
            yield from self._parse_response_with_parser_model_stream(
                session=session, run_response=run_response, stream_events=stream_events
            )

            # Yield RunContentCompletedEvent
            if stream_events:
                yield handle_event(  # type: ignore
                    create_run_content_completed_event(from_run_response=run_response),
                    run_response,
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

            # We should break out of the run function
            if any(tool_call.is_paused for tool_call in run_response.tools or []):
                yield from self._handle_agent_run_paused_stream(
                    run_response=run_response, session=session, user_id=user_id
                )
                return

            # Execute post-hooks
            if self.post_hooks is not None:
                yield from self._execute_post_hooks(
                    hooks=self.post_hooks,  # type: ignore
                    run_output=run_response,
                    session=session,
                    session_state=session_state,
                    dependencies=dependencies,
                    metadata=metadata,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    **kwargs,
                )

            # Check for cancellation before model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 4. Create session summary
            if self.session_summary_manager is not None:
                # Upsert the RunOutput to Agent Session before creating the session summary
                session.upsert_run(run=run_response)

                if stream_events:
                    yield handle_event(  # type: ignore
                        create_session_summary_started_event(from_run_response=run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )
                try:
                    self.session_summary_manager.create_session_summary(session=session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")

                if stream_events:
                    yield handle_event(  # type: ignore
                        create_session_summary_completed_event(
                            from_run_response=run_response, session_summary=session.summary
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )

            # Create the run completed event
            completed_event = handle_event(
                create_run_completed_event(run_response),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

            # Set the run status to completed
            run_response.status = RunStatus.completed

            # 5. Cleanup and store the run response and session
            self._cleanup_and_store(run_response=run_response, session=session, user_id=user_id)

            if stream_events:
                yield completed_event  # type: ignore

            # Log Agent Telemetry
            self._log_agent_telemetry(session_id=session.session_id, run_id=run_response.run_id)

            log_debug(f"Agent Run End: {run_response.run_id}", center=True, symbol="*")

        except RunCancelledException as e:
            # Handle run cancellation during async streaming
            log_info(f"Run {run_response.run_id} was cancelled during streaming")
            run_response.status = RunStatus.cancelled
            run_response.content = str(e)

            # Yield the cancellation event
            yield handle_event(  # type: ignore
                create_run_cancelled_event(from_run_response=run_response, reason=str(e)),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

            # Cleanup and store the run response and session
            self._cleanup_and_store(run_response=run_response, session=session, user_id=user_id)
        finally:
            # Always clean up the run tracking
            cleanup_run(run_response.run_id)  # type: ignore

    @overload
    async def acontinue_run(
        self,
        run_response: Optional[RunOutput] = None,
        *,
        stream: Literal[False] = False,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        run_id: Optional[str] = None,
        updated_tools: Optional[List[ToolExecution]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
    ) -> RunOutput: ...

    @overload
    def acontinue_run(
        self,
        run_response: Optional[RunOutput] = None,
        *,
        stream: Literal[True] = True,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        run_id: Optional[str] = None,
        updated_tools: Optional[List[ToolExecution]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
    ) -> AsyncIterator[Union[RunOutputEvent, RunOutput]]: ...

    def acontinue_run(  # type: ignore
        self,
        run_response: Optional[RunOutput] = None,
        *,
        run_id: Optional[str] = None,
        updated_tools: Optional[List[ToolExecution]] = None,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        yield_run_response: bool = False,
        **kwargs,
    ) -> Union[RunOutput, AsyncIterator[Union[RunOutputEvent, RunOutput]]]:
        """Continue a previous run.

        Args:
            run_response: The run response to continue.
            run_id: The run id to continue. Alternative to passing run_response.
            updated_tools: The updated tools to use for the run. Required to be used with `run_id`.
            stream: Whether to stream the response.
            stream_events: Whether to stream all events.
            user_id: The user id to continue the run for.
            session_id: The session id to continue the run for.
            retries: The number of retries to continue the run for.
            knowledge_filters: The knowledge filters to use for the run.
            dependencies: The dependencies to use for continuing the run.
            metadata: The metadata to use for continuing the run.
            debug_mode: Whether to enable debug mode.
            yield_run_response: Whether to yield the run response.
            (deprecated) stream_intermediate_steps: Whether to stream all steps.
        """
        if run_response is None and run_id is None:
            raise ValueError("Either run_response or run_id must be provided.")

        if run_response is None and (run_id is not None and (session_id is None and self.session_id is None)):
            raise ValueError("Session ID is required to continue a run from a run_id.")

        session_id, user_id = self._initialize_session(
            session_id=session_id,
            user_id=user_id,
        )

        # Initialize the Agent
        self.initialize_agent(debug_mode=debug_mode)

        run_dependencies = dependencies if dependencies is not None else self.dependencies

        # If no retries are set, use the agent's default retries
        retries = retries if retries is not None else self.retries

        # Use stream override value when necessary
        if stream is None:
            stream = False if self.stream is None else self.stream

        # Considering both stream_events and stream_intermediate_steps (deprecated)
        stream_events = stream_events or stream_intermediate_steps

        # Can't stream events if streaming is disabled
        if stream is False:
            stream_events = False

        if stream_events is None:
            stream_events = False if self.stream_events is None else self.stream_events

        # Can't have stream_intermediate_steps if stream is False
        if stream is False:
            stream_events = False

        self.stream = self.stream or stream
        self.stream_events = self.stream_events or stream_events

        # Get knowledge filters
        effective_filters = knowledge_filters
        if self.knowledge_filters or knowledge_filters:
            effective_filters = self._get_effective_filters(knowledge_filters)

        # Merge agent metadata with run metadata
        if self.metadata is not None:
            if metadata is None:
                metadata = self.metadata
            else:
                merge_dictionaries(metadata, self.metadata)

        # Prepare arguments for the model
        response_format = self._get_response_format()
        self.model = cast(Model, self.model)

        last_exception = None
        num_attempts = retries + 1
        for attempt in range(num_attempts):
            try:
                if stream:
                    return self._acontinue_run_stream(
                        run_response=run_response,
                        updated_tools=updated_tools,
                        knowledge_filters=effective_filters,
                        run_id=run_id,
                        user_id=user_id,
                        session_id=session_id,
                        response_format=response_format,
                        dependencies=run_dependencies,
                        stream_events=stream_events,
                        metadata=metadata,
                        yield_run_response=yield_run_response,
                        debug_mode=debug_mode,
                        **kwargs,
                    )
                else:
                    return self._acontinue_run(  # type: ignore
                        session_id=session_id,
                        run_response=run_response,
                        updated_tools=updated_tools,
                        knowledge_filters=effective_filters,
                        run_id=run_id,
                        user_id=user_id,
                        response_format=response_format,
                        dependencies=run_dependencies,
                        metadata=metadata,
                        debug_mode=debug_mode,
                        **kwargs,
                    )
            except ModelProviderError as e:
                log_warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")
                if isinstance(e, StopAgentRun):
                    raise e
                last_exception = e
                if attempt < num_attempts - 1:  # Don't sleep on the last attempt
                    if self.exponential_backoff:
                        delay = 2**attempt * self.delay_between_retries
                    else:
                        delay = self.delay_between_retries
                    import time

                    time.sleep(delay)
            except KeyboardInterrupt:
                run_response = cast(RunOutput, run_response)
                if stream:
                    return async_generator_wrapper(  # type: ignore
                        create_run_cancelled_event(run_response, "Operation cancelled by user")
                    )
                else:
                    run_response.content = "Operation cancelled by user"
                    run_response.status = RunStatus.cancelled
                    return run_response

        # If we get here, all retries failed
        if last_exception is not None:
            log_error(
                f"Failed after {num_attempts} attempts. Last error using {last_exception.model_name}({last_exception.model_id})"
            )
            if stream:
                return async_generator_wrapper(create_run_error_event(run_response, error=str(last_exception)))  # type: ignore
            raise last_exception
        else:
            if stream:
                return async_generator_wrapper(create_run_error_event(run_response, error=str(last_exception)))  # type: ignore
            raise Exception(f"Failed after {num_attempts} attempts.")

    async def _acontinue_run(
        self,
        session_id: str,
        run_response: Optional[RunOutput] = None,
        updated_tools: Optional[List[ToolExecution]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs,
    ) -> RunOutput:
        """Continue a previous run.

        Steps:
        1. Read existing session from db
        2. Resolve dependencies
        3. Update metadata and session state
        4. Prepare run response
        5. Determine tools for model
        6. Prepare run messages
        7. Handle the updated tools
        8. Get model response
        9. Update the RunOutput with the model response
        10. Convert response to structured format
        11. Store media if enabled
        12. Execute post-hooks
        13. Create session summary
        14. Cleanup and store (scrub, stop timer, save to file, add to session, calculate metrics, save session)
        """
        log_debug(f"Agent Run Continue: {run_response.run_id if run_response else run_id}", center=True)  # type: ignore

        # 1. Read existing session from db
        agent_session = await self._aread_or_create_session(session_id=session_id, user_id=user_id)

        # 2. Resolve dependencies
        if dependencies is not None:
            await self._aresolve_run_dependencies(dependencies=dependencies)

        # 3. Update metadata and session state
        self._update_metadata(session=agent_session)
        # Initialize session state
        session_state = self._initialize_session_state(
            session_state={}, user_id=user_id, session_id=session_id, run_id=run_id
        )
        # Update session state from DB
        if session_state is not None:
            session_state = self._load_session_state(session=agent_session, session_state=session_state)

        # 4. Prepare run response
        if run_response is not None:
            # The run is continued from a provided run_response. This contains the updated tools.
            input = run_response.messages or []
        elif run_id is not None:
            # The run is continued from a run_id. This requires the updated tools to be passed.
            if updated_tools is None:
                raise ValueError("Updated tools are required to continue a run from a run_id.")

            runs = agent_session.runs
            run_response = next((r for r in runs if r.run_id == run_id), None)  # type: ignore
            if run_response is None:
                raise RuntimeError(f"No runs found for run ID {run_id}")
            run_response.tools = updated_tools
            input = run_response.messages or []
        else:
            raise ValueError("Either run_response or run_id must be provided.")

        run_response = cast(RunOutput, run_response)
        run_response.status = RunStatus.running

        # 5. Determine tools for model
        self.model = cast(Model, self.model)
        processed_tools = await self.aget_tools(
            run_response=run_response,
            session=agent_session,
            user_id=user_id,
            knowledge_filters=knowledge_filters,
        )
        _tools = self._determine_tools_for_model(
            model=self.model,
            processed_tools=processed_tools,
            run_response=run_response,
            session=agent_session,
            session_state=session_state,
            dependencies=dependencies,
        )

        # 6. Prepare run messages
        run_messages: RunMessages = self._get_continue_run_messages(
            input=input,
        )

        # Register run for cancellation tracking
        register_run(run_response.run_id)  # type: ignore

        try:
            # 7. Handle the updated tools
            await self._ahandle_tool_call_updates(run_response=run_response, run_messages=run_messages, tools=_tools)

            # 8. Get model response
            model_response: ModelResponse = await self.model.aresponse(
                messages=run_messages.messages,
                response_format=response_format,
                tools=_tools,
                tool_choice=self.tool_choice,
                tool_call_limit=self.tool_call_limit,
                per_tool_call_limits=self._build_per_tool_call_limits(_tools) if _tools else None,
            )
            # Check for cancellation after model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # If an output model is provided, generate output using the output model
            await self._agenerate_response_with_output_model(model_response=model_response, run_messages=run_messages)

            # If a parser model is provided, structure the response separately
            await self._aparse_response_with_parser_model(model_response=model_response, run_messages=run_messages)

            # 9. Update the RunOutput with the model response
            self._update_run_response(
                model_response=model_response,
                run_response=run_response,
                run_messages=run_messages,
            )

            # Break out of the run function if a tool call is paused
            if any(tool_call.is_paused for tool_call in run_response.tools or []):
                return await self._ahandle_agent_run_paused(
                    run_response=run_response, session=agent_session, user_id=user_id
                )

            # 10. Convert the response to the structured format if needed
            self._convert_response_to_structured_format(run_response)

            # 11. Store media if enabled
            if self.store_media:
                self._store_media(run_response, model_response)

            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 12. Execute post-hooks
            if self.post_hooks is not None:
                async for _ in self._aexecute_post_hooks(
                    hooks=self.post_hooks,  # type: ignore
                    run_output=run_response,
                    session=agent_session,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    session_state=session_state,
                    dependencies=dependencies,
                    metadata=metadata,
                    **kwargs,
                ):
                    pass

            # Check for cancellation
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 13. Create session summary
            if self.session_summary_manager is not None:
                # Upsert the RunOutput to Agent Session before creating the session summary
                agent_session.upsert_run(run=run_response)

                try:
                    await self.session_summary_manager.acreate_session_summary(session=agent_session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")

            # Set the run status to completed
            run_response.status = RunStatus.completed

            # 14. Cleanup and store the run response and session
            await self._acleanup_and_store(run_response=run_response, session=agent_session, user_id=user_id)

            # Log Agent Telemetry
            await self._alog_agent_telemetry(session_id=agent_session.session_id, run_id=run_response.run_id)

            log_debug(f"Agent Run End: {run_response.run_id}", center=True, symbol="*")

            return run_response

        except RunCancelledException as e:
            # Handle run cancellation
            log_info(f"Run {run_response.run_id} was cancelled")
            run_response.content = str(e)
            run_response.status = RunStatus.cancelled

            # Cleanup and store the run response and session
            await self._acleanup_and_store(run_response=run_response, session=agent_session, user_id=user_id)

            return run_response
        finally:
            # Always disconnect MCP tools
            await self._disconnect_mcp_tools()

            # Always clean up the run tracking
            cleanup_run(run_response.run_id)  # type: ignore

    async def _acontinue_run_stream(
        self,
        session_id: str,
        run_response: Optional[RunOutput] = None,
        updated_tools: Optional[List[ToolExecution]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        stream_events: bool = False,
        yield_run_response: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs,
    ) -> AsyncIterator[Union[RunOutputEvent, RunOutput]]:
        """Continue a previous run.

        Steps:
        1. Resolve dependencies
        2. Read existing session from db
        3. Update session state and metadata
        4. Prepare run response
        5. Determine tools for model
        6. Prepare run messages
        7. Handle the updated tools
        8. Process model response
        9. Create session summary
        10. Execute post-hooks
        11. Cleanup and store the run response and session
        """
        log_debug(f"Agent Run Continue: {run_response.run_id if run_response else run_id}", center=True)  # type: ignore

        # 1. Resolve dependencies
        if dependencies is not None:
            await self._aresolve_run_dependencies(dependencies=dependencies)

        # 2. Read existing session from db
        agent_session = await self._aread_or_create_session(session_id=session_id, user_id=user_id)

        # 3. Update session state and metadata
        self._update_metadata(session=agent_session)
        # Initialize session state
        session_state = self._initialize_session_state(
            session_state={}, user_id=user_id, session_id=session_id, run_id=run_id
        )
        # Update session state from DB
        if session_state is not None:
            session_state = self._load_session_state(session=agent_session, session_state=session_state)

        # 4. Prepare run response
        if run_response is not None:
            # The run is continued from a provided run_response. This contains the updated tools.
            input = run_response.messages or []
        elif run_id is not None:
            # The run is continued from a run_id. This requires the updated tools to be passed.
            if updated_tools is None:
                raise ValueError("Updated tools are required to continue a run from a run_id.")

            runs = agent_session.runs
            run_response = next((r for r in runs if r.run_id == run_id), None)  # type: ignore
            if run_response is None:
                raise RuntimeError(f"No runs found for run ID {run_id}")
            run_response.tools = updated_tools
            input = run_response.messages or []
        else:
            raise ValueError("Either run_response or run_id must be provided.")

        run_response = cast(RunOutput, run_response)
        run_response.status = RunStatus.running

        # 5. Determine tools for model
        self.model = cast(Model, self.model)
        processed_tools = await self.aget_tools(
            run_response=run_response,
            session=agent_session,
            user_id=user_id,
            knowledge_filters=knowledge_filters,
        )
        _tools = self._determine_tools_for_model(
            model=self.model,
            processed_tools=processed_tools,
            run_response=run_response,
            session=agent_session,
            session_state=session_state,
            dependencies=dependencies,
        )

        # 6. Prepare run messages
        run_messages: RunMessages = self._get_continue_run_messages(
            input=input,
        )

        # Register run for cancellation tracking
        register_run(run_response.run_id)  # type: ignore

        try:
            # Start the Run by yielding a RunContinued event
            if stream_events:
                yield handle_event(  # type: ignore
                    create_run_continued_event(run_response),
                    run_response,
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

            # 7. Handle the updated tools
            async for event in self._ahandle_tool_call_updates_stream(
                run_response=run_response, run_messages=run_messages, tools=_tools, stream_events=stream_events
            ):
                raise_if_cancelled(run_response.run_id)  # type: ignore
                yield event

            # 8. Process model response
            if self.output_model is None:
                async for event in self._ahandle_model_response_stream(
                    session=agent_session,
                    run_response=run_response,
                    run_messages=run_messages,
                    tools=_tools,
                    response_format=response_format,
                    stream_events=stream_events,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    yield event
            else:
                from agno.run.agent import (
                    IntermediateRunContentEvent,
                    RunContentEvent,
                )  # type: ignore

                async for event in self._ahandle_model_response_stream(
                    session=agent_session,
                    run_response=run_response,
                    run_messages=run_messages,
                    tools=_tools,
                    response_format=response_format,
                    stream_events=stream_events,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    if isinstance(event, RunContentEvent):
                        if stream_events:
                            yield IntermediateRunContentEvent(
                                content=event.content,
                                content_type=event.content_type,
                            )
                    else:
                        yield event

                # If an output model is provided, generate output using the output model
                async for event in self._agenerate_response_with_output_model_stream(
                    session=agent_session,
                    run_response=run_response,
                    run_messages=run_messages,
                    stream_events=stream_events,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    yield event

            # Check for cancellation after model processing
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # Parse response with parser model if provided
            async for event in self._aparse_response_with_parser_model_stream(
                session=agent_session, run_response=run_response, stream_events=stream_events
            ):
                yield event

            # Yield RunContentCompletedEvent
            if stream_events:
                yield handle_event(  # type: ignore
                    create_run_content_completed_event(from_run_response=run_response),
                    run_response,
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

            # Break out of the run function if a tool call is paused
            if any(tool_call.is_paused for tool_call in run_response.tools or []):
                async for item in self._ahandle_agent_run_paused_stream(
                    run_response=run_response, session=agent_session, user_id=user_id
                ):
                    yield item
                return

            # 8. Execute post-hooks
            if self.post_hooks is not None:
                async for event in self._aexecute_post_hooks(
                    hooks=self.post_hooks,  # type: ignore
                    run_output=run_response,
                    session=agent_session,
                    user_id=user_id,
                    session_state=session_state,
                    dependencies=dependencies,
                    metadata=metadata,
                    debug_mode=debug_mode,
                    **kwargs,
                ):
                    yield event
            # Check for cancellation before model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 9. Create session summary
            if self.session_summary_manager is not None:
                # Upsert the RunOutput to Agent Session before creating the session summary
                agent_session.upsert_run(run=run_response)

                if stream_events:
                    yield handle_event(  # type: ignore
                        create_session_summary_started_event(from_run_response=run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )
                try:
                    await self.session_summary_manager.acreate_session_summary(session=agent_session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")
                if stream_events:
                    yield handle_event(  # type: ignore
                        create_session_summary_completed_event(
                            from_run_response=run_response, session_summary=agent_session.summary
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )

            # Create the run completed event
            completed_event = handle_event(
                create_run_completed_event(run_response),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

            # Set the run status to completed
            run_response.status = RunStatus.completed

            # 10. Cleanup and store the run response and session
            await self._acleanup_and_store(run_response=run_response, session=agent_session, user_id=user_id)

            if stream_events:
                yield completed_event  # type: ignore

            if yield_run_response:
                yield run_response

            # Log Agent Telemetry
            await self._alog_agent_telemetry(session_id=agent_session.session_id, run_id=run_response.run_id)

            log_debug(f"Agent Run End: {run_response.run_id}", center=True, symbol="*")
        except RunCancelledException as e:
            # Handle run cancellation during streaming
            log_info(f"Run {run_response.run_id} was cancelled during streaming")
            run_response.status = RunStatus.cancelled
            run_response.content = str(e)

            # Yield the cancellation event
            yield handle_event(  # type: ignore
                create_run_cancelled_event(from_run_response=run_response, reason=str(e)),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

            # Cleanup and store the run response and session
            await self._acleanup_and_store(run_response=run_response, session=agent_session, user_id=user_id)
        finally:
            # Always disconnect MCP tools
            await self._disconnect_mcp_tools()

            # Always clean up the run tracking
            cleanup_run(run_response.run_id)  # type: ignore

    def _execute_pre_hooks(
        self,
        hooks: Optional[List[Callable[..., Any]]],
        run_response: RunOutput,
        run_input: RunInput,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[RunOutputEvent]:
        """Execute multiple pre-hook functions in succession."""
        if hooks is None:
            return

        # Prepare all possible arguments once
        all_args = {
            "run_input": run_input,
            "agent": self,
            "session": session,
            "session_state": session_state,
            "dependencies": dependencies,
            "metadata": metadata,
            "user_id": user_id,
            "debug_mode": debug_mode or self.debug_mode,
        }
        all_args.update(kwargs)

        for i, hook in enumerate(hooks):
            yield handle_event(  # type: ignore
                run_response=run_response,
                event=create_pre_hook_started_event(
                    from_run_response=run_response,
                    run_input=run_input,
                    pre_hook_name=hook.__name__,
                ),
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )
            try:
                # Filter arguments to only include those that the hook accepts
                filtered_args = filter_hook_args(hook, all_args)

                hook(**filtered_args)

                yield handle_event(  # type: ignore
                    run_response=run_response,
                    event=create_pre_hook_completed_event(
                        from_run_response=run_response,
                        run_input=run_input,
                        pre_hook_name=hook.__name__,
                    ),
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

            except (InputCheckError, OutputCheckError) as e:
                raise e
            except Exception as e:
                log_error(f"Pre-hook #{i + 1} execution failed: {str(e)}")
                log_exception(e)
            finally:
                # Reset global log mode incase an agent in the pre-hook changed it
                self._set_debug(debug_mode=debug_mode)

        # Update the input on the run_response
        run_response.input = run_input

    async def _aexecute_pre_hooks(
        self,
        hooks: Optional[List[Callable[..., Any]]],
        run_response: RunOutput,
        run_input: RunInput,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunOutputEvent]:
        """Execute multiple pre-hook functions in succession (async version)."""
        if hooks is None:
            return

        # Prepare all possible arguments once
        all_args = {
            "run_input": run_input,
            "agent": self,
            "session": session,
            "session_state": session_state,
            "dependencies": dependencies,
            "metadata": metadata,
            "user_id": user_id,
            "debug_mode": debug_mode or self.debug_mode,
        }
        all_args.update(kwargs)

        for i, hook in enumerate(hooks):
            yield handle_event(  # type: ignore
                run_response=run_response,
                event=create_pre_hook_started_event(
                    from_run_response=run_response,
                    run_input=run_input,
                    pre_hook_name=hook.__name__,
                ),
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )
            try:
                # Filter arguments to only include those that the hook accepts
                filtered_args = filter_hook_args(hook, all_args)

                if iscoroutinefunction(hook):
                    await hook(**filtered_args)
                else:
                    # Synchronous function
                    hook(**filtered_args)

                yield handle_event(  # type: ignore
                    run_response=run_response,
                    event=create_pre_hook_completed_event(
                        from_run_response=run_response,
                        run_input=run_input,
                        pre_hook_name=hook.__name__,
                    ),
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

            except (InputCheckError, OutputCheckError) as e:
                raise e
            except Exception as e:
                log_error(f"Pre-hook #{i + 1} execution failed: {str(e)}")
                log_exception(e)
            finally:
                # Reset global log mode incase an agent in the pre-hook changed it
                self._set_debug(debug_mode=debug_mode)

        # Update the input on the run_response
        run_response.input = run_input

    def _execute_post_hooks(
        self,
        hooks: Optional[List[Callable[..., Any]]],
        run_output: RunOutput,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[RunOutputEvent]:
        """Execute multiple post-hook functions in succession."""
        if hooks is None:
            return

        # Prepare all possible arguments once
        all_args = {
            "run_output": run_output,
            "agent": self,
            "session": session,
            "user_id": user_id,
            "session_state": session_state,
            "dependencies": dependencies,
            "metadata": metadata,
            "debug_mode": debug_mode or self.debug_mode,
        }
        all_args.update(kwargs)

        for i, hook in enumerate(hooks):
            yield handle_event(  # type: ignore
                run_response=run_output,
                event=create_post_hook_started_event(
                    from_run_response=run_output,
                    post_hook_name=hook.__name__,
                ),
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )
            try:
                # Filter arguments to only include those that the hook accepts
                filtered_args = filter_hook_args(hook, all_args)

                hook(**filtered_args)

                yield handle_event(  # type: ignore
                    run_response=run_output,
                    event=create_post_hook_completed_event(
                        from_run_response=run_output,
                        post_hook_name=hook.__name__,
                    ),
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )
            except (InputCheckError, OutputCheckError) as e:
                raise e
            except Exception as e:
                log_error(f"Post-hook #{i + 1} execution failed: {str(e)}")
                log_exception(e)
            finally:
                # Reset global log mode incase an agent in the pre-hook changed it
                self._set_debug(debug_mode=debug_mode)

    async def _aexecute_post_hooks(
        self,
        hooks: Optional[List[Callable[..., Any]]],
        run_output: RunOutput,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunOutputEvent]:
        """Execute multiple post-hook functions in succession (async version)."""
        if hooks is None:
            return

        # Prepare all possible arguments once
        all_args = {
            "run_output": run_output,
            "agent": self,
            "session": session,
            "user_id": user_id,
            "session_state": session_state,
            "dependencies": dependencies,
            "metadata": metadata,
            "debug_mode": debug_mode or self.debug_mode,
        }
        all_args.update(kwargs)

        for i, hook in enumerate(hooks):
            yield handle_event(  # type: ignore
                run_response=run_output,
                event=create_post_hook_started_event(
                    from_run_response=run_output,
                    post_hook_name=hook.__name__,
                ),
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )
            try:
                # Filter arguments to only include those that the hook accepts
                filtered_args = filter_hook_args(hook, all_args)
                from inspect import iscoroutinefunction

                if iscoroutinefunction(hook):
                    await hook(**filtered_args)
                else:
                    hook(**filtered_args)

                yield handle_event(  # type: ignore
                    run_response=run_output,
                    event=create_post_hook_completed_event(
                        from_run_response=run_output,
                        post_hook_name=hook.__name__,
                    ),
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

            except (InputCheckError, OutputCheckError) as e:
                raise e
            except Exception as e:
                log_error(f"Post-hook #{i + 1} execution failed: {str(e)}")
                log_exception(e)
            finally:
                # Reset global log mode incase an agent in the pre-hook changed it
                self._set_debug(debug_mode=debug_mode)

    def _handle_agent_run_paused(
        self,
        run_response: RunOutput,
        session: AgentSession,
        user_id: Optional[str] = None,
    ) -> RunOutput:
        # Set the run response to paused

        run_response.status = RunStatus.paused
        if not run_response.content:
            run_response.content = get_paused_content(run_response)

        self._cleanup_and_store(run_response=run_response, session=session, user_id=user_id)

        log_debug(f"Agent Run Paused: {run_response.run_id}", center=True, symbol="*")

        # We return and await confirmation/completion for the tools that require it
        return run_response

    def _handle_agent_run_paused_stream(
        self,
        run_response: RunOutput,
        session: AgentSession,
        user_id: Optional[str] = None,
    ) -> Iterator[RunOutputEvent]:
        # Set the run response to paused

        run_response.status = RunStatus.paused
        if not run_response.content:
            run_response.content = get_paused_content(run_response)

        # We return and await confirmation/completion for the tools that require it
        pause_event = handle_event(
            create_run_paused_event(
                from_run_response=run_response,
                tools=run_response.tools,
            ),
            run_response,
            events_to_skip=self.events_to_skip,  # type: ignore
            store_events=self.store_events,
        )

        self._cleanup_and_store(run_response=run_response, session=session, user_id=user_id)

        yield pause_event  # type: ignore

        log_debug(f"Agent Run Paused: {run_response.run_id}", center=True, symbol="*")

    async def _ahandle_agent_run_paused(
        self,
        run_response: RunOutput,
        session: AgentSession,
        user_id: Optional[str] = None,
    ) -> RunOutput:
        # Set the run response to paused

        run_response.status = RunStatus.paused
        if not run_response.content:
            run_response.content = get_paused_content(run_response)

        await self._acleanup_and_store(run_response=run_response, session=session, user_id=user_id)

        log_debug(f"Agent Run Paused: {run_response.run_id}", center=True, symbol="*")

        # We return and await confirmation/completion for the tools that require it
        return run_response

    async def _ahandle_agent_run_paused_stream(
        self,
        run_response: RunOutput,
        session: AgentSession,
        user_id: Optional[str] = None,
    ) -> AsyncIterator[RunOutputEvent]:
        # Set the run response to paused

        run_response.status = RunStatus.paused
        if not run_response.content:
            run_response.content = get_paused_content(run_response)

        # We return and await confirmation/completion for the tools that require it
        pause_event = handle_event(
            create_run_paused_event(
                from_run_response=run_response,
                tools=run_response.tools,
            ),
            run_response,
            events_to_skip=self.events_to_skip,  # type: ignore
            store_events=self.store_events,
        )

        await self._acleanup_and_store(run_response=run_response, session=session, user_id=user_id)

        yield pause_event  # type: ignore

        log_debug(f"Agent Run Paused: {run_response.run_id}", center=True, symbol="*")

    def _convert_response_to_structured_format(self, run_response: Union[RunOutput, ModelResponse]):
        # Convert the response to the structured format if needed
        if self.output_schema is not None and not isinstance(run_response.content, self.output_schema):
            if isinstance(run_response.content, str) and self.parse_response:
                try:
                    structured_output = parse_response_model_str(run_response.content, self.output_schema)

                    # Update RunOutput
                    if structured_output is not None:
                        run_response.content = structured_output
                        if isinstance(run_response, RunOutput):
                            run_response.content_type = self.output_schema.__name__
                    else:
                        log_warning("Failed to convert response to output_schema")
                except Exception as e:
                    log_warning(f"Failed to convert response to output model: {e}")
            else:
                log_warning("Something went wrong. Run response content is not a string")

    def _handle_external_execution_update(self, run_messages: RunMessages, tool: ToolExecution):
        self.model = cast(Model, self.model)

        if tool.result is not None:
            for msg in run_messages.messages:
                # Skip if the message is already in the run_messages
                if msg.tool_call_id == tool.tool_call_id:
                    break

            run_messages.messages.append(
                Message(
                    role=self.model.tool_message_role,
                    content=tool.result,
                    tool_call_id=tool.tool_call_id,
                    tool_name=tool.tool_name,
                    tool_args=tool.tool_args,
                    tool_call_error=tool.tool_call_error,
                    stop_after_tool_call=tool.stop_after_tool_call,
                )
            )
            tool.external_execution_required = False
        else:
            raise ValueError(f"Tool {tool.tool_name} requires external execution, cannot continue run")

    def _handle_user_input_update(self, tool: ToolExecution):
        for field in tool.user_input_schema or []:
            if not tool.tool_args:
                tool.tool_args = {}
            tool.tool_args[field.name] = field.value

    def _handle_get_user_input_tool_update(self, run_messages: RunMessages, tool: ToolExecution):
        import json

        self.model = cast(Model, self.model)
        # Skipping tool without user_input_schema so that tool_call_id is not repeated
        if not hasattr(tool, "user_input_schema") or not tool.user_input_schema:
            return
        user_input_result = [
            {"name": user_input_field.name, "value": user_input_field.value}
            for user_input_field in tool.user_input_schema or []
        ]
        # Add the tool call result to the run_messages
        run_messages.messages.append(
            Message(
                role=self.model.tool_message_role,
                content=f"User inputs retrieved: {json.dumps(user_input_result)}",
                tool_call_id=tool.tool_call_id,
                tool_name=tool.tool_name,
                tool_args=tool.tool_args,
                metrics=Metrics(duration=0),
            )
        )

    def _run_tool(
        self,
        run_response: RunOutput,
        run_messages: RunMessages,
        tool: ToolExecution,
        functions: Optional[Dict[str, Function]] = None,
        stream_events: bool = False,
    ) -> Iterator[RunOutputEvent]:
        self.model = cast(Model, self.model)
        # Execute the tool
        function_call = self.model.get_function_call_to_run_from_tool_execution(tool, functions)
        function_call_results: List[Message] = []

        for call_result in self.model.run_function_call(
            function_call=function_call,
            function_call_results=function_call_results,
        ):
            if isinstance(call_result, ModelResponse):
                if call_result.event == ModelResponseEvent.tool_call_started.value:
                    if stream_events:
                        yield handle_event(  # type: ignore
                            create_tool_call_started_event(from_run_response=run_response, tool=tool),
                            run_response,
                            events_to_skip=self.events_to_skip,  # type: ignore
                            store_events=self.store_events,
                        )

                if call_result.event == ModelResponseEvent.tool_call_completed.value and call_result.tool_executions:
                    tool_execution = call_result.tool_executions[0]
                    tool.result = tool_execution.result
                    tool.tool_call_error = tool_execution.tool_call_error
                    if stream_events:
                        yield handle_event(  # type: ignore
                            create_tool_call_completed_event(
                                from_run_response=run_response, tool=tool, content=call_result.content
                            ),
                            run_response,
                            events_to_skip=self.events_to_skip,  # type: ignore
                            store_events=self.store_events,
                        )

        if len(function_call_results) > 0:
            run_messages.messages.extend(function_call_results)

    def _reject_tool_call(
        self, run_messages: RunMessages, tool: ToolExecution, functions: Optional[Dict[str, Function]] = None
    ):
        self.model = cast(Model, self.model)
        function_call = self.model.get_function_call_to_run_from_tool_execution(tool, functions)
        function_call.error = tool.confirmation_note or "Function call was rejected by the user"
        function_call_result = self.model.create_function_call_result(
            function_call=function_call,
            success=False,
        )
        run_messages.messages.append(function_call_result)

    async def _arun_tool(
        self,
        run_response: RunOutput,
        run_messages: RunMessages,
        tool: ToolExecution,
        functions: Optional[Dict[str, Function]] = None,
        stream_events: bool = False,
    ) -> AsyncIterator[RunOutputEvent]:
        self.model = cast(Model, self.model)

        # Execute the tool
        function_call = self.model.get_function_call_to_run_from_tool_execution(tool, functions)
        function_call_results: List[Message] = []

        async for call_result in self.model.arun_function_calls(
            function_calls=[function_call],
            function_call_results=function_call_results,
            skip_pause_check=True,
        ):
            if isinstance(call_result, ModelResponse):
                if call_result.event == ModelResponseEvent.tool_call_started.value:
                    if stream_events:
                        yield handle_event(  # type: ignore
                            create_tool_call_started_event(from_run_response=run_response, tool=tool),
                            run_response,
                            events_to_skip=self.events_to_skip,  # type: ignore
                            store_events=self.store_events,
                        )
                if call_result.event == ModelResponseEvent.tool_call_completed.value and call_result.tool_executions:
                    tool_execution = call_result.tool_executions[0]
                    tool.result = tool_execution.result
                    tool.tool_call_error = tool_execution.tool_call_error
                    if stream_events:
                        yield handle_event(  # type: ignore
                            create_tool_call_completed_event(
                                from_run_response=run_response, tool=tool, content=call_result.content
                            ),
                            run_response,
                            events_to_skip=self.events_to_skip,  # type: ignore
                            store_events=self.store_events,
                        )
        if len(function_call_results) > 0:
            run_messages.messages.extend(function_call_results)

    def _handle_tool_call_updates(
        self, run_response: RunOutput, run_messages: RunMessages, tools: List[Union[Function, dict]]
    ):
        self.model = cast(Model, self.model)
        _functions = {tool.name: tool for tool in tools if isinstance(tool, Function)}

        for _t in run_response.tools or []:
            # Case 1: Handle confirmed tools and execute them
            if _t.requires_confirmation is not None and _t.requires_confirmation is True and _functions:
                # Tool is confirmed and hasn't been run before
                if _t.confirmed is not None and _t.confirmed is True and _t.result is None:
                    # Consume the generator without yielding
                    deque(self._run_tool(run_response, run_messages, _t, functions=_functions), maxlen=0)
                else:
                    self._reject_tool_call(run_messages, _t, functions=_functions)
                    _t.confirmed = False
                    _t.confirmation_note = _t.confirmation_note or "Tool call was rejected"
                    _t.tool_call_error = True
                _t.requires_confirmation = False

            # Case 2: Handle external execution required tools
            elif _t.external_execution_required is not None and _t.external_execution_required is True:
                self._handle_external_execution_update(run_messages=run_messages, tool=_t)

            # Case 3: Agentic user input required
            elif (
                _t.tool_name == "get_user_input"
                and _t.requires_user_input is not None
                and _t.requires_user_input is True
            ):
                self._handle_get_user_input_tool_update(run_messages=run_messages, tool=_t)
                _t.requires_user_input = False

            # Case 4: Handle user input required tools
            elif _t.requires_user_input is not None and _t.requires_user_input is True:
                self._handle_user_input_update(tool=_t)
                _t.requires_user_input = False
                _t.answered = True
                # Consume the generator without yielding
                deque(self._run_tool(run_response, run_messages, _t, functions=_functions), maxlen=0)

    def _handle_tool_call_updates_stream(
        self,
        run_response: RunOutput,
        run_messages: RunMessages,
        tools: List[Union[Function, dict]],
        stream_events: bool = False,
    ) -> Iterator[RunOutputEvent]:
        self.model = cast(Model, self.model)
        _functions = {tool.name: tool for tool in tools if isinstance(tool, Function)}

        for _t in run_response.tools or []:
            # Case 1: Handle confirmed tools and execute them
            if _t.requires_confirmation is not None and _t.requires_confirmation is True and _functions:
                # Tool is confirmed and hasn't been run before
                if _t.confirmed is not None and _t.confirmed is True and _t.result is None:
                    yield from self._run_tool(
                        run_response, run_messages, _t, functions=_functions, stream_events=stream_events
                    )
                else:
                    self._reject_tool_call(run_messages, _t, functions=_functions)
                    _t.confirmed = False
                    _t.confirmation_note = _t.confirmation_note or "Tool call was rejected"
                    _t.tool_call_error = True
                _t.requires_confirmation = False

            # Case 2: Handle external execution required tools
            elif _t.external_execution_required is not None and _t.external_execution_required is True:
                self._handle_external_execution_update(run_messages=run_messages, tool=_t)

            # Case 3: Agentic user input required
            elif (
                _t.tool_name == "get_user_input"
                and _t.requires_user_input is not None
                and _t.requires_user_input is True
            ):
                self._handle_get_user_input_tool_update(run_messages=run_messages, tool=_t)
                _t.requires_user_input = False
                _t.answered = True

            # Case 4: Handle user input required tools
            elif _t.requires_user_input is not None and _t.requires_user_input is True:
                self._handle_user_input_update(tool=_t)
                yield from self._run_tool(
                    run_response, run_messages, _t, functions=_functions, stream_events=stream_events
                )
                _t.requires_user_input = False
                _t.answered = True

    async def _ahandle_tool_call_updates(
        self, run_response: RunOutput, run_messages: RunMessages, tools: List[Union[Function, dict]]
    ):
        self.model = cast(Model, self.model)
        _functions = {tool.name: tool for tool in tools if isinstance(tool, Function)}

        for _t in run_response.tools or []:
            # Case 1: Handle confirmed tools and execute them
            if _t.requires_confirmation is not None and _t.requires_confirmation is True and _functions:
                # Tool is confirmed and hasn't been run before
                if _t.confirmed is not None and _t.confirmed is True and _t.result is None:
                    async for _ in self._arun_tool(run_response, run_messages, _t, functions=_functions):
                        pass
                else:
                    self._reject_tool_call(run_messages, _t, functions=_functions)
                    _t.confirmed = False
                    _t.confirmation_note = _t.confirmation_note or "Tool call was rejected"
                    _t.tool_call_error = True
                _t.requires_confirmation = False

            # Case 2: Handle external execution required tools
            elif _t.external_execution_required is not None and _t.external_execution_required is True:
                self._handle_external_execution_update(run_messages=run_messages, tool=_t)
            # Case 3: Agentic user input required
            elif (
                _t.tool_name == "get_user_input"
                and _t.requires_user_input is not None
                and _t.requires_user_input is True
            ):
                self._handle_get_user_input_tool_update(run_messages=run_messages, tool=_t)
                _t.requires_user_input = False
                _t.answered = True
            # Case 4: Handle user input required tools
            elif _t.requires_user_input is not None and _t.requires_user_input is True:
                self._handle_user_input_update(tool=_t)
                async for _ in self._arun_tool(run_response, run_messages, _t, functions=_functions):
                    pass
                _t.requires_user_input = False
                _t.answered = True

    async def _ahandle_tool_call_updates_stream(
        self,
        run_response: RunOutput,
        run_messages: RunMessages,
        tools: List[Union[Function, dict]],
        stream_events: bool = False,
    ) -> AsyncIterator[RunOutputEvent]:
        self.model = cast(Model, self.model)
        _functions = {tool.name: tool for tool in tools if isinstance(tool, Function)}

        for _t in run_response.tools or []:
            # Case 1: Handle confirmed tools and execute them
            if _t.requires_confirmation is not None and _t.requires_confirmation is True and _functions:
                # Tool is confirmed and hasn't been run before
                if _t.confirmed is not None and _t.confirmed is True and _t.result is None:
                    async for event in self._arun_tool(
                        run_response, run_messages, _t, functions=_functions, stream_events=stream_events
                    ):
                        yield event
                else:
                    self._reject_tool_call(run_messages, _t, functions=_functions)
                    _t.confirmed = False
                    _t.confirmation_note = _t.confirmation_note or "Tool call was rejected"
                    _t.tool_call_error = True
                _t.requires_confirmation = False

            # Case 2: Handle external execution required tools
            elif _t.external_execution_required is not None and _t.external_execution_required is True:
                self._handle_external_execution_update(run_messages=run_messages, tool=_t)
            # Case 3: Agentic user input required
            elif (
                _t.tool_name == "get_user_input"
                and _t.requires_user_input is not None
                and _t.requires_user_input is True
            ):
                self._handle_get_user_input_tool_update(run_messages=run_messages, tool=_t)
                _t.requires_user_input = False
                _t.answered = True
            # # Case 4: Handle user input required tools
            elif _t.requires_user_input is not None and _t.requires_user_input is True:
                self._handle_user_input_update(tool=_t)
                async for event in self._arun_tool(
                    run_response, run_messages, _t, functions=_functions, stream_events=stream_events
                ):
                    yield event
                _t.requires_user_input = False
                _t.answered = True

    def _store_media(self, run_response: RunOutput, model_response: ModelResponse):
        """Store media from model response in run_response for persistence"""
        # Handle generated media fields from ModelResponse (generated media)
        if model_response.images is not None:
            for image in model_response.images:
                self._add_image(image, run_response)  # Generated images go to run_response.images

        if model_response.videos is not None:
            for video in model_response.videos:
                self._add_video(video, run_response)  # Generated videos go to run_response.videos

        if model_response.audios is not None:
            for audio in model_response.audios:
                self._add_audio(audio, run_response)  # Generated audio go to run_response.audio

        if model_response.files is not None:
            for file in model_response.files:
                self._add_file(file, run_response)  # Generated files go to run_response.files

    def _update_run_response(
        self,
        model_response: ModelResponse,
        run_response: RunOutput,
        run_messages: RunMessages,
    ):
        # Handle structured outputs
        if self.output_schema is not None and model_response.parsed is not None:
            # We get native structured outputs from the model
            if self._model_should_return_structured_output():
                # Update the run_response content with the structured output
                run_response.content = model_response.parsed
                # Update the run_response content_type with the structured output class name
                run_response.content_type = self.output_schema.__name__
        else:
            # Update the run_response content with the model response content
            run_response.content = model_response.content

        # Update the run_response reasoning content with the model response reasoning content
        if model_response.reasoning_content is not None:
            run_response.reasoning_content = model_response.reasoning_content
        if model_response.redacted_reasoning_content is not None:
            if run_response.reasoning_content is None:
                run_response.reasoning_content = model_response.redacted_reasoning_content
            else:
                run_response.reasoning_content += model_response.redacted_reasoning_content

        # Update the run_response citations with the model response citations
        if model_response.citations is not None:
            run_response.citations = model_response.citations
        if model_response.provider_data is not None:
            run_response.model_provider_data = model_response.provider_data

        # Update the run_response tools with the model response tool_executions
        if model_response.tool_executions is not None:
            if run_response.tools is None:
                run_response.tools = model_response.tool_executions
            else:
                run_response.tools.extend(model_response.tool_executions)

            # For Reasoning/Thinking/Knowledge Tools update reasoning_content in RunOutput
            for tool_call in model_response.tool_executions:
                tool_name = tool_call.tool_name or ""
                if tool_name.lower() in ["think", "analyze"]:
                    tool_args = tool_call.tool_args or {}
                    self._update_reasoning_content_from_tool_call(
                        run_response=run_response,
                        tool_name=tool_name,
                        tool_args=tool_args,
                    )

        # Update the run_response audio with the model response audio
        if model_response.audio is not None:
            run_response.response_audio = model_response.audio

        # Update the run_response created_at with the model response created_at
        run_response.created_at = model_response.created_at

        # Build a list of messages that should be added to the RunOutput
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        # Update the RunOutput messages
        run_response.messages = messages_for_run_response
        # Update the RunOutput metrics
        run_response.metrics = self._calculate_run_metrics(
            messages=messages_for_run_response, current_run_metrics=run_response.metrics
        )

    def _update_session_metrics(self, session: AgentSession, run_response: RunOutput):
        """Calculate session metrics"""
        session_metrics = self._get_session_metrics(session=session)
        # Add the metrics for the current run to the session metrics
        if run_response.metrics is not None:
            session_metrics += run_response.metrics
        session_metrics.time_to_first_token = None
        if session.session_data is not None:
            session.session_data["session_metrics"] = session_metrics

    def _handle_model_response_stream(
        self,
        session: AgentSession,
        run_response: RunOutput,
        run_messages: RunMessages,
        tools: Optional[List[Union[Function, dict]]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        stream_events: bool = False,
    ) -> Iterator[RunOutputEvent]:
        self.model = cast(Model, self.model)

        reasoning_state = {
            "reasoning_started": False,
            "reasoning_time_taken": 0.0,
        }
        model_response = ModelResponse(content="")

        stream_model_response = True
        if self.should_parse_structured_output:
            log_debug("Response model set, model response is not streamed.")
            stream_model_response = False

        for model_response_event in self.model.response_stream(
            messages=run_messages.messages,
            response_format=response_format,
            tools=tools,
            tool_choice=self.tool_choice,
            tool_call_limit=self.tool_call_limit,
            per_tool_call_limits=self._build_per_tool_call_limits(tools) if tools else None,
            stream_model_response=stream_model_response,
            run_response=run_response,
            send_media_to_model=self.send_media_to_model,
        ):
            yield from self._handle_model_response_chunk(
                session=session,
                run_response=run_response,
                model_response=model_response,
                model_response_event=model_response_event,
                reasoning_state=reasoning_state,
                parse_structured_output=self.should_parse_structured_output,
                stream_events=stream_events,
            )

        # Determine reasoning completed
        if stream_events and reasoning_state["reasoning_started"]:
            all_reasoning_steps: List[ReasoningStep] = []
            if run_response and run_response.reasoning_steps:
                all_reasoning_steps = cast(List[ReasoningStep], run_response.reasoning_steps)

            if all_reasoning_steps:
                add_reasoning_metrics_to_metadata(
                    run_response=run_response,
                    reasoning_time_taken=reasoning_state["reasoning_time_taken"],
                )
                yield handle_event(  # type: ignore
                    create_reasoning_completed_event(
                        from_run_response=run_response,
                        content=ReasoningSteps(reasoning_steps=all_reasoning_steps),
                        content_type=ReasoningSteps.__name__,
                    ),
                    run_response,
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

        # Update RunOutput
        # Build a list of messages that should be added to the RunOutput
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        # Update the RunOutput messages
        run_response.messages = messages_for_run_response
        # Update the RunOutput metrics
        run_response.metrics = self._calculate_run_metrics(
            messages=messages_for_run_response, current_run_metrics=run_response.metrics
        )

        # Update the run_response audio if streaming
        if model_response.audio is not None:
            run_response.response_audio = model_response.audio

    async def _ahandle_model_response_stream(
        self,
        session: AgentSession,
        run_response: RunOutput,
        run_messages: RunMessages,
        tools: Optional[List[Union[Function, dict]]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        stream_events: bool = False,
    ) -> AsyncIterator[RunOutputEvent]:
        self.model = cast(Model, self.model)

        reasoning_state = {
            "reasoning_started": False,
            "reasoning_time_taken": 0.0,
        }
        model_response = ModelResponse(content="")

        stream_model_response = True
        if self.should_parse_structured_output:
            log_debug("Response model set, model response is not streamed.")
            stream_model_response = False

        model_response_stream = self.model.aresponse_stream(
            messages=run_messages.messages,
            response_format=response_format,
            tools=tools,
            tool_choice=self.tool_choice,
            tool_call_limit=self.tool_call_limit,
            per_tool_call_limits=self._build_per_tool_call_limits(tools) if tools else None,
            stream_model_response=stream_model_response,
            run_response=run_response,
            send_media_to_model=self.send_media_to_model,
        )  # type: ignore

        async for model_response_event in model_response_stream:  # type: ignore
            for event in self._handle_model_response_chunk(
                session=session,
                run_response=run_response,
                model_response=model_response,
                model_response_event=model_response_event,
                reasoning_state=reasoning_state,
                parse_structured_output=self.should_parse_structured_output,
                stream_events=stream_events,
            ):
                yield event

        if stream_events and reasoning_state["reasoning_started"]:
            all_reasoning_steps: List[ReasoningStep] = []
            if run_response and run_response.reasoning_steps:
                all_reasoning_steps = cast(List[ReasoningStep], run_response.reasoning_steps)

            if all_reasoning_steps:
                add_reasoning_metrics_to_metadata(
                    run_response=run_response,
                    reasoning_time_taken=reasoning_state["reasoning_time_taken"],
                )
                yield handle_event(  # type: ignore
                    create_reasoning_completed_event(
                        from_run_response=run_response,
                        content=ReasoningSteps(reasoning_steps=all_reasoning_steps),
                        content_type=ReasoningSteps.__name__,
                    ),
                    run_response,
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

        # Update RunOutput
        # Build a list of messages that should be added to the RunOutput
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        # Update the RunOutput messages
        run_response.messages = messages_for_run_response
        # Update the RunOutput metrics
        run_response.metrics = self._calculate_run_metrics(
            messages=messages_for_run_response, current_run_metrics=run_response.metrics
        )

        # Update the run_response audio if streaming
        if model_response.audio is not None:
            run_response.response_audio = model_response.audio

    def _handle_model_response_chunk(
        self,
        session: AgentSession,
        run_response: RunOutput,
        model_response: ModelResponse,
        model_response_event: Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent],
        reasoning_state: Optional[Dict[str, Any]] = None,
        parse_structured_output: bool = False,
        stream_events: bool = False,
    ) -> Iterator[RunOutputEvent]:
        if isinstance(model_response_event, tuple(get_args(RunOutputEvent))) or isinstance(
            model_response_event, tuple(get_args(TeamRunOutputEvent))
        ):
            if model_response_event.event == RunEvent.custom_event:  # type: ignore
                model_response_event.agent_id = self.id  # type: ignore
                model_response_event.agent_name = self.name  # type: ignore
                model_response_event.session_id = session.session_id  # type: ignore
                model_response_event.run_id = run_response.run_id  # type: ignore

            # We just bubble the event up
            yield handle_event(  # type: ignore
                model_response_event,  # type: ignore
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )
        else:
            model_response_event = cast(ModelResponse, model_response_event)
            # If the model response is an assistant_response, yield a RunOutput
            if model_response_event.event == ModelResponseEvent.assistant_response.value:
                content_type = "str"

                # Process content and thinking
                if model_response_event.content is not None:
                    if parse_structured_output:
                        model_response.content = model_response_event.content
                        self._convert_response_to_structured_format(model_response)

                        content_type = self.output_schema.__name__  # type: ignore
                        run_response.content = model_response.content
                        run_response.content_type = content_type
                    else:
                        model_response.content = (model_response.content or "") + model_response_event.content
                        run_response.content = model_response.content
                        run_response.content_type = "str"

                # Process reasoning content
                if model_response_event.reasoning_content is not None:
                    model_response.reasoning_content = (
                        model_response.reasoning_content or ""
                    ) + model_response_event.reasoning_content
                    run_response.reasoning_content = model_response.reasoning_content

                if model_response_event.redacted_reasoning_content is not None:
                    if not model_response.reasoning_content:
                        model_response.reasoning_content = model_response_event.redacted_reasoning_content
                    else:
                        model_response.reasoning_content += model_response_event.redacted_reasoning_content
                    run_response.reasoning_content = model_response.reasoning_content

                # Handle provider data (one chunk)
                if model_response_event.provider_data is not None:
                    run_response.model_provider_data = model_response_event.provider_data

                # Handle citations (one chunk)
                if model_response_event.citations is not None:
                    run_response.citations = model_response_event.citations

                # Only yield if we have content to show
                if content_type != "str":
                    yield handle_event(  # type: ignore
                        create_run_output_content_event(
                            from_run_response=run_response,
                            content=model_response.content,
                            content_type=content_type,
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )
                elif (
                    model_response_event.content is not None
                    or model_response_event.reasoning_content is not None
                    or model_response_event.redacted_reasoning_content is not None
                    or model_response_event.citations is not None
                    or model_response_event.provider_data is not None
                ):
                    yield handle_event(  # type: ignore
                        create_run_output_content_event(
                            from_run_response=run_response,
                            content=model_response_event.content,
                            reasoning_content=model_response_event.reasoning_content,
                            redacted_reasoning_content=model_response_event.redacted_reasoning_content,
                            citations=model_response_event.citations,
                            model_provider_data=model_response_event.provider_data,
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )

                # Process audio
                if model_response_event.audio is not None:
                    if model_response.audio is None:
                        model_response.audio = Audio(id=str(uuid4()), content=b"", transcript="")

                    if model_response_event.audio.id is not None:
                        model_response.audio.id = model_response_event.audio.id  # type: ignore

                    if model_response_event.audio.content is not None:
                        # Handle both base64 string and bytes content
                        if isinstance(model_response_event.audio.content, str):
                            # Decode base64 string to bytes
                            try:
                                import base64

                                decoded_content = base64.b64decode(model_response_event.audio.content)
                                if model_response.audio.content is None:
                                    model_response.audio.content = b""
                                model_response.audio.content += decoded_content
                            except Exception:
                                # If decode fails, encode string as bytes
                                if model_response.audio.content is None:
                                    model_response.audio.content = b""
                                model_response.audio.content += model_response_event.audio.content.encode("utf-8")
                        elif isinstance(model_response_event.audio.content, bytes):
                            # Content is already bytes
                            if model_response.audio.content is None:
                                model_response.audio.content = b""
                            model_response.audio.content += model_response_event.audio.content

                    if model_response_event.audio.transcript is not None:
                        model_response.audio.transcript += model_response_event.audio.transcript  # type: ignore

                    if model_response_event.audio.expires_at is not None:
                        model_response.audio.expires_at = model_response_event.audio.expires_at  # type: ignore
                    if model_response_event.audio.mime_type is not None:
                        model_response.audio.mime_type = model_response_event.audio.mime_type  # type: ignore
                    if model_response_event.audio.sample_rate is not None:
                        model_response.audio.sample_rate = model_response_event.audio.sample_rate
                    if model_response_event.audio.channels is not None:
                        model_response.audio.channels = model_response_event.audio.channels

                    # Yield the audio and transcript bit by bit
                    run_response.response_audio = Audio(
                        id=model_response_event.audio.id,
                        content=model_response_event.audio.content,
                        transcript=model_response_event.audio.transcript,
                        sample_rate=model_response_event.audio.sample_rate,
                        channels=model_response_event.audio.channels,
                    )
                    run_response.created_at = model_response_event.created_at

                    yield handle_event(  # type: ignore
                        create_run_output_content_event(
                            from_run_response=run_response,
                            response_audio=run_response.response_audio,
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )

                if model_response_event.images is not None:
                    yield handle_event(  # type: ignore
                        create_run_output_content_event(
                            from_run_response=run_response,
                            image=model_response_event.images[-1],
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )

                    if model_response.images is None:
                        model_response.images = []
                    model_response.images.extend(model_response_event.images)
                    # Store media in run_response if store_media is enabled
                    if self.store_media:
                        for image in model_response_event.images:
                            self._add_image(image, run_response)

            # Handle tool interruption events
            elif model_response_event.event == ModelResponseEvent.tool_call_paused.value:
                # Add tool calls to the run_response
                tool_executions_list = model_response_event.tool_executions
                if tool_executions_list is not None:
                    # Add tool calls to the agent.run_response
                    if run_response.tools is None:
                        run_response.tools = tool_executions_list
                    else:
                        run_response.tools.extend(tool_executions_list)

            # If the model response is a tool_call_started, add the tool call to the run_response
            elif (
                model_response_event.event == ModelResponseEvent.tool_call_started.value
            ):  # Add tool calls to the run_response
                tool_executions_list = model_response_event.tool_executions
                if tool_executions_list is not None:
                    # Add tool calls to the agent.run_response
                    if run_response.tools is None:
                        run_response.tools = tool_executions_list
                    else:
                        run_response.tools.extend(tool_executions_list)

                    # Yield each tool call started event
                    if stream_events:
                        for tool in tool_executions_list:
                            yield handle_event(  # type: ignore
                                create_tool_call_started_event(from_run_response=run_response, tool=tool),
                                run_response,
                                events_to_skip=self.events_to_skip,  # type: ignore
                                store_events=self.store_events,
                            )

            # If the model response is a tool_call_completed, update the existing tool call in the run_response
            elif model_response_event.event == ModelResponseEvent.tool_call_completed.value:
                if model_response_event.updated_session_state is not None and session.session_data is not None:
                    merge_dictionaries(
                        session.session_data["session_state"],
                        model_response_event.updated_session_state,
                    )

                if model_response_event.images is not None:
                    for image in model_response_event.images:
                        self._add_image(image, run_response)

                if model_response_event.videos is not None:
                    for video in model_response_event.videos:
                        self._add_video(video, run_response)

                if model_response_event.audios is not None:
                    for audio in model_response_event.audios:
                        self._add_audio(audio, run_response)

                reasoning_step: Optional[ReasoningStep] = None

                tool_executions_list = model_response_event.tool_executions
                if tool_executions_list is not None:
                    # Update the existing tool call in the run_response
                    if run_response.tools:
                        # Create a mapping of tool_call_id to index
                        tool_call_index_map = {
                            tc.tool_call_id: i for i, tc in enumerate(run_response.tools) if tc.tool_call_id is not None
                        }
                        # Process tool calls
                        for tool_call_dict in tool_executions_list:
                            tool_call_id = tool_call_dict.tool_call_id or ""
                            index = tool_call_index_map.get(tool_call_id)
                            if index is not None:
                                run_response.tools[index] = tool_call_dict
                    else:
                        run_response.tools = tool_executions_list

                    # Only iterate through new tool calls
                    for tool_call in tool_executions_list:
                        tool_name = tool_call.tool_name or ""
                        if tool_name.lower() in ["think", "analyze"]:
                            tool_args = tool_call.tool_args or {}

                            reasoning_step = self._update_reasoning_content_from_tool_call(
                                run_response=run_response,
                                tool_name=tool_name,
                                tool_args=tool_args,
                            )

                            tool_call_metrics = tool_call.metrics

                            if (
                                tool_call_metrics is not None
                                and tool_call_metrics.duration is not None
                                and reasoning_state is not None
                            ):
                                reasoning_state["reasoning_time_taken"] = reasoning_state[
                                    "reasoning_time_taken"
                                ] + float(tool_call_metrics.duration)

                        if stream_events:
                            yield handle_event(  # type: ignore
                                create_tool_call_completed_event(
                                    from_run_response=run_response, tool=tool_call, content=model_response_event.content
                                ),
                                run_response,
                                events_to_skip=self.events_to_skip,  # type: ignore
                                store_events=self.store_events,
                            )

                if stream_events:
                    if reasoning_step is not None:
                        if reasoning_state and not reasoning_state["reasoning_started"]:
                            yield handle_event(  # type: ignore
                                create_reasoning_started_event(from_run_response=run_response),
                                run_response,
                                events_to_skip=self.events_to_skip,  # type: ignore
                                store_events=self.store_events,
                            )
                            reasoning_state["reasoning_started"] = True

                        yield handle_event(  # type: ignore
                            create_reasoning_step_event(
                                from_run_response=run_response,
                                reasoning_step=reasoning_step,
                                reasoning_content=run_response.reasoning_content or "",
                            ),
                            run_response,
                            events_to_skip=self.events_to_skip,  # type: ignore
                            store_events=self.store_events,
                        )

    def _make_cultural_knowledge(
        self,
        run_messages: RunMessages,
    ):
        if (
            run_messages.user_message is not None
            and self.culture_manager is not None
            and self.update_cultural_knowledge
        ):
            log_debug("Creating cultural knowledge.")
            self.culture_manager.create_cultural_knowledge(message=run_messages.user_message.get_content_string())

    async def _acreate_cultural_knowledge(
        self,
        run_messages: RunMessages,
    ):
        if (
            run_messages.user_message is not None
            and self.culture_manager is not None
            and self.update_cultural_knowledge
        ):
            log_debug("Creating cultural knowledge.")
            await self.culture_manager.acreate_cultural_knowledge(
                message=run_messages.user_message.get_content_string()
            )

    def _make_memories(
        self,
        run_messages: RunMessages,
        user_id: Optional[str] = None,
    ):
        user_message_str = (
            run_messages.user_message.get_content_string() if run_messages.user_message is not None else None
        )
        if user_message_str is not None and user_message_str.strip() != "" and self.memory_manager is not None:
            log_debug("Managing user memories")
            self.memory_manager.create_user_memories(  # type: ignore
                message=user_message_str,
                user_id=user_id,
                agent_id=self.id,
            )

        if run_messages.extra_messages is not None and len(run_messages.extra_messages) > 0:
            parsed_messages = []
            for _im in run_messages.extra_messages:
                if isinstance(_im, Message):
                    parsed_messages.append(_im)
                elif isinstance(_im, dict):
                    try:
                        parsed_messages.append(Message(**_im))
                    except Exception as e:
                        log_warning(f"Failed to validate message during memory update: {e}")
                else:
                    log_warning(f"Unsupported message type: {type(_im)}")
                    continue

            # Filter out messages with empty content before passing to memory manager
            non_empty_messages = [
                msg
                for msg in parsed_messages
                if msg.content and (not isinstance(msg.content, str) or msg.content.strip() != "")
            ]
            if len(non_empty_messages) > 0 and self.memory_manager is not None:
                self.memory_manager.create_user_memories(messages=non_empty_messages, user_id=user_id, agent_id=self.id)  # type: ignore
            else:
                log_warning("Unable to add messages to memory")

    async def _amake_memories(
        self,
        run_messages: RunMessages,
        user_id: Optional[str] = None,
    ):
        user_message_str = (
            run_messages.user_message.get_content_string() if run_messages.user_message is not None else None
        )
        if user_message_str is not None and user_message_str.strip() != "" and self.memory_manager is not None:
            log_debug("Managing user memories")
            await self.memory_manager.acreate_user_memories(  # type: ignore
                message=user_message_str,
                user_id=user_id,
                agent_id=self.id,
            )

        if run_messages.extra_messages is not None and len(run_messages.extra_messages) > 0:
            parsed_messages = []
            for _im in run_messages.extra_messages:
                if isinstance(_im, Message):
                    parsed_messages.append(_im)
                elif isinstance(_im, dict):
                    try:
                        parsed_messages.append(Message(**_im))
                    except Exception as e:
                        log_warning(f"Failed to validate message during memory update: {e}")
                else:
                    log_warning(f"Unsupported message type: {type(_im)}")
                    continue

            # Filter out messages with empty content before passing to memory manager
            non_empty_messages = [
                msg
                for msg in parsed_messages
                if msg.content and (not isinstance(msg.content, str) or msg.content.strip() != "")
            ]
            if len(non_empty_messages) > 0 and self.memory_manager is not None:
                await self.memory_manager.acreate_user_memories(  # type: ignore
                    messages=non_empty_messages, user_id=user_id, agent_id=self.id
                )
            else:
                log_warning("Unable to add messages to memory")

    def _raise_if_async_tools(self) -> None:
        """Raise an exception if any tools contain async functions"""
        if self.tools is None:
            return

        from inspect import iscoroutinefunction

        for tool in self.tools:
            if isinstance(tool, Toolkit):
                for func in tool.functions:
                    if iscoroutinefunction(tool.functions[func].entrypoint):
                        raise Exception(
                            f"Async tool {tool.name} can't be used with synchronous agent.run() or agent.print_response(). "
                            "Use agent.arun() or agent.aprint_response() instead to use this tool."
                        )
            elif isinstance(tool, Function):
                if iscoroutinefunction(tool.entrypoint):
                    raise Exception(
                        f"Async function {tool.name} can't be used with synchronous agent.run() or agent.print_response(). "
                        "Use agent.arun() or agent.aprint_response() instead to use this tool."
                    )
            elif callable(tool):
                if iscoroutinefunction(tool):
                    raise Exception(
                        f"Async function {tool.__name__} can't be used with synchronous agent.run() or agent.print_response(). "
                        "Use agent.arun() or agent.aprint_response() instead to use this tool."
                    )

    def get_tools(
        self,
        run_response: RunOutput,
        session: AgentSession,
        user_id: Optional[str] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Union[Toolkit, Callable, Function, Dict]]:
        agent_tools: List[Union[Toolkit, Callable, Function, Dict]] = []

        # Add provided tools
        if self.tools is not None:
            # If not running in async mode, raise if any tool is async
            self._raise_if_async_tools()
            agent_tools.extend(self.tools)

        # Add tools for accessing memory
        if self.read_chat_history:
            agent_tools.append(self._get_chat_history_function(session=session))
        if self.read_tool_call_history:
            agent_tools.append(self._get_tool_call_history_function(session=session))
        if self.search_session_history:
            agent_tools.append(
                self._get_previous_sessions_messages_function(
                    num_history_sessions=self.num_history_sessions, user_id=user_id
                )
            )

        if self.enable_agentic_memory:
            agent_tools.append(self._get_update_user_memory_function(user_id=user_id, async_mode=False))

        if self.enable_agentic_culture:
            agent_tools.append(self._get_update_cultural_knowledge_function(async_mode=False))

        if self.enable_agentic_state:
            agent_tools.append(Function(name="update_session_state", entrypoint=self._update_session_state_tool))

        # Add tools for accessing knowledge
        if self.knowledge is not None or self.knowledge_retriever is not None:
            # Check if knowledge retriever is an async function but used in sync mode
            from inspect import iscoroutinefunction

            if self.knowledge_retriever and iscoroutinefunction(self.knowledge_retriever):
                log_warning(
                    "Async knowledge retriever function is being used with synchronous agent.run() or agent.print_response(). "
                    "It is recommended to use agent.arun() or agent.aprint_response() instead."
                )

            if self.search_knowledge:
                # Use async or sync search based on async_mode
                if self.enable_agentic_knowledge_filters:
                    agent_tools.append(
                        self._search_knowledge_base_with_agentic_filters_function(
                            run_response=run_response,
                            async_mode=False,
                            knowledge_filters=knowledge_filters,
                        )
                    )
                else:
                    agent_tools.append(
                        self._get_search_knowledge_base_function(
                            run_response=run_response,
                            async_mode=False,
                            knowledge_filters=knowledge_filters,
                        )
                    )

            if self.update_knowledge:
                agent_tools.append(self.add_to_knowledge)

        return agent_tools

    async def aget_tools(
        self,
        run_response: RunOutput,
        session: AgentSession,
        user_id: Optional[str] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        check_mcp_tools: bool = True,
    ) -> List[Union[Toolkit, Callable, Function, Dict]]:
        agent_tools: List[Union[Toolkit, Callable, Function, Dict]] = []

        # Connect MCP tools
        await self._connect_mcp_tools()

        # Add provided tools
        if self.tools is not None:
            for tool in self.tools:
                if tool.__class__.__name__ in ["MCPTools", "MultiMCPTools"]:
                    if tool.refresh_connection:  # type: ignore
                        try:
                            is_alive = await tool.is_alive()  # type: ignore
                            if not is_alive:
                                await tool.connect(force=True)  # type: ignore
                        except (RuntimeError, BaseException) as e:
                            log_warning(f"Failed to check if MCP tool is alive or to connect to it: {e}")
                            continue

                        try:
                            await tool.build_tools()  # type: ignore
                        except (RuntimeError, BaseException) as e:
                            log_warning(f"Failed to build tools for {str(tool)}: {e}")
                            continue

                    # Only add the tool if it successfully connected and built its tools
                    if check_mcp_tools and not tool.initialized:  # type: ignore
                        continue

                    agent_tools.append(tool)
                else:
                    agent_tools.append(tool)

        # Add tools for accessing memory
        if self.read_chat_history:
            agent_tools.append(self._get_chat_history_function(session=session))
        if self.read_tool_call_history:
            agent_tools.append(self._get_tool_call_history_function(session=session))
        if self.search_session_history:
            agent_tools.append(
                await self._aget_previous_sessions_messages_function(
                    num_history_sessions=self.num_history_sessions, user_id=user_id
                )
            )

        if self.enable_agentic_memory:
            agent_tools.append(self._get_update_user_memory_function(user_id=user_id, async_mode=True))

        if self.enable_agentic_state:
            agent_tools.append(Function(name="update_session_state", entrypoint=self._update_session_state_tool))

        # Add tools for accessing knowledge
        if self.knowledge is not None or self.knowledge_retriever is not None:
            if self.search_knowledge:
                # Use async or sync search based on async_mode
                if self.enable_agentic_knowledge_filters:
                    agent_tools.append(
                        self._search_knowledge_base_with_agentic_filters_function(
                            run_response=run_response,
                            async_mode=True,
                            knowledge_filters=knowledge_filters,
                        )
                    )
                else:
                    agent_tools.append(
                        self._get_search_knowledge_base_function(
                            run_response=run_response,
                            async_mode=True,
                            knowledge_filters=knowledge_filters,
                        )
                    )

            if self.update_knowledge:
                agent_tools.append(self.add_to_knowledge)

        return agent_tools

    def _determine_tools_for_model(
        self,
        model: Model,
        processed_tools: List[Union[Toolkit, Callable, Function, Dict]],
        run_response: RunOutput,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
    ) -> List[Union[Function, dict]]:
        _function_names = []
        _functions: List[Union[Function, dict]] = []
        self._tool_instructions = []

        # Get Agent tools
        if processed_tools is not None and len(processed_tools) > 0:
            log_debug("Processing tools for model")

            # Check if we need strict mode for the functions for the model
            strict = False
            if (
                self.output_schema is not None
                and (self.structured_outputs or (not self.use_json_mode))
                and model.supports_native_structured_outputs
            ):
                strict = True

            for tool in processed_tools:
                if isinstance(tool, Dict):
                    # If a dict is passed, it is a builtin tool
                    # that is run by the model provider and not the Agent
                    _functions.append(tool)
                    log_debug(f"Included builtin tool {tool}")

                elif isinstance(tool, Toolkit):
                    # For each function in the toolkit and process entrypoint
                    for name, _func in tool.functions.items():
                        if name in _function_names:
                            continue
                        _function_names.append(name)
                        _func = _func.model_copy(deep=True)
                        _func._agent = self
                        _func.process_entrypoint(strict=strict)
                        if strict and _func.strict is None:
                            _func.strict = True
                        if self.tool_hooks is not None:
                            _func.tool_hooks = self.tool_hooks
                        _functions.append(_func)
                        log_debug(f"Added tool {name} from {tool.name}")

                    # Add instructions from the toolkit
                    if tool.add_instructions and tool.instructions is not None:
                        self._tool_instructions.append(tool.instructions)

                elif isinstance(tool, Function):
                    if tool.name in _function_names:
                        continue
                    _function_names.append(tool.name)

                    tool.process_entrypoint(strict=strict)
                    tool = tool.model_copy(deep=True)

                    tool._agent = self
                    if strict and tool.strict is None:
                        tool.strict = True
                    if self.tool_hooks is not None:
                        tool.tool_hooks = self.tool_hooks
                    _functions.append(tool)
                    log_debug(f"Added tool {tool.name}")

                    # Add instructions from the Function
                    if tool.add_instructions and tool.instructions is not None:
                        self._tool_instructions.append(tool.instructions)

                elif callable(tool):
                    try:
                        function_name = tool.__name__

                        if function_name in _function_names:
                            continue
                        _function_names.append(function_name)

                        _func = Function.from_callable(tool, strict=strict)
                        _func = _func.model_copy(deep=True)
                        _func._agent = self
                        if strict:
                            _func.strict = True
                        if self.tool_hooks is not None:
                            _func.tool_hooks = self.tool_hooks
                        _functions.append(_func)
                        log_debug(f"Added tool {_func.name}")
                    except Exception as e:
                        log_warning(f"Could not add tool {tool}: {e}")

        # Update the session state for the functions
        if _functions:
            from inspect import signature

            # Check if any functions need media before collecting
            needs_media = any(
                any(param in signature(func.entrypoint).parameters for param in ["images", "videos", "audios", "files"])
                for func in _functions
                if isinstance(func, Function) and func.entrypoint is not None
            )

            # Only collect media if functions actually need them
            joint_images = collect_joint_images(run_response.input, session) if needs_media else None
            joint_files = collect_joint_files(run_response.input) if needs_media else None
            joint_audios = collect_joint_audios(run_response.input, session) if needs_media else None
            joint_videos = collect_joint_videos(run_response.input, session) if needs_media else None

            for func in _functions:  # type: ignore
                if isinstance(func, Function):
                    func._session_state = session_state
                    func._dependencies = dependencies
                    func._images = joint_images
                    func._files = joint_files
                    func._audios = joint_audios
                    func._videos = joint_videos

        return _functions

    def _build_per_tool_call_limits(self, tools: List[Union[Function, dict]]) -> Dict[str, int]:
        tool_limits: Dict[str, int] = {}

        for tool in tools:
            if isinstance(tool, Function) and tool.call_limit is not None:
                tool_limits[tool.name] = tool.call_limit

        return tool_limits

    def _model_should_return_structured_output(self):
        self.model = cast(Model, self.model)
        return bool(
            self.model.supports_native_structured_outputs
            and self.output_schema is not None
            and (not self.use_json_mode or self.structured_outputs)
        )

    def _get_response_format(self, model: Optional[Model] = None) -> Optional[Union[Dict, Type[BaseModel]]]:
        model = cast(Model, model or self.model)
        if self.output_schema is None:
            return None
        else:
            json_response_format = {"type": "json_object"}

            if model.supports_native_structured_outputs:
                if not self.use_json_mode or self.structured_outputs:
                    log_debug("Setting Model.response_format to Agent.output_schema")
                    return self.output_schema
                else:
                    log_debug(
                        "Model supports native structured outputs but it is not enabled. Using JSON mode instead."
                    )
                    return json_response_format

            elif model.supports_json_schema_outputs:
                if self.use_json_mode or (not self.structured_outputs):
                    log_debug("Setting Model.response_format to JSON response mode")
                    return {
                        "type": "json_schema",
                        "json_schema": {
                            "name": self.output_schema.__name__,
                            "schema": self.output_schema.model_json_schema(),
                        },
                    }
                else:
                    return None

            else:
                log_debug("Model does not support structured or JSON schema outputs.")
                return json_response_format

    def _resolve_run_dependencies(self, dependencies: Dict[str, Any]) -> None:
        from inspect import iscoroutine, iscoroutinefunction, signature

        # Dependencies should already be resolved in run() method
        log_debug("Resolving dependencies")
        if not isinstance(dependencies, dict):
            log_warning("Dependencies is not a dict")
            return

        for key, value in dependencies.items():
            if iscoroutine(value) or iscoroutinefunction(value):
                log_warning(f"Dependency {key} is a coroutine. Use agent.arun() or agent.aprint_response() instead.")
                continue
            elif callable(value):
                try:
                    sig = signature(value)
                    result = value(agent=self) if "agent" in sig.parameters else value()
                    if result is not None:
                        dependencies[key] = result
                except Exception as e:
                    log_warning(f"Failed to resolve dependencies for '{key}': {e}")
            else:
                dependencies[key] = value

    async def _aresolve_run_dependencies(self, dependencies: Dict[str, Any]) -> None:
        from inspect import iscoroutine, iscoroutinefunction, signature

        log_debug("Resolving context (async)")
        if not isinstance(dependencies, dict):
            log_warning("Context is not a dict")
            return

        for key, value in dependencies.items():
            if not callable(value):
                dependencies[key] = value
                continue
            try:
                sig = signature(value)
                result = value(agent=self) if "agent" in sig.parameters else value()

                if iscoroutine(result) or iscoroutinefunction(result):
                    result = await result
                dependencies[key] = result
            except Exception as e:
                log_warning(f"Failed to resolve context for '{key}': {e}")

    def _get_agent_data(self) -> Dict[str, Any]:
        agent_data: Dict[str, Any] = {}
        if self.name is not None:
            agent_data["name"] = self.name
        if self.id is not None:
            agent_data["agent_id"] = self.id
        if self.model is not None:
            agent_data["model"] = self.model.to_dict()
        return agent_data

    @staticmethod
    def cancel_run(run_id: str) -> bool:
        """Cancel a running agent execution.

        Args:
            run_id (str): The run_id to cancel.

        Returns:
            bool: True if the run was found and marked for cancellation, False otherwise.
        """
        return cancel_run_global(run_id)

    # -*- Session Database Functions
    def _read_session(
        self, session_id: str, session_type: SessionType = SessionType.AGENT
    ) -> Optional[Union[AgentSession, TeamSession, WorkflowSession]]:
        """Get a Session from the database."""
        try:
            if not self.db:
                raise ValueError("Db not initialized")
            return self.db.get_session(session_id=session_id, session_type=session_type)  # type: ignore
        except Exception as e:
            log_warning(f"Error getting session from db: {e}")
            return None

    async def _aread_session(
        self, session_id: str, session_type: SessionType = SessionType.AGENT
    ) -> Optional[Union[AgentSession, TeamSession, WorkflowSession]]:
        """Get a Session from the database."""
        try:
            if not self.db:
                raise ValueError("Db not initialized")
            return await self.db.get_session(session_id=session_id, session_type=SessionType.AGENT)  # type: ignore
        except Exception as e:
            log_warning(f"Error getting session from db: {e}")
            return None

    def _upsert_session(self, session: AgentSession) -> Optional[AgentSession]:
        """Upsert a Session into the database."""

        try:
            if not self.db:
                raise ValueError("Db not initialized")
            return self.db.upsert_session(session=session)  # type: ignore
        except Exception as e:
            log_warning(f"Error upserting session into db: {e}")
            return None

    async def _aupsert_session(self, session: AgentSession) -> Optional[AgentSession]:
        """Upsert a Session into the database."""
        try:
            if not self.db:
                raise ValueError("Db not initialized")
            return await self.db.upsert_session(session=session)  # type: ignore
        except Exception as e:
            log_warning(f"Error upserting session into db: {e}")
            return None

    def _load_session_state(self, session: AgentSession, session_state: Dict[str, Any]):
        """Load and return the stored session_state from the database, optionally merging it with the given one"""

        # Get the session_state from the database and merge with proper precedence
        # At this point session_state contains: agent_defaults + run_params
        if session.session_data is not None and "session_state" in session.session_data:
            session_state_from_db = session.session_data.get("session_state")

            if (
                session_state_from_db is not None
                and isinstance(session_state_from_db, dict)
                and len(session_state_from_db) > 0
                and not self.overwrite_db_session_state
            ):
                # This preserves precedence: run_params > db_state > agent_defaults
                merged_state = session_state_from_db.copy()
                merge_dictionaries(merged_state, session_state)
                session_state.clear()
                session_state.update(merged_state)

        # Update the session_state in the session
        if session.session_data is not None:
            session.session_data["session_state"] = session_state

        return session_state

    def _update_metadata(self, session: AgentSession):
        """Update the extra_data in the session"""
        # Read metadata from the database
        if session.metadata is not None:
            # If metadata is set in the agent, update the database metadata with the agent's metadata
            if self.metadata is not None:
                # Updates agent's session metadata in place
                merge_dictionaries(session.metadata, self.metadata)
            # Update the current metadata with the metadata from the database which is updated in place
            self.metadata = session.metadata

    def _get_session_metrics(self, session: AgentSession):
        # Get the session_metrics from the database
        if session.session_data is not None and "session_metrics" in session.session_data:
            session_metrics_from_db = session.session_data.get("session_metrics")
            if session_metrics_from_db is not None:
                if isinstance(session_metrics_from_db, dict):
                    return Metrics(**session_metrics_from_db)
                elif isinstance(session_metrics_from_db, Metrics):
                    return session_metrics_from_db
        else:
            return Metrics()

    def _read_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> AgentSession:
        from time import time

        # Returning cached session if we have one
        if self._cached_session is not None and self._cached_session.session_id == session_id:
            return self._cached_session

        # Try to load from database
        agent_session = None
        if self.db is not None and self.team_id is None and self.workflow_id is None:
            log_debug(f"Reading AgentSession: {session_id}")

            agent_session = cast(AgentSession, self._read_session(session_id=session_id))

        if agent_session is None:
            # Creating new session if none found
            log_debug(f"Creating new AgentSession: {session_id}")
            session_data = {}
            if self.session_state is not None:
                from copy import deepcopy

                session_data["session_state"] = deepcopy(self.session_state)
            agent_session = AgentSession(
                session_id=session_id,
                agent_id=self.id,
                user_id=user_id,
                agent_data=self._get_agent_data(),
                session_data=session_data,
                metadata=self.metadata,
                created_at=int(time()),
            )

        if self.cache_session:
            self._cached_session = agent_session

        return agent_session

    async def _aread_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> AgentSession:
        from time import time

        # Returning cached session if we have one
        if self._cached_session is not None and self._cached_session.session_id == session_id:
            return self._cached_session

        # Try to load from database
        agent_session = None
        if self.db is not None and self.team_id is None and self.workflow_id is None:
            log_debug(f"Reading AgentSession: {session_id}")
            if self._has_async_db():
                agent_session = cast(AgentSession, await self._aread_session(session_id=session_id))
            else:
                agent_session = cast(AgentSession, self._read_session(session_id=session_id))

        if agent_session is None:
            # Creating new session if none found
            log_debug(f"Creating new AgentSession: {session_id}")
            session_data = {}
            if self.session_state is not None:
                from copy import deepcopy

                session_data["session_state"] = deepcopy(self.session_state)
            agent_session = AgentSession(
                session_id=session_id,
                agent_id=self.id,
                user_id=user_id,
                agent_data=self._get_agent_data(),
                session_data=session_data,
                metadata=self.metadata,
                created_at=int(time()),
            )

        if self.cache_session:
            self._cached_session = agent_session

        return agent_session

    # -*- Public Convenience Functions
    def get_run_output(self, run_id: str, session_id: Optional[str] = None) -> Optional[RunOutput]:
        """
        Get a RunOutput from the database.

        Args:
            run_id (str): The run_id to load from storage.
            session_id (Optional[str]): The session_id to load from storage.
        Returns:
            Optional[RunOutput]: The RunOutput from the database or None if not found.
        """
        if not session_id and not self.session_id:
            raise Exception("No session_id provided")

        session_id_to_load = session_id or self.session_id
        return cast(RunOutput, get_run_output_util(self, run_id=run_id, session_id=session_id_to_load))

    async def aget_run_output(self, run_id: str, session_id: Optional[str] = None) -> Optional[RunOutput]:
        """
        Get a RunOutput from the database.

        Args:
            run_id (str): The run_id to load from storage.
            session_id (Optional[str]): The session_id to load from storage.
        Returns:
            Optional[RunOutput]: The RunOutput from the database or None if not found.
        """
        if not session_id and not self.session_id:
            raise Exception("No session_id provided")

        session_id_to_load = session_id or self.session_id
        return cast(RunOutput, await aget_run_output_util(self, run_id=run_id, session_id=session_id_to_load))

    def get_last_run_output(self, session_id: Optional[str] = None) -> Optional[RunOutput]:
        """
        Get the last run response from the database.

        Args:
            session_id (Optional[str]): The session_id to load from storage.

        Returns:
            Optional[RunOutput]: The last run response from the database or None if not found.
        """
        if not session_id and not self.session_id:
            raise Exception("No session_id provided")

        session_id_to_load = session_id or self.session_id
        return cast(RunOutput, get_last_run_output_util(self, session_id=session_id_to_load))

    async def aget_last_run_output(self, session_id: Optional[str] = None) -> Optional[RunOutput]:
        """
        Get the last run response from the database.

        Args:
            session_id (Optional[str]): The session_id to load from storage.

        Returns:
            Optional[RunOutput]: The last run response from the database or None if not found.
        """
        if not session_id and not self.session_id:
            raise Exception("No session_id provided")

        session_id_to_load = session_id or self.session_id
        return cast(RunOutput, await aget_last_run_output_util(self, session_id=session_id_to_load))

    def get_session(
        self,
        session_id: Optional[str] = None,
    ) -> Optional[AgentSession]:
        """Load an AgentSession from database or cache.

        Args:
            session_id: The session_id to load from storage.

        Returns:
            AgentSession: The AgentSession loaded from the database/cache or None if not found.
        """
        if not session_id and not self.session_id:
            raise Exception("No session_id provided")

        session_id_to_load = session_id or self.session_id

        # If there is a cached session, return it
        if self.cache_session and hasattr(self, "_cached_session") and self._cached_session is not None:
            if self._cached_session.session_id == session_id_to_load:
                return self._cached_session

        if self._has_async_db():
            raise ValueError("Async database not supported for get_session")

        # Load and return the session from the database
        if self.db is not None:
            loaded_session = None

            # We have a standalone agent, so we are loading an AgentSession
            if self.team_id is None and self.workflow_id is None:
                loaded_session = cast(
                    AgentSession,
                    self._read_session(session_id=session_id_to_load, session_type=SessionType.AGENT),  # type: ignore
                )

            # We have a team member agent, so we are loading a TeamSession
            if loaded_session is None and self.team_id is not None:
                # Load session for team member agents
                loaded_session = cast(
                    TeamSession,
                    self._read_session(session_id=session_id_to_load, session_type=SessionType.TEAM),  # type: ignore
                )

            # We have a workflow member agent, so we are loading a WorkflowSession
            if loaded_session is None and self.workflow_id is not None:
                # Load session for workflow memberagents
                loaded_session = cast(
                    WorkflowSession,
                    self._read_session(session_id=session_id_to_load, session_type=SessionType.WORKFLOW),  # type: ignore
                )

            # Cache the session if relevant
            if loaded_session is not None and self.cache_session:
                self._cached_session = loaded_session

            return loaded_session

        log_debug(f"Session {session_id_to_load} not found in db")
        return None

    async def aget_session(
        self,
        session_id: Optional[str] = None,
    ) -> Optional[AgentSession]:
        """Load an AgentSession from database or cache.

        Args:
            session_id: The session_id to load from storage.

        Returns:
            AgentSession: The AgentSession loaded from the database/cache or None if not found.
        """
        if not session_id and not self.session_id:
            raise Exception("No session_id provided")

        session_id_to_load = session_id or self.session_id

        # If there is a cached session, return it
        if self.cache_session and hasattr(self, "_cached_session") and self._cached_session is not None:
            if self._cached_session.session_id == session_id_to_load:
                return self._cached_session

        # Load and return the session from the database
        if self.db is not None:
            loaded_session = None

            # We have a standalone agent, so we are loading an AgentSession
            if self.team_id is None and self.workflow_id is None:
                loaded_session = cast(
                    AgentSession,
                    await self._aread_session(session_id=session_id_to_load, session_type=SessionType.AGENT),  # type: ignore
                )

            # We have a team member agent, so we are loading a TeamSession
            if loaded_session is None and self.team_id is not None:
                # Load session for team member agents
                loaded_session = cast(
                    TeamSession,
                    await self._aread_session(session_id=session_id_to_load, session_type=SessionType.TEAM),  # type: ignore
                )

            # We have a workflow member agent, so we are loading a WorkflowSession
            if loaded_session is None and self.workflow_id is not None:
                # Load session for workflow memberagents
                loaded_session = cast(
                    WorkflowSession,
                    await self._aread_session(session_id=session_id_to_load, session_type=SessionType.WORKFLOW),  # type: ignore
                )

            # Cache the session if relevant
            if loaded_session is not None and self.cache_session:
                self._cached_session = loaded_session

            return loaded_session

        log_debug(f"AgentSession {session_id_to_load} not found in db")
        return None

    def save_session(self, session: AgentSession) -> None:
        """
        Save the AgentSession to storage
        """
        if self._has_async_db():
            raise ValueError("Async database not supported for save_session")

        # If the agent is a member of a team, do not save the session to the database
        if (
            self.db is not None
            and self.team_id is None
            and self.workflow_id is None
            and session.session_data is not None
        ):
            if session.session_data is not None and "session_state" in session.session_data:
                session.session_data["session_state"].pop("current_session_id", None)
                session.session_data["session_state"].pop("current_user_id", None)
                session.session_data["session_state"].pop("current_run_id", None)

            self._upsert_session(session=session)
            log_debug(f"Created or updated AgentSession record: {session.session_id}")

    async def asave_session(self, session: AgentSession) -> None:
        """
        Save the AgentSession to storage
        """
        # If the agent is a member of a team, do not save the session to the database
        if (
            self.db is not None
            and self.team_id is None
            and self.workflow_id is None
            and session.session_data is not None
        ):
            if session.session_data is not None and "session_state" in session.session_data:
                session.session_data["session_state"].pop("current_session_id", None)
                session.session_data["session_state"].pop("current_user_id", None)
                session.session_data["session_state"].pop("current_run_id", None)
            if self._has_async_db():
                await self._aupsert_session(session=session)
            else:
                self._upsert_session(session=session)
            log_debug(f"Created or updated AgentSession record: {session.session_id}")

    def get_chat_history(self, session_id: Optional[str] = None) -> List[Message]:
        """Read the chat history from the session

        Args:
            session_id: The session ID to get the chat history for. If not provided, the current cached session ID is used.
        Returns:
            List[Message]: The chat history from the session.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            log_warning("Session ID is not set, cannot get chat history")
            return []

        return get_chat_history_util(self, session_id=session_id)

    async def aget_chat_history(self, session_id: Optional[str] = None) -> List[Message]:
        """Read the chat history from the session

        Args:
            session_id: The session ID to get the chat history for. If not provided, the current cached session ID is used.
        Returns:
            List[Message]: The chat history from the session.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            log_warning("Session ID is not set, cannot get chat history")
            return []

        return await aget_chat_history_util(self, session_id=session_id)

    # -*- Session Management Functions
    def rename(self, name: str, session_id: Optional[str] = None) -> None:
        """
        Rename the Agent and save to storage

        Args:
            name (str): The new name for the Agent.
            session_id (Optional[str]): The session_id of the session where to store the new name. If not provided, the current cached session ID is used.
        """

        session_id = session_id or self.session_id

        if session_id is None:
            raise Exception("Session ID is not set")

        if self._has_async_db():
            import asyncio

            session = asyncio.run(self.aget_session(session_id=session_id))
        else:
            session = self.get_session(session_id=session_id)

        if session is None:
            raise Exception("Session not found")

        # -*- Rename Agent
        self.name = name
        if session.agent_data is not None:
            session.agent_data["name"] = name
        else:
            session.agent_data = {"name": name}

        # -*- Save to storage
        if self._has_async_db():
            import asyncio

            asyncio.run(self.asave_session(session=session))
        else:
            self.save_session(session=session)

    def set_session_name(
        self,
        session_id: Optional[str] = None,
        autogenerate: bool = False,
        session_name: Optional[str] = None,
    ) -> AgentSession:
        """
        Set the session name and save to storage

        Args:
            session_id: The session ID to set the name for. If not provided, the current cached session ID is used.
            autogenerate: Whether to autogenerate the session name.
            session_name: The session name to set. If not provided, the session name will be autogenerated.
        Returns:
            AgentSession: The updated session.
        """
        session_id = session_id or self.session_id

        if session_id is None:
            raise Exception("Session ID is not set")

        return cast(
            AgentSession,
            set_session_name_util(self, session_id=session_id, autogenerate=autogenerate, session_name=session_name),
        )

    async def aset_session_name(
        self,
        session_id: Optional[str] = None,
        autogenerate: bool = False,
        session_name: Optional[str] = None,
    ) -> AgentSession:
        """
        Set the session name and save to storage

        Args:
            session_id: The session ID to set the name for. If not provided, the current cached session ID is used.
            autogenerate: Whether to autogenerate the session name.
            session_name: The session name to set. If not provided, the session name will be autogenerated.
        Returns:
            AgentSession: The updated session.
        """
        session_id = session_id or self.session_id

        if session_id is None:
            raise Exception("Session ID is not set")

        return cast(
            AgentSession,
            await aset_session_name_util(
                self, session_id=session_id, autogenerate=autogenerate, session_name=session_name
            ),
        )

    def generate_session_name(self, session: AgentSession) -> str:
        """
        Generate a name for the session using the first 6 messages from the memory

        Args:
            session (AgentSession): The session to generate a name for.
        Returns:
            str: The generated session name.
        """

        if self.model is None:
            raise Exception("Model not set")

        gen_session_name_prompt = "Conversation\n"

        messages_for_generating_session_name = session.get_messages_for_session()

        for message in messages_for_generating_session_name:
            gen_session_name_prompt += f"{message.role.upper()}: {message.content}\n"

        gen_session_name_prompt += "\n\nConversation Name: "

        system_message = Message(
            role=self.system_message_role,
            content="Please provide a suitable name for this conversation in maximum 5 words. "
            "Remember, do not exceed 5 words.",
        )
        user_message = Message(role=self.user_message_role, content=gen_session_name_prompt)
        generate_name_messages = [system_message, user_message]

        # Generate name
        generated_name = self.model.response(messages=generate_name_messages)
        content = generated_name.content
        if content is None:
            log_error("Generated name is None. Trying again.")
            return self.generate_session_name(session=session)

        if len(content.split()) > 5:
            log_error("Generated name is too long. It should be less than 5 words. Trying again.")
            return self.generate_session_name(session=session)
        return content.replace('"', "").strip()

    def get_session_name(self, session_id: Optional[str] = None) -> str:
        """
        Get the session name for the given session ID.

        Args:
            session_id: The session ID to get the name for. If not provided, the current cached session ID is used.
        Returns:
            str: The session name.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            raise Exception("Session ID is not set")
        return get_session_name_util(self, session_id=session_id)

    async def aget_session_name(self, session_id: Optional[str] = None) -> str:
        """
        Get the session name for the given session ID.

        Args:
            session_id: The session ID to get the name for. If not provided, the current cached session ID is used.
        Returns:
            str: The session name.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            raise Exception("Session ID is not set")
        return await aget_session_name_util(self, session_id=session_id)

    def get_session_state(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the session state for the given session ID.

        Args:
            session_id: The session ID to get the state for. If not provided, the current cached session ID is used.
        Returns:
            Dict[str, Any]: The session state.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            raise Exception("Session ID is not set")
        return get_session_state_util(self, session_id=session_id)

    async def aget_session_state(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the session state for the given session ID.

        Args:
            session_id: The session ID to get the state for. If not provided, the current cached session ID is used.
        Returns:
            Dict[str, Any]: The session state.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            raise Exception("Session ID is not set")
        return await aget_session_state_util(self, session_id=session_id)

    def update_session_state(self, session_state_updates: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """
        Update the session state for the given session ID and user ID.
        Args:
            session_state_updates: The updates to apply to the session state. Should be a dictionary of key-value pairs.
            session_id: The session ID to update. If not provided, the current cached session ID is used.
        Returns:
            dict: The updated session state.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            raise Exception("Session ID is not set")
        return update_session_state_util(self, session_state_updates=session_state_updates, session_id=session_id)

    async def aupdate_session_state(
        self, session_state_updates: Dict[str, Any], session_id: Optional[str] = None
    ) -> str:
        """
        Update the session state for the given session ID and user ID.
        Args:
            session_state_updates: The updates to apply to the session state. Should be a dictionary of key-value pairs.
            session_id: The session ID to update. If not provided, the current cached session ID is used.
        Returns:
            dict: The updated session state.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            raise Exception("Session ID is not set")
        return await aupdate_session_state_util(
            self, session_state_updates=session_state_updates, session_id=session_id
        )

    def get_session_metrics(self, session_id: Optional[str] = None) -> Optional[Metrics]:
        """Get the session metrics for the given session ID.

        Args:
            session_id: The session ID to get the metrics for. If not provided, the current cached session ID is used.
        Returns:
            Optional[Metrics]: The session metrics.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            raise Exception("Session ID is not set")

        return get_session_metrics_util(self, session_id=session_id)

    async def aget_session_metrics(self, session_id: Optional[str] = None) -> Optional[Metrics]:
        """Get the session metrics for the given session ID.

        Args:
            session_id: The session ID to get the metrics for. If not provided, the current cached session ID is used.
        Returns:
            Optional[Metrics]: The session metrics.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            raise Exception("Session ID is not set")

        return await aget_session_metrics_util(self, session_id=session_id)

    def delete_session(self, session_id: str):
        """Delete the current session and save to storage"""
        if self.db is None:
            return

        self.db.delete_session(session_id=session_id)

    async def adelete_session(self, session_id: str):
        """Delete the current session and save to storage"""
        if self.db is None:
            return
        await self.db.delete_session(session_id=session_id)  # type: ignore

    def get_messages_for_session(self, session_id: Optional[str] = None) -> List[Message]:
        """Get messages for a session

        Args:
            session_id: The session ID to get the messages for. If not provided, the current cached session ID is used.
        Returns:
            List[Message]: The messages for the session.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            log_warning("Session ID is not set, cannot get messages for session")
            return []

        session = self.get_session(session_id=session_id)

        if session is None:
            raise Exception("Session not found")

        # Only filter by agent_id if this is part of a team
        return session.get_messages_from_last_n_runs(
            agent_id=self.id if self.team_id is not None else None,
        )

    async def aget_messages_for_session(self, session_id: Optional[str] = None) -> List[Message]:
        """Get messages for a session

        Args:
            session_id: The session ID to get the messages for. If not provided, the current cached session ID is used.
        Returns:
            List[Message]: The messages for the session.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            log_warning("Session ID is not set, cannot get messages for session")
            return []

        session = await self.aget_session(session_id=session_id)

        if session is None:
            raise Exception("Session not found")

        # Only filter by agent_id if this is part of a team
        return session.get_messages_from_last_n_runs(
            agent_id=self.id if self.team_id is not None else None,
        )

    def get_session_summary(self, session_id: Optional[str] = None) -> Optional[SessionSummary]:
        """Get the session summary for the given session ID and user ID

        Args:
            session_id: The session ID to get the summary for. If not provided, the current cached session ID is used.
        Returns:
            SessionSummary: The session summary.
        """
        session_id = session_id if session_id is not None else self.session_id
        if session_id is None:
            raise ValueError("Session ID is required")

        session = self.get_session(session_id=session_id)

        if session is None:
            raise Exception(f"Session {session_id} not found")

        return session.get_session_summary()

    async def aget_session_summary(self, session_id: Optional[str] = None) -> Optional[SessionSummary]:
        """Get the session summary for the given session ID and user ID.

        Args:
            session_id: The session ID to get the summary for. If not provided, the current cached session ID is used.
        Returns:
            SessionSummary: The session summary.
        """
        session_id = session_id if session_id is not None else self.session_id
        if session_id is None:
            raise ValueError("Session ID is required")

        session = await self.aget_session(session_id=session_id)

        if session is None:
            raise Exception(f"Session {session_id} not found")

        return session.get_session_summary()

    def get_user_memories(self, user_id: Optional[str] = None) -> Optional[List[UserMemory]]:
        """Get the user memories for the given user ID.

        Args:
            user_id: The user ID to get the memories for. If not provided, the current cached user ID is used.
        Returns:
            Optional[List[UserMemory]]: The user memories.
        """
        if self.memory_manager is None:
            return None
        user_id = user_id if user_id is not None else self.user_id
        if user_id is None:
            user_id = "default"

        return self.memory_manager.get_user_memories(user_id=user_id)

    async def aget_user_memories(self, user_id: Optional[str] = None) -> Optional[List[UserMemory]]:
        """Get the user memories for the given user ID.

        Args:
            user_id: The user ID to get the memories for. If not provided, the current cached user ID is used.
        Returns:
            Optional[List[UserMemory]]: The user memories.
        """
        if self.memory_manager is None:
            return None
        user_id = user_id if user_id is not None else self.user_id
        if user_id is None:
            user_id = "default"

        return await self.memory_manager.aget_user_memories(user_id=user_id)

    def get_culture_knowledge(self) -> Optional[List[CulturalKnowledge]]:
        """Get the cultural knowledge the agent has access to

        Returns:
            Optional[List[CulturalKnowledge]]: The cultural knowledge.
        """
        if self.culture_manager is None:
            return None

        return self.culture_manager.get_all_knowledge()

    async def aget_culture_knowledge(self) -> Optional[List[CulturalKnowledge]]:
        """Get the cultural knowledge the agent has access to

        Returns:
            Optional[List[CulturalKnowledge]]: The cultural knowledge.
        """
        if self.culture_manager is None:
            return None

        return await self.culture_manager.aget_all_knowledge()

    # -*- System & User Message Functions
    def _format_message_with_state_variables(
        self,
        message: Any,
        user_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Format a message with the session state variables."""
        import re
        import string
        from copy import deepcopy

        if not isinstance(message, str):
            return message

        # Should already be resolved and passed from run() method
        format_variables = ChainMap(
            session_state or {},
            dependencies or {},
            metadata or {},
            {"user_id": user_id} if user_id is not None else {},
        )
        converted_msg = deepcopy(message)
        for var_name in format_variables.keys():
            # Only convert standalone {var_name} patterns, not nested ones
            pattern = r"\{" + re.escape(var_name) + r"\}"
            replacement = "${" + var_name + "}"
            converted_msg = re.sub(pattern, replacement, converted_msg)

        # Use Template to safely substitute variables
        template = string.Template(converted_msg)
        try:
            result = template.safe_substitute(format_variables)
            return result
        except Exception as e:
            log_warning(f"Template substitution failed: {e}")
            return message

    def get_system_message(
        self,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        tools: Optional[List[Union[Function, dict]]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        add_session_state_to_context: Optional[bool] = None,
    ) -> Optional[Message]:
        """Return the system message for the Agent.

        1. If the system_message is provided, use that.
        2. If build_context is False, return None.
        3. Build and return the default system message for the Agent.
        """

        # 1. If the system_message is provided, use that.
        if self.system_message is not None:
            if isinstance(self.system_message, Message):
                return self.system_message

            sys_message_content: str = ""
            if isinstance(self.system_message, str):
                sys_message_content = self.system_message
            elif callable(self.system_message):
                sys_message_content = self.system_message(agent=self)
                if not isinstance(sys_message_content, str):
                    raise Exception("system_message must return a string")

            # Format the system message with the session state variables
            if self.resolve_in_context:
                sys_message_content = self._format_message_with_state_variables(
                    sys_message_content,
                    user_id=user_id,
                    dependencies=dependencies,
                    metadata=metadata,
                    session_state=session_state,
                )

            # type: ignore
            return Message(role=self.system_message_role, content=sys_message_content)

        # 2. If build_context is False, return None.
        if not self.build_context:
            return None

        if self.model is None:
            raise Exception("model not set")

        # 3. Build and return the default system message for the Agent.
        # 3.1 Build the list of instructions for the system message
        instructions: List[str] = []
        if self.instructions is not None:
            _instructions = self.instructions
            if callable(self.instructions):
                import inspect

                signature = inspect.signature(self.instructions)
                instruction_args: Dict[str, Any] = {}

                # Check for agent parameter
                if "agent" in signature.parameters:
                    instruction_args["agent"] = self

                # Check for session_state parameter
                if "session_state" in signature.parameters:
                    instruction_args["session_state"] = session_state or {}

                _instructions = self.instructions(**instruction_args)

            if isinstance(_instructions, str):
                instructions.append(_instructions)
            elif isinstance(_instructions, list):
                instructions.extend(_instructions)

        # 3.1.1 Add instructions from the Model
        _model_instructions = self.model.get_instructions_for_model(tools)
        if _model_instructions is not None:
            instructions.extend(_model_instructions)

        # 3.2 Build a list of additional information for the system message
        additional_information: List[str] = []
        # 3.2.1 Add instructions for using markdown
        if self.markdown and self.output_schema is None:
            additional_information.append("Use markdown to format your answers.")
        # 3.2.2 Add the current datetime
        if self.add_datetime_to_context:
            from datetime import datetime

            tz = None

            if self.timezone_identifier:
                try:
                    from zoneinfo import ZoneInfo

                    tz = ZoneInfo(self.timezone_identifier)
                except Exception:
                    log_warning("Invalid timezone identifier")

            time = datetime.now(tz) if tz else datetime.now()

            additional_information.append(f"The current time is {time}.")

        # 3.2.3 Add the current location
        if self.add_location_to_context:
            from agno.utils.location import get_location

            location = get_location()
            if location:
                location_str = ", ".join(
                    filter(
                        None,
                        [
                            location.get("city"),
                            location.get("region"),
                            location.get("country"),
                        ],
                    )
                )
                if location_str:
                    additional_information.append(f"Your approximate location is: {location_str}.")

        # 3.2.4 Add agent name if provided
        if self.name is not None and self.add_name_to_context:
            additional_information.append(f"Your name is: {self.name}.")

        # 3.2.5 Add information about agentic filters if enabled
        if self.knowledge is not None and self.enable_agentic_knowledge_filters:
            valid_filters = self.knowledge.get_valid_filters()
            if valid_filters:
                valid_filters_str = ", ".join(valid_filters)
                additional_information.append(
                    dedent(
                        f"""
                    The knowledge base contains documents with these metadata filters: {valid_filters_str}.
                    Always use filters when the user query indicates specific metadata.

                    Examples:
                    1. If the user asks about a specific person like "Jordan Mitchell", you MUST use the search_knowledge_base tool with the filters parameter set to {{'<valid key like user_id>': '<valid value based on the user query>'}}.
                    2. If the user asks about a specific document type like "contracts", you MUST use the search_knowledge_base tool with the filters parameter set to {{'document_type': 'contract'}}.
                    4. If the user asks about a specific location like "documents from New York", you MUST use the search_knowledge_base tool with the filters parameter set to {{'<valid key like location>': 'New York'}}.

                    General Guidelines:
                    - Always analyze the user query to identify relevant metadata.
                    - Use the most specific filter(s) possible to narrow down results.
                    - If multiple filters are relevant, combine them in the filters parameter (e.g., {{'name': 'Jordan Mitchell', 'document_type': 'contract'}}).
                    - Ensure the filter keys match the valid metadata filters: {valid_filters_str}.

                    You can use the search_knowledge_base tool to search the knowledge base and get the most relevant documents. Make sure to pass the filters as [Dict[str: Any]] to the tool. FOLLOW THIS STRUCTURE STRICTLY.
                """
                    )
                )

        # 3.3 Build the default system message for the Agent.
        system_message_content: str = ""
        # 3.3.1 First add the Agent description if provided
        if self.description is not None:
            system_message_content += f"{self.description}\n"
        # 3.3.2 Then add the Agent role if provided
        if self.role is not None:
            system_message_content += f"\n<your_role>\n{self.role}\n</your_role>\n\n"
        # 3.3.4 Then add instructions for the Agent
        if len(instructions) > 0:
            system_message_content += "<instructions>"
            if len(instructions) > 1:
                for _upi in instructions:
                    system_message_content += f"\n- {_upi}"
            else:
                system_message_content += "\n" + instructions[0]
            system_message_content += "\n</instructions>\n\n"
        # 3.3.6 Add additional information
        if len(additional_information) > 0:
            system_message_content += "<additional_information>"
            for _ai in additional_information:
                system_message_content += f"\n- {_ai}"
            system_message_content += "\n</additional_information>\n\n"
        # 3.3.7 Then add instructions for the tools
        if self._tool_instructions is not None:
            for _ti in self._tool_instructions:
                system_message_content += f"{_ti}\n"

        # Format the system message with the session state variables
        if self.resolve_in_context:
            system_message_content = self._format_message_with_state_variables(
                system_message_content,
                user_id=user_id,
                session_state=session_state,
                dependencies=dependencies,
                metadata=metadata,
            )

        # 3.3.7 Then add the expected output
        if self.expected_output is not None:
            system_message_content += f"<expected_output>\n{self.expected_output.strip()}\n</expected_output>\n\n"
        # 3.3.8 Then add additional context
        if self.additional_context is not None:
            system_message_content += f"{self.additional_context}\n"
        # 3.3.9 Then add memories to the system prompt
        if self.add_memories_to_context:
            _memory_manager_not_set = False
            if not user_id:
                user_id = "default"
            if self.memory_manager is None:
                self._set_memory_manager()
                _memory_manager_not_set = True

            user_memories = self.memory_manager.get_user_memories(user_id=user_id)  # type: ignore

            if user_memories and len(user_memories) > 0:
                system_message_content += "You have access to user info and preferences from previous interactions that you can use to personalize your response:\n\n"
                system_message_content += "<memories_from_previous_interactions>"
                for _memory in user_memories:  # type: ignore
                    system_message_content += f"\n- {_memory.memory}"
                system_message_content += "\n</memories_from_previous_interactions>\n\n"
                system_message_content += (
                    "Note: this information is from previous interactions and may be updated in this conversation. "
                    "You should always prefer information from this conversation over the past memories.\n"
                )
            else:
                system_message_content += (
                    "You have the capability to retain memories from previous interactions with the user, "
                    "but have not had any interactions with the user yet.\n"
                )
            if _memory_manager_not_set:
                self.memory_manager = None

            if self.enable_agentic_memory:
                system_message_content += (
                    "\n<updating_user_memories>\n"
                    "- You have access to the `update_user_memory` tool that you can use to add new memories, update existing memories, delete memories, or clear all memories.\n"
                    "- If the user's message includes information that should be captured as a memory, use the `update_user_memory` tool to update your memory database.\n"
                    "- Memories should include details that could personalize ongoing interactions with the user.\n"
                    "- Use this tool to add new memories or update existing memories that you identify in the conversation.\n"
                    "- Use this tool if the user asks to update their memory, delete a memory, or clear all memories.\n"
                    "- If you use the `update_user_memory` tool, remember to pass on the response to the user.\n"
                    "</updating_user_memories>\n\n"
                )

        # 3.3.10 Then add cultural knowledge to the system prompt
        if self.add_culture_to_context:
            _culture_manager_not_set = None
            if not self.culture_manager:
                self._set_culture_manager()
                _culture_manager_not_set = True

            cultural_knowledge = self.culture_manager.get_all_knowledge()  # type: ignore

            if cultural_knowledge and len(cultural_knowledge) > 0:
                system_message_content += (
                    "You have access to shared **Cultural Knowledge**, which provides context, norms, rules and guidance "
                    "for your reasoning, communication, and decision-making. "
                    "Cultural Knowledge represents the collective understanding, values, rules and practices that have "
                    "emerged across agents and teams. It encodes collective experience — including preferred "
                    "approaches, common patterns, lessons learned, and ethical guardrails.\n\n"
                    "When performing any task:\n"
                    "- **Reference Cultural Knowledge** to align with shared norms and best practices.\n"
                    "- **Apply it contextually**, not mechanically — adapt principles to the current situation.\n"
                    "- **Preserve consistency** with cultural values (tone, reasoning, and style) unless explicitly told otherwise.\n"
                    "- **Extend it** when you discover new insights — your outputs may become future Cultural Knowledge.\n"
                    "- **Clarify conflicts** if Cultural Knowledge appears to contradict explicit user instructions.\n\n"
                    "Your goal is to act not only intelligently but also *culturally coherently* — reflecting the "
                    "collective intelligence of the system.\n\n"
                    "Below is the currently available Cultural Knowledge for this context:\n\n"
                )
                system_message_content += "<cultural_knowledge>"
                for _knowledge in cultural_knowledge:  # type: ignore
                    system_message_content += "\n---"
                    system_message_content += f"\nName: {_knowledge.name}"
                    system_message_content += f"\nSummary: {_knowledge.summary}"
                    system_message_content += f"\nContent: {_knowledge.content}"
                system_message_content += "\n</cultural_knowledge>\n"
            else:
                system_message_content += (
                    "You have the capability to access shared **Cultural Knowledge**, which normally provides "
                    "context, norms, and guidance for your behavior and reasoning. However, no cultural knowledge "
                    "is currently available in this session.\n"
                    "Proceed thoughtfully and document any useful insights you create — they may become future "
                    "Cultural Knowledge for others.\n\n"
                )

            if _culture_manager_not_set:
                self.culture_manager = None

            if self.enable_agentic_culture:
                system_message_content += (
                    "\n<contributing_to_culture>\n"
                    "When you discover an insight, pattern, rule, or best practice that will help future agents, use the `create_or_update_cultural_knowledge` tool to add or update entries in the shared cultural knowledge.\n"
                    "\n"
                    "When to contribute:\n"
                    "- You discover a reusable insight, pattern, rule, or best practice that will help future agents.\n"
                    "- You correct or clarify an existing cultural entry.\n"
                    "- You capture a guardrail, decision rationale, postmortem lesson, or example template.\n"
                    "- You identify missing context that should persist across sessions or teams.\n"
                    "\n"
                    "Cultural knowledge should capture reusable insights, best practices, or contextual knowledge that transcends individual conversations.\n"
                    "Mention your contribution to the user only if it is relevant to their request or they asked to be notified.\n"
                    "</contributing_to_culture>\n\n"
                )

        # 3.3.11 Then add a summary of the interaction to the system prompt
        if self.add_session_summary_to_context and session.summary is not None:
            system_message_content += "Here is a brief summary of your previous interactions:\n\n"
            system_message_content += "<summary_of_previous_interactions>\n"
            system_message_content += session.summary.summary
            system_message_content += "\n</summary_of_previous_interactions>\n\n"
            system_message_content += (
                "Note: this information is from previous interactions and may be outdated. "
                "You should ALWAYS prefer information from this conversation over the past summary.\n\n"
            )

        # 3.3.12 Add the system message from the Model
        system_message_from_model = self.model.get_system_message_for_model(tools)
        if system_message_from_model is not None:
            system_message_content += system_message_from_model

        # 3.3.13 Add the JSON output prompt if output_schema is provided and the model does not support native structured outputs or JSON schema outputs
        # or if use_json_mode is True
        if (
            self.output_schema is not None
            and self.parser_model is None
            and not (
                (self.model.supports_native_structured_outputs or self.model.supports_json_schema_outputs)
                and (not self.use_json_mode or self.structured_outputs is True)
            )
        ):
            system_message_content += f"{get_json_output_prompt(self.output_schema)}"  # type: ignore

        # 3.3.14 Add the response model format prompt if output_schema is provided
        if self.output_schema is not None and self.parser_model is not None:
            system_message_content += f"{get_response_model_format_prompt(self.output_schema)}"

        # 3.3.15 Add the session state to the system message
        if add_session_state_to_context and session_state is not None:
            system_message_content += f"\n<session_state>\n{session_state}\n</session_state>\n\n"

        # Return the system message
        return (
            Message(role=self.system_message_role, content=system_message_content.strip())  # type: ignore
            if system_message_content
            else None
        )

    async def aget_system_message(
        self,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        tools: Optional[List[Union[Function, dict]]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        """Return the system message for the Agent.

        1. If the system_message is provided, use that.
        2. If build_context is False, return None.
        3. Build and return the default system message for the Agent.
        """

        # 1. If the system_message is provided, use that.
        if self.system_message is not None:
            if isinstance(self.system_message, Message):
                return self.system_message

            sys_message_content: str = ""
            if isinstance(self.system_message, str):
                sys_message_content = self.system_message
            elif callable(self.system_message):
                sys_message_content = self.system_message(agent=self)
                if not isinstance(sys_message_content, str):
                    raise Exception("system_message must return a string")

            # Format the system message with the session state variables
            if self.resolve_in_context:
                sys_message_content = self._format_message_with_state_variables(
                    sys_message_content,
                    user_id=user_id,
                    dependencies=dependencies,
                    metadata=metadata,
                    session_state=session_state,
                )

            # type: ignore
            return Message(role=self.system_message_role, content=sys_message_content)

        # 2. If build_context is False, return None.
        if not self.build_context:
            return None

        if self.model is None:
            raise Exception("model not set")

        # 3. Build and return the default system message for the Agent.
        # 3.1 Build the list of instructions for the system message
        instructions: List[str] = []
        if self.instructions is not None:
            _instructions = self.instructions
            if callable(self.instructions):
                import inspect

                signature = inspect.signature(self.instructions)
                instruction_args: Dict[str, Any] = {}

                # Check for agent parameter
                if "agent" in signature.parameters:
                    instruction_args["agent"] = self

                # Check for session_state parameter
                if "session_state" in signature.parameters:
                    instruction_args["session_state"] = session_state or {}

                _instructions = self.instructions(**instruction_args)

            if isinstance(_instructions, str):
                instructions.append(_instructions)
            elif isinstance(_instructions, list):
                instructions.extend(_instructions)

        # 3.1.1 Add instructions from the Model
        _model_instructions = self.model.get_instructions_for_model(tools)
        if _model_instructions is not None:
            instructions.extend(_model_instructions)

        # 3.2 Build a list of additional information for the system message
        additional_information: List[str] = []
        # 3.2.1 Add instructions for using markdown
        if self.markdown and self.output_schema is None:
            additional_information.append("Use markdown to format your answers.")
        # 3.2.2 Add the current datetime
        if self.add_datetime_to_context:
            from datetime import datetime

            tz = None

            if self.timezone_identifier:
                try:
                    from zoneinfo import ZoneInfo

                    tz = ZoneInfo(self.timezone_identifier)
                except Exception:
                    log_warning("Invalid timezone identifier")

            time = datetime.now(tz) if tz else datetime.now()

            additional_information.append(f"The current time is {time}.")

        # 3.2.3 Add the current location
        if self.add_location_to_context:
            from agno.utils.location import get_location

            location = get_location()
            if location:
                location_str = ", ".join(
                    filter(
                        None,
                        [
                            location.get("city"),
                            location.get("region"),
                            location.get("country"),
                        ],
                    )
                )
                if location_str:
                    additional_information.append(f"Your approximate location is: {location_str}.")

        # 3.2.4 Add agent name if provided
        if self.name is not None and self.add_name_to_context:
            additional_information.append(f"Your name is: {self.name}.")

        # 3.2.5 Add information about agentic filters if enabled
        if self.knowledge is not None and self.enable_agentic_knowledge_filters:
            valid_filters = getattr(self.knowledge, "valid_metadata_filters", None)
            if valid_filters:
                valid_filters_str = ", ".join(valid_filters)
                additional_information.append(
                    dedent(
                        f"""
                    The knowledge base contains documents with these metadata filters: {valid_filters_str}.
                    Always use filters when the user query indicates specific metadata.

                    Examples:
                    1. If the user asks about a specific person like "Jordan Mitchell", you MUST use the search_knowledge_base tool with the filters parameter set to {{'<valid key like user_id>': '<valid value based on the user query>'}}.
                    2. If the user asks about a specific document type like "contracts", you MUST use the search_knowledge_base tool with the filters parameter set to {{'document_type': 'contract'}}.
                    4. If the user asks about a specific location like "documents from New York", you MUST use the search_knowledge_base tool with the filters parameter set to {{'<valid key like location>': 'New York'}}.

                    General Guidelines:
                    - Always analyze the user query to identify relevant metadata.
                    - Use the most specific filter(s) possible to narrow down results.
                    - If multiple filters are relevant, combine them in the filters parameter (e.g., {{'name': 'Jordan Mitchell', 'document_type': 'contract'}}).
                    - Ensure the filter keys match the valid metadata filters: {valid_filters_str}.

                    You can use the search_knowledge_base tool to search the knowledge base and get the most relevant documents. Make sure to pass the filters as [Dict[str: Any]] to the tool. FOLLOW THIS STRUCTURE STRICTLY.
                """
                    )
                )

        # 3.3 Build the default system message for the Agent.
        system_message_content: str = ""
        # 3.3.1 First add the Agent description if provided
        if self.description is not None:
            system_message_content += f"{self.description}\n"
        # 3.3.2 Then add the Agent role if provided
        if self.role is not None:
            system_message_content += f"\n<your_role>\n{self.role}\n</your_role>\n\n"
        # 3.3.4 Then add instructions for the Agent
        if len(instructions) > 0:
            system_message_content += "<instructions>"
            if len(instructions) > 1:
                for _upi in instructions:
                    system_message_content += f"\n- {_upi}"
            else:
                system_message_content += "\n" + instructions[0]
            system_message_content += "\n</instructions>\n\n"
        # 3.3.6 Add additional information
        if len(additional_information) > 0:
            system_message_content += "<additional_information>"
            for _ai in additional_information:
                system_message_content += f"\n- {_ai}"
            system_message_content += "\n</additional_information>\n\n"
        # 3.3.7 Then add instructions for the tools
        if self._tool_instructions is not None:
            for _ti in self._tool_instructions:
                system_message_content += f"{_ti}\n"

        # Format the system message with the session state variables
        if self.resolve_in_context:
            system_message_content = self._format_message_with_state_variables(
                system_message_content,
                user_id=user_id,
                session_state=session_state,
                dependencies=dependencies,
                metadata=metadata,
            )

        # 3.3.7 Then add the expected output
        if self.expected_output is not None:
            system_message_content += f"<expected_output>\n{self.expected_output.strip()}\n</expected_output>\n\n"
        # 3.3.8 Then add additional context
        if self.additional_context is not None:
            system_message_content += f"{self.additional_context}\n"
        # 3.3.9 Then add memories to the system prompt
        if self.add_memories_to_context:
            _memory_manager_not_set = False
            if not user_id:
                user_id = "default"
            if self.memory_manager is None:
                self._set_memory_manager()
                _memory_manager_not_set = True

            if self._has_async_db():
                user_memories = await self.memory_manager.aget_user_memories(user_id=user_id)  # type: ignore
            else:
                user_memories = self.memory_manager.get_user_memories(user_id=user_id)  # type: ignore

            if user_memories and len(user_memories) > 0:
                system_message_content += "You have access to user info and preferences from previous interactions that you can use to personalize your response:\n\n"
                system_message_content += "<memories_from_previous_interactions>"
                for _memory in user_memories:  # type: ignore
                    system_message_content += f"\n- {_memory.memory}"
                system_message_content += "\n</memories_from_previous_interactions>\n\n"
                system_message_content += (
                    "Note: this information is from previous interactions and may be updated in this conversation. "
                    "You should always prefer information from this conversation over the past memories.\n"
                )
            else:
                system_message_content += (
                    "You have the capability to retain memories from previous interactions with the user, "
                    "but have not had any interactions with the user yet.\n"
                )
            if _memory_manager_not_set:
                self.memory_manager = None

            if self.enable_agentic_memory:
                system_message_content += (
                    "\n<updating_user_memories>\n"
                    "- You have access to the `update_user_memory` tool that you can use to add new memories, update existing memories, delete memories, or clear all memories.\n"
                    "- If the user's message includes information that should be captured as a memory, use the `update_user_memory` tool to update your memory database.\n"
                    "- Memories should include details that could personalize ongoing interactions with the user.\n"
                    "- Use this tool to add new memories or update existing memories that you identify in the conversation.\n"
                    "- Use this tool if the user asks to update their memory, delete a memory, or clear all memories.\n"
                    "- If you use the `update_user_memory` tool, remember to pass on the response to the user.\n"
                    "</updating_user_memories>\n\n"
                )

        # 3.3.10 Then add cultural knowledge to the system prompt
        if self.add_culture_to_context:
            _culture_manager_not_set = None
            if not self.culture_manager:
                self._set_culture_manager()
                _culture_manager_not_set = True

            cultural_knowledge = await self.culture_manager.aget_all_knowledge()  # type: ignore

            if cultural_knowledge and len(cultural_knowledge) > 0:
                system_message_content += (
                    "You have access to shared **Cultural Knowledge**, which provides context, norms, rules and guidance "
                    "for your reasoning, communication, and decision-making.\n\n"
                    "Cultural Knowledge represents the collective understanding, values, rules and practices that have "
                    "emerged across agents and teams. It encodes collective experience — including preferred "
                    "approaches, common patterns, lessons learned, and ethical guardrails.\n\n"
                    "When performing any task:\n"
                    "- **Reference Cultural Knowledge** to align with shared norms and best practices.\n"
                    "- **Apply it contextually**, not mechanically — adapt principles to the current situation.\n"
                    "- **Preserve consistency** with cultural values (tone, reasoning, and style) unless explicitly told otherwise.\n"
                    "- **Extend it** when you discover new insights — your outputs may become future Cultural Knowledge.\n"
                    "- **Clarify conflicts** if Cultural Knowledge appears to contradict explicit user instructions.\n\n"
                    "Your goal is to act not only intelligently but also *culturally coherently* — reflecting the "
                    "collective intelligence of the system.\n\n"
                    "Below is the currently available Cultural Knowledge for this context:\n\n"
                )
                system_message_content += "<cultural_knowledge>"
                for _knowledge in cultural_knowledge:  # type: ignore
                    system_message_content += "\n---"
                    system_message_content += f"\nName: {_knowledge.name}"
                    system_message_content += f"\nSummary: {_knowledge.summary}"
                    system_message_content += f"\nContent: {_knowledge.content}"
                system_message_content += "\n</cultural_knowledge>\n"
            else:
                system_message_content += (
                    "You have the capability to access shared **Cultural Knowledge**, which normally provides "
                    "context, norms, and guidance for your behavior and reasoning. However, no cultural knowledge "
                    "is currently available in this session.\n"
                    "Proceed thoughtfully and document any useful insights you create — they may become future "
                    "Cultural Knowledge for others.\n\n"
                )

            if _culture_manager_not_set:
                self.culture_manager = None

            if self.enable_agentic_culture:
                system_message_content += (
                    "\n<contributing_to_culture>\n"
                    "When you discover an insight, pattern, rule, or best practice that will help future agents, use the `create_or_update_cultural_knowledge` tool to add or update entries in the shared cultural knowledge.\n"
                    "\n"
                    "When to contribute:\n"
                    "- You discover a reusable insight, pattern, rule, or best practice that will help future agents.\n"
                    "- You correct or clarify an existing cultural entry.\n"
                    "- You capture a guardrail, decision rationale, postmortem lesson, or example template.\n"
                    "- You identify missing context that should persist across sessions or teams.\n"
                    "\n"
                    "Cultural knowledge should capture reusable insights, best practices, or contextual knowledge that transcends individual conversations.\n"
                    "Mention your contribution to the user only if it is relevant to their request or they asked to be notified.\n"
                    "</contributing_to_culture>\n\n"
                )

        # 3.3.11 Then add a summary of the interaction to the system prompt
        if self.add_session_summary_to_context and session.summary is not None:
            system_message_content += "Here is a brief summary of your previous interactions:\n\n"
            system_message_content += "<summary_of_previous_interactions>\n"
            system_message_content += session.summary.summary
            system_message_content += "\n</summary_of_previous_interactions>\n\n"
            system_message_content += (
                "Note: this information is from previous interactions and may be outdated. "
                "You should ALWAYS prefer information from this conversation over the past summary.\n\n"
            )

        # 3.3.12 Add the system message from the Model
        system_message_from_model = self.model.get_system_message_for_model(tools)
        if system_message_from_model is not None:
            system_message_content += system_message_from_model

        # 3.3.13 Add the JSON output prompt if output_schema is provided and the model does not support native structured outputs or JSON schema outputs
        # or if use_json_mode is True
        if (
            self.output_schema is not None
            and self.parser_model is None
            and not (
                (self.model.supports_native_structured_outputs or self.model.supports_json_schema_outputs)
                and (not self.use_json_mode or self.structured_outputs is True)
            )
        ):
            system_message_content += f"{get_json_output_prompt(self.output_schema)}"  # type: ignore

        # 3.3.14 Add the response model format prompt if output_schema is provided
        if self.output_schema is not None and self.parser_model is not None:
            system_message_content += f"{get_response_model_format_prompt(self.output_schema)}"

        # 3.3.15 Add the session state to the system message
        if self.add_session_state_to_context and session_state is not None:
            system_message_content += self._get_formatted_session_state_for_system_message(session_state)

        # Return the system message
        return (
            Message(role=self.system_message_role, content=system_message_content.strip())  # type: ignore
            if system_message_content
            else None
        )

    def _get_formatted_session_state_for_system_message(self, session_state: Dict[str, Any]) -> str:
        return f"\n<session_state>\n{session_state}\n</session_state>\n\n"

    def _get_user_message(
        self,
        *,
        run_response: RunOutput,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        input: Optional[Union[str, List, Dict, Message, BaseModel, List[Message]]] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        add_dependencies_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[Message]:
        """Return the user message for the Agent.

        1. If the user_message is provided, use that.
        2. If build_user_context is False or if the message is a list, return the message as is.
        3. Build the default user message for the Agent
        """
        # Get references from the knowledge base to use in the user message
        references = None

        # 1. If build_user_context is False or message is a list, return the message as is.
        if not self.build_user_context:
            return Message(
                role=self.user_message_role or "user",
                content=input,
                images=None if not self.send_media_to_model else images,
                audio=None if not self.send_media_to_model else audio,
                videos=None if not self.send_media_to_model else videos,
                files=None if not self.send_media_to_model else files,
                **kwargs,
            )
        # 2. Build the user message for the Agent
        elif input is None:
            # If we have any media, return a message with empty content
            if images is not None or audio is not None or videos is not None or files is not None:
                return Message(
                    role=self.user_message_role or "user",
                    content="",
                    images=None if not self.send_media_to_model else images,
                    audio=None if not self.send_media_to_model else audio,
                    videos=None if not self.send_media_to_model else videos,
                    files=None if not self.send_media_to_model else files,
                    **kwargs,
                )
            else:
                # If the input is None, return None
                return None

        else:
            # Handle list messages by converting to string
            if isinstance(input, list):
                # Convert list to string (join with newlines if all elements are strings)
                if all(isinstance(item, str) for item in input):
                    message_content = "\n".join(input)  # type: ignore
                else:
                    message_content = str(input)

                return Message(
                    role=self.user_message_role,
                    content=message_content,
                    images=None if not self.send_media_to_model else images,
                    audio=None if not self.send_media_to_model else audio,
                    videos=None if not self.send_media_to_model else videos,
                    files=None if not self.send_media_to_model else files,
                    **kwargs,
                )

            # If message is provided as a Message, use it directly
            elif isinstance(input, Message):
                return input
            # If message is provided as a dict, try to validate it as a Message
            elif isinstance(input, dict):
                try:
                    return Message.model_validate(input)
                except Exception as e:
                    log_warning(f"Failed to validate message: {e}")
                    raise Exception(f"Failed to validate message: {e}")

            # If message is provided as a BaseModel, convert it to a Message
            elif isinstance(input, BaseModel):
                try:
                    # Create a user message with the BaseModel content
                    content = input.model_dump_json(indent=2, exclude_none=True)
                    return Message(role=self.user_message_role, content=content)
                except Exception as e:
                    log_warning(f"Failed to convert BaseModel to message: {e}")
                    raise Exception(f"Failed to convert BaseModel to message: {e}")
            else:
                user_msg_content = input
                if self.add_knowledge_to_context:
                    if isinstance(input, str):
                        user_msg_content = input
                    elif callable(input):
                        user_msg_content = input(agent=self)
                    else:
                        raise Exception("message must be a string or a callable when add_references is True")

                    try:
                        retrieval_timer = Timer()
                        retrieval_timer.start()
                        docs_from_knowledge = self.get_relevant_docs_from_knowledge(
                            query=user_msg_content, filters=knowledge_filters, **kwargs
                        )
                        if docs_from_knowledge is not None:
                            references = MessageReferences(
                                query=user_msg_content,
                                references=docs_from_knowledge,
                                time=round(retrieval_timer.elapsed, 4),
                            )
                            # Add the references to the run_response
                            if run_response.references is None:
                                run_response.references = []
                            run_response.references.append(references)
                        retrieval_timer.stop()
                        log_debug(f"Time to get references: {retrieval_timer.elapsed:.4f}s")
                    except Exception as e:
                        log_warning(f"Failed to get references: {e}")

                if self.resolve_in_context:
                    user_msg_content = self._format_message_with_state_variables(
                        user_msg_content,
                        user_id=user_id,
                        session_state=session_state,
                        dependencies=dependencies,
                        metadata=metadata,
                    )

                # Convert to string for concatenation operations
                user_msg_content_str = get_text_from_message(user_msg_content) if user_msg_content is not None else ""

                # 4.1 Add knowledge references to user message
                if (
                    self.add_knowledge_to_context
                    and references is not None
                    and references.references is not None
                    and len(references.references) > 0
                ):
                    user_msg_content_str += "\n\nUse the following references from the knowledge base if it helps:\n"
                    user_msg_content_str += "<references>\n"
                    user_msg_content_str += self._convert_documents_to_string(references.references) + "\n"
                    user_msg_content_str += "</references>"
                # 4.2 Add context to user message
                if add_dependencies_to_context and dependencies is not None:
                    user_msg_content_str += "\n\n<additional context>\n"
                    user_msg_content_str += self._convert_dependencies_to_string(dependencies) + "\n"
                    user_msg_content_str += "</additional context>"

                # Use the string version for the final content
                user_msg_content = user_msg_content_str

                # Return the user message
                return Message(
                    role=self.user_message_role,
                    content=user_msg_content,
                    audio=None if not self.send_media_to_model else audio,
                    images=None if not self.send_media_to_model else images,
                    videos=None if not self.send_media_to_model else videos,
                    files=None if not self.send_media_to_model else files,
                    **kwargs,
                )

    def _get_run_messages(
        self,
        *,
        run_response: RunOutput,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Union[Function, dict]]] = None,
        **kwargs: Any,
    ) -> RunMessages:
        """This function returns a RunMessages object with the following attributes:
            - system_message: The system message for this run
            - user_message: The user message for this run
            - messages: List of messages to send to the model

        To build the RunMessages object:
        1. Add system message to run_messages
        2. Add extra messages to run_messages if provided
        3. Add history to run_messages
        4. Add user message to run_messages (if input is single content)
        5. Add input messages to run_messages if provided (if input is List[Message])

        Returns:
            RunMessages object with the following attributes:
                - system_message: The system message for this run
                - user_message: The user message for this run
                - messages: List of all messages to send to the model

        Typical usage:
        run_messages = self._get_run_messages(
            input=input, session_id=session_id, user_id=user_id, audio=audio, images=images, videos=videos, files=files, **kwargs
        )
        """

        # Initialize the RunMessages object (no media here - that's in RunInput now)
        run_messages = RunMessages()

        # 1. Add system message to run_messages
        system_message = self.get_system_message(
            session=session,
            session_state=session_state,
            user_id=user_id,
            tools=tools,
            dependencies=dependencies,
            metadata=metadata,
            add_session_state_to_context=add_session_state_to_context,
        )
        if system_message is not None:
            run_messages.system_message = system_message
            run_messages.messages.append(system_message)

        # 2. Add extra messages to run_messages if provided
        if self.additional_input is not None:
            messages_to_add_to_run_response: List[Message] = []
            if run_messages.extra_messages is None:
                run_messages.extra_messages = []

            for _m in self.additional_input:
                if isinstance(_m, Message):
                    messages_to_add_to_run_response.append(_m)
                    run_messages.messages.append(_m)
                    run_messages.extra_messages.append(_m)
                elif isinstance(_m, dict):
                    try:
                        _m_parsed = Message.model_validate(_m)
                        messages_to_add_to_run_response.append(_m_parsed)
                        run_messages.messages.append(_m_parsed)
                        run_messages.extra_messages.append(_m_parsed)
                    except Exception as e:
                        log_warning(f"Failed to validate message: {e}")
            # Add the extra messages to the run_response
            if len(messages_to_add_to_run_response) > 0:
                log_debug(f"Adding {len(messages_to_add_to_run_response)} extra messages")
                if run_response.additional_input is None:
                    run_response.additional_input = messages_to_add_to_run_response
                else:
                    run_response.additional_input.extend(messages_to_add_to_run_response)

        # 3. Add history to run_messages
        if add_history_to_context:
            from copy import deepcopy

            # Only skip messages from history when system_message_role is NOT a standard conversation role.
            # Standard conversation roles ("user", "assistant", "tool") should never be filtered
            # to preserve conversation continuity.
            skip_role = (
                self.system_message_role if self.system_message_role not in ["user", "assistant", "tool"] else None
            )

            history: List[Message] = session.get_messages_from_last_n_runs(
                last_n=self.num_history_runs,
                last_n_messages=self.num_history_messages,
                skip_role=skip_role,
                agent_id=self.id if self.team_id is not None else None,
            )

            if len(history) > 0:
                # Create a deep copy of the history messages to avoid modifying the original messages
                history_copy = [deepcopy(msg) for msg in history]

                # Tag each message as coming from history
                for _msg in history_copy:
                    _msg.from_history = True

                # Filter tool calls from history if limit is set (before adding to run_messages)
                if self.max_tool_calls_from_history is not None:
                    filter_tool_calls(history_copy, self.max_tool_calls_from_history)

                log_debug(f"Adding {len(history_copy)} messages from history")

                run_messages.messages += history_copy

        # 4. Add user message to run_messages
        user_message: Optional[Message] = None

        # 4.1 Build user message if input is None, str or list and not a list of Message/dict objects
        if (
            input is None
            or isinstance(input, str)
            or (
                isinstance(input, list)
                and not (
                    len(input) > 0
                    and (isinstance(input[0], Message) or (isinstance(input[0], dict) and "role" in input[0]))
                )
            )
        ):
            user_message = self._get_user_message(
                run_response=run_response,
                session_state=session_state,
                input=input,
                audio=audio,
                images=images,
                videos=videos,
                files=files,
                knowledge_filters=knowledge_filters,
                dependencies=dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                metadata=metadata,
                **kwargs,
            )

        # 4.2 If input is provided as a Message, use it directly
        elif isinstance(input, Message):
            user_message = input

        # 4.3 If input is provided as a dict, try to validate it as a Message
        elif isinstance(input, dict):
            try:
                if self.input_schema and is_typed_dict(self.input_schema):
                    import json

                    content = json.dumps(input, indent=2, ensure_ascii=False)
                    user_message = Message(role=self.user_message_role, content=content)
                else:
                    user_message = Message.model_validate(input)
            except Exception as e:
                log_warning(f"Failed to validate message: {e}")

        # 4.4 If input is provided as a BaseModel, convert it to a Message
        elif isinstance(input, BaseModel):
            try:
                # Create a user message with the BaseModel content
                content = input.model_dump_json(indent=2, exclude_none=True)
                user_message = Message(role=self.user_message_role, content=content)
            except Exception as e:
                log_warning(f"Failed to convert BaseModel to message: {e}")

        # 5. Add input messages to run_messages if provided (List[Message] or List[Dict])
        if (
            isinstance(input, list)
            and len(input) > 0
            and (isinstance(input[0], Message) or (isinstance(input[0], dict) and "role" in input[0]))
        ):
            for _m in input:
                if isinstance(_m, Message):
                    run_messages.messages.append(_m)
                    if run_messages.extra_messages is None:
                        run_messages.extra_messages = []
                    run_messages.extra_messages.append(_m)
                elif isinstance(_m, dict):
                    try:
                        msg = Message.model_validate(_m)
                        run_messages.messages.append(msg)
                        if run_messages.extra_messages is None:
                            run_messages.extra_messages = []
                        run_messages.extra_messages.append(msg)
                    except Exception as e:
                        log_warning(f"Failed to validate message: {e}")

        # Add user message to run_messages
        if user_message is not None:
            run_messages.user_message = user_message
            run_messages.messages.append(user_message)

        return run_messages

    async def _aget_run_messages(
        self,
        *,
        run_response: RunOutput,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Union[Function, dict]]] = None,
        **kwargs: Any,
    ) -> RunMessages:
        """This function returns a RunMessages object with the following attributes:
            - system_message: The system message for this run
            - user_message: The user message for this run
            - messages: List of messages to send to the model

        To build the RunMessages object:
        1. Add system message to run_messages
        2. Add extra messages to run_messages if provided
        3. Add history to run_messages
        4. Add user message to run_messages (if input is single content)
        5. Add input messages to run_messages if provided (if input is List[Message])

        Returns:
            RunMessages object with the following attributes:
                - system_message: The system message for this run
                - user_message: The user message for this run
                - messages: List of all messages to send to the model

        Typical usage:
        run_messages = self._get_run_messages(
            input=input, session_id=session_id, user_id=user_id, audio=audio, images=images, videos=videos, files=files, **kwargs
        )
        """

        # Initialize the RunMessages object (no media here - that's in RunInput now)
        run_messages = RunMessages()

        # 1. Add system message to run_messages
        system_message = await self.aget_system_message(
            session=session,
            session_state=session_state,
            user_id=user_id,
            tools=tools,
            dependencies=dependencies,
            metadata=metadata,
        )
        if system_message is not None:
            run_messages.system_message = system_message
            run_messages.messages.append(system_message)

        # 2. Add extra messages to run_messages if provided
        if self.additional_input is not None:
            messages_to_add_to_run_response: List[Message] = []
            if run_messages.extra_messages is None:
                run_messages.extra_messages = []

            for _m in self.additional_input:
                if isinstance(_m, Message):
                    messages_to_add_to_run_response.append(_m)
                    run_messages.messages.append(_m)
                    run_messages.extra_messages.append(_m)
                elif isinstance(_m, dict):
                    try:
                        _m_parsed = Message.model_validate(_m)
                        messages_to_add_to_run_response.append(_m_parsed)
                        run_messages.messages.append(_m_parsed)
                        run_messages.extra_messages.append(_m_parsed)
                    except Exception as e:
                        log_warning(f"Failed to validate message: {e}")
            # Add the extra messages to the run_response
            if len(messages_to_add_to_run_response) > 0:
                log_debug(f"Adding {len(messages_to_add_to_run_response)} extra messages")
                if run_response.additional_input is None:
                    run_response.additional_input = messages_to_add_to_run_response
                else:
                    run_response.additional_input.extend(messages_to_add_to_run_response)

        # 3. Add history to run_messages
        if add_history_to_context:
            from copy import deepcopy

            # Only skip messages from history when system_message_role is NOT a standard conversation role.
            # Standard conversation roles ("user", "assistant", "tool") should never be filtered
            # to preserve conversation continuity.
            skip_role = (
                self.system_message_role if self.system_message_role not in ["user", "assistant", "tool"] else None
            )

            history: List[Message] = session.get_messages_from_last_n_runs(
                last_n=self.num_history_runs,
                last_n_messages=self.num_history_messages,
                skip_role=skip_role,
                agent_id=self.id if self.team_id is not None else None,
            )

            if len(history) > 0:
                # Create a deep copy of the history messages to avoid modifying the original messages
                history_copy = [deepcopy(msg) for msg in history]

                # Tag each message as coming from history
                for _msg in history_copy:
                    _msg.from_history = True

                # Filter tool calls from history if limit is set (before adding to run_messages)
                if self.max_tool_calls_from_history is not None:
                    filter_tool_calls(history_copy, self.max_tool_calls_from_history)

                log_debug(f"Adding {len(history_copy)} messages from history")

                run_messages.messages += history_copy

        # 4. Add user message to run_messages
        user_message: Optional[Message] = None

        # 4.1 Build user message if input is None, str or list and not a list of Message/dict objects
        if (
            input is None
            or isinstance(input, str)
            or (
                isinstance(input, list)
                and not (
                    len(input) > 0
                    and (isinstance(input[0], Message) or (isinstance(input[0], dict) and "role" in input[0]))
                )
            )
        ):
            user_message = self._get_user_message(
                run_response=run_response,
                session_state=session_state,
                input=input,
                audio=audio,
                images=images,
                videos=videos,
                files=files,
                knowledge_filters=knowledge_filters,
                dependencies=dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                metadata=metadata,
                **kwargs,
            )

        # 4.2 If input is provided as a Message, use it directly
        elif isinstance(input, Message):
            user_message = input

        # 4.3 If input is provided as a dict, try to validate it as a Message
        elif isinstance(input, dict):
            try:
                user_message = Message.model_validate(input)
            except Exception as e:
                log_warning(f"Failed to validate message: {e}")

        # 4.4 If input is provided as a BaseModel, convert it to a Message
        elif isinstance(input, BaseModel):
            try:
                # Create a user message with the BaseModel content
                content = input.model_dump_json(indent=2, exclude_none=True)
                user_message = Message(role=self.user_message_role, content=content)
            except Exception as e:
                log_warning(f"Failed to convert BaseModel to message: {e}")

        # 5. Add input messages to run_messages if provided (List[Message] or List[Dict])
        if (
            isinstance(input, list)
            and len(input) > 0
            and (isinstance(input[0], Message) or (isinstance(input[0], dict) and "role" in input[0]))
        ):
            for _m in input:
                if isinstance(_m, Message):
                    run_messages.messages.append(_m)
                    if run_messages.extra_messages is None:
                        run_messages.extra_messages = []
                    run_messages.extra_messages.append(_m)
                elif isinstance(_m, dict):
                    try:
                        msg = Message.model_validate(_m)
                        run_messages.messages.append(msg)
                        if run_messages.extra_messages is None:
                            run_messages.extra_messages = []
                        run_messages.extra_messages.append(msg)
                    except Exception as e:
                        log_warning(f"Failed to validate message: {e}")

        # Add user message to run_messages
        if user_message is not None:
            run_messages.user_message = user_message
            run_messages.messages.append(user_message)

        return run_messages

    def _get_continue_run_messages(
        self,
        input: List[Message],
    ) -> RunMessages:
        """This function returns a RunMessages object with the following attributes:
            - system_message: The system message for this run
            - user_message: The user message for this run
            - messages: List of messages to send to the model

        It continues from a previous run and completes a tool call that was paused.
        """

        # Initialize the RunMessages object
        run_messages = RunMessages()

        # Extract most recent user message from messages as the original user message
        user_message = None
        for msg in reversed(input):
            if msg.role == self.user_message_role:
                user_message = msg
                break

        # Extract system message from messages
        system_message = None
        for msg in input:
            if msg.role == self.system_message_role:
                system_message = msg
                break

        run_messages.system_message = system_message
        run_messages.user_message = user_message
        run_messages.messages = input

        return run_messages

    def _get_messages_for_parser_model(
        self,
        model_response: ModelResponse,
        response_format: Optional[Union[Dict, Type[BaseModel]]],
    ) -> List[Message]:
        """Get the messages for the parser model."""
        system_content = (
            self.parser_model_prompt
            if self.parser_model_prompt is not None
            else "You are tasked with creating a structured output from the provided user message."
        )

        if response_format == {"type": "json_object"} and self.output_schema is not None:
            system_content += f"{get_json_output_prompt(self.output_schema)}"  # type: ignore

        return [
            Message(role="system", content=system_content),
            Message(role="user", content=model_response.content),
        ]

    def _get_messages_for_parser_model_stream(
        self,
        run_response: RunOutput,
        response_format: Optional[Union[Dict, Type[BaseModel]]],
    ) -> List[Message]:
        """Get the messages for the parser model."""
        system_content = (
            self.parser_model_prompt
            if self.parser_model_prompt is not None
            else "You are tasked with creating a structured output from the provided data."
        )

        if response_format == {"type": "json_object"} and self.output_schema is not None:
            system_content += f"{get_json_output_prompt(self.output_schema)}"  # type: ignore

        return [
            Message(role="system", content=system_content),
            Message(role="user", content=run_response.content),
        ]

    def _get_messages_for_output_model(self, messages: List[Message]) -> List[Message]:
        """Get the messages for the output model."""

        if self.output_model_prompt is not None:
            system_message_exists = False
            for message in messages:
                if message.role == "system":
                    system_message_exists = True
                    message.content = self.output_model_prompt
                    break
            if not system_message_exists:
                messages.insert(0, Message(role="system", content=self.output_model_prompt))

        # Remove the last assistant message from the messages list
        messages.pop(-1)

        return messages

    def get_relevant_docs_from_knowledge(
        self,
        query: str,
        num_documents: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[List[Union[Dict[str, Any], str]]]:
        """Get relevant docs from the knowledge base to answer a query.

        Args:
            query (str): The query to search for.
            num_documents (Optional[int]): Number of documents to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.
            **kwargs: Additional keyword arguments.

        Returns:
            Optional[List[Dict[str, Any]]]: List of relevant document dicts.
        """
        from agno.knowledge.document import Document

        if num_documents is None and self.knowledge is not None:
            num_documents = self.knowledge.max_results
        # Validate the filters against known valid filter keys
        if self.knowledge is not None:
            valid_filters, invalid_keys = self.knowledge.validate_filters(filters)  # type: ignore

            # Warn about invalid filter keys
            if invalid_keys:
                # type: ignore
                log_warning(f"Invalid filter keys provided: {invalid_keys}. These filters will be ignored.")
                log_info(f"Valid filter keys are: {self.knowledge.valid_metadata_filters}")  # type: ignore

                # Only use valid filters
                filters = valid_filters
                if not filters:
                    log_warning("No valid filters remain after validation. Search will proceed without filters.")

        if self.knowledge_retriever is not None and callable(self.knowledge_retriever):
            from inspect import signature

            try:
                sig = signature(self.knowledge_retriever)
                knowledge_retriever_kwargs: Dict[str, Any] = {}
                if "agent" in sig.parameters:
                    knowledge_retriever_kwargs = {"agent": self}
                if "filters" in sig.parameters:
                    knowledge_retriever_kwargs["filters"] = filters
                knowledge_retriever_kwargs.update({"query": query, "num_documents": num_documents, **kwargs})
                return self.knowledge_retriever(**knowledge_retriever_kwargs)
            except Exception as e:
                log_warning(f"Knowledge retriever failed: {e}")
                raise e

        # Use knowledge base search
        try:
            if self.knowledge is None or (
                (getattr(self.knowledge, "vector_db", None)) is None
                and getattr(self.knowledge, "knowledge_retriever", None) is None
            ):
                return None

            if num_documents is None:
                if isinstance(self.knowledge, Knowledge):
                    num_documents = self.knowledge.max_results

            log_debug(f"Searching knowledge base with filters: {filters}")
            relevant_docs: List[Document] = self.knowledge.search(
                query=query, max_results=num_documents, filters=filters
            )

            if not relevant_docs or len(relevant_docs) == 0:
                log_debug("No relevant documents found for query")
                return None

            return [doc.to_dict() for doc in relevant_docs]
        except Exception as e:
            log_warning(f"Error searching knowledge base: {e}")
            raise e

    async def aget_relevant_docs_from_knowledge(
        self,
        query: str,
        num_documents: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[List[Union[Dict[str, Any], str]]]:
        """Get relevant documents from knowledge base asynchronously."""
        from agno.knowledge.document import Document

        if num_documents is None and self.knowledge is not None:
            num_documents = self.knowledge.max_results

        # Validate the filters against known valid filter keys
        if self.knowledge is not None:
            valid_filters, invalid_keys = self.knowledge.validate_filters(filters)  # type: ignore

            # Warn about invalid filter keys
            if invalid_keys:  # type: ignore
                log_warning(f"Invalid filter keys provided: {invalid_keys}. These filters will be ignored.")
                log_info(f"Valid filter keys are: {self.knowledge.valid_metadata_filters}")  # type: ignore

                # Only use valid filters
                filters = valid_filters
                if not filters:
                    log_warning("No valid filters remain after validation. Search will proceed without filters.")

        if self.knowledge_retriever is not None and callable(self.knowledge_retriever):
            from inspect import isawaitable, signature

            try:
                sig = signature(self.knowledge_retriever)
                knowledge_retriever_kwargs: Dict[str, Any] = {}
                if "agent" in sig.parameters:
                    knowledge_retriever_kwargs = {"agent": self}
                if "filters" in sig.parameters:
                    knowledge_retriever_kwargs["filters"] = filters
                knowledge_retriever_kwargs.update({"query": query, "num_documents": num_documents, **kwargs})
                result = self.knowledge_retriever(**knowledge_retriever_kwargs)

                if isawaitable(result):
                    result = await result

                return result
            except Exception as e:
                log_warning(f"Knowledge retriever failed: {e}")
                raise e

        # Use knowledge base search
        try:
            if self.knowledge is None or (
                getattr(self.knowledge, "vector_db", None) is None
                and getattr(self.knowledge, "knowledge_retriever", None) is None
            ):
                return None

            if num_documents is None:
                num_documents = self.knowledge.max_results

            log_debug(f"Searching knowledge base with filters: {filters}")
            relevant_docs: List[Document] = await self.knowledge.async_search(
                query=query, max_results=num_documents, filters=filters
            )

            if not relevant_docs or len(relevant_docs) == 0:
                log_debug("No relevant documents found for query")
                return None

            return [doc.to_dict() for doc in relevant_docs]
        except Exception as e:
            log_warning(f"Error searching knowledge base: {e}")
            raise e

    def _convert_documents_to_string(self, docs: List[Union[Dict[str, Any], str]]) -> str:
        if docs is None or len(docs) == 0:
            return ""

        if self.references_format == "yaml":
            import yaml

            return yaml.dump(docs)

        import json

        return json.dumps(docs, indent=2, ensure_ascii=False)

    def _convert_dependencies_to_string(self, context: Dict[str, Any]) -> str:
        """Convert the context dictionary to a string representation.

        Args:
            context: Dictionary containing context data

        Returns:
            String representation of the context, or empty string if conversion fails
        """
        if context is None:
            return ""

        import json

        try:
            return json.dumps(context, indent=2, default=str)
        except (TypeError, ValueError, OverflowError) as e:
            log_warning(f"Failed to convert context to JSON: {e}")
            # Attempt a fallback conversion for non-serializable objects
            sanitized_context = {}
            for key, value in context.items():
                try:
                    # Try to serialize each value individually
                    json.dumps({key: value}, default=str)
                    sanitized_context[key] = value
                except Exception:
                    # If serialization fails, convert to string representation
                    sanitized_context[key] = str(value)

            try:
                return json.dumps(sanitized_context, indent=2)
            except Exception as e:
                log_error(f"Failed to convert sanitized context to JSON: {e}")
                return str(context)

    def deep_copy(self, *, update: Optional[Dict[str, Any]] = None) -> Agent:
        """Create and return a deep copy of this Agent, optionally updating fields.

        Args:
            update (Optional[Dict[str, Any]]): Optional dictionary of fields for the new Agent.

        Returns:
            Agent: A new Agent instance.
        """
        from dataclasses import fields

        # Extract the fields to set for the new Agent
        fields_for_new_agent: Dict[str, Any] = {}

        for f in fields(self):
            field_value = getattr(self, f.name)
            if field_value is not None:
                fields_for_new_agent[f.name] = self._deep_copy_field(f.name, field_value)

        # Update fields if provided
        if update:
            fields_for_new_agent.update(update)
        # Create a new Agent
        new_agent = self.__class__(**fields_for_new_agent)
        log_debug(f"Created new {self.__class__.__name__}")
        return new_agent

    def _deep_copy_field(self, field_name: str, field_value: Any) -> Any:
        """Helper method to deep copy a field based on its type."""
        from copy import copy, deepcopy

        # For memory and reasoning_agent, use their deep_copy methods
        if field_name == "reasoning_agent":
            return field_value.deep_copy()

        # For storage, model and reasoning_model, use a deep copy
        elif field_name in ("db", "model", "reasoning_model"):
            try:
                return deepcopy(field_value)
            except Exception:
                try:
                    return copy(field_value)
                except Exception as e:
                    log_warning(f"Failed to copy field: {field_name} - {e}")
                    return field_value

        # For compound types, attempt a deep copy
        elif isinstance(field_value, (list, dict, set)):
            try:
                return deepcopy(field_value)
            except Exception:
                try:
                    return copy(field_value)
                except Exception as e:
                    log_warning(f"Failed to copy field: {field_name} - {e}")
                    return field_value

        # For pydantic models, attempt a model_copy
        elif isinstance(field_value, BaseModel):
            try:
                return field_value.model_copy(deep=True)
            except Exception:
                try:
                    return field_value.model_copy(deep=False)
                except Exception as e:
                    log_warning(f"Failed to copy field: {field_name} - {e}")
                    return field_value

        # For other types, attempt a shallow copy first
        try:
            from copy import copy

            return copy(field_value)
        except Exception:
            # If copy fails, return as is
            return field_value

    def save_run_response_to_file(
        self,
        run_response: RunOutput,
        input: Optional[Union[str, List, Dict, Message, List[Message]]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        if self.save_response_to_file is not None and run_response is not None:
            message_str = None
            if input is not None:
                if isinstance(input, str):
                    message_str = input
                else:
                    log_warning("Did not use input in output file name: input is not a string")
            try:
                from pathlib import Path

                fn = self.save_response_to_file.format(
                    name=self.name,
                    session_id=session_id,
                    user_id=user_id,
                    message=message_str,
                    run_id=run_response.run_id,
                )
                fn_path = Path(fn)
                if not fn_path.parent.exists():
                    fn_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(run_response.content, str):
                    fn_path.write_text(run_response.content)
                else:
                    import json

                    fn_path.write_text(json.dumps(run_response.content, indent=2))
            except Exception as e:
                log_warning(f"Failed to save output to file: {e}")

    def _calculate_run_metrics(self, messages: List[Message], current_run_metrics: Optional[Metrics] = None) -> Metrics:
        """Sum the metrics of the given messages into a Metrics object"""
        metrics = current_run_metrics or Metrics()

        assistant_message_role = self.model.assistant_message_role if self.model is not None else "assistant"
        for m in messages:
            if m.role == assistant_message_role and m.metrics is not None and m.from_history is False:
                metrics += m.metrics

        # If the run metrics were already initialized, keep the time related metrics
        if current_run_metrics is not None:
            metrics.timer = current_run_metrics.timer
            metrics.duration = current_run_metrics.duration
            metrics.time_to_first_token = current_run_metrics.time_to_first_token

        return metrics

    ###########################################################################
    # Handle images, videos and audio
    ###########################################################################

    def _add_image(self, image: Image, run_response: RunOutput) -> None:
        """Add an image to both the agent's stateful storage and the current run response"""
        # Add to run response
        if run_response.images is None:
            run_response.images = []
        run_response.images.append(image)

    def _add_video(self, video: Video, run_response: RunOutput) -> None:
        """Add a video to both the agent's stateful storage and the current run response"""
        # Add to run response
        if run_response.videos is None:
            run_response.videos = []
        run_response.videos.append(video)

    def _add_audio(self, audio: Audio, run_response: RunOutput) -> None:
        """Add audio to both the agent's stateful storage and the current run response"""
        # Add to run response
        if run_response.audio is None:
            run_response.audio = []
        run_response.audio.append(audio)

    def _add_file(self, file: File, run_response: RunOutput) -> None:
        """Add file to both the agent's stateful storage and the current run response"""
        # Add to run response
        if run_response.files is None:
            run_response.files = []
        run_response.files.append(file)

    ###########################################################################
    # Reasoning
    ###########################################################################

    def _handle_reasoning(self, run_response: RunOutput, run_messages: RunMessages) -> None:
        if self.reasoning or self.reasoning_model is not None:
            reasoning_generator = self._reason(
                run_response=run_response, run_messages=run_messages, stream_events=False
            )

            # Consume the generator without yielding
            deque(reasoning_generator, maxlen=0)

    def _handle_reasoning_stream(
        self, run_response: RunOutput, run_messages: RunMessages, stream_events: Optional[bool] = None
    ) -> Iterator[RunOutputEvent]:
        if self.reasoning or self.reasoning_model is not None:
            reasoning_generator = self._reason(
                run_response=run_response,
                run_messages=run_messages,
                stream_events=stream_events,
            )
            yield from reasoning_generator

    async def _ahandle_reasoning(self, run_response: RunOutput, run_messages: RunMessages) -> None:
        if self.reasoning or self.reasoning_model is not None:
            reason_generator = self._areason(run_response=run_response, run_messages=run_messages, stream_events=False)
            # Consume the generator without yielding
            async for _ in reason_generator:
                pass

    async def _ahandle_reasoning_stream(
        self, run_response: RunOutput, run_messages: RunMessages, stream_events: Optional[bool] = None
    ) -> AsyncIterator[RunOutputEvent]:
        if self.reasoning or self.reasoning_model is not None:
            reason_generator = self._areason(
                run_response=run_response,
                run_messages=run_messages,
                stream_events=stream_events,
            )
            async for item in reason_generator:
                yield item

    def _format_reasoning_step_content(self, run_response: RunOutput, reasoning_step: ReasoningStep) -> str:
        """Format content for a reasoning step without changing any existing logic."""
        step_content = ""
        if reasoning_step.title:
            step_content += f"## {reasoning_step.title}\n"
        if reasoning_step.reasoning:
            step_content += f"{reasoning_step.reasoning}\n"
        if reasoning_step.action:
            step_content += f"Action: {reasoning_step.action}\n"
        if reasoning_step.result:
            step_content += f"Result: {reasoning_step.result}\n"
        step_content += "\n"

        # Get the current reasoning_content and append this step
        current_reasoning_content = ""
        if hasattr(run_response, "reasoning_content") and run_response.reasoning_content:  # type: ignore
            current_reasoning_content = run_response.reasoning_content  # type: ignore

        # Create updated reasoning_content
        updated_reasoning_content = current_reasoning_content + step_content

        return updated_reasoning_content

    def _reason(
        self, run_response: RunOutput, run_messages: RunMessages, stream_events: Optional[bool] = None
    ) -> Iterator[RunOutputEvent]:
        # Yield a reasoning started event
        if stream_events:
            yield handle_event(  # type: ignore
                create_reasoning_started_event(from_run_response=run_response),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

        use_default_reasoning = False

        # Get the reasoning model
        reasoning_model: Optional[Model] = self.reasoning_model
        reasoning_model_provided = reasoning_model is not None
        if reasoning_model is None and self.model is not None:
            from copy import deepcopy

            reasoning_model = deepcopy(self.model)
        if reasoning_model is None:
            log_warning("Reasoning error. Reasoning model is None, continuing regular session...")
            return

        # If a reasoning model is provided, use it to generate reasoning
        if reasoning_model_provided:
            from agno.reasoning.anthropic import is_anthropic_reasoning_model
            from agno.reasoning.azure_ai_foundry import is_ai_foundry_reasoning_model
            from agno.reasoning.deepseek import is_deepseek_reasoning_model
            from agno.reasoning.gemini import is_gemini_reasoning_model
            from agno.reasoning.groq import is_groq_reasoning_model
            from agno.reasoning.helpers import get_reasoning_agent
            from agno.reasoning.ollama import is_ollama_reasoning_model
            from agno.reasoning.openai import is_openai_reasoning_model
            from agno.reasoning.vertexai import is_vertexai_reasoning_model

            reasoning_agent = self.reasoning_agent or get_reasoning_agent(
                reasoning_model=reasoning_model,
                telemetry=self.telemetry,
                debug_mode=self.debug_mode,
                debug_level=self.debug_level,
                session_state=self.session_state,
                dependencies=self.dependencies,
                metadata=self.metadata,
            )
            is_deepseek = is_deepseek_reasoning_model(reasoning_model)
            is_groq = is_groq_reasoning_model(reasoning_model)
            is_openai = is_openai_reasoning_model(reasoning_model)
            is_ollama = is_ollama_reasoning_model(reasoning_model)
            is_ai_foundry = is_ai_foundry_reasoning_model(reasoning_model)
            is_gemini = is_gemini_reasoning_model(reasoning_model)
            is_anthropic = is_anthropic_reasoning_model(reasoning_model)
            is_vertexai = is_vertexai_reasoning_model(reasoning_model)

            if (
                is_deepseek
                or is_groq
                or is_openai
                or is_ollama
                or is_ai_foundry
                or is_gemini
                or is_anthropic
                or is_vertexai
            ):
                reasoning_message: Optional[Message] = None
                if is_deepseek:
                    from agno.reasoning.deepseek import get_deepseek_reasoning

                    log_debug("Starting DeepSeek Reasoning", center=True, symbol="=")
                    reasoning_message = get_deepseek_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_groq:
                    from agno.reasoning.groq import get_groq_reasoning

                    log_debug("Starting Groq Reasoning", center=True, symbol="=")
                    reasoning_message = get_groq_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_openai:
                    from agno.reasoning.openai import get_openai_reasoning

                    log_debug("Starting OpenAI Reasoning", center=True, symbol="=")
                    reasoning_message = get_openai_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_ollama:
                    from agno.reasoning.ollama import get_ollama_reasoning

                    log_debug("Starting Ollama Reasoning", center=True, symbol="=")
                    reasoning_message = get_ollama_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_ai_foundry:
                    from agno.reasoning.azure_ai_foundry import get_ai_foundry_reasoning

                    log_debug("Starting Azure AI Foundry Reasoning", center=True, symbol="=")
                    reasoning_message = get_ai_foundry_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_gemini:
                    from agno.reasoning.gemini import get_gemini_reasoning

                    log_debug("Starting Gemini Reasoning", center=True, symbol="=")
                    reasoning_message = get_gemini_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_anthropic:
                    from agno.reasoning.anthropic import get_anthropic_reasoning

                    log_debug("Starting Anthropic Claude Reasoning", center=True, symbol="=")
                    reasoning_message = get_anthropic_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_vertexai:
                    from agno.reasoning.vertexai import get_vertexai_reasoning

                    log_debug("Starting VertexAI Reasoning", center=True, symbol="=")
                    reasoning_message = get_vertexai_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )

                if reasoning_message is None:
                    log_warning("Reasoning error. Reasoning response is None, continuing regular session...")
                    return
                run_messages.messages.append(reasoning_message)
                # Add reasoning step to the Agent's run_response
                update_run_output_with_reasoning(
                    run_response=run_response,
                    reasoning_steps=[ReasoningStep(result=reasoning_message.content)],
                    reasoning_agent_messages=[reasoning_message],
                )
                if stream_events:
                    yield handle_event(  # type: ignore
                        create_reasoning_completed_event(
                            from_run_response=run_response,
                            content=ReasoningSteps(reasoning_steps=[ReasoningStep(result=reasoning_message.content)]),
                            content_type=ReasoningSteps.__name__,
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )
            else:
                log_warning(
                    f"Reasoning model: {reasoning_model.__class__.__name__} is not a native reasoning model, defaulting to manual Chain-of-Thought reasoning"
                )
                use_default_reasoning = True
        # If no reasoning model is provided, use default reasoning
        else:
            use_default_reasoning = True

        if use_default_reasoning:
            from agno.reasoning.default import get_default_reasoning_agent
            from agno.reasoning.helpers import (
                get_next_action,
                update_messages_with_reasoning,
            )

            # Get default reasoning agent
            reasoning_agent: Optional[Agent] = self.reasoning_agent  # type: ignore
            if reasoning_agent is None:
                reasoning_agent = get_default_reasoning_agent(
                    reasoning_model=reasoning_model,
                    min_steps=self.reasoning_min_steps,
                    max_steps=self.reasoning_max_steps,
                    tools=self.tools,
                    tool_call_limit=self.tool_call_limit,
                    use_json_mode=self.use_json_mode,
                    telemetry=self.telemetry,
                    debug_mode=self.debug_mode,
                    debug_level=self.debug_level,
                    session_state=self.session_state,
                    dependencies=self.dependencies,
                    metadata=self.metadata,
                )

            # Validate reasoning agent
            if reasoning_agent is None:
                log_warning("Reasoning error. Reasoning agent is None, continuing regular session...")
                return
            # Ensure the reasoning agent response model is ReasoningSteps
            if (
                reasoning_agent.output_schema is not None
                and not isinstance(reasoning_agent.output_schema, type)
                and not issubclass(reasoning_agent.output_schema, ReasoningSteps)
            ):
                log_warning("Reasoning agent response model should be `ReasoningSteps`, continuing regular session...")
                return

            step_count = 1
            next_action = NextAction.CONTINUE
            reasoning_messages: List[Message] = []
            all_reasoning_steps: List[ReasoningStep] = []
            log_debug("Starting Reasoning", center=True, symbol="=")
            while next_action == NextAction.CONTINUE and step_count < self.reasoning_max_steps:
                log_debug(f"Step {step_count}", center=True, symbol="=")
                try:
                    # Run the reasoning agent
                    reasoning_agent_response: RunOutput = reasoning_agent.run(input=run_messages.get_input_messages())
                    if reasoning_agent_response.content is None or reasoning_agent_response.messages is None:
                        log_warning("Reasoning error. Reasoning response is empty, continuing regular session...")
                        break

                    if isinstance(reasoning_agent_response.content, str):
                        log_warning(
                            "Reasoning error. Content is a string, not structured output. Continuing regular session..."
                        )
                        break

                    if reasoning_agent_response.content is not None and (
                        reasoning_agent_response.content.reasoning_steps is None
                        or len(reasoning_agent_response.content.reasoning_steps) == 0
                    ):
                        log_warning("Reasoning error. Reasoning steps are empty, continuing regular session...")
                        break

                    reasoning_steps: List[ReasoningStep] = reasoning_agent_response.content.reasoning_steps
                    all_reasoning_steps.extend(reasoning_steps)
                    # Yield reasoning steps
                    if stream_events:
                        for reasoning_step in reasoning_steps:
                            updated_reasoning_content = self._format_reasoning_step_content(
                                run_response=run_response,
                                reasoning_step=reasoning_step,
                            )

                            yield handle_event(  # type: ignore
                                create_reasoning_step_event(
                                    from_run_response=run_response,
                                    reasoning_step=reasoning_step,
                                    reasoning_content=updated_reasoning_content,
                                ),
                                run_response,
                                events_to_skip=self.events_to_skip,  # type: ignore
                                store_events=self.store_events,
                            )

                    # Find the index of the first assistant message
                    first_assistant_index = next(
                        (i for i, m in enumerate(reasoning_agent_response.messages) if m.role == "assistant"),
                        len(reasoning_agent_response.messages),
                    )
                    # Extract reasoning messages starting from the message after the first assistant message
                    reasoning_messages = reasoning_agent_response.messages[first_assistant_index:]

                    # Add reasoning step to the Agent's run_response
                    update_run_output_with_reasoning(
                        run_response=run_response,
                        reasoning_steps=reasoning_steps,
                        reasoning_agent_messages=reasoning_agent_response.messages,
                    )
                    # Get the next action
                    next_action = get_next_action(reasoning_steps[-1])
                    if next_action == NextAction.FINAL_ANSWER:
                        break
                except Exception as e:
                    log_error(f"Reasoning error: {e}")
                    break

                step_count += 1

            log_debug(f"Total Reasoning steps: {len(all_reasoning_steps)}")
            log_debug("Reasoning finished", center=True, symbol="=")

            # Update the messages_for_model to include reasoning messages
            update_messages_with_reasoning(
                run_messages=run_messages,
                reasoning_messages=reasoning_messages,
            )

            # Yield the final reasoning completed event
            if stream_events:
                yield handle_event(  # type: ignore
                    create_reasoning_completed_event(
                        from_run_response=run_response,
                        content=ReasoningSteps(reasoning_steps=all_reasoning_steps),
                        content_type=ReasoningSteps.__name__,
                    ),
                    run_response,
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

    async def _areason(
        self, run_response: RunOutput, run_messages: RunMessages, stream_events: Optional[bool] = None
    ) -> Any:
        # Yield a reasoning started event
        if stream_events:
            yield handle_event(  # type: ignore
                create_reasoning_started_event(from_run_response=run_response),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

        use_default_reasoning = False

        # Get the reasoning model
        reasoning_model: Optional[Model] = self.reasoning_model
        reasoning_model_provided = reasoning_model is not None
        if reasoning_model is None and self.model is not None:
            from copy import deepcopy

            reasoning_model = deepcopy(self.model)
        if reasoning_model is None:
            log_warning("Reasoning error. Reasoning model is None, continuing regular session...")
            return

        # If a reasoning model is provided, use it to generate reasoning
        if reasoning_model_provided:
            from agno.reasoning.anthropic import is_anthropic_reasoning_model
            from agno.reasoning.azure_ai_foundry import is_ai_foundry_reasoning_model
            from agno.reasoning.deepseek import is_deepseek_reasoning_model
            from agno.reasoning.gemini import is_gemini_reasoning_model
            from agno.reasoning.groq import is_groq_reasoning_model
            from agno.reasoning.helpers import get_reasoning_agent
            from agno.reasoning.ollama import is_ollama_reasoning_model
            from agno.reasoning.openai import is_openai_reasoning_model
            from agno.reasoning.vertexai import is_vertexai_reasoning_model

            reasoning_agent = self.reasoning_agent or get_reasoning_agent(
                reasoning_model=reasoning_model,
                telemetry=self.telemetry,
                debug_mode=self.debug_mode,
                debug_level=self.debug_level,
                session_state=self.session_state,
                dependencies=self.dependencies,
                metadata=self.metadata,
            )
            is_deepseek = is_deepseek_reasoning_model(reasoning_model)
            is_groq = is_groq_reasoning_model(reasoning_model)
            is_openai = is_openai_reasoning_model(reasoning_model)
            is_ollama = is_ollama_reasoning_model(reasoning_model)
            is_ai_foundry = is_ai_foundry_reasoning_model(reasoning_model)
            is_gemini = is_gemini_reasoning_model(reasoning_model)
            is_anthropic = is_anthropic_reasoning_model(reasoning_model)
            is_vertexai = is_vertexai_reasoning_model(reasoning_model)

            if (
                is_deepseek
                or is_groq
                or is_openai
                or is_ollama
                or is_ai_foundry
                or is_gemini
                or is_anthropic
                or is_vertexai
            ):
                reasoning_message: Optional[Message] = None
                if is_deepseek:
                    from agno.reasoning.deepseek import aget_deepseek_reasoning

                    log_debug("Starting DeepSeek Reasoning", center=True, symbol="=")
                    reasoning_message = await aget_deepseek_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_groq:
                    from agno.reasoning.groq import aget_groq_reasoning

                    log_debug("Starting Groq Reasoning", center=True, symbol="=")
                    reasoning_message = await aget_groq_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_openai:
                    from agno.reasoning.openai import aget_openai_reasoning

                    log_debug("Starting OpenAI Reasoning", center=True, symbol="=")
                    reasoning_message = await aget_openai_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_ollama:
                    from agno.reasoning.ollama import get_ollama_reasoning

                    log_debug("Starting Ollama Reasoning", center=True, symbol="=")
                    reasoning_message = get_ollama_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_ai_foundry:
                    from agno.reasoning.azure_ai_foundry import get_ai_foundry_reasoning

                    log_debug("Starting Azure AI Foundry Reasoning", center=True, symbol="=")
                    reasoning_message = get_ai_foundry_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_gemini:
                    from agno.reasoning.gemini import aget_gemini_reasoning

                    log_debug("Starting Gemini Reasoning", center=True, symbol="=")
                    reasoning_message = await aget_gemini_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_anthropic:
                    from agno.reasoning.anthropic import aget_anthropic_reasoning

                    log_debug("Starting Anthropic Claude Reasoning", center=True, symbol="=")
                    reasoning_message = await aget_anthropic_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )
                elif is_vertexai:
                    from agno.reasoning.vertexai import aget_vertexai_reasoning

                    log_debug("Starting VertexAI Reasoning", center=True, symbol="=")
                    reasoning_message = await aget_vertexai_reasoning(
                        reasoning_agent=reasoning_agent,
                        messages=run_messages.get_input_messages(),
                    )

                if reasoning_message is None:
                    log_warning("Reasoning error. Reasoning response is None, continuing regular session...")
                    return
                run_messages.messages.append(reasoning_message)
                # Add reasoning step to the Agent's run_response
                update_run_output_with_reasoning(
                    run_response=run_response,
                    reasoning_steps=[ReasoningStep(result=reasoning_message.content)],
                    reasoning_agent_messages=[reasoning_message],
                )
                if stream_events:
                    yield handle_event(
                        create_reasoning_completed_event(
                            from_run_response=run_response,
                            content=ReasoningSteps(reasoning_steps=[ReasoningStep(result=reasoning_message.content)]),
                            content_type=ReasoningSteps.__name__,
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )
            else:
                log_warning(
                    f"Reasoning model: {reasoning_model.__class__.__name__} is not a native reasoning model, defaulting to manual Chain-of-Thought reasoning"
                )
                use_default_reasoning = True
        # If no reasoning model is provided, use default reasoning
        else:
            use_default_reasoning = True

        if use_default_reasoning:
            from agno.reasoning.default import get_default_reasoning_agent
            from agno.reasoning.helpers import (
                get_next_action,
                update_messages_with_reasoning,
            )

            # Get default reasoning agent
            reasoning_agent: Optional[Agent] = self.reasoning_agent  # type: ignore
            if reasoning_agent is None:
                reasoning_agent = get_default_reasoning_agent(
                    reasoning_model=reasoning_model,
                    min_steps=self.reasoning_min_steps,
                    max_steps=self.reasoning_max_steps,
                    tools=self.tools,
                    tool_call_limit=self.tool_call_limit,
                    use_json_mode=self.use_json_mode,
                    telemetry=self.telemetry,
                    debug_mode=self.debug_mode,
                    debug_level=self.debug_level,
                    session_state=self.session_state,
                    dependencies=self.dependencies,
                    metadata=self.metadata,
                )

            # Validate reasoning agent
            if reasoning_agent is None:
                log_warning("Reasoning error. Reasoning agent is None, continuing regular session...")
                return
            # Ensure the reasoning agent response model is ReasoningSteps
            if (
                reasoning_agent.output_schema is not None
                and not isinstance(reasoning_agent.output_schema, type)
                and not issubclass(reasoning_agent.output_schema, ReasoningSteps)
            ):
                log_warning("Reasoning agent response model should be `ReasoningSteps`, continuing regular session...")
                return

            step_count = 1
            next_action = NextAction.CONTINUE
            reasoning_messages: List[Message] = []
            all_reasoning_steps: List[ReasoningStep] = []
            log_debug("Starting Reasoning", center=True, symbol="=")
            while next_action == NextAction.CONTINUE and step_count < self.reasoning_max_steps:
                log_debug(f"Step {step_count}", center=True, symbol="=")
                step_count += 1
                try:
                    # Run the reasoning agent
                    reasoning_agent_response: RunOutput = await reasoning_agent.arun(
                        input=run_messages.get_input_messages()
                    )
                    if reasoning_agent_response.content is None or reasoning_agent_response.messages is None:
                        log_warning("Reasoning error. Reasoning response is empty, continuing regular session...")
                        break

                    if isinstance(reasoning_agent_response.content, str):
                        log_warning(
                            "Reasoning error. Content is a string, not structured output. Continuing regular session..."
                        )
                        break

                    if reasoning_agent_response.content.reasoning_steps is None:
                        log_warning("Reasoning error. Reasoning steps are empty, continuing regular session...")
                        break

                    reasoning_steps: List[ReasoningStep] = reasoning_agent_response.content.reasoning_steps
                    all_reasoning_steps.extend(reasoning_steps)
                    # Yield reasoning steps
                    if stream_events:
                        for reasoning_step in reasoning_steps:
                            updated_reasoning_content = self._format_reasoning_step_content(
                                run_response=run_response,
                                reasoning_step=reasoning_step,
                            )

                            # Yield the response with the updated reasoning_content
                            yield handle_event(
                                create_reasoning_step_event(
                                    from_run_response=run_response,
                                    reasoning_step=reasoning_step,
                                    reasoning_content=updated_reasoning_content,
                                ),
                                run_response,
                                events_to_skip=self.events_to_skip,  # type: ignore
                                store_events=self.store_events,
                            )

                    # Find the index of the first assistant message
                    first_assistant_index = next(
                        (i for i, m in enumerate(reasoning_agent_response.messages) if m.role == "assistant"),
                        len(reasoning_agent_response.messages),
                    )
                    # Extract reasoning messages starting from the message after the first assistant message
                    reasoning_messages = reasoning_agent_response.messages[first_assistant_index:]

                    # Add reasoning step to the Agent's run_response
                    update_run_output_with_reasoning(
                        run_response=run_response,
                        reasoning_steps=reasoning_steps,
                        reasoning_agent_messages=reasoning_agent_response.messages,
                    )

                    # Get the next action
                    next_action = get_next_action(reasoning_steps[-1])
                    if next_action == NextAction.FINAL_ANSWER:
                        break
                except Exception as e:
                    log_error(f"Reasoning error: {e}")
                    break

            log_debug(f"Total Reasoning steps: {len(all_reasoning_steps)}")
            log_debug("Reasoning finished", center=True, symbol="=")

            # Update the messages_for_model to include reasoning messages
            update_messages_with_reasoning(
                run_messages=run_messages,
                reasoning_messages=reasoning_messages,
            )

            # Yield the final reasoning completed event
            if stream_events:
                yield handle_event(
                    create_reasoning_completed_event(
                        from_run_response=run_response,
                        content=ReasoningSteps(reasoning_steps=all_reasoning_steps),
                        content_type=ReasoningSteps.__name__,
                    ),
                    run_response,
                    events_to_skip=self.events_to_skip,  # type: ignore
                    store_events=self.store_events,
                )

    def _process_parser_response(
        self,
        model_response: ModelResponse,
        run_messages: RunMessages,
        parser_model_response: ModelResponse,
        messages_for_parser_model: list,
    ) -> None:
        """Common logic for processing parser model response."""
        parser_model_response_message: Optional[Message] = None
        for message in reversed(messages_for_parser_model):
            if message.role == "assistant":
                parser_model_response_message = message
                break

        if parser_model_response_message is not None:
            run_messages.messages.append(parser_model_response_message)
            model_response.parsed = parser_model_response.parsed
            model_response.content = parser_model_response.content
        else:
            log_warning("Unable to parse response with parser model")

    def _parse_response_with_parser_model(self, model_response: ModelResponse, run_messages: RunMessages) -> None:
        """Parse the model response using the parser model."""
        if self.parser_model is None:
            return

        if self.output_schema is not None:
            parser_response_format = self._get_response_format(self.parser_model)
            messages_for_parser_model = self._get_messages_for_parser_model(model_response, parser_response_format)
            parser_model_response: ModelResponse = self.parser_model.response(
                messages=messages_for_parser_model,
                response_format=parser_response_format,
            )
            self._process_parser_response(
                model_response,
                run_messages,
                parser_model_response,
                messages_for_parser_model,
            )
        else:
            log_warning("A response model is required to parse the response with a parser model")

    async def _aparse_response_with_parser_model(
        self, model_response: ModelResponse, run_messages: RunMessages
    ) -> None:
        """Parse the model response using the parser model."""
        if self.parser_model is None:
            return

        if self.output_schema is not None:
            parser_response_format = self._get_response_format(self.parser_model)
            messages_for_parser_model = self._get_messages_for_parser_model(model_response, parser_response_format)
            parser_model_response: ModelResponse = await self.parser_model.aresponse(
                messages=messages_for_parser_model,
                response_format=parser_response_format,
            )
            self._process_parser_response(
                model_response,
                run_messages,
                parser_model_response,
                messages_for_parser_model,
            )
        else:
            log_warning("A response model is required to parse the response with a parser model")

    def _parse_response_with_parser_model_stream(
        self, session: AgentSession, run_response: RunOutput, stream_events: bool = True
    ):
        """Parse the model response using the parser model"""
        if self.parser_model is not None:
            if self.output_schema is not None:
                if stream_events:
                    yield handle_event(
                        create_parser_model_response_started_event(run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )

                parser_model_response = ModelResponse(content="")
                parser_response_format = self._get_response_format(self.parser_model)
                messages_for_parser_model = self._get_messages_for_parser_model_stream(
                    run_response, parser_response_format
                )
                for model_response_event in self.parser_model.response_stream(
                    messages=messages_for_parser_model,
                    response_format=parser_response_format,
                    stream_model_response=False,
                ):
                    yield from self._handle_model_response_chunk(
                        session=session,
                        run_response=run_response,
                        model_response=parser_model_response,
                        model_response_event=model_response_event,
                        parse_structured_output=True,
                        stream_events=stream_events,
                    )

                parser_model_response_message: Optional[Message] = None
                for message in reversed(messages_for_parser_model):
                    if message.role == "assistant":
                        parser_model_response_message = message
                        break
                if parser_model_response_message is not None:
                    if run_response.messages is not None:
                        run_response.messages.append(parser_model_response_message)
                else:
                    log_warning("Unable to parse response with parser model")

                if stream_events:
                    yield handle_event(
                        create_parser_model_response_completed_event(run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )

            else:
                log_warning("A response model is required to parse the response with a parser model")

    async def _aparse_response_with_parser_model_stream(
        self, session: AgentSession, run_response: RunOutput, stream_events: bool = True
    ):
        """Parse the model response using the parser model stream."""
        if self.parser_model is not None:
            if self.output_schema is not None:
                if stream_events:
                    yield handle_event(
                        create_parser_model_response_started_event(run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )

                parser_model_response = ModelResponse(content="")
                parser_response_format = self._get_response_format(self.parser_model)
                messages_for_parser_model = self._get_messages_for_parser_model_stream(
                    run_response, parser_response_format
                )
                model_response_stream = self.parser_model.aresponse_stream(
                    messages=messages_for_parser_model,
                    response_format=parser_response_format,
                    stream_model_response=False,
                )
                async for model_response_event in model_response_stream:  # type: ignore
                    for event in self._handle_model_response_chunk(
                        session=session,
                        run_response=run_response,
                        model_response=parser_model_response,
                        model_response_event=model_response_event,
                        parse_structured_output=True,
                        stream_events=stream_events,
                    ):
                        yield event

                parser_model_response_message: Optional[Message] = None
                for message in reversed(messages_for_parser_model):
                    if message.role == "assistant":
                        parser_model_response_message = message
                        break
                if parser_model_response_message is not None:
                    if run_response.messages is not None:
                        run_response.messages.append(parser_model_response_message)
                else:
                    log_warning("Unable to parse response with parser model")

                if stream_events:
                    yield handle_event(
                        create_parser_model_response_completed_event(run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,  # type: ignore
                        store_events=self.store_events,
                    )
            else:
                log_warning("A response model is required to parse the response with a parser model")

    def _generate_response_with_output_model(self, model_response: ModelResponse, run_messages: RunMessages) -> None:
        """Parse the model response using the output model."""
        if self.output_model is None:
            return

        messages_for_output_model = self._get_messages_for_output_model(run_messages.messages)
        output_model_response: ModelResponse = self.output_model.response(messages=messages_for_output_model)
        model_response.content = output_model_response.content

    def _generate_response_with_output_model_stream(
        self,
        session: AgentSession,
        run_response: RunOutput,
        run_messages: RunMessages,
        stream_events: bool = False,
    ):
        """Parse the model response using the output model."""
        from agno.utils.events import (
            create_output_model_response_completed_event,
            create_output_model_response_started_event,
        )

        if self.output_model is None:
            return

        if stream_events:
            yield handle_event(
                create_output_model_response_started_event(run_response),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

        messages_for_output_model = self._get_messages_for_output_model(run_messages.messages)

        model_response = ModelResponse(content="")

        for model_response_event in self.output_model.response_stream(messages=messages_for_output_model):
            yield from self._handle_model_response_chunk(
                session=session,
                run_response=run_response,
                model_response=model_response,
                model_response_event=model_response_event,
                stream_events=stream_events,
            )

        if stream_events:
            yield handle_event(
                create_output_model_response_completed_event(run_response),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

        # Build a list of messages that should be added to the RunResponse
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        # Update the RunResponse messages
        run_response.messages = messages_for_run_response
        # Update the RunResponse metrics
        run_response.metrics = self._calculate_run_metrics(messages_for_run_response)

    async def _agenerate_response_with_output_model(self, model_response: ModelResponse, run_messages: RunMessages):
        """Parse the model response using the output model."""
        if self.output_model is None:
            return

        messages_for_output_model = self._get_messages_for_output_model(run_messages.messages)
        output_model_response: ModelResponse = await self.output_model.aresponse(messages=messages_for_output_model)
        model_response.content = output_model_response.content

    async def _agenerate_response_with_output_model_stream(
        self,
        session: AgentSession,
        run_response: RunOutput,
        run_messages: RunMessages,
        stream_events: bool = False,
    ):
        """Parse the model response using the output model."""
        from agno.utils.events import (
            create_output_model_response_completed_event,
            create_output_model_response_started_event,
        )

        if self.output_model is None:
            return

        if stream_events:
            yield handle_event(
                create_output_model_response_started_event(run_response),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

        messages_for_output_model = self._get_messages_for_output_model(run_messages.messages)

        model_response = ModelResponse(content="")

        model_response_stream = self.output_model.aresponse_stream(messages=messages_for_output_model)

        async for model_response_event in model_response_stream:
            for event in self._handle_model_response_chunk(
                session=session,
                run_response=run_response,
                model_response=model_response,
                model_response_event=model_response_event,
                stream_events=stream_events,
            ):
                yield event

        if stream_events:
            yield handle_event(
                create_output_model_response_completed_event(run_response),
                run_response,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

        # Build a list of messages that should be added to the RunResponse
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        # Update the RunResponse messages
        run_response.messages = messages_for_run_response
        # Update the RunResponse metrics
        run_response.metrics = self._calculate_run_metrics(messages_for_run_response)

    ###########################################################################
    # Default Tools
    ###########################################################################

    def _get_update_user_memory_function(self, user_id: Optional[str] = None, async_mode: bool = False) -> Function:
        def update_user_memory(task: str) -> str:
            """Use this function to submit a task to modify the Agent's memory.
            Describe the task in detail and be specific.
            The task can include adding a memory, updating a memory, deleting a memory, or clearing all memories.

            Args:
                task: The task to update the memory. Be specific and describe the task in detail.

            Returns:
                str: A string indicating the status of the task.
            """
            self.memory_manager = cast(MemoryManager, self.memory_manager)
            response = self.memory_manager.update_memory_task(task=task, user_id=user_id)

            return response

        async def aupdate_user_memory(task: str) -> str:
            """Use this function to update the Agent's memory of a user.
            Describe the task in detail and be specific.
            The task can include adding a memory, updating a memory, deleting a memory, or clearing all memories.

            Args:
                task: The task to update the memory. Be specific and describe the task in detail.

            Returns:
                str: A string indicating the status of the task.
            """
            self.memory_manager = cast(MemoryManager, self.memory_manager)
            response = await self.memory_manager.aupdate_memory_task(task=task, user_id=user_id)
            return response

        if async_mode:
            update_user_memory_function = aupdate_user_memory
        else:
            update_user_memory_function = update_user_memory  # type: ignore

        return Function.from_callable(update_user_memory_function, name="update_user_memory")

    def _get_update_cultural_knowledge_function(self, async_mode: bool = False) -> Function:
        def update_cultural_knowledge(task: str) -> str:
            """Use this function to update a cultural knowledge."""
            self.culture_manager = cast(CultureManager, self.culture_manager)
            response = self.culture_manager.update_culture_task(task=task)

            return response

        async def aupdate_cultural_knowledge(task: str) -> str:
            """Use this function to update a cultural knowledge asynchronously."""
            self.culture_manager = cast(CultureManager, self.culture_manager)
            response = await self.culture_manager.aupdate_culture_task(task=task)
            return response

        if async_mode:
            update_cultural_knowledge_function = aupdate_cultural_knowledge
        else:
            update_cultural_knowledge_function = update_cultural_knowledge  # type: ignore

        return Function.from_callable(
            update_cultural_knowledge_function,
            name="create_or_update_cultural_knowledge",
        )

    def _get_chat_history_function(self, session: AgentSession) -> Callable:
        def get_chat_history(num_chats: Optional[int] = None) -> str:
            """Use this function to get the chat history between the user and agent.

            Args:
                num_chats: The number of chats to return.
                    Each chat contains 2 messages. One from the user and one from the agent.
                    Default: None

            Returns:
                str: A JSON of a list of dictionaries representing the chat history.

            Example:
                - To get the last chat, use num_chats=1.
                - To get the last 5 chats, use num_chats=5.
                - To get all chats, use num_chats=None.
                - To get the first chat, use num_chats=None and pick the first message.
            """
            import json

            history: List[Dict[str, Any]] = []
            all_chats = session.get_messages_for_session()

            if len(all_chats) == 0:
                return ""

            for chat in all_chats[::-1]:  # type: ignore
                history.insert(0, chat.to_dict())  # type: ignore

            if num_chats is not None:
                history = history[:num_chats]

            return json.dumps(history)

        return get_chat_history

    def _get_tool_call_history_function(self, session: AgentSession) -> Callable:
        def get_tool_call_history(num_calls: int = 3) -> str:
            """Use this function to get the tools called by the agent in reverse chronological order.

            Args:
                num_calls: The number of tool calls to return.
                    Default: 3

            Returns:
                str: A JSON of a list of dictionaries representing the tool call history.

            Example:
                - To get the last tool call, use num_calls=1.
                - To get all tool calls, use num_calls=None.
            """
            import json

            tool_calls = session.get_tool_calls(num_calls=num_calls)
            if len(tool_calls) == 0:
                return ""
            return json.dumps(tool_calls)

        return get_tool_call_history

    def _update_session_state_tool(self, session_state, session_state_updates: dict) -> str:
        """
        Update the shared session state.  Provide any updates as a dictionary of key-value pairs.
        Example:
            "session_state_updates": {"shopping_list": ["milk", "eggs", "bread"]}

        Args:
            session_state_updates (dict): The updates to apply to the shared session state. Should be a dictionary of key-value pairs.
        """
        for key, value in session_state_updates.items():
            session_state[key] = value

        return f"Updated session state: {session_state}"

    def _get_search_knowledge_base_function(
        self,
        run_response: RunOutput,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        async_mode: bool = False,
    ) -> Function:
        """Factory function to create a search_knowledge_base function with filters."""

        def search_knowledge_base(query: str) -> str:
            """Use this function to search the knowledge base for information about a query.

            Args:
                query: The query to search for.

            Returns:
                str: A string containing the response from the knowledge base.
            """

            # Get the relevant documents from the knowledge base, passing filters
            retrieval_timer = Timer()
            retrieval_timer.start()
            docs_from_knowledge = self.get_relevant_docs_from_knowledge(query=query, filters=knowledge_filters)
            if docs_from_knowledge is not None:
                references = MessageReferences(
                    query=query,
                    references=docs_from_knowledge,
                    time=round(retrieval_timer.elapsed, 4),
                )
                # Add the references to the run_response
                if run_response.references is None:
                    run_response.references = []
                run_response.references.append(references)
            retrieval_timer.stop()
            from agno.utils.log import log_debug

            log_debug(f"Time to get references: {retrieval_timer.elapsed:.4f}s")

            if docs_from_knowledge is None:
                return "No documents found"
            return self._convert_documents_to_string(docs_from_knowledge)

        async def asearch_knowledge_base(query: str) -> str:
            """Use this function to search the knowledge base for information about a query asynchronously.

            Args:
                query: The query to search for.

            Returns:
                str: A string containing the response from the knowledge base.
            """
            retrieval_timer = Timer()
            retrieval_timer.start()
            docs_from_knowledge = await self.aget_relevant_docs_from_knowledge(query=query, filters=knowledge_filters)
            if docs_from_knowledge is not None:
                references = MessageReferences(
                    query=query,
                    references=docs_from_knowledge,
                    time=round(retrieval_timer.elapsed, 4),
                )
                if run_response.references is None:
                    run_response.references = []
                run_response.references.append(references)
            retrieval_timer.stop()
            log_debug(f"Time to get references: {retrieval_timer.elapsed:.4f}s")

            if docs_from_knowledge is None:
                return "No documents found"
            return self._convert_documents_to_string(docs_from_knowledge)

        if async_mode:
            search_knowledge_base_function = asearch_knowledge_base
        else:
            search_knowledge_base_function = search_knowledge_base  # type: ignore

        search_func = Function.from_callable(search_knowledge_base_function, name="search_knowledge_base")
        if self.search_knowledge_call_limit is not None:
            search_func.call_limit = self.search_knowledge_call_limit
        return search_func

    def _search_knowledge_base_with_agentic_filters_function(
        self,
        run_response: RunOutput,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        async_mode: bool = False,
    ) -> Function:
        """Factory function to create a search_knowledge_base function with filters."""

        def search_knowledge_base(query: str, filters: Optional[List[KnowledgeFilter]] = None) -> str:
            """Use this function to search the knowledge base for information about a query.

            Args:
                query: The query to search for.
                filters (optional): The filters to apply to the search. This is a list of KnowledgeFilter objects.

            Returns:
                str: A string containing the response from the knowledge base.
            """
            filters_dict = {filt.key: filt.value for filt in filters} if filters else None
            search_filters = get_agentic_or_user_search_filters(filters_dict, knowledge_filters)

            # Get the relevant documents from the knowledge base, passing filters
            retrieval_timer = Timer()
            retrieval_timer.start()
            docs_from_knowledge = self.get_relevant_docs_from_knowledge(query=query, filters=search_filters)
            if docs_from_knowledge is not None:
                references = MessageReferences(
                    query=query,
                    references=docs_from_knowledge,
                    time=round(retrieval_timer.elapsed, 4),
                )
                # Add the references to the run_response
                if run_response.references is None:
                    run_response.references = []
                run_response.references.append(references)
            retrieval_timer.stop()
            from agno.utils.log import log_debug

            log_debug(f"Time to get references: {retrieval_timer.elapsed:.4f}s")

            if docs_from_knowledge is None:
                return "No documents found"
            return self._convert_documents_to_string(docs_from_knowledge)

        async def asearch_knowledge_base(query: str, filters: Optional[List[KnowledgeFilter]] = None) -> str:
            """Use this function to search the knowledge base for information about a query asynchronously.

            Args:
                query: The query to search for.
                filters (optional): The filters to apply to the search. This is a list of KnowledgeFilter objects.

            Returns:
                str: A string containing the response from the knowledge base.
            """
            filters_dict = {filt.key: filt.value for filt in filters} if filters else None
            search_filters = get_agentic_or_user_search_filters(filters_dict, knowledge_filters)

            retrieval_timer = Timer()
            retrieval_timer.start()
            docs_from_knowledge = await self.aget_relevant_docs_from_knowledge(query=query, filters=search_filters)
            if docs_from_knowledge is not None:
                references = MessageReferences(
                    query=query,
                    references=docs_from_knowledge,
                    time=round(retrieval_timer.elapsed, 4),
                )
                if run_response.references is None:
                    run_response.references = []
                run_response.references.append(references)
            retrieval_timer.stop()
            log_debug(f"Time to get references: {retrieval_timer.elapsed:.4f}s")

            if docs_from_knowledge is None:
                return "No documents found"
            return self._convert_documents_to_string(docs_from_knowledge)

        if async_mode:
            search_knowledge_base_function = asearch_knowledge_base
        else:
            search_knowledge_base_function = search_knowledge_base  # type: ignore

        search_func = Function.from_callable(
            search_knowledge_base_function,
            name="search_knowledge_base_with_agentic_filters",
        )
        if self.search_knowledge_call_limit is not None:
            search_func.call_limit = self.search_knowledge_call_limit
        return search_func

    def add_to_knowledge(self, query: str, result: str) -> str:
        """Use this function to add information to the knowledge base for future use.

        Args:
            query: The query to add.
            result: The result of the query.

        Returns:
            str: A string indicating the status of the addition.
        """
        import json

        if self.knowledge is None:
            return "Knowledge not available"
        document_name = query.replace(" ", "_").replace("?", "").replace("!", "").replace(".", "")
        document_content = json.dumps({"query": query, "result": result})
        log_info(f"Adding document to Knowledge: {document_name}: {document_content}")
        import asyncio

        from agno.knowledge.reader.text_reader import TextReader

        asyncio.run(
            self.knowledge.add_content_async(name=document_name, text_content=document_content, reader=TextReader())
        )
        return "Successfully added to knowledge base"

    def _get_previous_sessions_messages_function(
        self, num_history_sessions: Optional[int] = 2, user_id: Optional[str] = None
    ) -> Callable:
        """Factory function to create a get_previous_session_messages function.

        Args:
            num_history_sessions: The last n sessions to be taken from db
            user_id: The user ID to filter sessions by

        Returns:
            Callable: A function that retrieves messages from previous sessions
        """

        def get_previous_session_messages() -> str:
            """Use this function to retrieve messages from previous chat sessions.
            USE THIS TOOL ONLY WHEN THE QUESTION IS EITHER "What was my last conversation?" or "What was my last question?" and similar to it.

            Returns:
                str: JSON formatted list of message pairs from previous sessions
            """
            # TODO: Review and Test this function
            import json

            if self.db is None:
                return "Previous session messages not available"

            self.db = cast(BaseDb, self.db)

            selected_sessions = self.db.get_sessions(
                session_type=SessionType.AGENT,
                limit=num_history_sessions,
                user_id=user_id,
            )

            all_messages = []
            seen_message_pairs = set()

            for session in selected_sessions:
                if isinstance(session, AgentSession) and session.runs:
                    message_count = 0
                    for run in session.runs:
                        messages = run.messages
                        if messages is not None:
                            for i in range(0, len(messages) - 1, 2):
                                if i + 1 < len(messages):
                                    try:
                                        user_msg = messages[i]
                                        assistant_msg = messages[i + 1]
                                        user_content = user_msg.content
                                        assistant_content = assistant_msg.content
                                        if user_content is None or assistant_content is None:
                                            continue  # Skip this pair if either message has no content

                                        msg_pair_id = f"{user_content}:{assistant_content}"
                                        if msg_pair_id not in seen_message_pairs:
                                            seen_message_pairs.add(msg_pair_id)
                                            all_messages.append(Message.model_validate(user_msg))
                                            all_messages.append(Message.model_validate(assistant_msg))
                                            message_count += 1
                                    except Exception as e:
                                        log_warning(f"Error processing message pair: {e}")
                                        continue

            return json.dumps([msg.to_dict() for msg in all_messages]) if all_messages else "No history found"

        return get_previous_session_messages

    async def _aget_previous_sessions_messages_function(
        self, num_history_sessions: Optional[int] = 2, user_id: Optional[str] = None
    ) -> Function:
        """Factory function to create a get_previous_session_messages function.

        Args:
            num_history_sessions: The last n sessions to be taken from db
            user_id: The user ID to filter sessions by
        Returns:
            Callable: A function that retrieves messages from previous sessions
        """

        async def aget_previous_session_messages() -> str:
            """Use this function to retrieve messages from previous chat sessions.
            USE THIS TOOL ONLY WHEN THE QUESTION IS EITHER "What was my last conversation?" or "What was my last question?" and similar to it.

            Returns:
                str: JSON formatted list of message pairs from previous sessions
            """
            # TODO: Review and Test this function
            import json

            if self.db is None:
                return "Previous session messages not available"

            if self._has_async_db():
                selected_sessions = await self.db.get_sessions(  # type: ignore
                    session_type=SessionType.AGENT,
                    limit=num_history_sessions,
                    user_id=user_id,
                    sort_by="created_at",
                    sort_order="desc",
                )
            else:
                selected_sessions = self.db.get_sessions(
                    session_type=SessionType.AGENT,
                    limit=num_history_sessions,
                    user_id=user_id,
                    sort_by="created_at",
                    sort_order="desc",
                )

            all_messages = []
            seen_message_pairs = set()

            for session in selected_sessions:
                if isinstance(session, AgentSession) and session.runs:
                    message_count = 0
                    for run in session.runs:
                        messages = run.messages
                        if messages is not None:
                            for i in range(0, len(messages) - 1, 2):
                                if i + 1 < len(messages):
                                    try:
                                        user_msg = messages[i]
                                        assistant_msg = messages[i + 1]
                                        user_content = user_msg.content
                                        assistant_content = assistant_msg.content
                                        if user_content is None or assistant_content is None:
                                            continue  # Skip this pair if either message has no content

                                        msg_pair_id = f"{user_content}:{assistant_content}"
                                        if msg_pair_id not in seen_message_pairs:
                                            seen_message_pairs.add(msg_pair_id)
                                            all_messages.append(Message.model_validate(user_msg))
                                            all_messages.append(Message.model_validate(assistant_msg))
                                            message_count += 1
                                    except Exception as e:
                                        log_warning(f"Error processing message pair: {e}")
                                        continue

            return json.dumps([msg.to_dict() for msg in all_messages]) if all_messages else "No history found"

        return Function.from_callable(aget_previous_session_messages, name="get_previous_session_messages")

    ###########################################################################
    # Print Response
    ###########################################################################

    def print_response(
        self,
        input: Union[List, Dict, str, Message, BaseModel, List[Message]],
        *,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        markdown: Optional[bool] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        add_session_state_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        show_message: bool = True,
        show_reasoning: bool = True,
        show_full_reasoning: bool = False,
        console: Optional[Any] = None,
        # Add tags to include in markdown content
        tags_to_include_in_markdown: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> None:
        if self._has_async_db():
            raise Exception(
                "This method is not supported with an async DB. Please use the async version of this method."
            )

        if not tags_to_include_in_markdown:
            tags_to_include_in_markdown = {"think", "thinking"}

        if markdown is None:
            markdown = self.markdown

        if self.output_schema is not None:
            markdown = False

        # Use stream override value when necessary
        if stream is None:
            stream = False if self.stream is None else self.stream

        # Considering both stream_events and stream_intermediate_steps (deprecated)
        stream_events = stream_events or stream_intermediate_steps

        # Can't stream events if streaming is disabled
        if stream is False:
            stream_events = False

        if stream_events is None:
            stream_events = False if self.stream_events is None else self.stream_events

        if stream:
            print_response_stream(
                agent=self,
                input=input,
                session_id=session_id,
                session_state=session_state,
                user_id=user_id,
                audio=audio,
                images=images,
                videos=videos,
                files=files,
                stream_events=stream_events,
                knowledge_filters=knowledge_filters,
                debug_mode=debug_mode,
                markdown=markdown,
                show_message=show_message,
                show_reasoning=show_reasoning,
                show_full_reasoning=show_full_reasoning,
                tags_to_include_in_markdown=tags_to_include_in_markdown,
                console=console,
                add_history_to_context=add_history_to_context,
                dependencies=dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=metadata,
                **kwargs,
            )

        else:
            print_response(
                agent=self,
                input=input,
                session_id=session_id,
                session_state=session_state,
                user_id=user_id,
                audio=audio,
                images=images,
                videos=videos,
                files=files,
                stream_events=stream_events,
                knowledge_filters=knowledge_filters,
                debug_mode=debug_mode,
                markdown=markdown,
                show_message=show_message,
                show_reasoning=show_reasoning,
                show_full_reasoning=show_full_reasoning,
                tags_to_include_in_markdown=tags_to_include_in_markdown,
                console=console,
                add_history_to_context=add_history_to_context,
                dependencies=dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=metadata,
                **kwargs,
            )

    async def aprint_response(
        self,
        input: Union[List, Dict, str, Message, BaseModel, List[Message]],
        *,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        markdown: Optional[bool] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        show_message: bool = True,
        show_reasoning: bool = True,
        show_full_reasoning: bool = False,
        console: Optional[Any] = None,
        # Add tags to include in markdown content
        tags_to_include_in_markdown: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> None:
        if not tags_to_include_in_markdown:
            tags_to_include_in_markdown = {"think", "thinking"}

        if markdown is None:
            markdown = self.markdown

        if self.output_schema is not None:
            markdown = False

        if stream is None:
            stream = self.stream or False

        # Considering both stream_events and stream_intermediate_steps (deprecated)
        stream_events = stream_events or stream_intermediate_steps

        # Can't stream events if streaming is disabled
        if stream is False:
            stream_events = False

        if stream_events is None:
            stream_events = False if self.stream_events is None else self.stream_events

        if stream:
            await aprint_response_stream(
                agent=self,
                input=input,
                session_id=session_id,
                session_state=session_state,
                user_id=user_id,
                audio=audio,
                images=images,
                videos=videos,
                files=files,
                stream_events=stream_events,
                knowledge_filters=knowledge_filters,
                debug_mode=debug_mode,
                markdown=markdown,
                show_message=show_message,
                show_reasoning=show_reasoning,
                show_full_reasoning=show_full_reasoning,
                tags_to_include_in_markdown=tags_to_include_in_markdown,
                console=console,
                add_history_to_context=add_history_to_context,
                dependencies=dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=metadata,
                **kwargs,
            )
        else:
            await aprint_response(
                agent=self,
                input=input,
                session_id=session_id,
                session_state=session_state,
                user_id=user_id,
                audio=audio,
                images=images,
                videos=videos,
                files=files,
                knowledge_filters=knowledge_filters,
                debug_mode=debug_mode,
                markdown=markdown,
                show_message=show_message,
                show_reasoning=show_reasoning,
                show_full_reasoning=show_full_reasoning,
                tags_to_include_in_markdown=tags_to_include_in_markdown,
                console=console,
                add_history_to_context=add_history_to_context,
                dependencies=dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=metadata,
                **kwargs,
            )

    def _update_reasoning_content_from_tool_call(
        self, run_response: RunOutput, tool_name: str, tool_args: Dict[str, Any]
    ) -> Optional[ReasoningStep]:
        """Update reasoning_content based on tool calls that look like thinking or reasoning tools."""

        # Case 1: ReasoningTools.think (has title, thought, optional action and confidence)
        if tool_name.lower() == "think" and "title" in tool_args and "thought" in tool_args:
            title = tool_args["title"]
            thought = tool_args["thought"]
            action = tool_args.get("action", "")
            confidence = tool_args.get("confidence", None)

            # Create a reasoning step
            reasoning_step = ReasoningStep(
                title=title,
                reasoning=thought,
                action=action,
                next_action=NextAction.CONTINUE,
                confidence=confidence,
            )

            # Add the step to the run response
            add_reasoning_step_to_metadata(run_response=run_response, reasoning_step=reasoning_step)

            formatted_content = f"## {title}\n{thought}\n"
            if action:
                formatted_content += f"Action: {action}\n"
            if confidence is not None:
                formatted_content += f"Confidence: {confidence}\n"
            formatted_content += "\n"

            append_to_reasoning_content(run_response=run_response, content=formatted_content)
            return reasoning_step

        # Case 2: ReasoningTools.analyze (has title, result, analysis, optional next_action and confidence)
        elif tool_name.lower() == "analyze" and "title" in tool_args:
            title = tool_args["title"]
            result = tool_args.get("result", "")
            analysis = tool_args.get("analysis", "")
            next_action = tool_args.get("next_action", "")
            confidence = tool_args.get("confidence", None)

            # Map string next_action to enum
            next_action_enum = NextAction.CONTINUE
            if next_action.lower() == "validate":
                next_action_enum = NextAction.VALIDATE
            elif next_action.lower() in ["final", "final_answer", "finalize"]:
                next_action_enum = NextAction.FINAL_ANSWER

            # Create a reasoning step
            reasoning_step = ReasoningStep(
                title=title,
                result=result,
                reasoning=analysis,
                next_action=next_action_enum,
                confidence=confidence,
            )

            # Add the step to the run response
            add_reasoning_step_to_metadata(run_response=run_response, reasoning_step=reasoning_step)

            formatted_content = f"## {title}\n"
            if result:
                formatted_content += f"Result: {result}\n"
            if analysis:
                formatted_content += f"{analysis}\n"
            if next_action and next_action.lower() != "continue":
                formatted_content += f"Next Action: {next_action}\n"
            if confidence is not None:
                formatted_content += f"Confidence: {confidence}\n"
            formatted_content += "\n"

            append_to_reasoning_content(run_response=run_response, content=formatted_content)
            return reasoning_step

        # Case 3: ReasoningTool.think (simple format, just has 'thought')
        elif tool_name.lower() == "think" and "thought" in tool_args:
            thought = tool_args["thought"]
            reasoning_step = ReasoningStep(
                title="Thinking",
                reasoning=thought,
                confidence=None,
            )
            formatted_content = f"## Thinking\n{thought}\n\n"
            add_reasoning_step_to_metadata(run_response=run_response, reasoning_step=reasoning_step)
            append_to_reasoning_content(run_response=run_response, content=formatted_content)
            return reasoning_step

        return None

    def _get_effective_filters(self, knowledge_filters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Determine which knowledge filters to use, with priority to run-level filters.

        Args:
            knowledge_filters: Filters passed at run time

        Returns:
            The effective filters to use, with run-level filters taking priority
        """
        effective_filters = None

        # If agent has filters, use those as a base
        if self.knowledge_filters:
            effective_filters = self.knowledge_filters.copy()

        # If run has filters, they override agent filters
        if knowledge_filters:
            if effective_filters:
                # Merge filters, with run filters taking priority
                effective_filters.update(knowledge_filters)
            else:
                effective_filters = knowledge_filters

        if effective_filters:
            log_debug(f"Using knowledge filters: {effective_filters}")

        return effective_filters

    def _cleanup_and_store(self, run_response: RunOutput, session: AgentSession, user_id: Optional[str] = None) -> None:
        #  Scrub the stored run based on storage flags
        self._scrub_run_output_for_storage(run_response)

        # Stop the timer for the Run duration
        if run_response.metrics:
            run_response.metrics.stop_timer()

        # Optional: Save output to file if save_response_to_file is set
        self.save_run_response_to_file(
            run_response=run_response,
            input=run_response.input.input_content_string() if run_response.input else "",
            session_id=session.session_id,
            user_id=user_id,
        )

        # Add RunOutput to Agent Session
        session.upsert_run(run=run_response)

        # Calculate session metrics
        self._update_session_metrics(session=session, run_response=run_response)

        # Save session to memory
        self.save_session(session=session)

    async def _acleanup_and_store(
        self, run_response: RunOutput, session: AgentSession, user_id: Optional[str] = None
    ) -> None:
        #  Scrub the stored run based on storage flags
        self._scrub_run_output_for_storage(run_response)

        # Stop the timer for the Run duration
        if run_response.metrics:
            run_response.metrics.stop_timer()

        # Optional: Save output to file if save_response_to_file is set
        self.save_run_response_to_file(
            run_response=run_response,
            input=run_response.input.input_content_string() if run_response.input else "",
            session_id=session.session_id,
            user_id=user_id,
        )

        # Add RunOutput to Agent Session
        session.upsert_run(run=run_response)

        # Calculate session metrics
        self._update_session_metrics(session=session, run_response=run_response)

        # Save session to storage
        await self.asave_session(session=session)

    def _scrub_run_output_for_storage(self, run_response: RunOutput) -> None:
        """
        Scrub run output based on storage flags before persisting to database.
        """
        if not self.store_media:
            scrub_media_from_run_output(run_response)

        if not self.store_tool_messages:
            scrub_tool_results_from_run_output(run_response)

        if not self.store_history_messages:
            scrub_history_messages_from_run_output(run_response)

    def _validate_media_object_id(
        self,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        audios: Optional[Sequence[Audio]] = None,
        files: Optional[Sequence[File]] = None,
    ) -> tuple:
        """Convert raw Image/Video/Audio objects - now unified, so just return as-is."""
        # With unified classes, no conversion needed - just ensure IDs are set
        image_list = None
        if images:
            image_list = []
            for img in images:
                # Ensure ID is set (validation should handle this, but double-check)
                if not img.id:
                    from uuid import uuid4

                    img.id = str(uuid4())
                image_list.append(img)

        video_list = None
        if videos:
            video_list = []
            for vid in videos:
                if not vid.id:
                    from uuid import uuid4

                    vid.id = str(uuid4())
                video_list.append(vid)

        audio_list = None
        if audios:
            audio_list = []
            for aud in audios:
                if not aud.id:
                    from uuid import uuid4

                    aud.id = str(uuid4())
                audio_list.append(aud)

        file_list = None
        if files:
            file_list = []
            for file in files:
                if not file.id:
                    from uuid import uuid4

                    file.id = str(uuid4())
                file_list.append(file)

        return image_list, video_list, audio_list, file_list

    def cli_app(
        self,
        input: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user: str = "User",
        emoji: str = ":sunglasses:",
        stream: bool = False,
        markdown: bool = False,
        exit_on: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run an interactive command-line interface to interact with the agent."""

        from inspect import isawaitable

        from rich.prompt import Prompt

        # Ensuring the agent is not using our async MCP tools
        if self.tools is not None:
            for tool in self.tools:
                if isawaitable(tool):
                    raise NotImplementedError("Use `acli_app` to use async tools.")
                if tool.__class__.__name__ in ["MCPTools", "MultiMCPTools"]:
                    raise NotImplementedError("Use `acli_app` to use MCP tools.")

        if input:
            self.print_response(
                input=input,
                stream=stream,
                markdown=markdown,
                user_id=user_id,
                session_id=session_id,
                **kwargs,
            )

        _exit_on = exit_on or ["exit", "quit", "bye"]
        while True:
            message = Prompt.ask(f"[bold] {emoji} {user} [/bold]")
            if message in _exit_on:
                break

            self.print_response(
                input=message,
                stream=stream,
                markdown=markdown,
                user_id=user_id,
                session_id=session_id,
                **kwargs,
            )

    async def acli_app(
        self,
        input: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user: str = "User",
        emoji: str = ":sunglasses:",
        stream: bool = False,
        markdown: bool = False,
        exit_on: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Run an interactive command-line interface to interact with the agent.
        Works with agent dependencies requiring async logic.
        """
        from rich.prompt import Prompt

        if input:
            await self.aprint_response(
                input=input,
                stream=stream,
                markdown=markdown,
                user_id=user_id,
                session_id=session_id,
                **kwargs,
            )

        _exit_on = exit_on or ["exit", "quit", "bye"]
        while True:
            message = Prompt.ask(f"[bold] {emoji} {user} [/bold]")
            if message in _exit_on:
                break

            await self.aprint_response(
                input=message,
                stream=stream,
                markdown=markdown,
                user_id=user_id,
                session_id=session_id,
                **kwargs,
            )

    ###########################################################################
    # Api functions
    ###########################################################################

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """Get the telemetry data for the agent"""
        return {
            "agent_id": self.id,
            "db_type": self.db.__class__.__name__ if self.db else None,
            "model_provider": self.model.provider if self.model else None,
            "model_name": self.model.name if self.model else None,
            "model_id": self.model.id if self.model else None,
            "parser_model": self.parser_model.to_dict() if self.parser_model else None,
            "output_model": self.output_model.to_dict() if self.output_model else None,
            "has_tools": self.tools is not None,
            "has_memory": self.enable_user_memories is True
            or self.enable_agentic_memory is True
            or self.memory_manager is not None,
            "has_culture": self.enable_agentic_culture is True
            or self.update_cultural_knowledge is True
            or self.culture_manager is not None,
            "has_reasoning": self.reasoning is True,
            "has_knowledge": self.knowledge is not None,
            "has_input_schema": self.input_schema is not None,
            "has_output_schema": self.output_schema is not None,
            "has_team": self.team_id is not None,
        }

    def _log_agent_telemetry(self, session_id: str, run_id: Optional[str] = None) -> None:
        """Send a telemetry event to the API for a created Agent run"""

        self._set_telemetry()
        if not self.telemetry:
            return

        from agno.api.agent import AgentRunCreate, create_agent_run

        try:
            create_agent_run(
                run=AgentRunCreate(
                    session_id=session_id,
                    run_id=run_id,
                    data=self._get_telemetry_data(),
                ),
            )
        except Exception as e:
            log_debug(f"Could not create Agent run telemetry event: {e}")

    async def _alog_agent_telemetry(self, session_id: str, run_id: Optional[str] = None) -> None:
        """Send a telemetry event to the API for a created Agent async run"""

        self._set_telemetry()
        if not self.telemetry:
            return

        from agno.api.agent import AgentRunCreate, acreate_agent_run

        try:
            await acreate_agent_run(
                run=AgentRunCreate(
                    session_id=session_id,
                    run_id=run_id,
                    data=self._get_telemetry_data(),
                )
            )

        except Exception as e:
            log_debug(f"Could not create Agent run telemetry event: {e}")
