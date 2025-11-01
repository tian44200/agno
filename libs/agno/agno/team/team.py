import asyncio
import contextlib
import json
from collections import ChainMap, deque
from copy import copy
from dataclasses import dataclass
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

from agno.agent import Agent
from agno.db.base import AsyncBaseDb, BaseDb, SessionType, UserMemory
from agno.exceptions import (
    InputCheckError,
    ModelProviderError,
    OutputCheckError,
    RunCancelledException,
)
from agno.guardrails import BaseGuardrail
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.types import KnowledgeFilter
from agno.media import Audio, File, Image, Video
from agno.memory import MemoryManager
from agno.models.base import Model
from agno.models.message import Message, MessageReferences
from agno.models.metrics import Metrics
from agno.models.response import ModelResponse, ModelResponseEvent
from agno.models.utils import get_model
from agno.reasoning.step import NextAction, ReasoningStep, ReasoningSteps
from agno.run.agent import RunEvent, RunOutput, RunOutputEvent
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
from agno.run.team import TeamRunEvent, TeamRunInput, TeamRunOutput, TeamRunOutputEvent
from agno.session import SessionSummaryManager, TeamSession, WorkflowSession
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
    create_team_parser_model_response_completed_event,
    create_team_parser_model_response_started_event,
    create_team_post_hook_completed_event,
    create_team_post_hook_started_event,
    create_team_pre_hook_completed_event,
    create_team_pre_hook_started_event,
    create_team_reasoning_completed_event,
    create_team_reasoning_started_event,
    create_team_reasoning_step_event,
    create_team_run_cancelled_event,
    create_team_run_completed_event,
    create_team_run_content_completed_event,
    create_team_run_error_event,
    create_team_run_output_content_event,
    create_team_run_started_event,
    create_team_session_summary_completed_event,
    create_team_session_summary_started_event,
    create_team_tool_call_completed_event,
    create_team_tool_call_started_event,
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
    use_agent_logger,
    use_team_logger,
)
from agno.utils.merge_dict import merge_dictionaries
from agno.utils.message import filter_tool_calls, get_text_from_message
from agno.utils.print_response.team import (
    aprint_response,
    aprint_response_stream,
    print_response,
    print_response_stream,
)
from agno.utils.reasoning import (
    add_reasoning_metrics_to_metadata,
    add_reasoning_step_to_metadata,
    append_to_reasoning_content,
    update_run_output_with_reasoning,
)
from agno.utils.response import (
    async_generator_wrapper,
    check_if_run_cancelled,
    generator_wrapper,
)
from agno.utils.safe_formatter import SafeFormatter
from agno.utils.string import generate_id_from_name, parse_response_model_str
from agno.utils.team import (
    add_interaction_to_team_run_context,
    format_member_agent_task,
    get_member_id,
    get_team_member_interactions_str,
    get_team_run_context_audio,
    get_team_run_context_files,
    get_team_run_context_images,
    get_team_run_context_videos,
)
from agno.utils.timer import Timer


@dataclass(init=False)
class Team:
    """
    A class representing a team of agents.
    """

    members: List[Union[Agent, "Team"]]

    # Model for this Team
    model: Optional[Model] = None

    # --- Team settings ---
    # Team UUID (autogenerated if not set)
    id: Optional[str] = None
    # Name of the team
    name: Optional[str] = None
    # If this team is part of a team itself, this is the role of the team
    role: Optional[str] = None

    # --- If this Team is part of a team itself ---
    # If this team is part of a team itself, this is the ID of the parent team. This is set automatically.
    parent_team_id: Optional[str] = None

    # --- If this Team is part of a workflow ---
    # Optional workflow ID. Indicates this team is part of a workflow. This is set automatically.
    workflow_id: Optional[str] = None

    # --- Team execution settings ---
    # If True, the team leader won't process responses from the members and instead will return them directly
    # Should not be used in combination with delegate_task_to_all_members
    respond_directly: bool = False
    # If True, the team leader will delegate the task to all members, instead of deciding for a subset
    delegate_task_to_all_members: bool = False
    # Set to false if you want to send the run input directly to the member agents
    determine_input_for_members: bool = True

    # --- User settings ---
    # Default user ID for this team
    user_id: Optional[str] = None

    # --- Session settings ---
    # Default Session ID for this team (autogenerated if not set)
    session_id: Optional[str] = None
    # Session state (stored in the database to persist across runs)
    session_state: Optional[Dict[str, Any]] = None
    # Set to True to add the session_state to the context
    add_session_state_to_context: bool = False
    # Set to True to give the team tools to update the session_state dynamically
    enable_agentic_state: bool = False
    # Set to True to overwrite the stored session_state with the session_state provided in the run
    overwrite_db_session_state: bool = False
    # If True, cache the current Team session in memory for faster access
    cache_session: bool = False

    # Add this flag to control if the workflow should send the team history to the members. This means sending the team-level history to the members, not the agent-level history.
    add_team_history_to_members: bool = False
    # Number of historical runs to include in the messages sent to the members
    num_team_history_runs: int = 3
    # If True, send all member interactions (request/response) during the current run to members that have been delegated a task to
    share_member_interactions: bool = False

    # If True, adds a tool to allow searching through previous sessions
    search_session_history: Optional[bool] = False
    # Number of past sessions to include in the search
    num_history_sessions: Optional[int] = None

    # If True, adds a tool to allow the team to read the team history (this is deprecated and will be removed in a future version)
    read_team_history: bool = False
    # If True, adds a tool to allow the team to read the chat history
    read_chat_history: bool = False

    # --- System message settings ---
    # A description of the Team that is added to the start of the system message.
    description: Optional[str] = None
    # List of instructions for the team.
    instructions: Optional[Union[str, List[str], Callable]] = None
    # Provide the expected output from the Team.
    expected_output: Optional[str] = None
    # Additional context added to the end of the system message.
    additional_context: Optional[str] = None
    # If markdown=true, add instructions to format the output using markdown
    markdown: bool = False
    # If True, add the current datetime to the instructions to give the team a sense of time
    # This allows for relative times like "tomorrow" to be used in the prompt
    add_datetime_to_context: bool = False
    # If True, add the current location to the instructions to give the team a sense of location
    add_location_to_context: bool = False
    # Allows for custom timezone for datetime instructions following the TZ Database format (e.g. "Etc/UTC")
    timezone_identifier: Optional[str] = None
    # If True, add the team name to the instructions
    add_name_to_context: bool = False
    # If True, add the tools available to team members to the context
    add_member_tools_to_context: bool = True

    # Provide the system message as a string or function
    system_message: Optional[Union[str, Callable, Message]] = None
    # Role for the system message
    system_message_role: str = "system"

    # If True, resolve the session_state, dependencies, and metadata in the user and system messages
    resolve_in_context: bool = True

    # --- Extra Messages ---
    # A list of extra messages added after the system message and before the user message.
    # Use these for few-shot learning or to provide additional context to the Model.
    # Note: these are not retained in memory, they are added directly to the messages sent to the model.
    additional_input: Optional[List[Union[str, Dict, BaseModel, Message]]] = None

    # --- Database ---
    # Database to use for this agent
    db: Optional[Union[BaseDb, AsyncBaseDb]] = None

    # Memory manager to use for this agent
    memory_manager: Optional[MemoryManager] = None

    # --- User provided dependencies ---
    # User provided dependencies
    dependencies: Optional[Dict[str, Any]] = None
    # If True, add the dependencies to the user prompt
    add_dependencies_to_context: bool = False

    # --- Agent Knowledge ---
    knowledge: Optional[Knowledge] = None
    # Add knowledge_filters to the Agent class attributes
    knowledge_filters: Optional[Dict[str, Any]] = None
    # Let the agent choose the knowledge filters
    enable_agentic_knowledge_filters: Optional[bool] = False
    # Add a tool that allows the Team to update Knowledge.
    update_knowledge: bool = False
    # If True, add references to the user prompt
    add_knowledge_to_context: bool = False
    # Retrieval function to get references
    # This function, if provided, is used instead of the default search_knowledge function
    # Signature:
    # def knowledge_retriever(team: Team, query: str, num_documents: Optional[int], **kwargs) -> Optional[list[dict]]:
    #     ...
    knowledge_retriever: Optional[Callable[..., Optional[List[Union[Dict, str]]]]] = None
    references_format: Literal["json", "yaml"] = "json"

    # --- Tools ---
    # If True, add a tool to get information about the team members
    get_member_information_tool: bool = False
    # Add a tool to search the knowledge base (aka Agentic RAG)
    # Only added if knowledge is provided.
    search_knowledge: bool = True

    # If False, media (images, videos, audio, files) is only available to tools and not sent to the LLM
    send_media_to_model: bool = True
    # If True, store media in run output
    store_media: bool = True
    # If True, store tool results in run output
    store_tool_messages: bool = True
    # If True, store history messages in run output
    store_history_messages: bool = True

    # --- Team Tools ---
    # A list of tools provided to the Model.
    # Tools are functions the model may generate JSON inputs for.
    tools: Optional[List[Union[Toolkit, Callable, Function, Dict]]] = None

    # Controls which (if any) tool is called by the team model.
    # "none" means the model will not call a tool and instead generates a message.
    # "auto" means the model can pick between generating a message or calling a tool.
    # Specifying a particular function via {"type: "function", "function": {"name": "my_function"}}
    #   forces the model to call that tool.
    # "none" is the default when no tools are present. "auto" is the default if tools are present.
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # Maximum number of tool calls allowed.
    tool_call_limit: Optional[int] = None
    # A list of hooks to be called before and after the tool call
    tool_hooks: Optional[List[Callable]] = None

    # --- Team Hooks ---
    # Functions called right after team session is loaded, before processing starts
    pre_hooks: Optional[Union[List[Callable[..., Any]], List[BaseGuardrail]]] = None
    # Functions called after output is generated but before the response is returned
    post_hooks: Optional[Union[List[Callable[..., Any]], List[BaseGuardrail]]] = None

    # --- Structured output ---
    # Input schema for validating input
    input_schema: Optional[Type[BaseModel]] = None
    # Output schema for the team response
    output_schema: Optional[Type[BaseModel]] = None
    # Provide a secondary model to parse the response from the primary model
    parser_model: Optional[Model] = None
    # Provide a prompt for the parser model
    parser_model_prompt: Optional[str] = None
    # Provide an output model to parse the response from the team
    output_model: Optional[Model] = None
    # Provide a prompt for the output model
    output_model_prompt: Optional[str] = None
    # If `output_schema` is set, sets the response mode of the model, i.e. if the model should explicitly respond with a JSON object instead of a Pydantic model
    use_json_mode: bool = False
    # If True, parse the response
    parse_response: bool = True

    # --- History ---
    # Enable the agent to manage memories of the user
    enable_agentic_memory: bool = False
    # If True, the agent creates/updates user memories at the end of runs
    enable_user_memories: bool = False
    # If True, the agent adds a reference to the user memories in the response
    add_memories_to_context: Optional[bool] = None
    # If True, the agent creates/updates session summaries at the end of runs
    enable_session_summaries: bool = False
    # # Session summary model
    # session_summary_model: Optional[Model] = None
    # # Session summary prompt
    # session_summary_prompt: Optional[str] = None
    session_summary_manager: Optional[SessionSummaryManager] = None
    # If True, the team adds session summaries to the context
    add_session_summary_to_context: Optional[bool] = None

    # --- Team History ---
    # add_history_to_context=true adds messages from the chat history to the messages list sent to the Model.
    add_history_to_context: bool = False
    # Number of historical runs to include in the messages
    num_history_runs: Optional[int] = None
    # Number of historical messages to include in the messages list sent to the Model.
    num_history_messages: Optional[int] = None
    # Maximum number of tool calls to include from history (None = no limit)
    max_tool_calls_from_history: Optional[int] = None

    # --- Team Storage ---
    # Metadata stored with this team
    metadata: Optional[Dict[str, Any]] = None

    # --- Team Reasoning ---
    reasoning: bool = False
    reasoning_model: Optional[Model] = None
    reasoning_agent: Optional[Agent] = None
    reasoning_min_steps: int = 1
    reasoning_max_steps: int = 10

    # --- Team Streaming ---
    # Stream the response from the Team
    stream: Optional[bool] = None
    # Stream the intermediate steps from the Agent
    stream_events: Optional[bool] = None
    # [Deprecated] Stream the intermediate steps from the Agent
    stream_intermediate_steps: Optional[bool] = None
    # Stream the member events from the Team
    stream_member_events: bool = True

    # Store the events from the Team
    store_events: bool = False
    # List of events to skip from the Team
    events_to_skip: Optional[List[Union[RunEvent, TeamRunEvent]]] = None
    # Store member agent runs inside the team's RunOutput
    store_member_responses: bool = False

    # --- Debug ---
    # Enable debug logs
    debug_mode: bool = False
    # Debug level: 1 = basic, 2 = detailed
    debug_level: Literal[1, 2] = 1
    # Enable member logs - Sets the debug_mode for team and members
    show_members_responses: bool = False

    # --- Team Response Settings ---
    # Number of retries to attempt
    retries: int = 0
    # Delay between retries (in seconds)
    delay_between_retries: int = 1
    # Exponential backoff: if True, the delay between retries is doubled each time
    exponential_backoff: bool = False

    # --- Telemetry ---
    # telemetry=True logs minimal telemetry for analytics
    # This helps us improve the Teams implementation and provide better support
    telemetry: bool = True

    def __init__(
        self,
        members: List[Union[Agent, "Team"]],
        id: Optional[str] = None,
        model: Optional[Union[Model, str]] = None,
        name: Optional[str] = None,
        role: Optional[str] = None,
        respond_directly: bool = False,
        determine_input_for_members: bool = True,
        delegate_task_to_all_members: bool = False,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        add_session_state_to_context: bool = False,
        enable_agentic_state: bool = False,
        overwrite_db_session_state: bool = False,
        resolve_in_context: bool = True,
        cache_session: bool = False,
        add_team_history_to_members: bool = False,
        num_team_history_runs: int = 3,
        search_session_history: Optional[bool] = False,
        num_history_sessions: Optional[int] = None,
        description: Optional[str] = None,
        instructions: Optional[Union[str, List[str], Callable]] = None,
        expected_output: Optional[str] = None,
        additional_context: Optional[str] = None,
        markdown: bool = False,
        add_datetime_to_context: bool = False,
        add_location_to_context: bool = False,
        timezone_identifier: Optional[str] = None,
        add_name_to_context: bool = False,
        add_member_tools_to_context: bool = True,
        system_message: Optional[Union[str, Callable, Message]] = None,
        system_message_role: str = "system",
        additional_input: Optional[List[Union[str, Dict, BaseModel, Message]]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        add_dependencies_to_context: bool = False,
        knowledge: Optional[Knowledge] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_knowledge_to_context: bool = False,
        enable_agentic_knowledge_filters: Optional[bool] = False,
        update_knowledge: bool = False,
        knowledge_retriever: Optional[Callable[..., Optional[List[Union[Dict, str]]]]] = None,
        references_format: Literal["json", "yaml"] = "json",
        share_member_interactions: bool = False,
        get_member_information_tool: bool = False,
        search_knowledge: bool = True,
        read_team_history: bool = False,
        read_chat_history: bool = False,
        store_media: bool = True,
        store_tool_messages: bool = True,
        store_history_messages: bool = True,
        send_media_to_model: bool = True,
        add_history_to_context: bool = False,
        num_history_runs: Optional[int] = None,
        num_history_messages: Optional[int] = None,
        max_tool_calls_from_history: Optional[int] = None,
        tools: Optional[List[Union[Toolkit, Callable, Function, Dict]]] = None,
        tool_call_limit: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_hooks: Optional[List[Callable]] = None,
        pre_hooks: Optional[Union[List[Callable[..., Any]], List[BaseGuardrail]]] = None,
        post_hooks: Optional[Union[List[Callable[..., Any]], List[BaseGuardrail]]] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        parser_model: Optional[Union[Model, str]] = None,
        parser_model_prompt: Optional[str] = None,
        output_model: Optional[Union[Model, str]] = None,
        output_model_prompt: Optional[str] = None,
        use_json_mode: bool = False,
        parse_response: bool = True,
        db: Optional[Union[BaseDb, AsyncBaseDb]] = None,
        enable_agentic_memory: bool = False,
        enable_user_memories: bool = False,
        add_memories_to_context: Optional[bool] = None,
        memory_manager: Optional[MemoryManager] = None,
        enable_session_summaries: bool = False,
        session_summary_manager: Optional[SessionSummaryManager] = None,
        add_session_summary_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reasoning: bool = False,
        reasoning_model: Optional[Union[Model, str]] = None,
        reasoning_agent: Optional[Agent] = None,
        reasoning_min_steps: int = 1,
        reasoning_max_steps: int = 10,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        store_events: bool = False,
        events_to_skip: Optional[List[Union[RunEvent, TeamRunEvent]]] = None,
        store_member_responses: bool = False,
        stream_member_events: bool = True,
        debug_mode: bool = False,
        debug_level: Literal[1, 2] = 1,
        show_members_responses: bool = False,
        retries: int = 0,
        delay_between_retries: int = 1,
        exponential_backoff: bool = False,
        telemetry: bool = True,
    ):
        self.members = members

        self.model = model  # type: ignore[assignment]

        self.name = name
        self.id = id
        self.role = role

        self.respond_directly = respond_directly
        self.determine_input_for_members = determine_input_for_members
        self.delegate_task_to_all_members = delegate_task_to_all_members

        self.user_id = user_id
        self.session_id = session_id
        self.session_state = session_state
        self.add_session_state_to_context = add_session_state_to_context
        self.enable_agentic_state = enable_agentic_state
        self.overwrite_db_session_state = overwrite_db_session_state
        self.resolve_in_context = resolve_in_context
        self.cache_session = cache_session

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

        self.add_team_history_to_members = add_team_history_to_members
        self.num_team_history_runs = num_team_history_runs
        self.search_session_history = search_session_history
        self.num_history_sessions = num_history_sessions

        self.description = description
        self.instructions = instructions
        self.expected_output = expected_output
        self.additional_context = additional_context
        self.markdown = markdown
        self.add_datetime_to_context = add_datetime_to_context
        self.add_location_to_context = add_location_to_context
        self.add_name_to_context = add_name_to_context
        self.timezone_identifier = timezone_identifier
        self.add_member_tools_to_context = add_member_tools_to_context
        self.system_message = system_message
        self.system_message_role = system_message_role
        self.additional_input = additional_input

        self.dependencies = dependencies
        self.add_dependencies_to_context = add_dependencies_to_context

        self.knowledge = knowledge
        self.knowledge_filters = knowledge_filters
        self.enable_agentic_knowledge_filters = enable_agentic_knowledge_filters
        self.update_knowledge = update_knowledge
        self.add_knowledge_to_context = add_knowledge_to_context
        self.knowledge_retriever = knowledge_retriever
        self.references_format = references_format

        self.share_member_interactions = share_member_interactions
        self.get_member_information_tool = get_member_information_tool
        self.search_knowledge = search_knowledge
        self.read_chat_history = read_chat_history or read_team_history

        self.store_media = store_media
        self.store_tool_messages = store_tool_messages
        self.store_history_messages = store_history_messages
        self.send_media_to_model = send_media_to_model

        self.tools = tools
        self.tool_choice = tool_choice
        self.tool_call_limit = tool_call_limit
        self.tool_hooks = tool_hooks

        # Initialize hooks with backward compatibility
        self.pre_hooks = pre_hooks
        self.post_hooks = post_hooks

        self.input_schema = input_schema
        self.output_schema = output_schema
        self.parser_model = parser_model  # type: ignore[assignment]
        self.parser_model_prompt = parser_model_prompt
        self.output_model = output_model  # type: ignore[assignment]
        self.output_model_prompt = output_model_prompt
        self.use_json_mode = use_json_mode
        self.parse_response = parse_response

        self.db = db

        self.enable_agentic_memory = enable_agentic_memory
        self.enable_user_memories = enable_user_memories
        self.add_memories_to_context = add_memories_to_context
        self.memory_manager = memory_manager
        self.enable_session_summaries = enable_session_summaries
        self.session_summary_manager = session_summary_manager
        self.add_session_summary_to_context = add_session_summary_to_context
        self.metadata = metadata

        self.reasoning = reasoning
        self.reasoning_model = reasoning_model  # type: ignore[assignment]
        self.reasoning_agent = reasoning_agent
        self.reasoning_min_steps = reasoning_min_steps
        self.reasoning_max_steps = reasoning_max_steps

        self.stream = stream
        self.stream_events = stream_events or stream_intermediate_steps
        self.store_events = store_events
        self.store_member_responses = store_member_responses

        self.events_to_skip = events_to_skip
        if self.events_to_skip is None:
            self.events_to_skip = [
                RunEvent.run_content,
                TeamRunEvent.run_content,
            ]
        self.stream_member_events = stream_member_events

        self.debug_mode = debug_mode
        if debug_level not in [1, 2]:
            log_warning(f"Invalid debug level: {debug_level}. Setting to 1.")
            debug_level = 1
        self.debug_level = debug_level
        self.show_members_responses = show_members_responses

        self.retries = retries
        self.delay_between_retries = delay_between_retries
        self.exponential_backoff = exponential_backoff

        self.telemetry = telemetry

        # TODO: Remove these
        # Images generated during this session
        self.images: Optional[List[Image]] = None
        # Audio generated during this session
        self.audio: Optional[List[Audio]] = None
        # Videos generated during this session
        self.videos: Optional[List[Video]] = None

        # Team session
        self._cached_session: Optional[TeamSession] = None

        self._tool_instructions: Optional[List[str]] = None

        # True if we should parse a member response model
        self._member_response_model: Optional[Type[BaseModel]] = None

        self._formatter: Optional[SafeFormatter] = None

        self._hooks_normalised = False

        # List of MCP tools that were initialized on the last run
        self._mcp_tools_initialized_on_run: List[Any] = []

        # Lazy-initialized shared thread pool executor for background tasks (memory, cultural knowledge, etc.)
        self._background_executor: Optional[Any] = None

        self._resolve_models()

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
    def cached_session(self) -> Optional[TeamSession]:
        return self._cached_session

    def set_id(self) -> None:
        """Set the ID of the team if not set yet.

        If the ID is not provided, generate a deterministic UUID from the name.
        If the name is not provided, generate a random UUID.
        """
        if self.id is None:
            self.id = generate_id_from_name(self.name)

    def _set_debug(self, debug_mode: Optional[bool] = None) -> None:
        if self.debug_mode or debug_mode or getenv("AGNO_DEBUG", "false").lower() == "true":
            set_log_level_to_debug(source_type="team", level=self.debug_level)
        else:
            set_log_level_to_info(source_type="team")

    def _set_telemetry(self) -> None:
        """Override telemetry settings based on environment variables."""

        telemetry_env = getenv("AGNO_TELEMETRY")
        if telemetry_env is not None:
            self.telemetry = telemetry_env.lower() == "true"

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

    def _initialize_member(self, member: Union["Team", Agent], debug_mode: Optional[bool] = None) -> None:
        # Set debug mode for all members
        if debug_mode:
            member.debug_mode = True
            member.debug_level = self.debug_level

        if isinstance(member, Agent):
            member.team_id = self.id
            member.set_id()

            # Inherit team models if agent has no explicit model
            for model_type in ["model", "reasoning_model", "parser_model", "output_model"]:
                if getattr(member, model_type) is None and getattr(self, model_type) is not None:
                    setattr(member, model_type, getattr(self, model_type))
                    log_info(
                        f"Agent '{member.name or member.id}' inheriting {model_type} from Team: {getattr(self, model_type).id}"
                    )

        elif isinstance(member, Team):
            member.parent_team_id = self.id
            # Initialize the sub-team's model first so it has its model set
            member._set_default_model()
            # Then let the sub-team initialize its own members so they inherit from the sub-team
            for sub_member in member.members:
                member._initialize_member(sub_member, debug_mode=debug_mode)

    def _set_default_model(self) -> None:
        # Set the default model
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

    def _initialize_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """Initialize the session for the team."""

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
        """Initialize the session state for the team."""
        if user_id:
            session_state["current_user_id"] = user_id
        if session_id is not None:
            session_state["current_session_id"] = session_id
        if run_id is not None:
            session_state["current_run_id"] = run_id
        return session_state

    def _has_async_db(self) -> bool:
        """Return True if the db the team is equipped with is an Async implementation"""
        return self.db is not None and isinstance(self.db, AsyncBaseDb)

    def _resolve_models(self) -> None:
        """Resolve model strings to Model instances."""
        if self.model is not None:
            self.model = get_model(self.model)
        if self.reasoning_model is not None:
            self.reasoning_model = get_model(self.reasoning_model)
        if self.parser_model is not None:
            self.parser_model = get_model(self.parser_model)
        if self.output_model is not None:
            self.output_model = get_model(self.output_model)

    def initialize_team(self, debug_mode: Optional[bool] = None) -> None:
        # Make sure for the team, we are using the team logger
        use_team_logger()

        if self.delegate_task_to_all_members and self.respond_directly:
            log_warning(
                "`delegate_task_to_all_members` and `respond_directly` are both enabled. The task will be delegated to all members, but `respond_directly` will be disabled."
            )
            self.respond_directly = False

        self._set_default_model()

        # Set debug mode
        self._set_debug(debug_mode=debug_mode)

        # Set the team ID if not set
        self.set_id()

        # Set the memory manager and session summary manager
        if self.enable_user_memories or self.enable_agentic_memory or self.memory_manager is not None:
            self._set_memory_manager()
        if self.enable_session_summaries or self.session_summary_manager is not None:
            self._set_session_summary_manager()

        log_debug(f"Team ID: {self.id}", center=True)

        # Initialize formatter
        if self._formatter is None:
            self._formatter = SafeFormatter()

        for member in self.members:
            self._initialize_member(member, debug_mode=self.debug_mode)

    def add_tool(self, tool: Union[Toolkit, Callable, Function, Dict]):
        if not self.tools:
            self.tools = []
        self.tools.append(tool)

    def set_tools(self, tools: List[Union[Toolkit, Callable, Function, Dict]]):
        self.tools = tools

    @staticmethod
    def cancel_run(run_id: str) -> bool:
        """Cancel a running team execution.

        Args:
            run_id (str): The run_id to cancel.

        Returns:
            bool: True if the run was found and marked for cancellation, False otherwise.
        """
        return cancel_run_global(run_id)

    async def _connect_mcp_tools(self) -> None:
        """Connect the MCP tools to the agent."""
        if self.tools is not None:
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

    def _execute_pre_hooks(
        self,
        hooks: Optional[List[Callable[..., Any]]],
        run_response: TeamRunOutput,
        run_input: TeamRunInput,
        session: TeamSession,
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[TeamRunOutputEvent]:
        """Execute multiple pre-hook functions in succession."""
        if hooks is None:
            return

        # Prepare all possible arguments once
        all_args = {
            "run_input": run_input,
            "team": self,
            "session": session,
            "user_id": user_id,
            "metadata": metadata,
            "session_state": session_state,
            "dependencies": dependencies,
            "debug_mode": debug_mode or self.debug_mode,
        }
        all_args.update(kwargs)

        for i, hook in enumerate(hooks):
            yield handle_event(  # type: ignore
                run_response=run_response,
                event=create_team_pre_hook_started_event(
                    from_run_response=run_response, run_input=run_input, pre_hook_name=hook.__name__
                ),
                events_to_skip=self.events_to_skip,
                store_events=self.store_events,
            )
            try:
                # Filter arguments to only include those that the hook accepts
                filtered_args = filter_hook_args(hook, all_args)

                hook(**filtered_args)

                yield handle_event(  # type: ignore
                    run_response=run_response,
                    event=create_team_pre_hook_completed_event(
                        from_run_response=run_response, run_input=run_input, pre_hook_name=hook.__name__
                    ),
                    events_to_skip=self.events_to_skip,
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
        run_response: TeamRunOutput,
        run_input: TeamRunInput,
        session: TeamSession,
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[TeamRunOutputEvent]:
        """Execute multiple pre-hook functions in succession (async version)."""
        if hooks is None:
            return

        # Prepare all possible arguments once
        all_args = {
            "run_input": run_input,
            "team": self,
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
                run_response=run_response,
                event=create_team_pre_hook_started_event(
                    from_run_response=run_response, run_input=run_input, pre_hook_name=hook.__name__
                ),
                events_to_skip=self.events_to_skip,
                store_events=self.store_events,
            )
            try:
                # Filter arguments to only include those that the hook accepts
                filtered_args = filter_hook_args(hook, all_args)

                from inspect import iscoroutinefunction

                if iscoroutinefunction(hook):
                    await hook(**filtered_args)
                else:
                    # Synchronous function
                    hook(**filtered_args)

                yield handle_event(  # type: ignore
                    run_response=run_response,
                    event=create_team_pre_hook_completed_event(
                        from_run_response=run_response, run_input=run_input, pre_hook_name=hook.__name__
                    ),
                    events_to_skip=self.events_to_skip,
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
        run_output: TeamRunOutput,
        session: TeamSession,
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[TeamRunOutputEvent]:
        """Execute multiple post-hook functions in succession."""
        if hooks is None:
            return

        # Prepare all possible arguments once
        all_args = {
            "run_output": run_output,
            "team": self,
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
                event=create_team_post_hook_started_event(  # type: ignore
                    from_run_response=run_output,
                    post_hook_name=hook.__name__,
                ),
                events_to_skip=self.events_to_skip,
                store_events=self.store_events,
            )
            try:
                # Filter arguments to only include those that the hook accepts
                filtered_args = filter_hook_args(hook, all_args)

                hook(**filtered_args)

                yield handle_event(  # type: ignore
                    run_response=run_output,
                    event=create_team_post_hook_completed_event(  # type: ignore
                        from_run_response=run_output,
                        post_hook_name=hook.__name__,
                    ),
                    events_to_skip=self.events_to_skip,
                    store_events=self.store_events,
                )

            except (InputCheckError, OutputCheckError) as e:
                raise e
            except Exception as e:
                log_error(f"Post-hook #{i + 1} execution failed: {str(e)}")
                log_exception(e)

    async def _aexecute_post_hooks(
        self,
        hooks: Optional[List[Callable[..., Any]]],
        run_output: TeamRunOutput,
        session: TeamSession,
        user_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[TeamRunOutputEvent]:
        """Execute multiple post-hook functions in succession (async version)."""
        if hooks is None:
            return

        # Prepare all possible arguments once
        all_args = {
            "run_output": run_output,
            "team": self,
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
                event=create_team_post_hook_started_event(  # type: ignore
                    from_run_response=run_output,
                    post_hook_name=hook.__name__,
                ),
                events_to_skip=self.events_to_skip,
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
                    event=create_team_post_hook_completed_event(  # type: ignore
                        from_run_response=run_output,
                        post_hook_name=hook.__name__,
                    ),
                    events_to_skip=self.events_to_skip,
                    store_events=self.store_events,
                )
            except (InputCheckError, OutputCheckError) as e:
                raise e
            except Exception as e:
                log_error(f"Post-hook #{i + 1} execution failed: {str(e)}")
                log_exception(e)

    def _run(
        self,
        run_response: TeamRunOutput,
        session: TeamSession,
        session_state: Dict[str, Any],
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
    ) -> TeamRunOutput:
        """Run the Team and return the response.

        Steps:
        1. Execute pre-hooks
        2. Determine tools for model
        3. Prepare run messages
        4. Start memory creation in background thread
        5. Reason about the task if reasoning is enabled
        6. Get a response from the model
        7. Update TeamRunOutput with the model response
        8. Store media if enabled
        9. Convert response to structured format
        10. Execute post-hooks
        11. Wait for background memory creation
        12. Create session summary
        13. Cleanup and store (scrub, stop timer, add to session, calculate metrics, save session)
        """

        # Register run for cancellation tracking
        register_run(run_response.run_id)  # type: ignore

        # 1. Execute pre-hooks
        run_input = cast(TeamRunInput, run_response.input)
        self.model = cast(Model, self.model)
        if self.pre_hooks is not None:
            # Can modify the run input
            pre_hook_iterator = self._execute_pre_hooks(
                hooks=self.pre_hooks,  # type: ignore
                run_response=run_response,
                run_input=run_input,
                session=session,
                session_state=session_state,
                dependencies=dependencies,
                metadata=metadata,
                user_id=user_id,
                debug_mode=debug_mode,
                **kwargs,
            )
            # Consume the generator without yielding
            deque(pre_hook_iterator, maxlen=0)

        # 2. Determine tools for model
        # Initialize team run context
        team_run_context: Dict[str, Any] = {}

        _tools = self._determine_tools_for_model(
            model=self.model,
            run_response=run_response,
            team_run_context=team_run_context,
            session=session,
            session_state=session_state,
            user_id=user_id,
            async_mode=False,
            knowledge_filters=knowledge_filters,
            input_message=run_input.input_content,
            images=run_input.images,
            videos=run_input.videos,
            audio=run_input.audios,
            files=run_input.files,
            debug_mode=debug_mode,
            add_history_to_context=add_history_to_context,
            add_session_state_to_context=add_session_state_to_context,
            dependencies=dependencies,
            add_dependencies_to_context=add_dependencies_to_context,
            metadata=metadata,
        )

        # 3. Prepare run messages
        run_messages: RunMessages = self._get_run_messages(
            run_response=run_response,
            session=session,
            session_state=session_state,
            user_id=user_id,
            input_message=run_input.input_content,
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

        log_debug(f"Team Run Start: {run_response.run_id}", center=True)

        # 4. Start memory creation in background thread
        memory_future = None
        if run_messages.user_message is not None and self.memory_manager is not None and not self.enable_agentic_memory:
            log_debug("Starting memory creation in background thread.")
            memory_future = self.background_executor.submit(
                self._make_memories, run_messages=run_messages, user_id=user_id
            )

        try:
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 5. Reason about the task if reasoning is enabled
            self._handle_reasoning(run_response=run_response, run_messages=run_messages)

            # Check for cancellation before model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 6. Get the model response for the team leader
            self.model = cast(Model, self.model)
            model_response: ModelResponse = self.model.response(
                messages=run_messages.messages,
                response_format=response_format,
                tools=_tools,
                tool_choice=self.tool_choice,
                tool_call_limit=self.tool_call_limit,
                send_media_to_model=self.send_media_to_model,
            )

            # Check for cancellation after model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # If an output model is provided, generate output using the output model
            self._parse_response_with_output_model(model_response, run_messages)

            # If a parser model is provided, structure the response separately
            self._parse_response_with_parser_model(model_response, run_messages)

            # 7. Update TeamRunOutput with the model response
            self._update_run_response(
                model_response=model_response, run_response=run_response, run_messages=run_messages
            )

            # 8. Store media if enabled
            if self.store_media:
                self._store_media(run_response, model_response)

            # 9. Convert response to structured format
            self._convert_response_to_structured_format(run_response=run_response)

            # 10. Execute post-hooks after output is generated but before response is returned
            if self.post_hooks is not None:
                iterator = self._execute_post_hooks(
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
                deque(iterator, maxlen=0)
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 11. Wait for background memory creation
            wait_for_background_tasks(memory_future=memory_future)

            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 12. Create session summary
            if self.session_summary_manager is not None:
                # Upsert the RunOutput to Team Session before creating the session summary
                session.upsert_run(run_response=run_response)
                try:
                    self.session_summary_manager.create_session_summary(session=session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")

            raise_if_cancelled(run_response.run_id)  # type: ignore

            # Set the run status to completed
            run_response.status = RunStatus.completed

            # 13. Cleanup and store the run response
            self._cleanup_and_store(run_response=run_response, session=session)

            # Log Team Telemetry
            self._log_team_telemetry(session_id=session.session_id, run_id=run_response.run_id)

            log_debug(f"Team Run End: {run_response.run_id}", center=True, symbol="*")

            return run_response

        except RunCancelledException as e:
            # Handle run cancellation during streaming
            log_info(f"Team run {run_response.run_id} was cancelled")
            run_response.status = RunStatus.cancelled
            run_response.content = str(e)

            # Add the RunOutput to Team Session even when cancelled
            self._cleanup_and_store(run_response=run_response, session=session)
            return run_response
        finally:
            cleanup_run(run_response.run_id)  # type: ignore

    def _run_stream(
        self,
        run_response: TeamRunOutput,
        session: TeamSession,
        session_state: Dict[str, Any],
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
    ) -> Iterator[Union[TeamRunOutputEvent, RunOutputEvent, TeamRunOutput]]:
        """Run the Team and return the response iterator.

        Steps:
        1. Execute pre-hooks
        2. Determine tools for model
        3. Prepare run messages
        4. Start memory creation in background thread
        5. Reason about the task if reasoning is enabled
        6. Get a response from the model
        7. Parse response with parser model if provided
        8. Wait for background memory creation
        9. Create session summary
        10. Cleanup and store (scrub, add to session, calculate metrics, save session)
        """
        # Register run for cancellation tracking
        register_run(run_response.run_id)  # type: ignore

        # 1. Execute pre-hooks
        run_input = cast(TeamRunInput, run_response.input)
        self.model = cast(Model, self.model)
        if self.pre_hooks is not None:
            # Can modify the run input
            pre_hook_iterator = self._execute_pre_hooks(
                hooks=self.pre_hooks,  # type: ignore
                run_response=run_response,
                run_input=run_input,
                session=session,
                session_state=session_state,
                dependencies=dependencies,
                metadata=metadata,
                user_id=user_id,
                debug_mode=debug_mode,
                **kwargs,
            )
            for pre_hook_event in pre_hook_iterator:
                yield pre_hook_event

        # 2. Determine tools for model
        # Initialize team run context
        team_run_context: Dict[str, Any] = {}

        _tools = self._determine_tools_for_model(
            model=self.model,
            run_response=run_response,
            team_run_context=team_run_context,
            session=session,
            session_state=session_state,
            user_id=user_id,
            async_mode=False,
            knowledge_filters=knowledge_filters,
            input_message=run_input.input_content,
            images=run_input.images,
            videos=run_input.videos,
            audio=run_input.audios,
            files=run_input.files,
            debug_mode=debug_mode,
            add_history_to_context=add_history_to_context,
            add_session_state_to_context=add_session_state_to_context,
            dependencies=dependencies,
            add_dependencies_to_context=add_dependencies_to_context,
            metadata=metadata,
        )

        # 3. Prepare run messages
        run_messages: RunMessages = self._get_run_messages(
            run_response=run_response,
            session=session,
            session_state=session_state,
            user_id=user_id,
            input_message=run_input.input_content,
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

        log_debug(f"Team Run Start: {run_response.run_id}", center=True)

        # 4. Start memory creation in background thread
        memory_future = None
        if run_messages.user_message is not None and self.memory_manager is not None and not self.enable_agentic_memory:
            log_debug("Starting memory creation in background thread.")
            memory_future = self.background_executor.submit(
                self._make_memories, run_messages=run_messages, user_id=user_id
            )

        try:
            # Start the Run by yielding a RunStarted event
            if stream_events:
                yield handle_event(  # type: ignore
                    create_team_run_started_event(run_response),
                    run_response,
                    events_to_skip=self.events_to_skip,
                    store_events=self.store_events,
                )

            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 5. Reason about the task if reasoning is enabled
            yield from self._handle_reasoning_stream(
                run_response=run_response,
                run_messages=run_messages,
                stream_events=stream_events,
            )

            # Check for cancellation before model processing
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 6. Get a response from the model
            if self.output_model is None:
                for event in self._handle_model_response_stream(
                    session=session,
                    run_response=run_response,
                    run_messages=run_messages,
                    tools=_tools,
                    response_format=response_format,
                    stream_events=stream_events,
                    session_state=session_state,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    yield event
            else:
                for event in self._handle_model_response_stream(
                    session=session,
                    run_response=run_response,
                    run_messages=run_messages,
                    tools=_tools,
                    response_format=response_format,
                    stream_events=stream_events,
                    session_state=session_state,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    from agno.run.team import IntermediateRunContentEvent, RunContentEvent

                    if isinstance(event, RunContentEvent):
                        if stream_events:
                            yield IntermediateRunContentEvent(
                                content=event.content,
                                content_type=event.content_type,
                            )
                    else:
                        yield event

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

            # Yield RunContentCompletedEvent
            if stream_events:
                yield handle_event(  # type: ignore
                    create_team_run_content_completed_event(from_run_response=run_response),
                    run_response,
                    events_to_skip=self.events_to_skip,
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
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 8. Wait for background memory creation
            yield from wait_for_background_tasks_stream(
                run_response=run_response,
                memory_future=memory_future,
                stream_events=stream_events,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            )

            raise_if_cancelled(run_response.run_id)  # type: ignore
            # 9. Create session summary
            if self.session_summary_manager is not None:
                # Upsert the RunOutput to Team Session before creating the session summary
                session.upsert_run(run_response=run_response)

                if stream_events:
                    yield handle_event(  # type: ignore
                        create_team_session_summary_started_event(from_run_response=run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,
                        store_events=self.store_events,
                    )
                try:
                    self.session_summary_manager.create_session_summary(session=session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")
                if stream_events:
                    yield handle_event(  # type: ignore
                        create_team_session_summary_completed_event(
                            from_run_response=run_response, session_summary=session.summary
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,
                        store_events=self.store_events,
                    )

            raise_if_cancelled(run_response.run_id)  # type: ignore
            # Create the run completed event
            completed_event = handle_event(
                create_team_run_completed_event(
                    from_run_response=run_response,
                ),
                run_response,
                events_to_skip=self.events_to_skip,
                store_events=self.store_events,
            )

            # Set the run status to completed
            run_response.status = RunStatus.completed

            # 10. Cleanup and store the run response
            self._cleanup_and_store(run_response=run_response, session=session)

            if stream_events:
                yield completed_event

            if yield_run_response:
                yield run_response

            # Log Team Telemetry
            self._log_team_telemetry(session_id=session.session_id, run_id=run_response.run_id)

            log_debug(f"Team Run End: {run_response.run_id}", center=True, symbol="*")

        except RunCancelledException as e:
            # Handle run cancellation during streaming
            log_info(f"Team run {run_response.run_id} was cancelled during streaming")
            run_response.status = RunStatus.cancelled
            run_response.content = str(e)

            # Yield the cancellation event
            yield handle_event(  # type: ignore
                create_team_run_cancelled_event(from_run_response=run_response, reason=str(e)),
                run_response,
                events_to_skip=self.events_to_skip,
                store_events=self.store_events,
            )

            # Add the RunOutput to Team Session even when cancelled
            self._cleanup_and_store(run_response=run_response, session=session)
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
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        retries: Optional[int] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> TeamRunOutput: ...

    @overload
    def run(
        self,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        *,
        stream: Literal[True] = True,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        retries: Optional[int] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        yield_run_response: bool = False,
        **kwargs: Any,
    ) -> Iterator[Union[RunOutputEvent, TeamRunOutputEvent]]: ...

    def run(
        self,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        *,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        retries: Optional[int] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        yield_run_response: bool = False,
        **kwargs: Any,
    ) -> Union[TeamRunOutput, Iterator[Union[RunOutputEvent, TeamRunOutputEvent]]]:
        """Run the Team and return the response."""
        if self._has_async_db():
            raise Exception("run() is not supported with an async DB. Please use arun() instead.")

        # Initialize Team
        self.initialize_team(debug_mode=debug_mode)

        if (add_history_to_context or self.add_history_to_context) and not self.db and not self.parent_team_id:
            log_warning(
                "add_history_to_context is True, but no database has been assigned to the team. History will not be added to the context."
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

        image_artifacts, video_artifacts, audio_artifacts, file_artifacts = self._validate_media_object_id(
            images=images, videos=videos, audios=audio, files=files
        )

        # Create RunInput to capture the original user input
        run_input = TeamRunInput(
            input_content=validated_input,
            images=image_artifacts,
            videos=video_artifacts,
            audios=audio_artifacts,
            files=file_artifacts,
        )

        # Read existing session from database
        team_session = self._read_or_create_session(session_id=session_id, user_id=user_id)
        self._update_metadata(session=team_session)

        # Initialize session state
        session_state = self._initialize_session_state(
            session_state=session_state or {}, user_id=user_id, session_id=session_id, run_id=run_id
        )
        # Update session state from DB
        session_state = self._load_session_state(session=team_session, session_state=session_state)

        # Determine runtime dependencies
        run_dependencies = dependencies if dependencies is not None else self.dependencies

        # Resolve callable dependencies if present
        if run_dependencies is not None:
            self._resolve_run_dependencies(run_dependencies)

        # Determine runtime context parameters
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

        # Configure the model for runs
        response_format: Optional[Union[Dict, Type[BaseModel]]] = (
            self._get_response_format() if self.parser_model is None else None
        )
        self.model = cast(Model, self.model)

        if self.metadata is not None:
            if metadata is None:
                metadata = self.metadata
            else:
                merge_dictionaries(metadata, self.metadata)

        # Create a new run_response for this attempt
        run_response = TeamRunOutput(
            run_id=run_id,
            session_id=session_id,
            team_id=self.id,
            team_name=self.name,
            metadata=metadata,
            session_state=session_state,
            input=run_input,
        )

        run_response.model = self.model.id if self.model is not None else None
        run_response.model_provider = self.model.provider if self.model is not None else None

        # Start the run metrics timer, to calculate the run duration
        run_response.metrics = Metrics()
        run_response.metrics.start_timer()

        # If no retries are set, use the team's default retries
        retries = retries if retries is not None else self.retries

        # Run the team
        last_exception = None
        num_attempts = retries + 1

        for attempt in range(num_attempts):
            # Initialize the current run

            # Run the team
            try:
                if stream:
                    response_iterator = self._run_stream(
                        run_response=run_response,
                        session=team_session,
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

                    return response_iterator  # type: ignore
                else:
                    return self._run(
                        run_response=run_response,
                        session=team_session,
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

            except (InputCheckError, OutputCheckError) as e:
                log_error(f"Validation failed: {str(e)} | Check: {e.check_trigger}")
                raise e
            except ModelProviderError as e:
                import time

                log_warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")

                last_exception = e
                if attempt < num_attempts - 1:  # Don't sleep on the last attempt
                    if self.exponential_backoff:
                        delay = 2**attempt * self.delay_between_retries
                    else:
                        delay = self.delay_between_retries
                    time.sleep(delay)
            except KeyboardInterrupt:
                run_response.content = "Operation cancelled by user"
                run_response.status = RunStatus.cancelled

                if stream:
                    return generator_wrapper(  # type: ignore
                        create_team_run_cancelled_event(
                            from_run_response=run_response, reason="Operation cancelled by user"
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
                return generator_wrapper(create_team_run_error_event(run_response, error=str(last_exception)))

            raise last_exception
        else:
            if stream:
                return generator_wrapper(create_team_run_error_event(run_response, error=str(last_exception)))

            raise Exception(f"Failed after {num_attempts} attempts.")

    async def _arun(
        self,
        run_response: TeamRunOutput,
        session_id: str,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        add_history_to_context: Optional[bool] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TeamRunOutput:
        """Run the Team and return the response.

        Steps:
        1. Read or create session
        2. Update metadata and session state
        3. Execute pre-hooks
        4. Determine tools for model
        5. Prepare run messages
        6. Start memory creation in background task
        7. Reason about the task if reasoning is enabled
        8. Get a response from the Model
        9. Update TeamRunOutput with the model response
        10. Store media if enabled
        11. Convert response to structured format
        12. Execute post-hooks
        13. Wait for background memory creation
        14. Create session summary
        15. Cleanup and store (scrub, add to session, calculate metrics, save session)
        """
        log_debug(f"Team Run Start: {run_response.run_id}", center=True)

        register_run(run_response.run_id)  # type: ignore

        if dependencies is not None:
            await self._aresolve_run_dependencies(dependencies=dependencies)

        # 1. Read or create session. Reads from the database if provided.
        if self._has_async_db():
            team_session = await self._aread_or_create_session(session_id=session_id, user_id=user_id)
        else:
            team_session = self._read_or_create_session(session_id=session_id, user_id=user_id)

        # 2. Update metadata and session state
        self._update_metadata(session=team_session)
        # Initialize session state
        session_state = self._initialize_session_state(
            session_state=session_state or {}, user_id=user_id, session_id=session_id, run_id=run_response.run_id
        )
        # Update session state from DB
        session_state = self._load_session_state(session=team_session, session_state=session_state)  # type: ignore

        run_input = cast(TeamRunInput, run_response.input)

        # 3. Execute pre-hooks after session is loaded but before processing starts
        if self.pre_hooks is not None:
            pre_hook_iterator = self._aexecute_pre_hooks(
                hooks=self.pre_hooks,  # type: ignore
                run_response=run_response,
                run_input=run_input,
                session=team_session,
                user_id=user_id,
                debug_mode=debug_mode,
                session_state=session_state,
                dependencies=dependencies,
                metadata=metadata,
                **kwargs,
            )

            # Consume the async iterator without yielding
            async for _ in pre_hook_iterator:
                pass

        # 4. Determine tools for model
        team_run_context: Dict[str, Any] = {}
        self.model = cast(Model, self.model)
        await self._check_and_refresh_mcp_tools()
        _tools = self._determine_tools_for_model(
            model=self.model,
            run_response=run_response,
            team_run_context=team_run_context,
            session=team_session,
            session_state=session_state,
            user_id=user_id,
            async_mode=True,
            knowledge_filters=knowledge_filters,
            input_message=run_input.input_content,
            images=run_input.images,
            videos=run_input.videos,
            audio=run_input.audios,
            files=run_input.files,
            debug_mode=debug_mode,
            add_history_to_context=add_history_to_context,
            add_dependencies_to_context=add_dependencies_to_context,
            add_session_state_to_context=add_session_state_to_context,
            dependencies=dependencies,
            metadata=metadata,
        )

        # 5. Prepare run messages
        run_messages = await self._aget_run_messages(
            run_response=run_response,
            session=team_session,  # type: ignore
            session_state=session_state,
            user_id=user_id,
            input_message=run_input.input_content,
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

        self.model = cast(Model, self.model)
        log_debug(f"Team Run Start: {run_response.run_id}", center=True)

        # 6. Start memory creation in background task
        memory_task = None
        if run_messages.user_message is not None and self.memory_manager is not None and not self.enable_agentic_memory:
            log_debug("Starting memory creation in background task.")
            memory_task = asyncio.create_task(self._amake_memories(run_messages=run_messages, user_id=user_id))

        # Register run for cancellation tracking
        register_run(run_response.run_id)  # type: ignore

        try:
            raise_if_cancelled(run_response.run_id)  # type: ignore
            # 7. Reason about the task if reasoning is enabled
            await self._ahandle_reasoning(run_response=run_response, run_messages=run_messages)

            # Check for cancellation before model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 8. Get the model response for the team leader
            model_response = await self.model.aresponse(
                messages=run_messages.messages,
                tools=_tools,
                tool_choice=self.tool_choice,
                tool_call_limit=self.tool_call_limit,
                response_format=response_format,
                send_media_to_model=self.send_media_to_model,
            )  # type: ignore

            # Check for cancellation after model call
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # If an output model is provided, generate output using the output model
            await self._agenerate_response_with_output_model(model_response=model_response, run_messages=run_messages)

            # If a parser model is provided, structure the response separately
            await self._aparse_response_with_parser_model(model_response=model_response, run_messages=run_messages)

            # 9. Update TeamRunOutput with the model response
            self._update_run_response(
                model_response=model_response, run_response=run_response, run_messages=run_messages
            )

            # 10. Store media if enabled
            if self.store_media:
                self._store_media(run_response, model_response)

            # 11. Convert response to structured format
            self._convert_response_to_structured_format(run_response=run_response)

            # 12. Execute post-hooks after output is generated but before response is returned
            if self.post_hooks is not None:
                async for _ in self._aexecute_post_hooks(
                    hooks=self.post_hooks,  # type: ignore
                    run_output=run_response,
                    session=team_session,
                    session_state=session_state,
                    dependencies=dependencies,
                    metadata=metadata,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    **kwargs,
                ):
                    pass

            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 13. Wait for background memory creation
            await await_for_background_tasks(memory_task=memory_task)

            raise_if_cancelled(run_response.run_id)  # type: ignore
            # 14. Create session summary
            if self.session_summary_manager is not None:
                # Upsert the RunOutput to Team Session before creating the session summary
                team_session.upsert_run(run_response=run_response)
                try:
                    await self.session_summary_manager.acreate_session_summary(session=team_session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")

            raise_if_cancelled(run_response.run_id)  # type: ignore
            run_response.status = RunStatus.completed

            # 15. Cleanup and store the run response and session
            await self._acleanup_and_store(run_response=run_response, session=team_session)

            # Log Team Telemetry
            await self._alog_team_telemetry(session_id=team_session.session_id, run_id=run_response.run_id)

            log_debug(f"Team Run End: {run_response.run_id}", center=True, symbol="*")

            return run_response
        except RunCancelledException as e:
            # Handle run cancellation
            log_info(f"Run {run_response.run_id} was cancelled")
            run_response.content = str(e)
            run_response.status = RunStatus.cancelled

            # Cleanup and store the run response and session
            await self._acleanup_and_store(run_response=run_response, session=team_session)

            return run_response
        finally:
            await self._disconnect_mcp_tools()
            # Cancel the memory task if it's still running
            if memory_task is not None and not memory_task.done():
                memory_task.cancel()
                try:
                    await memory_task
                except asyncio.CancelledError:
                    pass
            # Always clean up the run tracking
            cleanup_run(run_response.run_id)  # type: ignore

    async def _arun_stream(
        self,
        run_response: TeamRunOutput,
        session_id: str,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        stream_events: bool = False,
        stream_intermediate_steps: bool = False,
        yield_run_response: bool = False,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        add_history_to_context: Optional[bool] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[TeamRunOutputEvent, RunOutputEvent, TeamRunOutput]]:
        """Run the Team and return the response.

        Steps:
        1. Resolve dependencies
        2. Read or create session
        3. Update metadata and session state
        4. Execute pre-hooks
        5. Determine tools for model
        6. Prepare run messages
        7. Start memory creation in background task
        8. Reason about the task if reasoning is enabled
        9. Get a response from the model
        10. Parse response with parser model if provided
        11. Wait for background memory creation
        12. Create session summary
        13. Cleanup and store (scrub, add to session, calculate metrics, save session)
        """

        # 1. Resolve dependencies
        if dependencies is not None:
            await self._aresolve_run_dependencies(dependencies=dependencies)

        # 2. Read or create session. Reads from the database if provided.
        if self._has_async_db():
            team_session = await self._aread_or_create_session(session_id=session_id, user_id=user_id)
        else:
            team_session = self._read_or_create_session(session_id=session_id, user_id=user_id)

        # 3. Update metadata and session state
        self._update_metadata(session=team_session)
        # Initialize session state
        session_state = self._initialize_session_state(
            session_state=session_state or {}, user_id=user_id, session_id=session_id, run_id=run_response.run_id
        )
        # Update session state from DB
        session_state = self._load_session_state(session=team_session, session_state=session_state)  # type: ignore

        # 4. Execute pre-hooks
        run_input = cast(TeamRunInput, run_response.input)
        self.model = cast(Model, self.model)
        if self.pre_hooks is not None:
            pre_hook_iterator = self._aexecute_pre_hooks(
                hooks=self.pre_hooks,  # type: ignore
                run_response=run_response,
                run_input=run_input,
                session=team_session,
                user_id=user_id,
                debug_mode=debug_mode,
                session_state=session_state,
                dependencies=dependencies,
                metadata=metadata,
                **kwargs,
            )
            async for pre_hook_event in pre_hook_iterator:
                yield pre_hook_event

        # 5. Determine tools for model
        team_run_context: Dict[str, Any] = {}
        self.model = cast(Model, self.model)
        await self._check_and_refresh_mcp_tools()
        _tools = self._determine_tools_for_model(
            model=self.model,
            run_response=run_response,
            team_run_context=team_run_context,
            session=team_session,  # type: ignore
            session_state=session_state,
            user_id=user_id,
            async_mode=True,
            knowledge_filters=knowledge_filters,
            input_message=run_input.input_content,
            images=run_input.images,
            videos=run_input.videos,
            audio=run_input.audios,
            files=run_input.files,
            debug_mode=debug_mode,
            add_history_to_context=add_history_to_context,
            dependencies=dependencies,
            metadata=metadata,
        )

        # 6. Prepare run messages
        run_messages = await self._aget_run_messages(
            run_response=run_response,
            session=team_session,  # type: ignore
            session_state=session_state,
            user_id=user_id,
            input_message=run_input.input_content,
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

        log_debug(f"Team Run Start: {run_response.run_id}", center=True)

        # 7. Start memory creation in background task
        memory_task = None
        if run_messages.user_message is not None and self.memory_manager is not None and not self.enable_agentic_memory:
            log_debug("Starting memory creation in background task.")
            memory_task = asyncio.create_task(self._amake_memories(run_messages=run_messages, user_id=user_id))

        # Register run for cancellation tracking
        register_run(run_response.run_id)  # type: ignore

        try:
            # Considering both stream_events and stream_intermediate_steps (deprecated)
            stream_events = stream_events or stream_intermediate_steps

            # Yield the run started event
            if stream_events:
                yield handle_event(  # type: ignore
                    create_team_run_started_event(from_run_response=run_response),
                    run_response,
                    events_to_skip=self.events_to_skip,
                    store_events=self.store_events,
                )

            # 8. Reason about the task if reasoning is enabled
            async for item in self._ahandle_reasoning_stream(
                run_response=run_response,
                run_messages=run_messages,
                stream_events=stream_events,
            ):
                raise_if_cancelled(run_response.run_id)  # type: ignore
                yield item

            # Check for cancellation before model processing
            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 9. Get a response from the model
            if self.output_model is None:
                async for event in self._ahandle_model_response_stream(
                    session=team_session,
                    run_response=run_response,
                    run_messages=run_messages,
                    tools=_tools,
                    response_format=response_format,
                    stream_events=stream_events,
                    session_state=session_state,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    yield event
            else:
                async for event in self._ahandle_model_response_stream(
                    session=team_session,
                    run_response=run_response,
                    run_messages=run_messages,
                    tools=_tools,
                    response_format=response_format,
                    stream_events=stream_events,
                    session_state=session_state,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    from agno.run.team import IntermediateRunContentEvent, RunContentEvent

                    if isinstance(event, RunContentEvent):
                        if stream_events:
                            yield IntermediateRunContentEvent(
                                content=event.content,
                                content_type=event.content_type,
                            )
                    else:
                        yield event

                async for event in self._agenerate_response_with_output_model_stream(
                    session=team_session,
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
                session=team_session, run_response=run_response, stream_events=stream_events
            ):
                yield event

            # Yield RunContentCompletedEvent
            if stream_events:
                yield handle_event(  # type: ignore
                    create_team_run_content_completed_event(from_run_response=run_response),
                    run_response,
                    events_to_skip=self.events_to_skip,
                    store_events=self.store_events,
                )

            # Execute post-hooks after output is generated but before response is returned
            if self.post_hooks is not None:
                async for event in self._aexecute_post_hooks(
                    hooks=self.post_hooks,  # type: ignore
                    run_output=run_response,
                    session_state=session_state,
                    dependencies=dependencies,
                    metadata=metadata,
                    session=team_session,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    **kwargs,
                ):
                    yield event

            raise_if_cancelled(run_response.run_id)  # type: ignore
            # 11. Wait for background memory creation
            async for event in await_for_background_tasks_stream(
                run_response=run_response,
                memory_task=memory_task,
                stream_events=stream_events,
                events_to_skip=self.events_to_skip,  # type: ignore
                store_events=self.store_events,
            ):
                yield event

            raise_if_cancelled(run_response.run_id)  # type: ignore

            # 12. Create session summary
            if self.session_summary_manager is not None:
                # Upsert the RunOutput to Team Session before creating the session summary
                team_session.upsert_run(run_response=run_response)

                if stream_events:
                    yield handle_event(  # type: ignore
                        create_team_session_summary_started_event(from_run_response=run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,
                        store_events=self.store_events,
                    )
                try:
                    await self.session_summary_manager.acreate_session_summary(session=team_session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")
                if stream_events:
                    yield handle_event(  # type: ignore
                        create_team_session_summary_completed_event(
                            from_run_response=run_response, session_summary=team_session.summary
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,
                        store_events=self.store_events,
                    )

            raise_if_cancelled(run_response.run_id)  # type: ignore

            # Create the run completed event
            completed_event = handle_event(
                create_team_run_completed_event(from_run_response=run_response),
                run_response,
                events_to_skip=self.events_to_skip,
                store_events=self.store_events,
            )

            # Set the run status to completed
            run_response.status = RunStatus.completed

            # 13. Cleanup and store the run response and session
            await self._acleanup_and_store(run_response=run_response, session=team_session)

            if stream_events:
                yield completed_event

            if yield_run_response:
                yield run_response

            # Log Team Telemetry
            await self._alog_team_telemetry(session_id=team_session.session_id, run_id=run_response.run_id)

            log_debug(f"Team Run End: {run_response.run_id}", center=True, symbol="*")

        except RunCancelledException as e:
            # Handle run cancellation during async streaming
            log_info(f"Team run {run_response.run_id} was cancelled during async streaming")
            run_response.status = RunStatus.cancelled
            run_response.content = str(e)

            # Yield the cancellation event
            yield handle_event(  # type: ignore
                create_team_run_cancelled_event(from_run_response=run_response, reason=str(e)),
                run_response,
                events_to_skip=self.events_to_skip,
                store_events=self.store_events,
            )

            # Cleanup and store the run response and session
            await self._acleanup_and_store(run_response=run_response, session=team_session)

        finally:
            await self._disconnect_mcp_tools()
            # Cancel the memory task if it's still running
            if memory_task is not None and not memory_task.done():
                memory_task.cancel()
                try:
                    await memory_task
                except asyncio.CancelledError:
                    pass
            # Always clean up the run tracking
            cleanup_run(run_response.run_id)  # type: ignore

    @overload
    async def arun(
        self,
        input: Union[str, List, Dict, Message, BaseModel],
        *,
        stream: Literal[False] = False,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        retries: Optional[int] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> TeamRunOutput: ...

    @overload
    def arun(
        self,
        input: Union[str, List, Dict, Message, BaseModel],
        *,
        stream: Literal[True] = True,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        retries: Optional[int] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        yield_run_response: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[Union[RunOutputEvent, TeamRunOutputEvent]]: ...

    def arun(  # type: ignore
        self,
        input: Union[str, List, Dict, Message, BaseModel],
        *,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        retries: Optional[int] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        yield_run_response: bool = False,
        **kwargs: Any,
    ) -> Union[TeamRunOutput, AsyncIterator[Union[RunOutputEvent, TeamRunOutputEvent]]]:
        """Run the Team asynchronously and return the response."""

        if (add_history_to_context or self.add_history_to_context) and not self.db and not self.parent_team_id:
            log_warning(
                "add_history_to_context is True, but no database has been assigned to the team. History will not be added to the context."
            )

        # Create a run_id for this specific run
        run_id = str(uuid4())

        # Validate input against input_schema if provided
        validated_input = self._validate_input(input)

        # Normalise hook & guardails
        if not self._hooks_normalised:
            if self.pre_hooks:
                self.pre_hooks = normalize_hooks(self.pre_hooks, async_mode=True)
            if self.post_hooks:
                self.post_hooks = normalize_hooks(self.post_hooks, async_mode=True)
            self._hooks_normalised = True

        session_id, user_id = self._initialize_session(session_id=session_id, user_id=user_id)

        # Initialize Team
        self.initialize_team(debug_mode=debug_mode)

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
        run_input = TeamRunInput(
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

        # Configure the model for runs
        response_format: Optional[Union[Dict, Type[BaseModel]]] = (
            self._get_response_format() if self.parser_model is None else None
        )

        self.model = cast(Model, self.model)

        if self.metadata is not None:
            if metadata is None:
                metadata = self.metadata
            else:
                merge_dictionaries(metadata, self.metadata)

        #  Get knowledge filters
        effective_filters = knowledge_filters
        if self.knowledge_filters or knowledge_filters:
            effective_filters = self._get_effective_filters(knowledge_filters)

        # Create a new run_response for this attempt
        run_response = TeamRunOutput(
            run_id=run_id,
            session_id=session_id,
            team_id=self.id,
            team_name=self.name,
            metadata=metadata,
            session_state=session_state,
            input=run_input,
        )

        run_response.model = self.model.id if self.model is not None else None
        run_response.model_provider = self.model.provider if self.model is not None else None

        # Start the run metrics timer, to calculate the run duration
        run_response.metrics = Metrics()
        run_response.metrics.start_timer()

        # If no retries are set, use the team's default retries
        retries = retries if retries is not None else self.retries

        # Run the team
        last_exception = None
        num_attempts = retries + 1

        for attempt in range(num_attempts):
            # Run the team
            try:
                if stream:
                    response_iterator = self._arun_stream(
                        input=validated_input,
                        run_response=run_response,
                        session_id=session_id,
                        session_state=session_state,
                        user_id=user_id,
                        knowledge_filters=effective_filters,
                        add_history_to_context=add_history,
                        add_dependencies_to_context=add_dependencies,
                        add_session_state_to_context=add_session_state,
                        metadata=metadata,
                        response_format=response_format,
                        dependencies=run_dependencies,
                        stream_events=stream_events,
                        yield_run_response=yield_run_response,
                        debug_mode=debug_mode,
                        **kwargs,
                    )
                    return response_iterator  # type: ignore
                else:
                    return self._arun(  # type: ignore
                        input=validated_input,
                        run_response=run_response,
                        session_id=session_id,
                        session_state=session_state,
                        user_id=user_id,
                        knowledge_filters=effective_filters,
                        add_history_to_context=add_history,
                        add_dependencies_to_context=add_dependencies,
                        add_session_state_to_context=add_session_state,
                        metadata=metadata,
                        response_format=response_format,
                        debug_mode=debug_mode,
                        dependencies=run_dependencies,
                        **kwargs,
                    )

            except (InputCheckError, OutputCheckError) as e:
                log_error(f"Validation failed: {str(e)} | Check: {e.check_trigger}")
                raise e
            except ModelProviderError as e:
                log_warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")
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
                    return async_generator_wrapper(
                        create_team_run_cancelled_event(
                            from_run_response=run_response, reason="Operation cancelled by user"
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
                return async_generator_wrapper(create_team_run_error_event(run_response, error=str(last_exception)))

            raise last_exception
        else:
            if stream:
                return async_generator_wrapper(create_team_run_error_event(run_response, error=str(last_exception)))

            raise Exception(f"Failed after {num_attempts} attempts.")

    def _store_media(self, run_response: TeamRunOutput, model_response: ModelResponse):
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
        self, model_response: ModelResponse, run_response: TeamRunOutput, run_messages: RunMessages
    ):
        # Handle structured outputs
        if (self.output_schema is not None) and not self.use_json_mode and (model_response.parsed is not None):
            # Update the run_response content with the structured output
            run_response.content = model_response.parsed
            # Update the run_response content_type with the structured output class name
            run_response.content_type = self.output_schema.__name__
        else:
            # Update the run_response content with the model response content
            if not run_response.content:
                run_response.content = model_response.content
            else:
                run_response.content += model_response.content

        # Update the run_response thinking with the model response thinking
        if model_response.reasoning_content is not None:
            if not run_response.reasoning_content:
                run_response.reasoning_content = model_response.reasoning_content
            else:
                run_response.reasoning_content += model_response.reasoning_content
        # Update provider data
        if model_response.provider_data is not None:
            run_response.model_provider_data = model_response.provider_data
        # Update citations
        if model_response.citations is not None:
            run_response.citations = model_response.citations

        # Update the run_response tools with the model response tool_executions
        if model_response.tool_executions is not None:
            if run_response.tools is None:
                run_response.tools = model_response.tool_executions
            else:
                run_response.tools.extend(model_response.tool_executions)

        # Update the run_response audio with the model response audio
        if model_response.audio is not None:
            run_response.response_audio = model_response.audio

        # Update session_state with changes from model response
        if model_response.updated_session_state is not None and run_response.session_state is not None:
            from agno.utils.merge_dict import merge_dictionaries

            merge_dictionaries(run_response.session_state, model_response.updated_session_state)

        # Build a list of messages that should be added to the RunOutput
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]

        # Update the TeamRunOutput messages
        run_response.messages = messages_for_run_response

        # Update the TeamRunOutput metrics
        run_response.metrics = self._calculate_metrics(messages_for_run_response)

        if model_response.tool_executions:
            for tool_call in model_response.tool_executions:
                tool_name = tool_call.tool_name
                if tool_name and tool_name.lower() in ["think", "analyze"]:
                    tool_args = tool_call.tool_args or {}
                    self._update_reasoning_content_from_tool_call(run_response, tool_name, tool_args)

    def _handle_model_response_stream(
        self,
        session: TeamSession,
        run_response: TeamRunOutput,
        run_messages: RunMessages,
        tools: Optional[List[Union[Function, dict]]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        stream_events: bool = False,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Union[TeamRunOutputEvent, RunOutputEvent]]:
        self.model = cast(Model, self.model)

        reasoning_state = {
            "reasoning_started": False,
            "reasoning_time_taken": 0.0,
        }

        stream_model_response = True
        if self.should_parse_structured_output:
            log_debug("Response model set, model response is not streamed.")
            stream_model_response = False

        full_model_response = ModelResponse()
        for model_response_event in self.model.response_stream(
            messages=run_messages.messages,
            response_format=response_format,
            tools=tools,
            tool_choice=self.tool_choice,
            tool_call_limit=self.tool_call_limit,
            stream_model_response=stream_model_response,
            send_media_to_model=self.send_media_to_model,
        ):
            yield from self._handle_model_response_chunk(
                session=session,
                run_response=run_response,
                full_model_response=full_model_response,
                model_response_event=model_response_event,
                reasoning_state=reasoning_state,
                stream_events=stream_events,
                parse_structured_output=self.should_parse_structured_output,
                session_state=session_state,
            )

        # 3. Update TeamRunOutput
        if full_model_response.content is not None:
            run_response.content = full_model_response.content
        if full_model_response.reasoning_content is not None:
            run_response.reasoning_content = full_model_response.reasoning_content
        if full_model_response.audio is not None:
            run_response.response_audio = full_model_response.audio
        if full_model_response.citations is not None:
            run_response.citations = full_model_response.citations
        if full_model_response.provider_data is not None:
            run_response.model_provider_data = full_model_response.provider_data

        if stream_events and reasoning_state["reasoning_started"]:
            all_reasoning_steps: List[ReasoningStep] = []
            if run_response.reasoning_steps:
                all_reasoning_steps = cast(List[ReasoningStep], run_response.reasoning_steps)

            if all_reasoning_steps:
                add_reasoning_metrics_to_metadata(run_response, reasoning_state["reasoning_time_taken"])
                yield handle_event(  # type: ignore
                    create_team_reasoning_completed_event(
                        from_run_response=run_response,
                        content=ReasoningSteps(reasoning_steps=all_reasoning_steps),
                        content_type=ReasoningSteps.__name__,
                    ),
                    run_response,
                    events_to_skip=self.events_to_skip,
                    store_events=self.store_events,
                )

        # Build a list of messages that should be added to the RunOutput
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        # Update the TeamRunOutput messages
        run_response.messages = messages_for_run_response
        # Update the TeamRunOutput metrics
        run_response.metrics = self._calculate_metrics(messages_for_run_response)

        # Update the run_response audio if streaming
        if full_model_response.audio is not None:
            run_response.response_audio = full_model_response.audio

    async def _ahandle_model_response_stream(
        self,
        session: TeamSession,
        run_response: TeamRunOutput,
        run_messages: RunMessages,
        tools: Optional[List[Union[Function, dict]]] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        stream_events: bool = False,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Union[TeamRunOutputEvent, RunOutputEvent]]:
        self.model = cast(Model, self.model)

        reasoning_state = {
            "reasoning_started": False,
            "reasoning_time_taken": 0.0,
        }

        stream_model_response = True
        if self.should_parse_structured_output:
            log_debug("Response model set, model response is not streamed.")
            stream_model_response = False

        full_model_response = ModelResponse()
        model_stream = self.model.aresponse_stream(
            messages=run_messages.messages,
            response_format=response_format,
            tools=tools,
            tool_choice=self.tool_choice,
            tool_call_limit=self.tool_call_limit,
            stream_model_response=stream_model_response,
            send_media_to_model=self.send_media_to_model,
        )  # type: ignore
        async for model_response_event in model_stream:
            for event in self._handle_model_response_chunk(
                session=session,
                run_response=run_response,
                full_model_response=full_model_response,
                model_response_event=model_response_event,
                reasoning_state=reasoning_state,
                stream_events=stream_events,
                parse_structured_output=self.should_parse_structured_output,
                session_state=session_state,
            ):
                yield event

        # Handle structured outputs
        if (self.output_schema is not None) and not self.use_json_mode and (full_model_response.parsed is not None):
            # Update the run_response content with the structured output
            run_response.content = full_model_response.parsed

        # Update TeamRunOutput
        if full_model_response.content is not None:
            run_response.content = full_model_response.content
        if full_model_response.reasoning_content is not None:
            run_response.reasoning_content = full_model_response.reasoning_content
        if full_model_response.audio is not None:
            run_response.response_audio = full_model_response.audio
        if full_model_response.citations is not None:
            run_response.citations = full_model_response.citations
        if full_model_response.provider_data is not None:
            run_response.model_provider_data = full_model_response.provider_data

        # Build a list of messages that should be added to the RunOutput
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        # Update the TeamRunOutput messages
        run_response.messages = messages_for_run_response
        # Update the TeamRunOutput metrics
        run_response.metrics = self._calculate_metrics(messages_for_run_response)

        if stream_events and reasoning_state["reasoning_started"]:
            all_reasoning_steps: List[ReasoningStep] = []
            if run_response.reasoning_steps:
                all_reasoning_steps = cast(List[ReasoningStep], run_response.reasoning_steps)

            if all_reasoning_steps:
                add_reasoning_metrics_to_metadata(run_response, reasoning_state["reasoning_time_taken"])
                yield handle_event(  # type: ignore
                    create_team_reasoning_completed_event(
                        from_run_response=run_response,
                        content=ReasoningSteps(reasoning_steps=all_reasoning_steps),
                        content_type=ReasoningSteps.__name__,
                    ),
                    run_response,
                    events_to_skip=self.events_to_skip,
                    store_events=self.store_events,
                )

    def _handle_model_response_chunk(
        self,
        session: TeamSession,
        run_response: TeamRunOutput,
        full_model_response: ModelResponse,
        model_response_event: Union[ModelResponse, TeamRunOutputEvent, RunOutputEvent],
        reasoning_state: Optional[Dict[str, Any]] = None,
        stream_events: bool = False,
        parse_structured_output: bool = False,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Union[TeamRunOutputEvent, RunOutputEvent]]:
        if isinstance(model_response_event, tuple(get_args(RunOutputEvent))) or isinstance(
            model_response_event, tuple(get_args(TeamRunOutputEvent))
        ):
            if self.stream_member_events:
                if model_response_event.event == TeamRunEvent.custom_event:  # type: ignore
                    if hasattr(model_response_event, "team_id"):
                        model_response_event.team_id = self.id
                    if hasattr(model_response_event, "team_name"):
                        model_response_event.team_name = self.name
                    if not model_response_event.session_id:  # type: ignore
                        model_response_event.session_id = session.session_id  # type: ignore
                    if not model_response_event.run_id:  # type: ignore
                        model_response_event.run_id = run_response.run_id  # type: ignore
                # We just bubble the event up
                yield handle_event(  # type: ignore
                    model_response_event,  # type: ignore
                    run_response,
                    events_to_skip=self.events_to_skip,
                    store_events=self.store_events,
                )  # type: ignore
            else:
                # Don't yield anything
                return
        else:
            model_response_event = cast(ModelResponse, model_response_event)
            # If the model response is an assistant_response, yield a RunOutput
            if model_response_event.event == ModelResponseEvent.assistant_response.value:
                content_type = "str"

                should_yield = False
                # Process content
                if model_response_event.content is not None:
                    if parse_structured_output:
                        full_model_response.content = model_response_event.content
                        self._convert_response_to_structured_format(full_model_response)
                        content_type = self.output_schema.__name__  # type: ignore
                        run_response.content_type = content_type
                    elif self._member_response_model is not None:
                        full_model_response.content = model_response_event.content
                        self._convert_response_to_structured_format(full_model_response)
                        content_type = self._member_response_model.__name__  # type: ignore
                        run_response.content_type = content_type
                    elif isinstance(model_response_event.content, str):
                        full_model_response.content = (full_model_response.content or "") + model_response_event.content
                    should_yield = True

                # Process reasoning content
                if model_response_event.reasoning_content is not None:
                    full_model_response.reasoning_content = (
                        full_model_response.reasoning_content or ""
                    ) + model_response_event.reasoning_content
                    run_response.reasoning_content = full_model_response.reasoning_content
                    should_yield = True

                if model_response_event.redacted_reasoning_content is not None:
                    if not full_model_response.reasoning_content:
                        full_model_response.reasoning_content = model_response_event.redacted_reasoning_content
                    else:
                        full_model_response.reasoning_content += model_response_event.redacted_reasoning_content
                    run_response.reasoning_content = full_model_response.reasoning_content
                    should_yield = True

                # Handle provider data (one chunk)
                if model_response_event.provider_data is not None:
                    run_response.model_provider_data = model_response_event.provider_data

                # Handle citations (one chunk)
                if model_response_event.citations is not None:
                    run_response.citations = model_response_event.citations

                # Process audio
                if model_response_event.audio is not None:
                    if full_model_response.audio is None:
                        full_model_response.audio = Audio(id=str(uuid4()), content=b"", transcript="")

                    if model_response_event.audio.id is not None:
                        full_model_response.audio.id = model_response_event.audio.id  # type: ignore

                    if model_response_event.audio.content is not None:
                        # Handle both base64 string and bytes content
                        if isinstance(model_response_event.audio.content, str):
                            # Decode base64 string to bytes
                            try:
                                import base64

                                decoded_content = base64.b64decode(model_response_event.audio.content)
                                if full_model_response.audio.content is None:
                                    full_model_response.audio.content = b""
                                full_model_response.audio.content += decoded_content
                            except Exception:
                                # If decode fails, encode string as bytes
                                if full_model_response.audio.content is None:
                                    full_model_response.audio.content = b""
                                full_model_response.audio.content += model_response_event.audio.content.encode("utf-8")
                        elif isinstance(model_response_event.audio.content, bytes):
                            # Content is already bytes
                            if full_model_response.audio.content is None:
                                full_model_response.audio.content = b""
                            full_model_response.audio.content += model_response_event.audio.content

                    if model_response_event.audio.transcript is not None:
                        if full_model_response.audio.transcript is None:
                            full_model_response.audio.transcript = ""
                        full_model_response.audio.transcript += model_response_event.audio.transcript  # type: ignore
                    if model_response_event.audio.expires_at is not None:
                        full_model_response.audio.expires_at = model_response_event.audio.expires_at  # type: ignore
                    if model_response_event.audio.mime_type is not None:
                        full_model_response.audio.mime_type = model_response_event.audio.mime_type  # type: ignore
                    if model_response_event.audio.sample_rate is not None:
                        full_model_response.audio.sample_rate = model_response_event.audio.sample_rate
                    if model_response_event.audio.channels is not None:
                        full_model_response.audio.channels = model_response_event.audio.channels

                    # Yield the audio and transcript bit by bit
                    should_yield = True

                if model_response_event.images is not None:
                    for image in model_response_event.images:
                        self._add_image(image, run_response)

                    should_yield = True

                # Only yield the chunk
                if should_yield:
                    if content_type == "str":
                        yield handle_event(  # type: ignore
                            create_team_run_output_content_event(
                                from_run_response=run_response,
                                content=model_response_event.content,
                                reasoning_content=model_response_event.reasoning_content,
                                redacted_reasoning_content=model_response_event.redacted_reasoning_content,
                                response_audio=full_model_response.audio,
                                citations=model_response_event.citations,
                                model_provider_data=model_response_event.provider_data,
                                image=model_response_event.images[-1] if model_response_event.images else None,
                            ),
                            run_response,
                            events_to_skip=self.events_to_skip,
                            store_events=self.store_events,
                        )
                    else:
                        yield handle_event(  # type: ignore
                            create_team_run_output_content_event(
                                from_run_response=run_response,
                                content=full_model_response.content,
                                content_type=content_type,
                            ),
                            run_response,
                            events_to_skip=self.events_to_skip,
                            store_events=self.store_events,
                        )

            # If the model response is a tool_call_started, add the tool call to the run_response
            elif model_response_event.event == ModelResponseEvent.tool_call_started.value:
                # Add tool calls to the run_response
                tool_executions_list = model_response_event.tool_executions
                if tool_executions_list is not None:
                    # Add tool calls to the agent.run_response
                    if run_response.tools is None:
                        run_response.tools = tool_executions_list
                    else:
                        run_response.tools.extend(tool_executions_list)

                    for tool in tool_executions_list:
                        yield handle_event(  # type: ignore
                            create_team_tool_call_started_event(
                                from_run_response=run_response,
                                tool=tool,
                            ),
                            run_response,
                            events_to_skip=self.events_to_skip,
                            store_events=self.store_events,
                        )

            # If the model response is a tool_call_completed, update the existing tool call in the run_response
            elif model_response_event.event == ModelResponseEvent.tool_call_completed.value:
                if model_response_event.updated_session_state is not None:
                    # Update the session_state variable that TeamRunOutput references
                    if session_state is not None:
                        merge_dictionaries(session_state, model_response_event.updated_session_state)
                    # Also update the DB session object
                    if session.session_data is not None:
                        merge_dictionaries(
                            session.session_data["session_state"], model_response_event.updated_session_state
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
                        for tool_execution in tool_executions_list:
                            tool_call_id = tool_execution.tool_call_id or ""
                            index = tool_call_index_map.get(tool_call_id)
                            if index is not None:
                                if run_response.tools[index].child_run_id is not None:
                                    tool_execution.child_run_id = run_response.tools[index].child_run_id
                                run_response.tools[index] = tool_execution
                    else:
                        run_response.tools = tool_executions_list

                    # Only iterate through new tool calls
                    for tool_call in tool_executions_list:
                        tool_name = tool_call.tool_name or ""
                        if tool_name.lower() in ["think", "analyze"]:
                            tool_args = tool_call.tool_args or {}

                            reasoning_step = self._update_reasoning_content_from_tool_call(
                                run_response, tool_name, tool_args
                            )

                            metrics = tool_call.metrics
                            if metrics is not None and metrics.duration is not None and reasoning_state is not None:
                                reasoning_state["reasoning_time_taken"] = reasoning_state[
                                    "reasoning_time_taken"
                                ] + float(metrics.duration)

                        yield handle_event(  # type: ignore
                            create_team_tool_call_completed_event(
                                from_run_response=run_response,
                                tool=tool_call,
                                content=model_response_event.content,
                            ),
                            run_response,
                            events_to_skip=self.events_to_skip,
                            store_events=self.store_events,
                        )

                if stream_events:
                    if reasoning_step is not None:
                        if reasoning_state is not None and not reasoning_state["reasoning_started"]:
                            yield handle_event(  # type: ignore
                                create_team_reasoning_started_event(
                                    from_run_response=run_response,
                                ),
                                run_response,
                                events_to_skip=self.events_to_skip,
                                store_events=self.store_events,
                            )
                            reasoning_state["reasoning_started"] = True

                        yield handle_event(  # type: ignore
                            create_team_reasoning_step_event(
                                from_run_response=run_response,
                                reasoning_step=reasoning_step,
                                reasoning_content=run_response.reasoning_content or "",
                            ),
                            run_response,
                            events_to_skip=self.events_to_skip,
                            store_events=self.store_events,
                        )

    def _convert_response_to_structured_format(self, run_response: Union[TeamRunOutput, RunOutput, ModelResponse]):
        # Convert the response to the structured format if needed
        if self.output_schema is not None and not isinstance(run_response.content, self.output_schema):
            if isinstance(run_response.content, str) and self.parse_response:
                try:
                    parsed_response_content = parse_response_model_str(run_response.content, self.output_schema)

                    # Update TeamRunOutput
                    if parsed_response_content is not None:
                        run_response.content = parsed_response_content
                        if hasattr(run_response, "content_type"):
                            run_response.content_type = self.output_schema.__name__
                    else:
                        log_warning("Failed to convert response to output_schema")
                except Exception as e:
                    log_warning(f"Failed to convert response to output model: {e}")
            else:
                log_warning("Something went wrong. Team run response content is not a string")
        elif self._member_response_model is not None and not isinstance(
            run_response.content, self._member_response_model
        ):
            if isinstance(run_response.content, str):
                try:
                    parsed_response_content = parse_response_model_str(
                        run_response.content, self._member_response_model
                    )
                    # Update TeamRunOutput
                    if parsed_response_content is not None:
                        run_response.content = parsed_response_content
                        if hasattr(run_response, "content_type"):
                            run_response.content_type = self._member_response_model.__name__
                    else:
                        log_warning("Failed to convert response to output_schema")
                except Exception as e:
                    log_warning(f"Failed to convert response to output model: {e}")
            else:
                log_warning("Something went wrong. Member run response content is not a string")

    def _cleanup_and_store(self, run_response: TeamRunOutput, session: TeamSession) -> None:
        #  Scrub the stored run based on storage flags
        self._scrub_run_output_for_storage(run_response)

        # Stop the timer for the Run duration
        if run_response.metrics:
            run_response.metrics.stop_timer()

        # Add RunOutput to Agent Session
        session.upsert_run(run_response=run_response)

        # Calculate session metrics
        self._update_session_metrics(session=session)

        # Save session to memory
        self.save_session(session=session)

    async def _acleanup_and_store(self, run_response: TeamRunOutput, session: TeamSession) -> None:
        #  Scrub the stored run based on storage flags
        self._scrub_run_output_for_storage(run_response)

        # Stop the timer for the Run duration
        if run_response.metrics:
            run_response.metrics.stop_timer()

        # Add RunOutput to Agent Session
        session.upsert_run(run_response=run_response)

        # Calculate session metrics
        self._update_session_metrics(session=session)

        # Save session to memory
        await self.asave_session(session=session)

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
            self.memory_manager.create_user_memories(
                message=user_message_str,
                user_id=user_id,
                team_id=self.id,
            )

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
            await self.memory_manager.acreate_user_memories(
                message=user_message_str,
                user_id=user_id,
                team_id=self.id,
            )

    def _get_response_format(self, model: Optional[Model] = None) -> Optional[Union[Dict, Type[BaseModel]]]:
        model = cast(Model, model or self.model)
        if self.output_schema is None:
            return None
        else:
            json_response_format = {"type": "json_object"}

            if model.supports_native_structured_outputs:
                if not self.use_json_mode:
                    log_debug("Setting Model.response_format to Agent.output_schema")
                    return self.output_schema
                else:
                    log_debug(
                        "Model supports native structured outputs but it is not enabled. Using JSON mode instead."
                    )
                    return json_response_format

            elif model.supports_json_schema_outputs:
                if self.use_json_mode:
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
                model_response, run_messages, parser_model_response, messages_for_parser_model
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
                model_response, run_messages, parser_model_response, messages_for_parser_model
            )
        else:
            log_warning("A response model is required to parse the response with a parser model")

    def _parse_response_with_parser_model_stream(
        self,
        session: TeamSession,
        run_response: TeamRunOutput,
        stream_events: bool = False,
    ):
        """Parse the model response using the parser model"""
        if self.parser_model is not None:
            if self.output_schema is not None:
                if stream_events:
                    yield handle_event(  # type: ignore
                        create_team_parser_model_response_started_event(run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,
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
                        full_model_response=parser_model_response,
                        model_response_event=model_response_event,
                        parse_structured_output=True,
                        stream_events=stream_events,
                    )

                run_response.content = parser_model_response.content

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
                    yield handle_event(  # type: ignore
                        create_team_parser_model_response_completed_event(run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,
                        store_events=self.store_events,
                    )

            else:
                log_warning("A response model is required to parse the response with a parser model")

    async def _aparse_response_with_parser_model_stream(
        self, session: TeamSession, run_response: TeamRunOutput, stream_events: bool = False
    ):
        """Parse the model response using the parser model stream."""
        if self.parser_model is not None:
            if self.output_schema is not None:
                if stream_events:
                    yield handle_event(  # type: ignore
                        create_team_parser_model_response_started_event(run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,
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
                        full_model_response=parser_model_response,
                        model_response_event=model_response_event,
                        parse_structured_output=True,
                        stream_events=stream_events,
                    ):
                        yield event

                run_response.content = parser_model_response.content

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
                    yield handle_event(  # type: ignore
                        create_team_parser_model_response_completed_event(run_response),
                        run_response,
                        events_to_skip=self.events_to_skip,
                        store_events=self.store_events,
                    )
            else:
                log_warning("A response model is required to parse the response with a parser model")

    def _parse_response_with_output_model(self, model_response: ModelResponse, run_messages: RunMessages) -> None:
        """Parse the model response using the output model."""
        if self.output_model is None:
            return

        messages_for_output_model = self._get_messages_for_output_model(run_messages.messages)
        output_model_response: ModelResponse = self.output_model.response(messages=messages_for_output_model)
        model_response.content = output_model_response.content

    def _generate_response_with_output_model_stream(
        self,
        session: TeamSession,
        run_response: TeamRunOutput,
        run_messages: RunMessages,
        stream_events: bool = False,
    ):
        """Parse the model response using the output model stream."""
        from agno.utils.events import (
            create_team_output_model_response_completed_event,
            create_team_output_model_response_started_event,
        )

        if self.output_model is None:
            return

        if stream_events:
            yield handle_event(  # type: ignore
                create_team_output_model_response_started_event(run_response),
                run_response,
                events_to_skip=self.events_to_skip,
                store_events=self.store_events,
            )

        messages_for_output_model = self._get_messages_for_output_model(run_messages.messages)
        model_response = ModelResponse(content="")

        for model_response_event in self.output_model.response_stream(messages=messages_for_output_model):
            yield from self._handle_model_response_chunk(
                session=session,
                run_response=run_response,
                full_model_response=model_response,
                model_response_event=model_response_event,
            )

        # Update the TeamRunResponse content
        run_response.content = model_response.content

        if stream_events:
            yield handle_event(  # type: ignore
                create_team_output_model_response_completed_event(run_response),
                run_response,
                events_to_skip=self.events_to_skip,
                store_events=self.store_events,
            )

        # Build a list of messages that should be added to the RunResponse
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        # Update the RunResponse messages
        run_response.messages = messages_for_run_response
        # Update the RunResponse metrics
        run_response.metrics = self._calculate_metrics(messages_for_run_response)

    async def _agenerate_response_with_output_model(
        self, model_response: ModelResponse, run_messages: RunMessages
    ) -> None:
        """Parse the model response using the output model stream."""
        if self.output_model is None:
            return

        messages_for_output_model = self._get_messages_for_output_model(run_messages.messages)
        output_model_response: ModelResponse = await self.output_model.aresponse(messages=messages_for_output_model)
        model_response.content = output_model_response.content

    async def _agenerate_response_with_output_model_stream(
        self,
        session: TeamSession,
        run_response: TeamRunOutput,
        run_messages: RunMessages,
        stream_events: bool = False,
    ):
        """Parse the model response using the output model stream."""
        from agno.utils.events import (
            create_team_output_model_response_completed_event,
            create_team_output_model_response_started_event,
        )

        if self.output_model is None:
            return

        if stream_events:
            yield handle_event(  # type: ignore
                create_team_output_model_response_started_event(run_response),
                run_response,
                events_to_skip=self.events_to_skip,
                store_events=self.store_events,
            )

        messages_for_output_model = self._get_messages_for_output_model(run_messages.messages)
        model_response = ModelResponse(content="")

        async for model_response_event in self.output_model.aresponse_stream(messages=messages_for_output_model):
            for event in self._handle_model_response_chunk(
                session=session,
                run_response=run_response,
                full_model_response=model_response,
                model_response_event=model_response_event,
            ):
                yield event

        # Update the TeamRunResponse content
        run_response.content = model_response.content

        if stream_events:
            yield handle_event(  # type: ignore
                create_team_output_model_response_completed_event(run_response),
                run_response,
                events_to_skip=self.events_to_skip,
                store_events=self.store_events,
            )

        # Build a list of messages that should be added to the RunResponse
        messages_for_run_response = [m for m in run_messages.messages if m.add_to_agent_memory]
        # Update the RunResponse messages
        run_response.messages = messages_for_run_response
        # Update the RunResponse metrics
        run_response.metrics = self._calculate_metrics(messages_for_run_response)

    def _handle_event(
        self,
        event: Union[RunOutputEvent, TeamRunOutputEvent],
        run_response: TeamRunOutput,
    ):
        # We only store events that are not run_response_content events
        events_to_skip = [event.value for event in self.events_to_skip] if self.events_to_skip else []
        if self.store_events and event.event not in events_to_skip:
            if run_response.events is None:
                run_response.events = []
            run_response.events.append(event)
        return event

    ###########################################################################
    # Print Response
    ###########################################################################

    def print_response(
        self,
        input: Union[List, Dict, str, Message, BaseModel, List[Message]],
        *,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        markdown: Optional[bool] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
        show_message: bool = True,
        show_reasoning: bool = True,
        show_full_reasoning: bool = False,
        console: Optional[Any] = None,
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
            print_response_stream(
                team=self,
                input=input,
                console=console,
                show_message=show_message,
                show_reasoning=show_reasoning,
                show_full_reasoning=show_full_reasoning,
                tags_to_include_in_markdown=tags_to_include_in_markdown,
                session_id=session_id,
                session_state=session_state,
                user_id=user_id,
                audio=audio,
                images=images,
                videos=videos,
                files=files,
                markdown=markdown,
                stream_events=stream_events,
                knowledge_filters=knowledge_filters,
                add_history_to_context=add_history_to_context,
                dependencies=dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=metadata,
                debug_mode=debug_mode,
                **kwargs,
            )
        else:
            print_response(
                team=self,
                input=input,
                console=console,
                show_message=show_message,
                show_reasoning=show_reasoning,
                show_full_reasoning=show_full_reasoning,
                tags_to_include_in_markdown=tags_to_include_in_markdown,
                session_id=session_id,
                session_state=session_state,
                user_id=user_id,
                audio=audio,
                images=images,
                videos=videos,
                files=files,
                markdown=markdown,
                knowledge_filters=knowledge_filters,
                add_history_to_context=add_history_to_context,
                dependencies=dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=metadata,
                debug_mode=debug_mode,
                **kwargs,
            )

    async def aprint_response(
        self,
        input: Union[List, Dict, str, Message, BaseModel, List[Message]],
        *,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
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
                team=self,
                input=input,
                console=console,
                show_message=show_message,
                show_reasoning=show_reasoning,
                show_full_reasoning=show_full_reasoning,
                tags_to_include_in_markdown=tags_to_include_in_markdown,
                session_id=session_id,
                session_state=session_state,
                user_id=user_id,
                audio=audio,
                images=images,
                videos=videos,
                files=files,
                markdown=markdown,
                stream_events=stream_events,
                knowledge_filters=knowledge_filters,
                add_history_to_context=add_history_to_context,
                dependencies=dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=metadata,
                debug_mode=debug_mode,
                **kwargs,
            )
        else:
            await aprint_response(
                team=self,
                input=input,
                console=console,
                show_message=show_message,
                show_reasoning=show_reasoning,
                show_full_reasoning=show_full_reasoning,
                tags_to_include_in_markdown=tags_to_include_in_markdown,
                session_id=session_id,
                session_state=session_state,
                user_id=user_id,
                audio=audio,
                images=images,
                videos=videos,
                files=files,
                markdown=markdown,
                knowledge_filters=knowledge_filters,
                add_history_to_context=add_history_to_context,
                dependencies=dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=metadata,
                debug_mode=debug_mode,
                **kwargs,
            )

    def _get_member_name(self, entity_id: str) -> str:
        for member in self.members:
            if isinstance(member, Agent):
                if member.id == entity_id:
                    return member.name or entity_id
            elif isinstance(member, Team):
                if member.id == entity_id:
                    return member.name or entity_id
        return entity_id

    def _scrub_run_output_for_storage(self, run_response: TeamRunOutput) -> bool:
        """
        Scrub run output based on storage flags before persisting to database.
        Returns True if any scrubbing was done, False otherwise.
        """
        scrubbed = False

        if not self.store_media:
            scrub_media_from_run_output(run_response)
            scrubbed = True

        if not self.store_tool_messages:
            scrub_tool_results_from_run_output(run_response)
            scrubbed = True

        if not self.store_history_messages:
            scrub_history_messages_from_run_output(run_response)
            scrubbed = True

        return scrubbed

    def _scrub_member_responses(self, member_responses: List[Union[TeamRunOutput, RunOutput]]) -> None:
        """
        Scrub member responses based on each member's storage flags.
        This is called when saving the team session to ensure member data is scrubbed per member settings.
        Recursively handles nested team's member responses.
        """
        for member_response in member_responses:
            member_id = None
            if isinstance(member_response, RunOutput):
                member_id = member_response.agent_id
            elif isinstance(member_response, TeamRunOutput):
                member_id = member_response.team_id

            if not member_id:
                log_info("Skipping member response with no ID")
                continue

            member_result = self._find_member_by_id(member_id)
            if not member_result:
                log_debug(f"Could not find member with ID: {member_id}")
                continue

            _, member = member_result

            if not member.store_media or not member.store_tool_messages or not member.store_history_messages:
                member._scrub_run_output_for_storage(member_response)  # type: ignore

            # If this is a nested team, recursively scrub its member responses
            if isinstance(member_response, TeamRunOutput) and member_response.member_responses:
                member._scrub_member_responses(member_response.member_responses)  # type: ignore

    def _validate_media_object_id(
        self,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        audios: Optional[Sequence[Audio]] = None,
        files: Optional[Sequence[File]] = None,
    ) -> tuple:
        image_list = None
        if images:
            image_list = []
            for img in images:
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
        user: str = "User",
        emoji: str = ":sunglasses:",
        stream: bool = False,
        markdown: bool = False,
        exit_on: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run an interactive command-line interface to interact with the team."""

        from inspect import isawaitable

        from rich.prompt import Prompt

        # Ensuring the team is not using async tools
        if self.tools is not None:
            for tool in self.tools:
                if isawaitable(tool):
                    raise NotImplementedError("Use `acli_app` to use async tools.")
                if tool.__class__.__name__ in ["MCPTools", "MultiMCPTools"]:
                    raise NotImplementedError("Use `acli_app` to use MCP tools.")

        if input:
            self.print_response(input=input, stream=stream, markdown=markdown, **kwargs)

        _exit_on = exit_on or ["exit", "quit", "bye"]
        while True:
            user_input = Prompt.ask(f"[bold] {emoji} {user} [/bold]")
            if user_input in _exit_on:
                break

            self.print_response(input=user_input, stream=stream, markdown=markdown, **kwargs)

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
        Run an interactive command-line interface to interact with the team.
        Works with team dependencies requiring async logic.
        """
        from rich.prompt import Prompt

        if input:
            await self.aprint_response(
                input=input, stream=stream, markdown=markdown, user_id=user_id, session_id=session_id, **kwargs
            )

        _exit_on = exit_on or ["exit", "quit", "bye"]
        while True:
            message = Prompt.ask(f"[bold] {emoji} {user} [/bold]")
            if message in _exit_on:
                break

            await self.aprint_response(
                input=message, stream=stream, markdown=markdown, user_id=user_id, session_id=session_id, **kwargs
            )

    ###########################################################################
    # Helpers
    ###########################################################################

    def _handle_reasoning(self, run_response: TeamRunOutput, run_messages: RunMessages):
        if self.reasoning or self.reasoning_model is not None:
            reasoning_generator = self._reason(
                run_response=run_response, run_messages=run_messages, stream_events=False
            )

            # Consume the generator without yielding
            deque(reasoning_generator, maxlen=0)

    def _handle_reasoning_stream(
        self, run_response: TeamRunOutput, run_messages: RunMessages, stream_events: bool
    ) -> Iterator[TeamRunOutputEvent]:
        if self.reasoning or self.reasoning_model is not None:
            reasoning_generator = self._reason(
                run_response=run_response,
                run_messages=run_messages,
                stream_events=stream_events,
            )
            yield from reasoning_generator

    async def _ahandle_reasoning(self, run_response: TeamRunOutput, run_messages: RunMessages) -> None:
        if self.reasoning or self.reasoning_model is not None:
            reason_generator = self._areason(run_response=run_response, run_messages=run_messages, stream_events=False)
            # Consume the generator without yielding
            async for _ in reason_generator:
                pass

    async def _ahandle_reasoning_stream(
        self, run_response: TeamRunOutput, run_messages: RunMessages, stream_events: bool
    ) -> AsyncIterator[TeamRunOutputEvent]:
        if self.reasoning or self.reasoning_model is not None:
            reason_generator = self._areason(
                run_response=run_response,
                run_messages=run_messages,
                stream_events=stream_events,
            )
            async for item in reason_generator:
                yield item

    def _calculate_session_metrics(self, messages: List[Message]) -> Metrics:
        """Sum the metrics of the given messages into a Metrics object"""
        session_metrics = Metrics()
        assistant_message_role = self.model.assistant_message_role if self.model is not None else "assistant"

        # Get metrics of the team leader's messages
        for m in messages:
            if m.role == assistant_message_role and m.metrics is not None:
                session_metrics += m.metrics

        return session_metrics

    def _calculate_metrics(self, messages: List[Message]) -> Metrics:
        metrics = Metrics()
        assistant_message_role = self.model.assistant_message_role if self.model is not None else "assistant"

        for m in messages:
            if m.role == assistant_message_role and m.metrics is not None and m.from_history is False:
                metrics += m.metrics

        return metrics

    def _update_session_metrics(self, session: TeamSession):
        """Calculate session metrics"""

        session_messages: List[Message] = []
        for run in session.runs:  # type: ignore
            if run.messages is not None:
                for m in run.messages:
                    # Skipping messages from history to avoid duplicates
                    if not m.from_history:
                        session_messages.append(m)

        # Calculate initial metrics
        session_metrics = self._calculate_session_metrics(session_messages)

        if session.session_data is not None:
            session.session_data["session_metrics"] = session_metrics

    def _get_reasoning_agent(self, reasoning_model: Model) -> Optional[Agent]:
        return Agent(
            model=reasoning_model,
            telemetry=self.telemetry,
            debug_mode=self.debug_mode,
            debug_level=self.debug_level,
        )

    def _format_reasoning_step_content(self, run_response: TeamRunOutput, reasoning_step: ReasoningStep) -> str:
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
        if hasattr(run_response, "reasoning_content") and run_response.reasoning_content:
            current_reasoning_content = run_response.reasoning_content

        # Create updated reasoning_content
        updated_reasoning_content = current_reasoning_content + step_content

        return updated_reasoning_content

    def _reason(
        self,
        run_response: TeamRunOutput,
        run_messages: RunMessages,
        stream_events: bool,
    ) -> Iterator[TeamRunOutputEvent]:
        if stream_events:
            yield handle_event(  # type: ignore
                create_team_reasoning_started_event(from_run_response=run_response),
                run_response,
                events_to_skip=self.events_to_skip,
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
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_groq:
                    from agno.reasoning.groq import get_groq_reasoning

                    log_debug("Starting Groq Reasoning", center=True, symbol="=")
                    reasoning_message = get_groq_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_openai:
                    from agno.reasoning.openai import get_openai_reasoning

                    log_debug("Starting OpenAI Reasoning", center=True, symbol="=")
                    reasoning_message = get_openai_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_ollama:
                    from agno.reasoning.ollama import get_ollama_reasoning

                    log_debug("Starting Ollama Reasoning", center=True, symbol="=")
                    reasoning_message = get_ollama_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_ai_foundry:
                    from agno.reasoning.azure_ai_foundry import get_ai_foundry_reasoning

                    log_debug("Starting Azure AI Foundry Reasoning", center=True, symbol="=")
                    reasoning_message = get_ai_foundry_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_gemini:
                    from agno.reasoning.gemini import get_gemini_reasoning

                    log_debug("Starting Gemini Reasoning", center=True, symbol="=")
                    reasoning_message = get_gemini_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_anthropic:
                    from agno.reasoning.anthropic import get_anthropic_reasoning

                    log_debug("Starting Anthropic Claude Reasoning", center=True, symbol="=")
                    reasoning_message = get_anthropic_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_vertexai:
                    from agno.reasoning.vertexai import get_vertexai_reasoning

                    log_debug("Starting VertexAI Reasoning", center=True, symbol="=")
                    reasoning_message = get_vertexai_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
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
                        create_team_reasoning_completed_event(
                            from_run_response=run_response,
                            content=ReasoningSteps(reasoning_steps=[ReasoningStep(result=reasoning_message.content)]),
                            content_type=ReasoningSteps.__name__,
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,
                        store_events=self.store_events,
                    )
            else:
                log_info(
                    f"Reasoning model: {reasoning_model.__class__.__name__} is not a native reasoning model, defaulting to manual Chain-of-Thought reasoning"
                )
                use_default_reasoning = True
        # If no reasoning model is provided, use default reasoning
        else:
            use_default_reasoning = True

        if use_default_reasoning:
            from agno.reasoning.default import get_default_reasoning_agent
            from agno.reasoning.helpers import get_next_action, update_messages_with_reasoning

            # Get default reasoning agent
            use_json_mode: bool = self.use_json_mode

            reasoning_agent: Optional[Agent] = self.reasoning_agent  # type: ignore
            if reasoning_agent is None:
                reasoning_agent = get_default_reasoning_agent(
                    reasoning_model=reasoning_model,
                    min_steps=self.reasoning_min_steps,
                    max_steps=self.reasoning_max_steps,
                    tool_call_limit=self.tool_call_limit,
                    telemetry=self.telemetry,
                    debug_mode=self.debug_mode,
                    debug_level=self.debug_level,
                    use_json_mode=use_json_mode,
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
            # Ensure the reasoning model and agent do not show tool calls

            step_count = 1
            next_action = NextAction.CONTINUE
            reasoning_messages: List[Message] = []
            all_reasoning_steps: List[ReasoningStep] = []
            log_debug("Starting Reasoning", center=True, symbol="=")
            while next_action == NextAction.CONTINUE and step_count < self.reasoning_max_steps:
                log_debug(f"Step {step_count}", center=True, symbol="-")
                step_count += 1
                try:
                    # Run the reasoning agent
                    reasoning_agent_response: RunOutput = reasoning_agent.run(  # type: ignore
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
                                run_response, reasoning_step
                            )

                            yield handle_event(  # type: ignore
                                create_team_reasoning_step_event(
                                    from_run_response=run_response,
                                    reasoning_step=reasoning_step,
                                    reasoning_content=updated_reasoning_content,
                                ),
                                run_response,
                                events_to_skip=self.events_to_skip,
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
                yield handle_event(  # type: ignore
                    create_team_reasoning_completed_event(
                        from_run_response=run_response,
                        content=ReasoningSteps(reasoning_steps=all_reasoning_steps),
                        content_type=ReasoningSteps.__name__,
                    ),
                    run_response,
                    events_to_skip=self.events_to_skip,
                    store_events=self.store_events,
                )

    async def _areason(
        self,
        run_response: TeamRunOutput,
        run_messages: RunMessages,
        stream_events: bool,
    ) -> AsyncIterator[TeamRunOutputEvent]:
        if stream_events:
            yield handle_event(  # type: ignore
                create_team_reasoning_started_event(from_run_response=run_response),
                run_response,
                events_to_skip=self.events_to_skip,
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
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_groq:
                    from agno.reasoning.groq import aget_groq_reasoning

                    log_debug("Starting Groq Reasoning", center=True, symbol="=")
                    reasoning_message = await aget_groq_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_openai:
                    from agno.reasoning.openai import aget_openai_reasoning

                    log_debug("Starting OpenAI Reasoning", center=True, symbol="=")
                    reasoning_message = await aget_openai_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_ollama:
                    from agno.reasoning.ollama import get_ollama_reasoning

                    log_debug("Starting Ollama Reasoning", center=True, symbol="=")
                    reasoning_message = get_ollama_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_ai_foundry:
                    from agno.reasoning.azure_ai_foundry import get_ai_foundry_reasoning

                    log_debug("Starting Azure AI Foundry Reasoning", center=True, symbol="=")
                    reasoning_message = get_ai_foundry_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_gemini:
                    from agno.reasoning.gemini import aget_gemini_reasoning

                    log_debug("Starting Gemini Reasoning", center=True, symbol="=")
                    reasoning_message = await aget_gemini_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_anthropic:
                    from agno.reasoning.anthropic import aget_anthropic_reasoning

                    log_debug("Starting Anthropic Claude Reasoning", center=True, symbol="=")
                    reasoning_message = await aget_anthropic_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
                    )
                elif is_vertexai:
                    from agno.reasoning.vertexai import aget_vertexai_reasoning

                    log_debug("Starting VertexAI Reasoning", center=True, symbol="=")
                    reasoning_message = await aget_vertexai_reasoning(
                        reasoning_agent=reasoning_agent, messages=run_messages.get_input_messages()
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
                        create_team_reasoning_completed_event(
                            from_run_response=run_response,
                            content=ReasoningSteps(reasoning_steps=[ReasoningStep(result=reasoning_message.content)]),
                            content_type=ReasoningSteps.__name__,
                        ),
                        run_response,
                        events_to_skip=self.events_to_skip,
                        store_events=self.store_events,
                    )
            else:
                log_info(
                    f"Reasoning model: {reasoning_model.__class__.__name__} is not a native reasoning model, defaulting to manual Chain-of-Thought reasoning"
                )
                use_default_reasoning = True
        # If no reasoning model is provided, use default reasoning
        else:
            use_default_reasoning = True

        if use_default_reasoning:
            from agno.reasoning.default import get_default_reasoning_agent
            from agno.reasoning.helpers import get_next_action, update_messages_with_reasoning

            # Get default reasoning agent
            use_json_mode: bool = self.use_json_mode
            reasoning_agent: Optional[Agent] = self.reasoning_agent  # type: ignore
            if reasoning_agent is None:
                reasoning_agent = get_default_reasoning_agent(  # type: ignore
                    reasoning_model=reasoning_model,
                    min_steps=self.reasoning_min_steps,
                    max_steps=self.reasoning_max_steps,
                    telemetry=self.telemetry,
                    debug_mode=self.debug_mode,
                    debug_level=self.debug_level,
                    use_json_mode=use_json_mode,
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

            # Ensure the reasoning model and agent do not show tool calls

            step_count = 1
            next_action = NextAction.CONTINUE
            reasoning_messages: List[Message] = []
            all_reasoning_steps: List[ReasoningStep] = []
            log_debug("Starting Reasoning", center=True, symbol="=")
            while next_action == NextAction.CONTINUE and step_count < self.reasoning_max_steps:
                log_debug(f"Step {step_count}", center=True, symbol="-")
                step_count += 1
                try:
                    # Run the reasoning agent
                    reasoning_agent_response: RunOutput = await reasoning_agent.arun(  # type: ignore
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
                                run_response, reasoning_step
                            )

                            yield handle_event(  # type: ignore
                                create_team_reasoning_step_event(
                                    from_run_response=run_response,
                                    reasoning_step=reasoning_step,
                                    reasoning_content=updated_reasoning_content,
                                ),
                                run_response,
                                events_to_skip=self.events_to_skip,
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
                yield handle_event(  # type: ignore  # type: ignore
                    create_team_reasoning_completed_event(
                        from_run_response=run_response,
                        content=ReasoningSteps(reasoning_steps=all_reasoning_steps),
                        content_type=ReasoningSteps.__name__,
                    ),
                    run_response,
                    events_to_skip=self.events_to_skip,
                    store_events=self.store_events,
                )

    def _resolve_run_dependencies(self, dependencies: Optional[Dict[str, Any]] = None) -> None:
        from inspect import signature

        log_debug("Resolving dependencies")
        if not isinstance(dependencies, dict):
            log_warning("Dependencies is not a dict")
            return

        for key, value in dependencies.items():
            if not callable(value):
                dependencies[key] = value
                continue

            try:
                sig = signature(value)
                resolved_value = value(agent=self) if "agent" in sig.parameters else value()

                dependencies[key] = resolved_value
            except Exception as e:
                log_warning(f"Failed to resolve dependencies for {key}: {e}")

    async def _aresolve_run_dependencies(self, dependencies: Optional[Dict[str, Any]] = None) -> None:
        from inspect import iscoroutine, signature

        log_debug("Resolving context (async)")
        if not isinstance(dependencies, dict):
            log_warning("Dependencies is not a dict")
            return

        for key, value in dependencies.items():
            if not callable(value):
                dependencies[key] = value
                continue

            try:
                sig = signature(value)
                resolved_value = value(team=self) if "team" in sig.parameters else value()

                if iscoroutine(resolved_value):
                    resolved_value = await resolved_value

                dependencies[key] = resolved_value
            except Exception as e:
                log_warning(f"Failed to resolve context for '{key}': {e}")

    async def _check_and_refresh_mcp_tools(self) -> None:
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
                            log_warning(f"Failed to check if MCP tool is alive: {e}")
                            continue

                        try:
                            await tool.build_tools()  # type: ignore
                        except (RuntimeError, BaseException) as e:
                            log_warning(f"Failed to build tools for {str(tool)}: {e}")
                            continue

    def _determine_tools_for_model(
        self,
        model: Model,
        run_response: TeamRunOutput,
        team_run_context: Dict[str, Any],
        session: TeamSession,
        session_state: Dict[str, Any],
        user_id: Optional[str] = None,
        async_mode: bool = False,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        input_message: Optional[Union[str, List, Dict, Message, BaseModel, List[Message]]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        audio: Optional[Sequence[Audio]] = None,
        files: Optional[Sequence[File]] = None,
        debug_mode: Optional[bool] = None,
        add_history_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        check_mcp_tools: bool = True,
    ) -> List[Union[Function, dict]]:
        # Prepare tools
        _tools: List[Union[Toolkit, Callable, Function, Dict]] = []

        # Add provided tools
        if self.tools is not None:
            for tool in self.tools:
                if tool.__class__.__name__ in ["MCPTools", "MultiMCPTools"]:
                    # Only add the tool if it successfully connected and built its tools
                    if check_mcp_tools and not tool.initialized:  # type: ignore
                        continue
                _tools.append(tool)

        if self.read_chat_history:
            _tools.append(self._get_chat_history_function(session=session, async_mode=async_mode))

        if self.memory_manager is not None and self.enable_agentic_memory:
            _tools.append(self._get_update_user_memory_function(user_id=user_id, async_mode=async_mode))

        if self.enable_agentic_state:
            _tools.append(Function(name="update_session_state", entrypoint=self._update_session_state_tool))

        if self.search_session_history:
            _tools.append(
                self._get_previous_sessions_messages_function(
                    num_history_sessions=self.num_history_sessions, user_id=user_id, async_mode=async_mode
                )
            )

        if self.knowledge is not None or self.knowledge_retriever is not None:
            # Check if knowledge retriever is an async function but used in sync mode
            from inspect import iscoroutinefunction

            if self.knowledge_retriever is not None and iscoroutinefunction(self.knowledge_retriever):
                log_warning(
                    "Async knowledge retriever function is being used with synchronous agent.run() or agent.print_response(). "
                    "It is recommended to use agent.arun() or agent.aprint_response() instead."
                )

            if self.search_knowledge:
                # Use async or sync search based on async_mode
                if self.enable_agentic_knowledge_filters:
                    _tools.append(
                        self._get_search_knowledge_base_with_agentic_filters_function(
                            run_response=run_response, knowledge_filters=knowledge_filters, async_mode=async_mode
                        )
                    )
                else:
                    _tools.append(
                        self._get_search_knowledge_base_function(
                            run_response=run_response, knowledge_filters=knowledge_filters, async_mode=async_mode
                        )
                    )

        if self.knowledge is not None and self.update_knowledge:
            _tools.append(self.add_to_knowledge)

        if self.members:
            # Get the user message if we are using the input directly
            user_message_content = None
            if self.determine_input_for_members is False:
                user_message = self._get_user_message(
                    run_response=run_response,
                    session_state=session_state,
                    input_message=input_message,
                    user_id=user_id,
                    audio=audio,
                    images=images,
                    videos=videos,
                    files=files,
                    dependencies=dependencies,
                    add_dependencies_to_context=add_dependencies_to_context,
                    metadata=metadata,
                )
                user_message_content = user_message.content if user_message is not None else None

            delegate_task_func = self._get_delegate_task_function(
                run_response=run_response,
                session=session,
                session_state=session_state,
                team_run_context=team_run_context,
                input=user_message_content,
                user_id=user_id,
                stream=self.stream or False,
                stream_events=self.stream_events or False,
                async_mode=async_mode,
                images=images,  # type: ignore
                videos=videos,  # type: ignore
                audio=audio,  # type: ignore
                files=files,  # type: ignore
                knowledge_filters=knowledge_filters,
                add_history_to_context=add_history_to_context,
                dependencies=dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=metadata,
                debug_mode=debug_mode,
            )

            _tools.append(delegate_task_func)
            if self.get_member_information_tool:
                _tools.append(self.get_member_information)

        # Get Agent tools
        if len(_tools) > 0:
            log_debug("Processing tools for model")

        _function_names = []
        _functions: List[Union[Function, dict]] = []

        # Check if we need strict mode for the model
        strict = False
        if self.output_schema is not None and not self.use_json_mode and model.supports_native_structured_outputs:
            strict = True

        for tool in _tools:
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

                    _func._team = self
                    _func.process_entrypoint(strict=strict)
                    if strict:
                        _func.strict = True
                    if self.tool_hooks:
                        _func.tool_hooks = self.tool_hooks
                    _functions.append(_func)
                    log_debug(f"Added tool {_func.name} from {tool.name}")

                # Add instructions from the toolkit
                if tool.add_instructions and tool.instructions is not None:
                    if self._tool_instructions is None:
                        self._tool_instructions = []
                    self._tool_instructions.append(tool.instructions)

            elif isinstance(tool, Function):
                if tool.name in _function_names:
                    continue
                _function_names.append(tool.name)
                tool = tool.model_copy(deep=True)
                tool._team = self
                tool.process_entrypoint(strict=strict)
                if strict and tool.strict is None:
                    tool.strict = True
                if self.tool_hooks:
                    tool.tool_hooks = self.tool_hooks
                _functions.append(tool)
                log_debug(f"Added tool {tool.name}")

                # Add instructions from the Function
                if tool.add_instructions and tool.instructions is not None:
                    if self._tool_instructions is None:
                        self._tool_instructions = []
                    self._tool_instructions.append(tool.instructions)

            elif callable(tool):
                # We add the tools, which are callable functions
                try:
                    _func = Function.from_callable(tool, strict=strict)
                    _func = _func.model_copy(deep=True)
                    if _func.name in _function_names:
                        continue
                    _function_names.append(_func.name)

                    _func._team = self
                    if strict:
                        _func.strict = True
                    if self.tool_hooks:
                        _func.tool_hooks = self.tool_hooks
                    _functions.append(_func)
                    log_debug(f"Added tool {_func.name}")
                except Exception as e:
                    log_warning(f"Could not add tool {tool}: {e}")

        if _functions:
            from inspect import signature

            # Check if any functions need media before collecting
            needs_media = any(
                any(param in signature(func.entrypoint).parameters for param in ["images", "videos", "audios", "files"])
                for func in _functions
                if isinstance(func, Function) and func.entrypoint is not None
            )

            # Only collect media if functions actually need them
            joint_images = collect_joint_images(run_response.input, session) if needs_media else None  # type: ignore
            joint_files = collect_joint_files(run_response.input) if needs_media else None  # type: ignore
            joint_audios = collect_joint_audios(run_response.input, session) if needs_media else None  # type: ignore
            joint_videos = collect_joint_videos(run_response.input, session) if needs_media else None  # type: ignore

            for func in _functions:  # type: ignore
                if isinstance(func, Function):
                    func._session_state = session_state
                    func._dependencies = dependencies
                    func._images = joint_images
                    func._files = joint_files
                    func._audios = joint_audios
                    func._videos = joint_videos

        return _functions

    def get_members_system_message_content(self, indent: int = 0) -> str:
        system_message_content = ""
        for idx, member in enumerate(self.members):
            url_safe_member_id = get_member_id(member)

            if isinstance(member, Team):
                system_message_content += f"{indent * ' '} - Team: {member.name}\n"
                system_message_content += f"{indent * ' '} - ID: {url_safe_member_id}\n"
                if member.members is not None:
                    system_message_content += member.get_members_system_message_content(indent=indent + 2)
            else:
                system_message_content += f"{indent * ' '} - Agent {idx + 1}:\n"
                if url_safe_member_id is not None:
                    system_message_content += f"{indent * ' '}   - ID: {url_safe_member_id}\n"
                if member.name is not None:
                    system_message_content += f"{indent * ' '}   - Name: {member.name}\n"
                if member.role is not None:
                    system_message_content += f"{indent * ' '}   - Role: {member.role}\n"
                if member.tools is not None and member.tools != [] and self.add_member_tools_to_context:
                    system_message_content += f"{indent * ' '}   - Member tools:\n"
                    for _tool in member.tools:
                        if isinstance(_tool, Toolkit):
                            for _func in _tool.functions.values():
                                if _func.entrypoint:
                                    system_message_content += f"{indent * ' '}    - {_func.name}\n"
                        elif isinstance(_tool, Function) and _tool.entrypoint:
                            system_message_content += f"{indent * ' '}    - {_tool.name}\n"
                        elif callable(_tool):
                            system_message_content += f"{indent * ' '}    - {_tool.__name__}\n"

        return system_message_content

    def get_system_message(
        self,
        session: TeamSession,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Union[Function, dict]]] = None,
        add_session_state_to_context: Optional[bool] = None,
    ) -> Optional[Message]:
        """Get the system message for the team."""

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
                    session_state=session_state,
                    dependencies=dependencies,
                    metadata=metadata,
                )

            # type: ignore
            return Message(role=self.system_message_role, content=sys_message_content)

        # 1. Build and return the default system message for the Team.
        # 1.1 Build the list of instructions for the system message
        self.model = cast(Model, self.model)
        instructions: List[str] = []
        if self.instructions is not None:
            _instructions = self.instructions
            if callable(self.instructions):
                import inspect

                signature = inspect.signature(self.instructions)
                if "team" in signature.parameters:
                    _instructions = self.instructions(team=self)
                elif "agent" in signature.parameters:
                    _instructions = self.instructions(agent=self)
                else:
                    _instructions = self.instructions()

            if isinstance(_instructions, str):
                instructions.append(_instructions)
            elif isinstance(_instructions, list):
                instructions.extend(_instructions)

        # 1.2 Add instructions from the Model
        _model_instructions = self.model.get_instructions_for_model(tools)
        if _model_instructions is not None:
            instructions.extend(_model_instructions)

        # 1.3 Build a list of additional information for the system message
        additional_information: List[str] = []
        # 1.3.1 Add instructions for using markdown
        if self.markdown and self.output_schema is None:
            additional_information.append("Use markdown to format your answers.")
        # 1.3.2 Add the current datetime
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

        # 1.3.3 Add the current location
        if self.add_location_to_context:
            from agno.utils.location import get_location

            location = get_location()
            if location:
                location_str = ", ".join(
                    filter(None, [location.get("city"), location.get("region"), location.get("country")])
                )
                if location_str:
                    additional_information.append(f"Your approximate location is: {location_str}.")

        # 1.3.4 Add team name if provided
        if self.name is not None and self.add_name_to_context:
            additional_information.append(f"Your name is: {self.name}.")

        if self.knowledge is not None and self.enable_agentic_knowledge_filters:
            valid_filters = self.knowledge.get_valid_filters()
            if valid_filters:
                valid_filters_str = ", ".join(valid_filters)
                additional_information.append(
                    dedent(f"""
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
                """)
                )

        # 2 Build the default system message for the Agent.
        system_message_content: str = ""
        if self.members is not None and len(self.members) > 0:
            system_message_content += "You are the leader of a team and sub-teams of AI Agents.\n"
            system_message_content += "Your task is to coordinate the team to complete the user's request.\n"

            system_message_content += "\nHere are the members in your team:\n"
            system_message_content += "<team_members>\n"
            system_message_content += self.get_members_system_message_content()
            if self.get_member_information_tool:
                system_message_content += "If you need to get information about your team members, you can use the `get_member_information` tool at any time.\n"
            system_message_content += "</team_members>\n"

            system_message_content += "\n<how_to_respond>\n"

            if self.delegate_task_to_all_members:
                system_message_content += (
                    "- You can either respond directly or use the `delegate_task_to_members` tool to delegate a task to all members in your team to get a collaborative response.\n"
                    "- To delegate a task to all members in your team, call `delegate_task_to_members` ONLY once. This will delegate a task to all members in your team.\n"
                    "- Analyze the responses from all members and evaluate whether the task has been completed.\n"
                    "- If you feel the task has been completed, you can stop and respond to the user.\n"
                )
            else:
                system_message_content += (
                    "- Your role is to delegate tasks to members in your team with the highest likelihood of completing the user's request.\n"
                    "- Carefully analyze the tools available to the members and their roles before delegating tasks.\n"
                    "- You cannot use a member tool directly. You can only delegate tasks to members.\n"
                    "- When you delegate a task to another member, make sure to include:\n"
                    "  - member_id (str): The ID of the member to delegate the task to. Use only the ID of the member, not the ID of the team followed by the ID of the member.\n"
                    "  - task (str): A clear description of the task. Determine the best way to describe the task to the member.\n"
                    "- You can delegate tasks to multiple members at once.\n"
                    "- You must always analyze the responses from members before responding to the user.\n"
                    "- After analyzing the responses from the members, if you feel the task has been completed, you can stop and respond to the user.\n"
                    "- If you are NOT satisfied with the responses from the members, you should re-assign the task to a different member.\n"
                    "- For simple greetings, thanks, or questions about the team itself, you should respond directly.\n"
                    "- For all work requests, tasks, or questions requiring expertise, route to appropriate team members.\n"
                )
            system_message_content += "</how_to_respond>\n\n"

        # Attached media
        if audio is not None or images is not None or videos is not None or files is not None:
            system_message_content += "<attached_media>\n"
            system_message_content += "You have the following media attached to your message:\n"
            if audio is not None and len(audio) > 0:
                system_message_content += " - Audio\n"
            if images is not None and len(images) > 0:
                system_message_content += " - Images\n"
            if videos is not None and len(videos) > 0:
                system_message_content += " - Videos\n"
            if files is not None and len(files) > 0:
                system_message_content += " - Files\n"
            system_message_content += "</attached_media>\n\n"

        # Then add memories to the system prompt
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

        # Then add a summary of the interaction to the system prompt
        if self.add_session_summary_to_context and session.summary is not None:
            system_message_content += "Here is a brief summary of your previous interactions:\n\n"
            system_message_content += "<summary_of_previous_interactions>\n"
            system_message_content += session.summary.summary
            system_message_content += "\n</summary_of_previous_interactions>\n\n"
            system_message_content += (
                "Note: this information is from previous interactions and may be outdated. "
                "You should ALWAYS prefer information from this conversation over the past summary.\n\n"
            )

        if self.description is not None:
            system_message_content += f"<description>\n{self.description}\n</description>\n\n"

        if self.role is not None:
            system_message_content += f"\n<your_role>\n{self.role}\n</your_role>\n\n"

        # 3.3.5 Then add instructions for the Agent
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

        system_message_from_model = self.model.get_system_message_for_model(tools)
        if system_message_from_model is not None:
            system_message_content += system_message_from_model

        if self.expected_output is not None:
            system_message_content += f"<expected_output>\n{self.expected_output.strip()}\n</expected_output>\n\n"

        if self.additional_context is not None:
            system_message_content += (
                f"<additional_context>\n{self.additional_context.strip()}\n</additional_context>\n\n"
            )

        if add_session_state_to_context and session_state is not None:
            system_message_content += self._get_formatted_session_state_for_system_message(session_state)

        # Add the JSON output prompt if output_schema is provided and structured_outputs is False
        if (
            self.output_schema is not None
            and self.use_json_mode
            and self.model
            and self.model.supports_native_structured_outputs
        ):
            system_message_content += f"{self._get_json_output_prompt()}"

        return Message(role=self.system_message_role, content=system_message_content.strip())

    async def aget_system_message(
        self,
        session: TeamSession,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Union[Function, dict]]] = None,
        add_session_state_to_context: Optional[bool] = None,
    ) -> Optional[Message]:
        """Get the system message for the team."""

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
                    session_state=session_state,
                    dependencies=dependencies,
                    metadata=metadata,
                )

            # type: ignore
            return Message(role=self.system_message_role, content=sys_message_content)

        # 1. Build and return the default system message for the Team.
        # 1.1 Build the list of instructions for the system message
        self.model = cast(Model, self.model)
        instructions: List[str] = []
        if self.instructions is not None:
            _instructions = self.instructions
            if callable(self.instructions):
                import inspect

                signature = inspect.signature(self.instructions)
                if "team" in signature.parameters:
                    _instructions = self.instructions(team=self)
                elif "agent" in signature.parameters:
                    _instructions = self.instructions(agent=self)
                else:
                    _instructions = self.instructions()

            if isinstance(_instructions, str):
                instructions.append(_instructions)
            elif isinstance(_instructions, list):
                instructions.extend(_instructions)

        # 1.2 Add instructions from the Model
        _model_instructions = self.model.get_instructions_for_model(tools)
        if _model_instructions is not None:
            instructions.extend(_model_instructions)

        # 1.3 Build a list of additional information for the system message
        additional_information: List[str] = []
        # 1.3.1 Add instructions for using markdown
        if self.markdown and self.output_schema is None:
            additional_information.append("Use markdown to format your answers.")
        # 1.3.2 Add the current datetime
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

        # 1.3.3 Add the current location
        if self.add_location_to_context:
            from agno.utils.location import get_location

            location = get_location()
            if location:
                location_str = ", ".join(
                    filter(None, [location.get("city"), location.get("region"), location.get("country")])
                )
                if location_str:
                    additional_information.append(f"Your approximate location is: {location_str}.")

        # 1.3.4 Add team name if provided
        if self.name is not None and self.add_name_to_context:
            additional_information.append(f"Your name is: {self.name}.")

        if self.knowledge is not None and self.enable_agentic_knowledge_filters:
            valid_filters = getattr(self.knowledge, "valid_metadata_filters", None)
            if valid_filters:
                valid_filters_str = ", ".join(valid_filters)
                additional_information.append(
                    dedent(f"""
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
                """)
                )

        # 2 Build the default system message for the Agent.
        system_message_content: str = ""
        system_message_content += "You are the leader of a team and sub-teams of AI Agents.\n"
        system_message_content += "Your task is to coordinate the team to complete the user's request.\n"

        system_message_content += "\nHere are the members in your team:\n"
        system_message_content += "<team_members>\n"
        system_message_content += self.get_members_system_message_content()
        if self.get_member_information_tool:
            system_message_content += "If you need to get information about your team members, you can use the `get_member_information` tool at any time.\n"
        system_message_content += "</team_members>\n"

        system_message_content += "\n<how_to_respond>\n"

        if self.delegate_task_to_all_members:
            system_message_content += (
                "- Your role is to forward tasks to members in your team with the highest likelihood of completing the user's request.\n"
                "- You can either respond directly or use the `delegate_task_to_members` tool to delegate a task to all members in your team to get a collaborative response.\n"
                "- To delegate a task to all members in your team, call `delegate_task_to_members` ONLY once. This will delegate a task to all members in your team.\n"
                "- Analyze the responses from all members and evaluate whether the task has been completed.\n"
                "- If you feel the task has been completed, you can stop and respond to the user.\n"
            )
        else:
            system_message_content += (
                "- Your role is to delegate tasks to members in your team with the highest likelihood of completing the user's request.\n"
                "- Carefully analyze the tools available to the members and their roles before delegating tasks.\n"
                "- You cannot use a member tool directly. You can only delegate tasks to members.\n"
                "- When you delegate a task to another member, make sure to include:\n"
                "  - member_id (str): The ID of the member to delegate the task to. Use only the ID of the member, not the ID of the team followed by the ID of the member.\n"
                "  - task (str): A clear description of the task.\n"
                "- You can delegate tasks to multiple members at once.\n"
                "- You must always analyze the responses from members before responding to the user.\n"
                "- After analyzing the responses from the members, if you feel the task has been completed, you can stop and respond to the user.\n"
                "- If you are not satisfied with the responses from the members, you should re-assign the task.\n"
                "- For simple greetings, thanks, or questions about the team itself, you should respond directly.\n"
                "- For all work requests, tasks, or questions requiring expertise, route to appropriate team members.\n"
            )
        system_message_content += "</how_to_respond>\n\n"

        # Attached media
        if audio is not None or images is not None or videos is not None or files is not None:
            system_message_content += "<attached_media>\n"
            system_message_content += "You have the following media attached to your message:\n"
            if audio is not None and len(audio) > 0:
                system_message_content += " - Audio\n"
            if images is not None and len(images) > 0:
                system_message_content += " - Images\n"
            if videos is not None and len(videos) > 0:
                system_message_content += " - Videos\n"
            if files is not None and len(files) > 0:
                system_message_content += " - Files\n"
            system_message_content += "</attached_media>\n\n"

        # Then add memories to the system prompt
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

        # Then add a summary of the interaction to the system prompt
        if self.add_session_summary_to_context and session.summary is not None:
            system_message_content += "Here is a brief summary of your previous interactions:\n\n"
            system_message_content += "<summary_of_previous_interactions>\n"
            system_message_content += session.summary.summary
            system_message_content += "\n</summary_of_previous_interactions>\n\n"
            system_message_content += (
                "Note: this information is from previous interactions and may be outdated. "
                "You should ALWAYS prefer information from this conversation over the past summary.\n\n"
            )

        if self.description is not None:
            system_message_content += f"<description>\n{self.description}\n</description>\n\n"

        if self.role is not None:
            system_message_content += f"\n<your_role>\n{self.role}\n</your_role>\n\n"

        # 3.3.5 Then add instructions for the Agent
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

        system_message_from_model = self.model.get_system_message_for_model(tools)
        if system_message_from_model is not None:
            system_message_content += system_message_from_model

        if self.expected_output is not None:
            system_message_content += f"<expected_output>\n{self.expected_output.strip()}\n</expected_output>\n\n"

        if self.additional_context is not None:
            system_message_content += (
                f"<additional_context>\n{self.additional_context.strip()}\n</additional_context>\n\n"
            )

        if self.add_session_state_to_context:
            system_message_content += f"<session_state>\n{session_state}\n</session_state>\n\n"

        # Add the JSON output prompt if output_schema is provided and structured_outputs is False
        if (
            self.output_schema is not None
            and self.use_json_mode
            and self.model
            and self.model.supports_native_structured_outputs
        ):
            system_message_content += f"{self._get_json_output_prompt()}"

        return Message(role=self.system_message_role, content=system_message_content.strip())

    def _get_formatted_session_state_for_system_message(self, session_state: Dict[str, Any]) -> str:
        return f"\n<session_state>\n{session_state}\n</session_state>\n\n"

    def _get_run_messages(
        self,
        *,
        run_response: TeamRunOutput,
        session: TeamSession,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        input_message: Optional[Union[str, List, Dict, Message, BaseModel, List[Message]]] = None,
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
        2. Add extra messages to run_messages
        3. Add history to run_messages
        4. Add messages to run_messages if provided (messages parameter first)
        5. Add user message to run_messages (message parameter second)

        """
        # Initialize the RunMessages object
        run_messages = RunMessages()

        # 1. Add system message to run_messages
        system_message = self.get_system_message(
            session=session,
            session_state=session_state,
            user_id=user_id,
            images=images,
            audio=audio,
            videos=videos,
            files=files,
            dependencies=dependencies,
            metadata=metadata,
            add_session_state_to_context=add_session_state_to_context,
            tools=tools,
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

            history = session.get_messages_from_last_n_runs(
                last_n=self.num_history_runs,
                last_n_messages=self.num_history_messages,
                skip_role=skip_role,
                team_id=self.id if self.parent_team_id is not None else None,
            )

            if len(history) > 0:
                # Create a deep copy of the history messages to avoid modifying the original messages
                history_copy = [deepcopy(msg) for msg in history]

                # Tag each message as coming from history
                for _msg in history_copy:
                    _msg.from_history = True

                # Filter tool calls from history messages
                if self.max_tool_calls_from_history is not None:
                    filter_tool_calls(history_copy, self.max_tool_calls_from_history)

                log_debug(f"Adding {len(history_copy)} messages from history")

                # Extend the messages with the history
                run_messages.messages += history_copy

        # 5. Add user message to run_messages (message second as per Dirk's requirement)
        # 5.1 Build user message if message is None, str or list
        user_message = self._get_user_message(
            run_response=run_response,
            session_state=session_state,
            input_message=input_message,
            user_id=user_id,
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
        # Add user message to run_messages
        if user_message is not None:
            run_messages.user_message = user_message
            run_messages.messages.append(user_message)

        return run_messages

    async def _aget_run_messages(
        self,
        *,
        run_response: TeamRunOutput,
        session: TeamSession,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        input_message: Optional[Union[str, List, Dict, Message, BaseModel, List[Message]]] = None,
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
        2. Add extra messages to run_messages
        3. Add history to run_messages
        4. Add messages to run_messages if provided (messages parameter first)
        5. Add user message to run_messages (message parameter second)

        """
        # Initialize the RunMessages object
        run_messages = RunMessages()

        # 1. Add system message to run_messages
        system_message = await self.aget_system_message(
            session=session,
            session_state=session_state,
            user_id=user_id,
            images=images,
            audio=audio,
            videos=videos,
            files=files,
            dependencies=dependencies,
            metadata=metadata,
            add_session_state_to_context=add_session_state_to_context,
            tools=tools,
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
            history = session.get_messages_from_last_n_runs(
                last_n=self.num_history_runs,
                last_n_messages=self.num_history_messages,
                skip_role=skip_role,
                team_id=self.id,
            )

            if len(history) > 0:
                # Create a deep copy of the history messages to avoid modifying the original messages
                history_copy = [deepcopy(msg) for msg in history]

                # Tag each message as coming from history
                for _msg in history_copy:
                    _msg.from_history = True

                # Filter tool calls from history messages
                if self.max_tool_calls_from_history is not None:
                    filter_tool_calls(history_copy, self.max_tool_calls_from_history)

                log_debug(f"Adding {len(history_copy)} messages from history")

                # Extend the messages with the history
                run_messages.messages += history_copy

        # 5. Add user message to run_messages (message second as per Dirk's requirement)
        # 5.1 Build user message if message is None, str or list
        user_message = self._get_user_message(
            run_response=run_response,
            session_state=session_state,
            input_message=input_message,
            user_id=user_id,
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
        # Add user message to run_messages
        if user_message is not None:
            run_messages.user_message = user_message
            run_messages.messages.append(user_message)

        return run_messages

    def _get_user_message(
        self,
        *,
        run_response: TeamRunOutput,
        session_state: Optional[Dict[str, Any]] = None,
        input_message: Optional[Union[str, List, Dict, Message, BaseModel, List[Message]]] = None,
        user_id: Optional[str] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        add_dependencies_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Get references from the knowledge base to use in the user message
        references = None

        if input_message is None:
            # If we have any media, return a message with empty content
            if images is not None or audio is not None or videos is not None or files is not None:
                return Message(
                    role="user",
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
            if isinstance(input_message, list):
                input_content: Union[str, list[Any], list[Message]]
                if len(input_message) > 0 and isinstance(input_message[0], dict) and "type" in input_message[0]:
                    # This is multimodal content (text + images/audio/video), preserve the structure
                    input_content = input_message
                elif len(input_message) > 0 and isinstance(input_message[0], Message):
                    # This is a list of Message objects, extract text content from them
                    input_content = get_text_from_message(input_message)
                elif all(isinstance(item, str) for item in input_message):
                    input_content = "\n".join([str(item) for item in input_message])
                else:
                    input_content = str(input_message)

                return Message(
                    role="user",
                    content=input_content,
                    images=None if not self.send_media_to_model else images,
                    audio=None if not self.send_media_to_model else audio,
                    videos=None if not self.send_media_to_model else videos,
                    files=None if not self.send_media_to_model else files,
                    **kwargs,
                )

            # If message is provided as a Message, use it directly
            elif isinstance(input_message, Message):
                return input_message
            # If message is provided as a dict, try to validate it as a Message
            elif isinstance(input_message, dict):
                try:
                    if self.input_schema and is_typed_dict(self.input_schema):
                        import json

                        content = json.dumps(input_message, indent=2, ensure_ascii=False)
                        return Message(role="user", content=content)
                    else:
                        return Message.model_validate(input_message)
                except Exception as e:
                    log_warning(f"Failed to validate input: {e}")

            # If message is provided as a BaseModel, convert it to a Message
            elif isinstance(input_message, BaseModel):
                try:
                    # Create a user message with the BaseModel content
                    content = input_message.model_dump_json(indent=2, exclude_none=True)
                    return Message(role="user", content=content)
                except Exception as e:
                    log_warning(f"Failed to convert BaseModel to message: {e}")
            else:
                user_msg_content = input_message
                if self.add_knowledge_to_context:
                    if isinstance(input_message, str):
                        user_msg_content = input_message
                    elif callable(input_message):
                        user_msg_content = input_message(agent=self)
                    else:
                        raise Exception("input must be a string or a callable when add_references is True")

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
                    role="user",
                    content=user_msg_content,
                    images=None if not self.send_media_to_model else images,
                    audio=None if not self.send_media_to_model else audio,
                    videos=None if not self.send_media_to_model else videos,
                    files=None if not self.send_media_to_model else files,
                    **kwargs,
                )

    def _get_messages_for_parser_model(
        self, model_response: ModelResponse, response_format: Optional[Union[Dict, Type[BaseModel]]]
    ) -> List[Message]:
        from agno.utils.prompts import get_json_output_prompt

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
        self, run_response: TeamRunOutput, response_format: Optional[Union[Dict, Type[BaseModel]]]
    ) -> List[Message]:
        """Get the messages for the parser model."""
        from agno.utils.prompts import get_json_output_prompt

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

        if not isinstance(message, str):
            return message
        # Should already be resolved and passed from run() method
        format_variables = ChainMap(
            session_state or {},
            dependencies or {},
            metadata or {},
            {"user_id": user_id} if user_id is not None else {},
        )
        converted_msg = message
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
                except Exception as e:
                    log_error(f"Failed to serialize to JSON: {e}")
                    # If serialization fails, convert to string representation
                    sanitized_context[key] = str(value)

            try:
                return json.dumps(sanitized_context, indent=2)
            except Exception as e:
                log_error(f"Failed to convert sanitized context to JSON: {e}")
                return str(context)

    def _get_json_output_prompt(self) -> str:
        """Return the JSON output prompt for the Agent.

        This is added to the system prompt when the output_schema is set and structured_outputs is False.
        """
        import json

        json_output_prompt = "Provide your output as a JSON containing the following fields:"
        if self.output_schema is not None:
            if isinstance(self.output_schema, str):
                json_output_prompt += "\n<json_fields>"
                json_output_prompt += f"\n{self.output_schema}"
                json_output_prompt += "\n</json_fields>"
            elif isinstance(self.output_schema, list):
                json_output_prompt += "\n<json_fields>"
                json_output_prompt += f"\n{json.dumps(self.output_schema)}"
                json_output_prompt += "\n</json_fields>"
            elif issubclass(self.output_schema, BaseModel):
                json_schema = self.output_schema.model_json_schema()
                if json_schema is not None:
                    response_model_properties = {}
                    json_schema_properties = json_schema.get("properties")
                    if json_schema_properties is not None:
                        for field_name, field_properties in json_schema_properties.items():
                            formatted_field_properties = {
                                prop_name: prop_value
                                for prop_name, prop_value in field_properties.items()
                                if prop_name != "title"
                            }
                            response_model_properties[field_name] = formatted_field_properties
                    json_schema_defs = json_schema.get("$defs")
                    if json_schema_defs is not None:
                        response_model_properties["$defs"] = {}
                        for def_name, def_properties in json_schema_defs.items():
                            def_fields = def_properties.get("properties")
                            formatted_def_properties = {}
                            if def_fields is not None:
                                for field_name, field_properties in def_fields.items():
                                    formatted_field_properties = {
                                        prop_name: prop_value
                                        for prop_name, prop_value in field_properties.items()
                                        if prop_name != "title"
                                    }
                                    formatted_def_properties[field_name] = formatted_field_properties
                            if len(formatted_def_properties) > 0:
                                response_model_properties["$defs"][def_name] = formatted_def_properties

                    if len(response_model_properties) > 0:
                        json_output_prompt += "\n<json_fields>"
                        json_output_prompt += (
                            f"\n{json.dumps([key for key in response_model_properties.keys() if key != '$defs'])}"
                        )
                        json_output_prompt += "\n</json_fields>"
                        json_output_prompt += "\n\nHere are the properties for each field:"
                        json_output_prompt += "\n<json_field_properties>"
                        json_output_prompt += f"\n{json.dumps(response_model_properties, indent=2)}"
                        json_output_prompt += "\n</json_field_properties>"
            else:
                log_warning(f"Could not build json schema for {self.output_schema}")
        else:
            json_output_prompt += "Provide the output as JSON."

        json_output_prompt += "\nStart your response with `{` and end it with `}`."
        json_output_prompt += "\nYour output will be passed to json.loads() to convert it to a Python object."
        json_output_prompt += "\nMake sure it only contains valid JSON."
        return json_output_prompt

    def _update_team_media(self, run_response: Union[TeamRunOutput, RunOutput]) -> None:
        """Update the team state with the run response."""
        if run_response.images is not None:
            if self.images is None:
                self.images = []
            self.images.extend(run_response.images)
        if run_response.videos is not None:
            if self.videos is None:
                self.videos = []
            self.videos.extend(run_response.videos)
        if run_response.audio is not None:
            if self.audio is None:
                self.audio = []
            self.audio.extend(run_response.audio)

    ###########################################################################
    # Built-in Tools
    ###########################################################################

    def _get_update_user_memory_function(self, user_id: Optional[str] = None, async_mode: bool = False) -> Function:
        def update_user_memory(task: str) -> str:
            """
            Use this function to submit a task to modify the Agent's memory.
            Describe the task in detail and be specific.
            The task can include adding a memory, updating a memory, deleting a memory, or clearing all memories.

            Args:
                task: The task to update the memory. Be specific and describe the task in detail.

            Returns:
                str: A string indicating the status of the update.
            """
            self.memory_manager = cast(MemoryManager, self.memory_manager)
            response = self.memory_manager.update_memory_task(task=task, user_id=user_id)
            return response

        async def aupdate_user_memory(task: str) -> str:
            """
            Use this function to submit a task to modify the Agent's memory.
            Describe the task in detail and be specific.
            The task can include adding a memory, updating a memory, deleting a memory, or clearing all memories.

            Args:
                task: The task to update the memory. Be specific and describe the task in detail.

            Returns:
                str: A string indicating the status of the update.
            """
            self.memory_manager = cast(MemoryManager, self.memory_manager)
            response = await self.memory_manager.aupdate_memory_task(task=task, user_id=user_id)
            return response

        if async_mode:
            update_memory_function = aupdate_user_memory
        else:
            update_memory_function = update_user_memory  # type: ignore

        return Function.from_callable(update_memory_function, name="update_user_memory")

    def get_member_information(self) -> str:
        """Get information about the members of the team, including their IDs, names, and roles."""
        return self.get_members_system_message_content(indent=0)

    def _get_chat_history_function(self, session: TeamSession, async_mode: bool = False):
        def get_chat_history(num_chats: Optional[int] = None) -> str:
            """
            Use this function to get the team chat history in reverse chronological order.
            Leave the num_chats parameter blank to get the entire chat history.
            Example:
                - To get the last chat, use num_chats=1
                - To get the last 5 chats, use num_chats=5
                - To get all chats, leave num_chats blank

            Args:
                num_chats: The number of chats to return.
                    Each chat contains 2 messages. One from the team and one from the user.
                    Default: None

            Returns:
                str: A JSON string containing a list of dictionaries representing the team chat history.
            """
            import json

            history: List[Dict[str, Any]] = []

            all_chats = session.get_messages_from_last_n_runs(
                team_id=self.id,
            )

            if len(all_chats) == 0:
                return ""

            for chat in all_chats[::-1]:  # type: ignore
                history.insert(0, chat.to_dict())  # type: ignore

            if num_chats is not None:
                history = history[:num_chats]

            return json.dumps(history)

        async def aget_chat_history(num_chats: Optional[int] = None) -> str:
            """
            Use this function to get the team chat history in reverse chronological order.
            Leave the num_chats parameter blank to get the entire chat history.
            Example:
                - To get the last chat, use num_chats=1
                - To get the last 5 chats, use num_chats=5
                - To get all chats, leave num_chats blank

            Args:
                num_chats: The number of chats to return.
                    Each chat contains 2 messages. One from the team and one from the user.
                    Default: None

            Returns:
                str: A JSON string containing a list of dictionaries representing the team chat history.
            """
            import json

            history: List[Dict[str, Any]] = []

            all_chats = session.get_messages_from_last_n_runs(
                team_id=self.id,
            )

            if len(all_chats) == 0:
                return ""

            for chat in all_chats[::-1]:  # type: ignore
                history.insert(0, chat.to_dict())  # type: ignore

            if num_chats is not None:
                history = history[:num_chats]

            return json.dumps(history)

        if async_mode:
            get_chat_history_func = aget_chat_history
        else:
            get_chat_history_func = get_chat_history  # type: ignore
        return Function.from_callable(get_chat_history_func, name="get_chat_history")

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

    def _get_previous_sessions_messages_function(
        self, num_history_sessions: Optional[int] = 2, user_id: Optional[str] = None, async_mode: bool = False
    ):
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
            import json

            if self.db is None:
                return "Previous session messages not available"

            self.db = cast(BaseDb, self.db)
            selected_sessions = self.db.get_sessions(
                session_type=SessionType.TEAM,
                limit=num_history_sessions,
                user_id=user_id,
                sort_by="created_at",
                sort_order="desc",
            )

            all_messages = []
            seen_message_pairs = set()

            for session in selected_sessions:
                if isinstance(session, TeamSession) and session.runs:
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

        async def aget_previous_session_messages() -> str:
            """Use this function to retrieve messages from previous chat sessions.
            USE THIS TOOL ONLY WHEN THE QUESTION IS EITHER "What was my last conversation?" or "What was my last question?" and similar to it.

            Returns:
                str: JSON formatted list of message pairs from previous sessions
            """
            import json

            if self.db is None:
                return "Previous session messages not available"

            self.db = cast(AsyncBaseDb, self.db)
            if self._has_async_db():
                selected_sessions = await self.db.get_sessions(  # type: ignore
                    session_type=SessionType.TEAM,
                    limit=num_history_sessions,
                    user_id=user_id,
                    sort_by="created_at",
                    sort_order="desc",
                )
            else:
                selected_sessions = self.db.get_sessions(  # type: ignore
                    session_type=SessionType.TEAM,
                    limit=num_history_sessions,
                    user_id=user_id,
                    sort_by="created_at",
                    sort_order="desc",
                )

            all_messages = []
            seen_message_pairs = set()

            for session in selected_sessions:
                if isinstance(session, TeamSession) and session.runs:
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

        if self._has_async_db():
            return Function.from_callable(aget_previous_session_messages, name="get_previous_session_messages")
        else:
            return Function.from_callable(get_previous_session_messages, name="get_previous_session_messages")

    def _get_history_for_member_agent(self, session: TeamSession, member_agent: Union[Agent, "Team"]) -> List[Message]:
        from copy import deepcopy

        log_debug(f"Adding messages from history for {member_agent.name}")

        member_agent_id = member_agent.id if isinstance(member_agent, Agent) else None
        member_team_id = member_agent.id if isinstance(member_agent, Team) else None

        # Only skip messages from history when system_message_role is NOT a standard conversation role.
        # Standard conversation roles ("user", "assistant", "tool") should never be filtered
        # to preserve conversation continuity.
        skip_role = self.system_message_role if self.system_message_role not in ["user", "assistant", "tool"] else None

        history = session.get_messages_from_last_n_runs(
            last_n=member_agent.num_history_runs or self.num_history_runs,
            last_n_messages=member_agent.num_history_messages,
            skip_role=skip_role,
            agent_id=member_agent_id,
            team_id=member_team_id,
            member_runs=True,
        )

        if len(history) > 0:
            # Create a deep copy of the history messages to avoid modifying the original messages
            history_copy = [deepcopy(msg) for msg in history]

            # Tag each message as coming from history
            for _msg in history_copy:
                _msg.from_history = True

            return history_copy
        return []

    def _determine_team_member_interactions(
        self,
        team_run_context: Dict[str, Any],
        images: List[Image],
        videos: List[Video],
        audio: List[Audio],
        files: List[File],
    ) -> Optional[str]:
        team_member_interactions_str = None
        if self.share_member_interactions:
            team_member_interactions_str = get_team_member_interactions_str(team_run_context=team_run_context)  # type: ignore
            if context_images := get_team_run_context_images(team_run_context=team_run_context):  # type: ignore
                images.extend(context_images)
            if context_videos := get_team_run_context_videos(team_run_context=team_run_context):  # type: ignore
                videos.extend(context_videos)
            if context_audio := get_team_run_context_audio(team_run_context=team_run_context):  # type: ignore
                audio.extend(context_audio)
            if context_files := get_team_run_context_files(team_run_context=team_run_context):  # type: ignore
                files.extend(context_files)
        return team_member_interactions_str

    def _find_member_by_id(self, member_id: str) -> Optional[Tuple[int, Union[Agent, "Team"]]]:
        """
        Recursively search through team members and subteams to find an agent by name.

        Args:
            member_id (str): ID of the agent to find

        Returns:
            Optional[Tuple[int, Union[Agent, "Team"], Optional[str]]]: Tuple containing:
                - Index of the member in its immediate parent team
                - The top-level leader agent
        """
        # First check direct members
        for i, member in enumerate(self.members):
            url_safe_member_id = get_member_id(member)
            if url_safe_member_id == member_id:
                return i, member

            # If this member is a team, search its members recursively
            if isinstance(member, Team):
                result = member._find_member_by_id(member_id)
                if result is not None:
                    # Found in subteam, return with the top-level team member's name
                    return i, member

        return None

    def _get_delegate_task_function(
        self,
        run_response: TeamRunOutput,
        session: TeamSession,
        session_state: Dict[str, Any],
        team_run_context: Dict[str, Any],
        user_id: Optional[str] = None,
        stream: bool = False,
        stream_events: bool = False,
        async_mode: bool = False,
        input: Optional[str] = None,  # Used for determine_input_for_members=False
        images: Optional[List[Image]] = None,
        videos: Optional[List[Video]] = None,
        audio: Optional[List[Audio]] = None,
        files: Optional[List[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        debug_mode: Optional[bool] = None,
    ) -> Function:
        if not images:
            images = []
        if not videos:
            videos = []
        if not audio:
            audio = []
        if not files:
            files = []

        def _setup_delegate_task_to_member(member_agent: Union[Agent, "Team"], task_description: str):
            # 1. Initialize the member agent
            self._initialize_member(member_agent)

            # If team has send_media_to_model=False, ensure member agent also has it set to False
            # This allows tools to access files while preventing models from receiving them
            if not self.send_media_to_model:
                member_agent.send_media_to_model = False

            # 2. Handle respond_directly nuances
            if self.respond_directly:
                # Since we return the response directly from the member agent, we need to set the output schema from the team down.
                if not member_agent.output_schema and self.output_schema:
                    member_agent.output_schema = self.output_schema

                # If the member will produce structured output, we need to parse the response
                if member_agent.output_schema is not None:
                    self._member_response_model = member_agent.output_schema

            # 3. Handle enable_agentic_knowledge_filters on the member agent
            if self.enable_agentic_knowledge_filters and not member_agent.enable_agentic_knowledge_filters:
                member_agent.enable_agentic_knowledge_filters = self.enable_agentic_knowledge_filters

            # 4. Determine team context to send
            team_member_interactions_str = self._determine_team_member_interactions(
                team_run_context, images=images, videos=videos, audio=audio, files=files
            )

            # 5. Get the team history
            team_history_str = None
            if self.add_team_history_to_members and session:
                team_history_str = session.get_team_history_context(num_runs=self.num_team_history_runs)

            # 6. Create the member agent task or use the input directly
            if self.determine_input_for_members is False:
                member_agent_task = input  # type: ignore
            else:
                member_agent_task = task_description

            if team_history_str or team_member_interactions_str:
                member_agent_task = format_member_agent_task(  # type: ignore
                    task_description=member_agent_task or "",
                    team_member_interactions_str=team_member_interactions_str,
                    team_history_str=team_history_str,
                )

            # 7. Add member-level history for the member if enabled (because we won't load the session for the member, so history won't be loaded automatically)
            history = None
            if member_agent.add_history_to_context:
                history = self._get_history_for_member_agent(session, member_agent)
                if history:
                    if isinstance(member_agent_task, str):
                        history.append(Message(role="user", content=member_agent_task))

            return member_agent_task, history

        def _process_delegate_task_to_member(
            member_agent_run_response: Optional[Union[TeamRunOutput, RunOutput]],
            member_agent: Union[Agent, "Team"],
            member_agent_task: Union[str, Message],
            member_session_state_copy: Dict[str, Any],
        ):
            # Add team run id to the member run
            if member_agent_run_response is not None:
                member_agent_run_response.parent_run_id = run_response.run_id  # type: ignore

            # Update the top-level team run_response tool call to have the run_id of the member run
            if run_response.tools is not None:
                for tool in run_response.tools:
                    if tool.tool_name and tool.tool_name.lower() == "delegate_task_to_member":
                        tool.child_run_id = member_agent_run_response.run_id  # type: ignore

            # Update the team run context
            member_name = member_agent.name if member_agent.name else member_agent.id if member_agent.id else "Unknown"
            if isinstance(member_agent_task, str):
                normalized_task = member_agent_task
            elif member_agent_task.content:
                normalized_task = str(member_agent_task.content)
            else:
                normalized_task = ""
            add_interaction_to_team_run_context(
                team_run_context=team_run_context,
                member_name=member_name,
                task=normalized_task,
                run_response=member_agent_run_response,  # type: ignore
            )

            # Add the member run to the team run response if enabled
            if run_response and member_agent_run_response:
                run_response.add_member_run(member_agent_run_response)

            # Scrub the member run based on that member's storage flags before storing
            if member_agent_run_response:
                if (
                    not member_agent.store_media
                    or not member_agent.store_tool_messages
                    or not member_agent.store_history_messages
                ):
                    member_agent._scrub_run_output_for_storage(member_agent_run_response)  # type: ignore

                # Add the member run to the team session
                session.upsert_run(member_agent_run_response)

            # Update team session state
            merge_dictionaries(session_state, member_session_state_copy)  # type: ignore

            # Update the team media
            if member_agent_run_response is not None:
                self._update_team_media(member_agent_run_response)  # type: ignore

        def delegate_task_to_member(
            member_id: str, task: str
        ) -> Iterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
            """Use this function to delegate a task to the selected team member.
            You must provide a clear and concise description of the task the member should achieve AND the expected output.

            Args:
                member_id (str): The ID of the member to delegate the task to. Use only the ID of the member, not the ID of the team followed by the ID of the member.
                task (str): A clear and concise description of the task the member should achieve.
            Returns:
                str: The result of the delegated task.
            """

            # Find the member agent using the helper function
            result = self._find_member_by_id(member_id)
            if result is None:
                yield f"Member with ID {member_id} not found in the team or any subteams. Please choose the correct member from the list of members:\n\n{self.get_members_system_message_content(indent=0)}"
                return

            _, member_agent = result
            member_agent_task, history = _setup_delegate_task_to_member(
                member_agent=member_agent, task_description=task
            )

            # Make sure for the member agent, we are using the agent logger
            use_agent_logger()

            member_session_state_copy = copy(session_state)

            if stream:
                member_agent_run_response_stream = member_agent.run(
                    input=member_agent_task if not history else history,
                    user_id=user_id,
                    # All members have the same session_id
                    session_id=session.session_id,
                    session_state=member_session_state_copy,  # Send a copy to the agent
                    images=images,
                    videos=videos,
                    audio=audio,
                    files=files,
                    stream=True,
                    stream_events=stream_events,
                    debug_mode=debug_mode,
                    dependencies=dependencies,
                    add_dependencies_to_context=add_dependencies_to_context,
                    metadata=metadata,
                    add_session_state_to_context=add_session_state_to_context,
                    knowledge_filters=knowledge_filters
                    if not member_agent.knowledge_filters and member_agent.knowledge
                    else None,
                    yield_run_response=True,
                )
                member_agent_run_response = None
                for member_agent_run_output_event in member_agent_run_response_stream:
                    # If we get the final response, we can break out of the loop
                    if isinstance(member_agent_run_output_event, TeamRunOutput) or isinstance(
                        member_agent_run_output_event, RunOutput
                    ):
                        member_agent_run_response = member_agent_run_output_event  # type: ignore
                        break

                    # Check if the run is cancelled
                    check_if_run_cancelled(member_agent_run_output_event)

                    # Yield the member event directly
                    member_agent_run_output_event.parent_run_id = (
                        member_agent_run_output_event.parent_run_id or run_response.run_id
                    )
                    yield member_agent_run_output_event
            else:
                member_agent_run_response = member_agent.run(  # type: ignore
                    input=member_agent_task if not history else history,
                    user_id=user_id,
                    # All members have the same session_id
                    session_id=session.session_id,
                    session_state=member_session_state_copy,  # Send a copy to the agent
                    images=images,
                    videos=videos,
                    audio=audio,
                    files=files,
                    stream=False,
                    debug_mode=debug_mode,
                    dependencies=dependencies,
                    add_dependencies_to_context=add_dependencies_to_context,
                    add_session_state_to_context=add_session_state_to_context,
                    metadata=metadata,
                    knowledge_filters=knowledge_filters
                    if not member_agent.knowledge_filters and member_agent.knowledge
                    else None,
                )

                check_if_run_cancelled(member_agent_run_response)  # type: ignore

                try:
                    if member_agent_run_response.content is None and (  # type: ignore
                        member_agent_run_response.tools is None or len(member_agent_run_response.tools) == 0  # type: ignore
                    ):
                        yield "No response from the member agent."
                    elif isinstance(member_agent_run_response.content, str):  # type: ignore
                        content = member_agent_run_response.content.strip()  # type: ignore
                        if len(content) > 0:
                            yield content

                        # If the content is empty but we have tool calls
                        elif member_agent_run_response.tools is not None and len(member_agent_run_response.tools) > 0:  # type: ignore
                            tool_str = ""
                            for tool in member_agent_run_response.tools:  # type: ignore
                                if tool.result:
                                    tool_str += f"{tool.result},"
                            yield tool_str.rstrip(",")

                    elif issubclass(type(member_agent_run_response.content), BaseModel):  # type: ignore
                        yield member_agent_run_response.content.model_dump_json(indent=2)  # type: ignore
                    else:
                        import json

                        yield json.dumps(member_agent_run_response.content, indent=2)  # type: ignore
                except Exception as e:
                    yield str(e)

            # Afterward, switch back to the team logger
            use_team_logger()

            _process_delegate_task_to_member(
                member_agent_run_response, member_agent, member_agent_task, member_session_state_copy
            )

        async def adelegate_task_to_member(
            member_id: str, task: str
        ) -> AsyncIterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
            """Use this function to delegate a task to the selected team member.
            You must provide a clear and concise description of the task the member should achieve AND the expected output.

            Args:
                member_id (str): The ID of the member to delegate the task to. Use only the ID of the member, not the ID of the team followed by the ID of the member.
                task (str): A clear and concise description of the task the member should achieve.
            Returns:
                str: The result of the delegated task.
            """

            # Find the member agent using the helper function
            result = self._find_member_by_id(member_id)
            if result is None:
                yield f"Member with ID {member_id} not found in the team or any subteams. Please choose the correct member from the list of members:\n\n{self.get_members_system_message_content(indent=0)}"
                return

            _, member_agent = result
            member_agent_task, history = _setup_delegate_task_to_member(
                member_agent=member_agent, task_description=task
            )

            # Make sure for the member agent, we are using the agent logger
            use_agent_logger()

            member_session_state_copy = copy(session_state)

            if stream:
                member_agent_run_response_stream = member_agent.arun(  # type: ignore
                    input=member_agent_task if not history else history,
                    user_id=user_id,
                    # All members have the same session_id
                    session_id=session.session_id,
                    session_state=member_session_state_copy,  # Send a copy to the agent
                    images=images,
                    videos=videos,
                    audio=audio,
                    files=files,
                    stream=True,
                    stream_events=stream_events,
                    debug_mode=debug_mode,
                    dependencies=dependencies,
                    add_dependencies_to_context=add_dependencies_to_context,
                    add_session_state_to_context=add_session_state_to_context,
                    metadata=metadata,
                    knowledge_filters=knowledge_filters
                    if not member_agent.knowledge_filters and member_agent.knowledge
                    else None,
                    yield_run_response=True,
                )
                member_agent_run_response = None
                async for member_agent_run_response_event in member_agent_run_response_stream:
                    # If we get the final response, we can break out of the loop
                    if isinstance(member_agent_run_response_event, TeamRunOutput) or isinstance(
                        member_agent_run_response_event, RunOutput
                    ):
                        member_agent_run_response = member_agent_run_response_event  # type: ignore
                        break

                    # Check if the run is cancelled
                    check_if_run_cancelled(member_agent_run_response_event)

                    # Yield the member event directly
                    member_agent_run_response_event.parent_run_id = (
                        member_agent_run_response_event.parent_run_id or run_response.run_id
                    )
                    yield member_agent_run_response_event
            else:
                member_agent_run_response = await member_agent.arun(  # type: ignore
                    input=member_agent_task if not history else history,
                    user_id=user_id,
                    # All members have the same session_id
                    session_id=session.session_id,
                    session_state=member_session_state_copy,  # Send a copy to the agent
                    images=images,
                    videos=videos,
                    audio=audio,
                    files=files,
                    stream=False,
                    debug_mode=debug_mode,
                    dependencies=dependencies,
                    add_dependencies_to_context=add_dependencies_to_context,
                    add_session_state_to_context=add_session_state_to_context,
                    metadata=metadata,
                    knowledge_filters=knowledge_filters
                    if not member_agent.knowledge_filters and member_agent.knowledge
                    else None,
                )
                check_if_run_cancelled(member_agent_run_response)  # type: ignore

                try:
                    if member_agent_run_response.content is None and (  # type: ignore
                        member_agent_run_response.tools is None or len(member_agent_run_response.tools) == 0  # type: ignore
                    ):
                        yield "No response from the member agent."
                    elif isinstance(member_agent_run_response.content, str):  # type: ignore
                        if len(member_agent_run_response.content.strip()) > 0:  # type: ignore
                            yield member_agent_run_response.content  # type: ignore

                        # If the content is empty but we have tool calls
                        elif (
                            member_agent_run_response.tools is not None  # type: ignore
                            and len(member_agent_run_response.tools) > 0  # type: ignore
                        ):
                            yield ",".join([tool.result for tool in member_agent_run_response.tools if tool.result])  # type: ignore
                    elif issubclass(type(member_agent_run_response.content), BaseModel):  # type: ignore
                        yield member_agent_run_response.content.model_dump_json(indent=2)  # type: ignore
                    else:
                        import json

                        yield json.dumps(member_agent_run_response.content, indent=2)  # type: ignore
                except Exception as e:
                    yield str(e)

            # Afterward, switch back to the team logger
            use_team_logger()

            _process_delegate_task_to_member(
                member_agent_run_response, member_agent, member_agent_task, member_session_state_copy
            )

        # When the task should be delegated to all members
        def delegate_task_to_members(task: str) -> Iterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
            """
            Use this function to delegate a task to all the member agents and return a response.
            You must provide a clear and concise description of the task the member should achieve AND the expected output.

            Args:
                task (str): A clear and concise description of the task to send to member agents.
            Returns:
                str: The result of the delegated task.
            """

            # Run all the members sequentially
            for _, member_agent in enumerate(self.members):
                member_agent_task, history = _setup_delegate_task_to_member(
                    member_agent=member_agent, task_description=task
                )

                member_session_state_copy = copy(session_state)
                if stream:
                    member_agent_run_response_stream = member_agent.run(
                        input=member_agent_task if not history else history,
                        user_id=user_id,
                        # All members have the same session_id
                        session_id=session.session_id,
                        session_state=member_session_state_copy,  # Send a copy to the agent
                        images=images,
                        videos=videos,
                        audio=audio,
                        files=files,
                        stream=True,
                        stream_events=stream_events,
                        knowledge_filters=knowledge_filters
                        if not member_agent.knowledge_filters and member_agent.knowledge
                        else None,
                        debug_mode=debug_mode,
                        dependencies=dependencies,
                        add_dependencies_to_context=add_dependencies_to_context,
                        add_session_state_to_context=add_session_state_to_context,
                        metadata=metadata,
                        yield_run_response=True,
                    )
                    member_agent_run_response = None
                    for member_agent_run_response_chunk in member_agent_run_response_stream:
                        # If we get the final response, we can break out of the loop
                        if isinstance(member_agent_run_response_chunk, TeamRunOutput) or isinstance(
                            member_agent_run_response_chunk, RunOutput
                        ):
                            member_agent_run_response = member_agent_run_response_chunk  # type: ignore
                            break

                        # Check if the run is cancelled
                        check_if_run_cancelled(member_agent_run_response_chunk)

                        # Yield the member event directly
                        member_agent_run_response_chunk.parent_run_id = (
                            member_agent_run_response_chunk.parent_run_id or run_response.run_id
                        )
                        yield member_agent_run_response_chunk

                else:
                    member_agent_run_response = member_agent.run(  # type: ignore
                        input=member_agent_task if not history else history,
                        user_id=user_id,
                        # All members have the same session_id
                        session_id=session.session_id,
                        session_state=member_session_state_copy,  # Send a copy to the agent
                        images=images,
                        videos=videos,
                        audio=audio,
                        files=files,
                        stream=False,
                        knowledge_filters=knowledge_filters
                        if not member_agent.knowledge_filters and member_agent.knowledge
                        else None,
                        debug_mode=debug_mode,
                        dependencies=dependencies,
                        add_dependencies_to_context=add_dependencies_to_context,
                        add_session_state_to_context=add_session_state_to_context,
                        metadata=metadata,
                    )

                    check_if_run_cancelled(member_agent_run_response)  # type: ignore

                    try:
                        if member_agent_run_response.content is None and (  # type: ignore
                            member_agent_run_response.tools is None or len(member_agent_run_response.tools) == 0  # type: ignore
                        ):
                            yield f"Agent {member_agent.name}: No response from the member agent."
                        elif isinstance(member_agent_run_response.content, str):  # type: ignore
                            if len(member_agent_run_response.content.strip()) > 0:  # type: ignore
                                yield f"Agent {member_agent.name}: {member_agent_run_response.content}"  # type: ignore
                            elif (
                                member_agent_run_response.tools is not None and len(member_agent_run_response.tools) > 0  # type: ignore
                            ):
                                yield f"Agent {member_agent.name}: {','.join([tool.result for tool in member_agent_run_response.tools])}"  # type: ignore
                        elif issubclass(type(member_agent_run_response.content), BaseModel):  # type: ignore
                            yield f"Agent {member_agent.name}: {member_agent_run_response.content.model_dump_json(indent=2)}"  # type: ignore
                        else:
                            import json

                            yield f"Agent {member_agent.name}: {json.dumps(member_agent_run_response.content, indent=2)}"  # type: ignore
                    except Exception as e:
                        yield f"Agent {member_agent.name}: Error - {str(e)}"

                _process_delegate_task_to_member(
                    member_agent_run_response, member_agent, member_agent_task, member_session_state_copy
                )

            # After all the member runs, switch back to the team logger
            use_team_logger()

        # When the task should be delegated to all members
        async def adelegate_task_to_members(task: str) -> AsyncIterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
            """Use this function to delegate a task to all the member agents and return a response.
            You must provide a clear and concise description of the task to send to member agents.

            Args:
                task (str): A clear and concise description of the task to send to member agents.
            Returns:
                str: The result of the delegated task.
            """

            if stream:
                # Concurrent streaming: launch each member as a streaming worker and merge events
                done_marker = object()
                queue: "asyncio.Queue[Union[RunOutputEvent, TeamRunOutputEvent, str, object]]" = asyncio.Queue()

                async def stream_member(agent: Union[Agent, "Team"]) -> None:
                    member_agent_task, history = _setup_delegate_task_to_member(
                        member_agent=agent, task_description=task
                    )  # type: ignore
                    member_session_state_copy = copy(session_state)

                    member_stream = agent.arun(  # type: ignore
                        input=member_agent_task if not history else history,
                        user_id=user_id,
                        session_id=session.session_id,
                        session_state=member_session_state_copy,  # Send a copy to the agent
                        images=images,
                        videos=videos,
                        audio=audio,
                        files=files,
                        stream=True,
                        stream_events=stream_events,
                        debug_mode=debug_mode,
                        knowledge_filters=knowledge_filters
                        if not member_agent.knowledge_filters and member_agent.knowledge
                        else None,
                        dependencies=dependencies,
                        add_dependencies_to_context=add_dependencies_to_context,
                        add_session_state_to_context=add_session_state_to_context,
                        metadata=metadata,
                        yield_run_response=True,
                    )
                    member_agent_run_response = None
                    try:
                        async for member_agent_run_output_event in member_stream:
                            if isinstance(member_agent_run_output_event, TeamRunOutput) or isinstance(
                                member_agent_run_output_event, RunOutput
                            ):
                                member_agent_run_response = member_agent_run_output_event  # type: ignore
                                break
                            check_if_run_cancelled(member_agent_run_output_event)
                            member_agent_run_output_event.parent_run_id = (
                                member_agent_run_output_event.parent_run_id or run_response.run_id
                            )
                            await queue.put(member_agent_run_output_event)
                    finally:
                        _process_delegate_task_to_member(
                            member_agent_run_response, member_agent, member_agent_task, member_session_state_copy
                        )
                    await queue.put(done_marker)

                # Initialize and launch all members
                tasks: List[asyncio.Task[None]] = []
                for member_agent in self.members:
                    current_agent = member_agent
                    self._initialize_member(current_agent)
                    tasks.append(asyncio.create_task(stream_member(current_agent)))

                # Drain queue until all members reported done
                completed = 0
                try:
                    while completed < len(tasks):
                        item = await queue.get()
                        if item is done_marker:
                            completed += 1
                        else:
                            yield item  # type: ignore
                finally:
                    # Ensure tasks do not leak on cancellation
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    # Await cancellation to suppress warnings
                    for t in tasks:
                        with contextlib.suppress(Exception):
                            await t
            else:
                # Non-streaming concurrent run of members; collect results when done
                tasks = []
                for member_agent_index, member_agent in enumerate(self.members):
                    current_agent = member_agent
                    member_agent_task, history = _setup_delegate_task_to_member(
                        member_agent=current_agent, task_description=task
                    )

                    async def run_member_agent(agent=current_agent) -> str:
                        member_session_state_copy = copy(session_state)

                        member_agent_run_response = await agent.arun(
                            input=member_agent_task if not history else history,
                            user_id=user_id,
                            # All members have the same session_id
                            session_id=session.session_id,
                            session_state=member_session_state_copy,  # Send a copy to the agent
                            images=images,
                            videos=videos,
                            audio=audio,
                            files=files,
                            stream=False,
                            stream_events=stream_events,
                            debug_mode=debug_mode,
                            knowledge_filters=knowledge_filters
                            if not member_agent.knowledge_filters and member_agent.knowledge
                            else None,
                            dependencies=dependencies,
                            add_dependencies_to_context=add_dependencies_to_context,
                            add_session_state_to_context=add_session_state_to_context,
                            metadata=metadata,
                        )
                        check_if_run_cancelled(member_agent_run_response)

                        _process_delegate_task_to_member(
                            member_agent_run_response, member_agent, member_agent_task, member_session_state_copy
                        )

                        member_name = member_agent.name if member_agent.name else f"agent_{member_agent_index}"
                        try:
                            if member_agent_run_response.content is None and (
                                member_agent_run_response.tools is None or len(member_agent_run_response.tools) == 0
                            ):
                                return f"Agent {member_name}: No response from the member agent."
                            elif isinstance(member_agent_run_response.content, str):
                                if len(member_agent_run_response.content.strip()) > 0:
                                    return f"Agent {member_name}: {member_agent_run_response.content}"
                                elif (
                                    member_agent_run_response.tools is not None
                                    and len(member_agent_run_response.tools) > 0
                                ):
                                    return f"Agent {member_name}: {','.join([tool.result for tool in member_agent_run_response.tools])}"
                            elif issubclass(type(member_agent_run_response.content), BaseModel):
                                return f"Agent {member_name}: {member_agent_run_response.content.model_dump_json(indent=2)}"  # type: ignore
                            else:
                                import json

                                return f"Agent {member_name}: {json.dumps(member_agent_run_response.content, indent=2)}"
                        except Exception as e:
                            return f"Agent {member_name}: Error - {str(e)}"

                        return f"Agent {member_name}: No Response"

                    tasks.append(run_member_agent)  # type: ignore

                results = await asyncio.gather(*[task() for task in tasks])  # type: ignore
                for result in results:
                    yield result

            # After all the member runs, switch back to the team logger
            use_team_logger()

        if self.delegate_task_to_all_members:
            if async_mode:
                delegate_function = adelegate_task_to_members  # type: ignore
            else:
                delegate_function = delegate_task_to_members  # type: ignore

            delegate_func = Function.from_callable(delegate_function, name="delegate_task_to_members")
        else:
            if async_mode:
                delegate_function = adelegate_task_to_member  # type: ignore
            else:
                delegate_function = delegate_task_to_member  # type: ignore

            delegate_func = Function.from_callable(delegate_function, name="delegate_task_to_member")

        if self.respond_directly:
            delegate_func.stop_after_tool_call = True
            delegate_func.show_result = True

        return delegate_func

    ###########################################################################
    # Session Management
    ###########################################################################
    def _read_session(
        self, session_id: str, session_type: SessionType = SessionType.TEAM
    ) -> Optional[Union[TeamSession, WorkflowSession]]:
        """Get a Session from the database."""
        try:
            if not self.db:
                raise ValueError("Db not initialized")
            session = self.db.get_session(session_id=session_id, session_type=session_type)
            return session  # type: ignore
        except Exception as e:
            log_warning(f"Error getting session from db: {e}")
            return None

    async def _aread_session(
        self, session_id: str, session_type: SessionType = SessionType.TEAM
    ) -> Optional[Union[TeamSession, WorkflowSession]]:
        """Get a Session from the database."""
        try:
            if not self.db:
                raise ValueError("Db not initialized")
            self.db = cast(AsyncBaseDb, self.db)
            session = await self.db.get_session(session_id=session_id, session_type=session_type)
            return session  # type: ignore
        except Exception as e:
            log_warning(f"Error getting session from db: {e}")
            return None

    def _upsert_session(self, session: TeamSession) -> Optional[TeamSession]:
        """Upsert a Session into the database."""

        try:
            if not self.db:
                raise ValueError("Db not initialized")
            return self.db.upsert_session(session=session)  # type: ignore
        except Exception as e:
            log_warning(f"Error upserting session into db: {e}")
        return None

    async def _aupsert_session(self, session: TeamSession) -> Optional[TeamSession]:
        """Upsert a Session into the database."""

        try:
            if not self.db:
                raise ValueError("Db not initialized")
            return await self.db.upsert_session(session=session)  # type: ignore
        except Exception as e:
            log_warning(f"Error upserting session into db: {e}")
        return None

    def _read_or_create_session(self, session_id: str, user_id: Optional[str] = None) -> TeamSession:
        """Load the TeamSession from storage

        Returns:
            Optional[TeamSession]: The loaded TeamSession or None if not found.
        """
        from time import time

        from agno.session.team import TeamSession

        # Return existing session if we have one
        if self._cached_session is not None and self._cached_session.session_id == session_id:
            return self._cached_session

        # Try to load from database
        team_session = None
        if self.db is not None and self.parent_team_id is None and self.workflow_id is None:
            team_session = cast(TeamSession, self._read_session(session_id=session_id))

        # Create new session if none found
        if team_session is None:
            log_debug(f"Creating new TeamSession: {session_id}")
            session_data = {}
            if self.session_state is not None:
                from copy import deepcopy

                session_data["session_state"] = deepcopy(self.session_state)
            team_session = TeamSession(
                session_id=session_id,
                team_id=self.id,
                user_id=user_id,
                team_data=self._get_team_data(),
                session_data=session_data,
                metadata=self.metadata,
                created_at=int(time()),
            )

        # Cache the session if relevant
        if team_session is not None and self.cache_session:
            self._cached_session = team_session

        return team_session

    async def _aread_or_create_session(self, session_id: str, user_id: Optional[str] = None) -> TeamSession:
        """Load the TeamSession from storage

        Returns:
            Optional[TeamSession]: The loaded TeamSession or None if not found.
        """
        from time import time

        from agno.session.team import TeamSession

        # Return existing session if we have one
        if self._cached_session is not None and self._cached_session.session_id == session_id:
            return self._cached_session

        # Try to load from database
        team_session = None
        if self.db is not None and self.parent_team_id is None and self.workflow_id is None:
            if self._has_async_db():
                team_session = cast(TeamSession, await self._aread_session(session_id=session_id))
            else:
                team_session = cast(TeamSession, self._read_session(session_id=session_id))

        # Create new session if none found
        if team_session is None:
            log_debug(f"Creating new TeamSession: {session_id}")
            team_session = TeamSession(
                session_id=session_id,
                team_id=self.id,
                user_id=user_id,
                team_data=self._get_team_data(),
                session_data={},
                metadata=self.metadata,
                created_at=int(time()),
            )

        # Cache the session if relevant
        if team_session is not None and self.cache_session:
            self._cached_session = team_session

        return team_session

    def _load_session_state(self, session: TeamSession, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Load and return the stored session_state from the database, optionally merging it with the given one"""

        from agno.utils.merge_dict import merge_dictionaries

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

    def _update_metadata(self, session: TeamSession):
        """Update the extra_data in the session"""
        from agno.utils.merge_dict import merge_dictionaries

        # Read metadata from the database
        if session.metadata is not None:
            # If metadata is set in the agent, update the database metadata with the agent's metadata
            if self.metadata is not None:
                # Updates agent's session metadata in place
                merge_dictionaries(session.metadata, self.metadata)
            # Update the current metadata with the metadata from the database which is updated in place
            self.metadata = session.metadata

    # -*- Public convenience functions
    def get_run_output(
        self, run_id: str, session_id: Optional[str] = None
    ) -> Optional[Union[TeamRunOutput, RunOutput]]:
        """
        Get a RunOutput or TeamRunOutput from the database.  Handles cached sessions.

        Args:
            run_id (str): The run_id to load from storage.
            session_id (Optional[str]): The session_id to load from storage.
        """
        if not session_id and not self.session_id:
            raise Exception("No session_id provided")

        session_id_to_load = session_id or self.session_id
        return get_run_output_util(self, run_id=run_id, session_id=session_id_to_load)

    async def aget_run_output(
        self, run_id: str, session_id: Optional[str] = None
    ) -> Optional[Union[TeamRunOutput, RunOutput]]:
        """
        Get a RunOutput or TeamRunOutput from the database.  Handles cached sessions.

        Args:
            run_id (str): The run_id to load from storage.
            session_id (Optional[str]): The session_id to load from storage.
        """
        if not session_id and not self.session_id:
            raise Exception("No session_id provided")

        session_id_to_load = session_id or self.session_id
        return await aget_run_output_util(self, run_id=run_id, session_id=session_id_to_load)

    def get_last_run_output(self, session_id: Optional[str] = None) -> Optional[TeamRunOutput]:
        """
        Get the last run response from the database.

        Args:
            session_id (Optional[str]): The session_id to load from storage.

        Returns:
            RunOutput: The last run response from the database.
        """
        if not session_id and not self.session_id:
            raise Exception("No session_id provided")

        session_id_to_load = session_id or self.session_id
        return cast(TeamRunOutput, get_last_run_output_util(self, session_id=session_id_to_load))

    async def aget_last_run_output(self, session_id: Optional[str] = None) -> Optional[TeamRunOutput]:
        """
        Get the last run response from the database.

        Args:
            session_id (Optional[str]): The session_id to load from storage.

        Returns:
            RunOutput: The last run response from the database.
        """
        if not session_id and not self.session_id:
            raise Exception("No session_id provided")

        session_id_to_load = session_id or self.session_id
        return cast(TeamRunOutput, await aget_last_run_output_util(self, session_id=session_id_to_load))

    def get_session(
        self,
        session_id: Optional[str] = None,
    ) -> Optional[TeamSession]:
        """Load an TeamSession from database.

        Args:
            session_id: The session_id to load from storage.

        Returns:
            TeamSession: The TeamSession loaded from the database or created if it does not exist.
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
            # We have a standalone team, so we are loading a TeamSession
            if self.workflow_id is None:
                loaded_session = cast(TeamSession, self._read_session(session_id=session_id_to_load))  # type: ignore
            # We have a workflow team, so we are loading a WorkflowSession
            else:
                loaded_session = cast(WorkflowSession, self._read_session(session_id=session_id_to_load))  # type: ignore

            # Cache the session if relevant
            if loaded_session is not None and self.cache_session:
                self._agent_session = loaded_session

            return loaded_session

        log_debug(f"TeamSession {session_id_to_load} not found in db")
        return None

    async def aget_session(
        self,
        session_id: Optional[str] = None,
    ) -> Optional[TeamSession]:
        """Load an TeamSession from database.

        Args:
            session_id: The session_id to load from storage.

        Returns:
            TeamSession: The TeamSession loaded from the database or created if it does not exist.
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
            # We have a standalone team, so we are loading a TeamSession
            if self.workflow_id is None:
                loaded_session = cast(TeamSession, await self._aread_session(session_id=session_id_to_load))  # type: ignore
            # We have a workflow team, so we are loading a WorkflowSession
            else:
                loaded_session = cast(WorkflowSession, await self._aread_session(session_id=session_id_to_load))  # type: ignore

            # Cache the session if relevant
            if loaded_session is not None and self.cache_session:
                self._cached_session = loaded_session

            return loaded_session

        log_debug(f"TeamSession {session_id_to_load} not found in db")
        return None

    def save_session(self, session: TeamSession) -> None:
        """
        Save the TeamSession to storage

        Args:
            session: The TeamSession to save.
        """
        if self._has_async_db():
            raise ValueError("Async database not supported for save_session")

        if self.db is not None and self.parent_team_id is None and self.workflow_id is None:
            if session.session_data is not None and "session_state" in session.session_data:
                session.session_data["session_state"].pop("current_session_id", None)  # type: ignore
                session.session_data["session_state"].pop("current_user_id", None)  # type: ignore
                session.session_data["session_state"].pop("current_run_id", None)  # type: ignore

            # scrub the member responses based on storage settings
            if session.runs is not None:
                for run in session.runs:
                    if hasattr(run, "member_responses"):
                        if not self.store_member_responses:
                            # Remove all member responses
                            run.member_responses = []
                        else:
                            # Scrub individual member responses based on their storage flags
                            self._scrub_member_responses(run.member_responses)
            self._upsert_session(session=session)
            log_debug(f"Created or updated TeamSession record: {session.session_id}")

    async def asave_session(self, session: TeamSession) -> None:
        """
        Save the TeamSession to storage

        Args:
            session: The TeamSession to save.
        """
        if self.db is not None and self.parent_team_id is None and self.workflow_id is None:
            if session.session_data is not None and "session_state" in session.session_data:
                session.session_data["session_state"].pop("current_session_id", None)  # type: ignore
                session.session_data["session_state"].pop("current_user_id", None)  # type: ignore
                session.session_data["session_state"].pop("current_run_id", None)  # type: ignore

            # scrub the member responses if not storing them
            if not self.store_member_responses and session.runs is not None:
                for run in session.runs:
                    if hasattr(run, "member_responses"):
                        run.member_responses = []

            if self._has_async_db():
                await self._aupsert_session(session=session)
            else:
                self._upsert_session(session=session)
            log_debug(f"Created or updated TeamSession record: {session.session_id}")

    def generate_session_name(self, session: TeamSession) -> str:
        """
        Generate a name for the team session

        Args:
            session: The TeamSession to generate a name for.
        Returns:
            str: The generated session name.
        """

        if self.model is None:
            raise Exception("Model not set")

        gen_session_name_prompt = "Team Conversation\n"

        # Get team session messages for generating the name
        messages_for_generating_session_name = self.get_messages_for_session()

        for message in messages_for_generating_session_name:
            gen_session_name_prompt += f"{message.role.upper()}: {message.content}\n"

        gen_session_name_prompt += "\n\nTeam Session Name: "

        system_message = Message(
            role=self.system_message_role,
            content="Please provide a suitable name for this conversation in maximum 5 words. "
            "Remember, do not exceed 5 words.",
        )
        user_message = Message(role="user", content=gen_session_name_prompt)
        generate_name_messages = [system_message, user_message]

        # Generate name
        generated_name = self.model.response(messages=generate_name_messages)
        content = generated_name.content
        if content is None:
            log_error("Generated name is None. Trying again.")
            return self.generate_session_name(session=session)
        if len(content.split()) > 15:
            log_error("Generated name is too long. Trying again.")
            return self.generate_session_name(session=session)
        return content.replace('"', "").strip()

    def set_session_name(
        self, session_id: Optional[str] = None, autogenerate: bool = False, session_name: Optional[str] = None
    ) -> TeamSession:
        """
        Set the session name and save to storage

        Args:
            session_id: The session ID to set the name for. If not provided, the current cached session ID is used.
            autogenerate: Whether to autogenerate the session name.
            session_name: The session name to set. If not provided, the session name will be autogenerated.
        Returns:
            TeamSession: The updated session.
        """
        session_id = session_id or self.session_id

        if session_id is None:
            raise Exception("Session ID is not set")

        return cast(
            TeamSession,
            set_session_name_util(self, session_id=session_id, autogenerate=autogenerate, session_name=session_name),
        )

    async def aset_session_name(
        self, session_id: Optional[str] = None, autogenerate: bool = False, session_name: Optional[str] = None
    ) -> TeamSession:
        """
        Set the session name and save to storage

        Args:
            session_id: The session ID to set the name for. If not provided, the current cached session ID is used.
            autogenerate: Whether to autogenerate the session name.
            session_name: The session name to set. If not provided, the session name will be autogenerated.
        Returns:
            TeamSession: The updated session.
        """
        session_id = session_id or self.session_id

        if session_id is None:
            raise Exception("Session ID is not set")

        return cast(
            TeamSession,
            await aset_session_name_util(
                self, session_id=session_id, autogenerate=autogenerate, session_name=session_name
            ),
        )

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
        """Get the session state for the given session ID.

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
        """Get the session state for the given session ID.

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
            entity=self, session_state_updates=session_state_updates, session_id=session_id
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

    def get_chat_history(self, session_id: Optional[str] = None) -> List[Message]:
        """Read the chat history from the session

        Args:
            session_id: The session ID to get the chat history for. If not provided, the current cached session ID is used.
        Returns:
            List[Message]: The chat history from the session.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            raise Exception("Session ID is not set")

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
            raise Exception("Session ID is not set")

        return await aget_chat_history_util(self, session_id=session_id)

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

        session = self.get_session(session_id=session_id)  # type: ignore

        if session is None:
            log_warning(f"Session {session_id} not found")
            return []

        # Only filter by agent_id if this is part of a team
        return session.get_messages_from_last_n_runs(
            team_id=self.id,
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

        session = await self.aget_session(session_id=session_id)  # type: ignore

        if session is None:
            log_warning(f"Session {session_id} not found")
            return []

        # Only filter by agent_id if this is part of a team
        return session.get_messages_from_last_n_runs(
            team_id=self.id,
        )

    def get_session_summary(self, session_id: Optional[str] = None) -> Optional[SessionSummary]:
        """Get the session summary for the given session ID and user ID.

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

        return session.get_session_summary()  # type: ignore

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

        return session.get_session_summary()  # type: ignore

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

    ###########################################################################
    # Handle images, videos and audio
    ###########################################################################

    def _add_image(self, image: Image, run_response: TeamRunOutput) -> None:
        """Add an image to both the agent's stateful storage and the current run response"""
        # Add to run response
        if run_response.images is None:
            run_response.images = []
        run_response.images.append(image)

    def _add_video(self, video: Video, run_response: TeamRunOutput) -> None:
        """Add a video to both the agent's stateful storage and the current run response"""
        # Add to run response
        if run_response.videos is None:
            run_response.videos = []
        run_response.videos.append(video)

    def _add_audio(self, audio: Audio, run_response: TeamRunOutput) -> None:
        """Add audio to both the agent's stateful storage and the current run response"""
        # Add to run response
        if run_response.audio is None:
            run_response.audio = []
        run_response.audio.append(audio)

    def _add_file(self, file: File, run_response: TeamRunOutput) -> None:
        """Add file to both the agent's stateful storage and the current run response"""
        # Add to run response
        if run_response.files is None:
            run_response.files = []
        run_response.files.append(file)

    def _update_reasoning_content_from_tool_call(
        self, run_response: TeamRunOutput, tool_name: str, tool_args: Dict[str, Any]
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
            add_reasoning_step_to_metadata(run_response, reasoning_step)

            formatted_content = f"## {title}\n{thought}\n"
            if action:
                formatted_content += f"Action: {action}\n"
            if confidence is not None:
                formatted_content += f"Confidence: {confidence}\n"
            formatted_content += "\n"

            append_to_reasoning_content(run_response, formatted_content)
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
            add_reasoning_step_to_metadata(run_response, reasoning_step)

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

            append_to_reasoning_content(run_response, formatted_content)
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
            add_reasoning_step_to_metadata(run_response, reasoning_step)
            append_to_reasoning_content(run_response, formatted_content)
            return reasoning_step

        return None

    ###########################################################################
    # Knowledge
    ###########################################################################

    def add_to_knowledge(self, query: str, result: str) -> str:
        """Use this function to add information to the knowledge base for future use.

        Args:
            query: The query to add.
            result: The result of the query.

        Returns:
            str: A string indicating the status of the addition.
        """
        if self.knowledge is None:
            log_warning("Knowledge is not set, cannot add to knowledge")
            return "Knowledge is not set, cannot add to knowledge"

        if self.knowledge.vector_db is None:
            log_warning("Knowledge vector database is not set, cannot add to knowledge")
            return "Knowledge vector database is not set, cannot add to knowledge"

        document_name = query.replace(" ", "_").replace("?", "").replace("!", "").replace(".", "")
        document_content = json.dumps({"query": query, "result": result})
        log_info(f"Adding document to Knowledge: {document_name}: {document_content}")
        import asyncio

        from agno.knowledge.reader.text_reader import TextReader

        asyncio.run(
            self.knowledge.add_content_async(name=document_name, text_content=document_content, reader=TextReader())
        )
        return "Successfully added to knowledge base"

    def get_relevant_docs_from_knowledge(
        self, query: str, num_documents: Optional[int] = None, filters: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Optional[List[Union[Dict[str, Any], str]]]:
        """Return a list of references from the knowledge base"""
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
                if "team" in sig.parameters:
                    knowledge_retriever_kwargs = {"team": self}
                if "filters" in sig.parameters:
                    knowledge_retriever_kwargs["filters"] = filters
                knowledge_retriever_kwargs.update({"query": query, "num_documents": num_documents, **kwargs})
                return self.knowledge_retriever(**knowledge_retriever_kwargs)
            except Exception as e:
                log_warning(f"Knowledge retriever failed: {e}")
                raise e
        try:
            if self.knowledge is None or self.knowledge.vector_db is None:
                return None

            if num_documents is None:
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
        self, query: str, num_documents: Optional[int] = None, filters: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Optional[List[Union[Dict[str, Any], str]]]:
        """Get relevant documents from knowledge base asynchronously."""
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
                # type: ignore
                log_info(f"Valid filter keys are: {self.knowledge.valid_metadata_filters}")

                # Only use valid filters
                filters = valid_filters
                if not filters:
                    log_warning("No valid filters remain after validation. Search will proceed without filters.")

        if self.knowledge_retriever is not None and callable(self.knowledge_retriever):
            from inspect import isawaitable, signature

            try:
                sig = signature(self.knowledge_retriever)
                knowledge_retriever_kwargs: Dict[str, Any] = {}
                if "team" in sig.parameters:
                    knowledge_retriever_kwargs = {"team": self}
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

        try:
            if self.knowledge is None or self.knowledge.vector_db is None:
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

        return json.dumps(docs, indent=2)

    def _get_effective_filters(self, knowledge_filters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Determine effective filters for the team, considering:
        1. Team-level filters (self.knowledge_filters)
        2. Run-time filters (knowledge_filters)

        Priority: Run-time filters > Team filters
        """
        effective_filters = None

        # Start with team-level filters if they exist
        if self.knowledge_filters:
            effective_filters = self.knowledge_filters.copy()

        # Apply run-time filters if they exist
        if knowledge_filters:
            if effective_filters:
                effective_filters.update(knowledge_filters)
            else:
                effective_filters = knowledge_filters

        return effective_filters

    def _get_search_knowledge_base_function(
        self,
        run_response: TeamRunOutput,
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
                    query=query, references=docs_from_knowledge, time=round(retrieval_timer.elapsed, 4)
                )
                # Add the references to the run_response
                if run_response.references is None:
                    run_response.references = []
                run_response.references.append(references)
            retrieval_timer.stop()
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
                    query=query, references=docs_from_knowledge, time=round(retrieval_timer.elapsed, 4)
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

        return Function.from_callable(search_knowledge_base_function, name="search_knowledge_base")

    def _get_search_knowledge_base_with_agentic_filters_function(
        self,
        run_response: TeamRunOutput,
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
                    query=query, references=docs_from_knowledge, time=round(retrieval_timer.elapsed, 4)
                )
                # Add the references to the run_response
                if run_response.references is None:
                    run_response.references = []
                run_response.references.append(references)
            retrieval_timer.stop()
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
                    query=query, references=docs_from_knowledge, time=round(retrieval_timer.elapsed, 4)
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

        return Function.from_callable(search_knowledge_base_function, name="search_knowledge_base")

    ###########################################################################
    # Logging
    ###########################################################################

    def _get_team_data(self) -> Dict[str, Any]:
        team_data: Dict[str, Any] = {}
        if self.name is not None:
            team_data["name"] = self.name
        if self.id is not None:
            team_data["team_id"] = self.id
        if self.model is not None:
            team_data["model"] = self.model.to_dict()
        return team_data

    ###########################################################################
    # Api functions
    ###########################################################################

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """Get the telemetry data for the team"""
        return {
            "team_id": self.id,
            "db_type": self.db.__class__.__name__ if self.db else None,
            "model_provider": self.model.provider if self.model else None,
            "model_name": self.model.name if self.model else None,
            "model_id": self.model.id if self.model else None,
            "parser_model": self.parser_model.to_dict() if self.parser_model else None,
            "output_model": self.output_model.to_dict() if self.output_model else None,
            "member_count": len(self.members) if self.members else 0,
            "has_knowledge": self.knowledge is not None,
            "has_tools": self.tools is not None,
        }

    def _log_team_telemetry(self, session_id: str, run_id: Optional[str] = None) -> None:
        """Send a telemetry event to the API for a created Team run"""

        self._set_telemetry()
        if not self.telemetry:
            return

        from agno.api.team import TeamRunCreate, create_team_run

        try:
            create_team_run(
                run=TeamRunCreate(session_id=session_id, run_id=run_id, data=self._get_telemetry_data()),
            )
        except Exception as e:
            log_debug(f"Could not create Team run telemetry event: {e}")

    async def _alog_team_telemetry(self, session_id: str, run_id: Optional[str] = None) -> None:
        """Send a telemetry event to the API for a created Team async run"""

        self._set_telemetry()
        if not self.telemetry:
            return

        from agno.api.team import TeamRunCreate, acreate_team_run

        try:
            await acreate_team_run(
                run=TeamRunCreate(session_id=session_id, run_id=run_id, data=self._get_telemetry_data())
            )
        except Exception as e:
            log_debug(f"Could not create Team run telemetry event: {e}")
