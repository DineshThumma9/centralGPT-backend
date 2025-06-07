from typing import List,Literal,Optional
from pydantic import BaseModel, Field


from pydantic import BaseModel
from typing import Optional, Literal


class Message(BaseModel):
    content: str


class UserMessage(Message):
    role: Literal["user"]
    name: Optional[str] = None


class SystemMessage(Message):
    role: Literal["system"]
    name: Optional[str] = None


class AssistantMessage(Message):
    role: Literal["assistant"]
    prefix: bool  # This is a custom field you introduced
    reasoning_content: Optional[str] = None
    name: Optional[str] = None


class ToolMessage(Message):
    role: Literal["tool"]
    tool_call_id: str


--- Streaming Options ---

class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False

# --- Tool and Function Support ---

class FunctionTool(BaseModel):
    type: Literal["function"]
    function: dict  # Replace with detailed function schema as needed

class ToolChoice(BaseModel):
    type: Literal["function"]
    function: dict  # Example: {"name": "function_name"}

# --- Main Request Schema ---

class DeepSeekRequest(BaseModel):
    # Conversation history
    messages: List[Union[UserMessage, SystemMessage, AssistantMessage, ToolMessage]]

    # Model selection
    model: Literal["deepseek-chat", "deepseek-reasoner"]

    # Sampling controls
    temperature: Optional[float] = Field(
        default=1.0, le=2.0,
        description="Controls randomness. Lower is more deterministic."
    )
    top_p: Optional[float] = Field(
        default=1.0, le=1.0,
        description="Nucleus sampling cutoff. Use instead of temperature."
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0, ge=-2.0, le=2.0,
        description="Penalizes repeated tokens."
    )
    presence_penalty: Optional[float] = Field(
        default=0.0, ge=-2.0, le=2.0,
        description="Penalizes tokens that already appear, promoting topic diversity."
    )

    # Output configuration
    max_tokens: Optional[int] = Field(
        default=4096, gt=1, le=8192,
        description="Maximum tokens for model output."
    )
    response_format: Optional[Union[Literal["text"], dict]] = Field(
        default="text",
        description='Either "text" or {"type": "json_object"} for structured responses.'
    )

    # Streaming setup
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None

    # Tool usage (function calling)
    tools: Optional[List[FunctionTool]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], ToolChoice]] = "auto"

    # Log probability debugging
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = Field(
        default=None, le=20,
        description="Number of top logprobs to return per token (requires logprobs=True)."
    )


from pydantic import BaseModel, Field
from typing import List, Optional


class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: List[int]


class LogprobContentItem(BaseModel):
    token: str
    logprob: float
    bytes: List[int]
    top_logprobs: List[TopLogprob]


class Logprobs(BaseModel):
    content: List[LogprobContentItem]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str  # usually "function"
    function: FunctionCall


class Message(BaseModel):
    content: str
    reasoning_content: str
    tool_calls: List[ToolCall]
    role: str  # should be "assistant"


class Choice(BaseModel):
    finish_reason: str  # e.g., "stop", "length", etc.
    index: int
    message: Message
    logprobs: Optional[Logprobs]


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    prompt_cache_hit_tokens: int
    prompt_cache_miss_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails


class DeekSeekeResponse(BaseModel):
    id: str
    object: str  # typically "chat.completion"
    created: int
    model: str
    system_fingerprint: str
    choices: List[Choice]
    usage: Usage


from typing import Optional, List, Literal, Union
from pydantic import BaseModel


class Delta(BaseModel):
    content: Optional[str] = None
    role: Optional[Literal["assistant"]] = None


class Choice(BaseModel):
    index: int
    delta: Delta
    finish_reason: Optional[str] = None
    logprobs: Optional[dict] = None  # You can replace with exact schema if needed


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    system_fingerprint: Optional[str]
    choices: List[Choice]
    usage: Optional[Usage] = None
