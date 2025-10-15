import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

# Step 3.1: Configure LLM models
# Assumes OAI_CONFIG_LIST.json with your API keys/models; alternatively, hardcode here
config_list_text = [
    {
        "model": "gpt-4",
        "api_key": autogen.oai.get_config_list()["OPENAI_API_KEY"]  # Fallback to env var
    }
]

config_list_multimodal = [
    {
        "model": "gpt-4-vision-preview",
        "api_key": autogen.oai.get_config_list()["OPENAI_API_KEY"]
    }
]

# Step 3.2: Define Agents
# User Proxy: Simulates user, sends multimodal inputs
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    code_execution_config=False,
    system_message="You are a human user providing multimodal inputs (text + images) for analysis."
)

# Text Assistant: Handles synthesis of responses
text_assistant = AssistantAgent(
    name="TextAssistant",
    llm_config={"config_list": config_list_text},
    system_message="You are a helpful assistant. Synthesize insights from others' reasoning."
)

# Vision Agent: Processes images with initial description
vision_agent = MultimodalConversableAgent(
    name="VisionAgent",
    llm_config={"config_list": config_list_multimodal},
    system_message="You are a vision expert. Describe images accurately, then pass to ReasoningAgent for deeper analysis."
)

# New: Reasoning Agent: Applies chain-of-thought reasoning to multimodal data
reasoning_agent = AssistantAgent(
    name="ReasoningAgent",
    llm_config={"config_list": config_list_text},
    system_message="""
    You are a reasoning expert. Use chain-of-thought (CoT) for any input:
    1. Break down the problem or data (e.g., image description).
    2. Analyze step-by-step (e.g., identify key elements, infer relationships).
    3. Draw logical conclusions.
    4. Suggest next steps if needed.
    Apply this to multimodal information, like combining text queries with visual analysis.
    """
)

# Step 3.3: Set up Group Chat for Multi-Agent Collaboration
groupchat = GroupChat(
    agents=[user_proxy, vision_agent, reasoning_agent, text_assistant],
    messages=[],
    max_round=10  # Limit rounds to avoid long conversations
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list_text}
)

# Step 3.4: Start the Conversation with Multimodal + Reasoning Example
user_proxy.initiate_chat(
    manager,
    message="""
    Analyze this image: https://example.com/sample_image.jpg (imagine it's a photo of a crowded city street).
    VisionAgent: Describe it.
    ReasoningAgent: Reason step-by-step about potential environmental impacts (e.g., pollution, urban planning).
    TextAssistant: Summarize the discussion.
    """
)

# Output: Agents will collaborate, e.g.,
# - VisionAgent: Describes the image.
# - ReasoningAgent: "Step 1: Key elements include cars and buildings. Step 2: Infer high traffic leads to air pollution. Step 3: Conclusion: Suggests need for green spaces."
# - TextAssistant: Integrates into a coherent response.