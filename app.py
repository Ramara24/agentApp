# -*- coding: utf-8 -*-
"""Data Analyst Agent Assignment"""

import streamlit as st
from datasets import load_dataset
import pandas as pd
from typing import Dict, List, Any
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# --- Environment Setup ---
#load_dotenv()
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#if not OPENAI_API_KEY:
#    st.error("Missing OPENAI_API_KEY in environment variables")
#    st.stop()
    

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
st.write("Key starts with:", OPENAI_API_KEY[:10])
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Constants ---
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
MODEL_NAME = "gpt-4-turbo"


# --- Load dataset ---
@st.cache_data
def load_dataset_to_df() -> pd.DataFrame:
    dataset = load_dataset(DATASET_NAME, split="train")
    df = pd.DataFrame(dataset)
    df['category'] = df['category'].str.upper().str.strip()
    df['intent'] = df['intent'].str.lower().str.strip()
    return df[['instruction', 'response', 'category', 'intent']].dropna()

# --- Tools ---
class CustomerServiceTools:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def select_semantic_intent(self, intent_names: List[str]) -> pd.DataFrame:
        return self.df[self.df['intent'].isin([i.lower() for i in intent_names])]

    def select_semantic_category(self, category_names: List[str]) -> pd.DataFrame:
        return self.df[self.df['category'].isin([c.upper() for c in category_names])]

    def count_intent(self, intent: str) -> int:
        return len(self.df[self.df['intent'] == intent.lower()])

    def count_category(self, category: str) -> int:
        return len(self.df[self.df['category'] == category.upper()])

    def show_examples(self, n: int, category: str = None, intent: str = None) -> List[Dict[str, Any]]:
        filtered = self.df
        if category:
            filtered = filtered[filtered['category'] == category.upper()]
        if intent:
            filtered = filtered[filtered['intent'] == intent.lower()]
        sample_size = min(n, len(filtered))
        return filtered.sample(sample_size).to_dict('records')

    def summarize(self, user_request: str) -> str:
        user_request_lower = user_request.lower()
        possible_categories = [cat for cat in self.get_all_categories() if cat.lower() in user_request_lower]
        possible_intents = [intent for intent in self.get_all_intents() if intent.lower() in user_request_lower]

        filtered_df = self.df

        # Heuristic: check if the user explicitly mentioned category or intent
        if "intent" in user_request_lower and possible_intents:
            filtered_df = filtered_df[filtered_df['intent'].isin([i.lower() for i in possible_intents])]
        elif "category" in user_request_lower and possible_categories:
            filtered_df = filtered_df[filtered_df['category'].isin([c.upper() for c in possible_categories])]
        # fallback: if only one match is found
        elif possible_intents and not possible_categories:
            filtered_df = filtered_df[filtered_df['intent'].isin([i.lower() for i in possible_intents])]
        elif possible_categories and not possible_intents:
            filtered_df = filtered_df[filtered_df['category'].isin([c.upper() for c in possible_categories])]

        if filtered_df.empty:
            return "No relevant examples found to summarize."

        context_df = filtered_df.sample(min(10, len(filtered_df)))[['instruction', 'response']]
        text_block = "\n\n".join([f"Customer: {row['instruction']}\nAgent: {row['response']}" for _, row in context_df.iterrows()])

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a customer service analyst. Summarize the tone, common patterns, and key concerns based on the following conversations."},
                {"role": "user", "content": text_block}
            ]
        )
        return completion.choices[0].message.content

        
    def get_all_categories(self) -> List[str]:
        """Return a list of unique categories in the dataset"""
        return sorted(self.df['category'].unique().tolist())
        
    def get_all_intents(self) -> List[str]:
        return sorted(self.df['intent'].unique().tolist())
        
    def get_intent_distribution(self) -> Dict[str, int]:
        return self.df['intent'].value_counts().to_dict()

    def get_top_categories(self, n: int = 5) -> List[str]:
        return self.df['category'].value_counts().head(n).index.tolist()

    def sum_values(self, a: int, b: int) -> int:
        return a + b

    def finish(self) -> str:
        return "This question is outside the scope of the customer service dataset. Please ask something related to customer interactions."


# --- Helper Functions ---
def format_examples(examples: List[Dict]) -> str:
    return "\n\n".join([f"**Example {i+1}**\n**Customer**: {e['instruction']}\n**Agent**: {e['response']}" 
                       for i, e in enumerate(examples)])

def run_tool(tool_call, tools: CustomerServiceTools) -> str:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    try:
        if name == "get_all_categories":
            categories = tools.get_all_categories()
            return f"Available categories:\n- " + "\n- ".join(categories)
        elif name == "select_semantic_intent":
            df = tools.select_semantic_intent(args['intent_names'])
            return f"Found {len(df)} matches for intents: {', '.join(args['intent_names'])}"
        elif name == "select_semantic_category":
            df = tools.select_semantic_category(args['category_names'])
            return f"Found {len(df)} matches for categories: {', '.join(args['category_names'])}"
        elif name == "count_intent":
            count = tools.count_intent(args['intent'])
            return f"Count: {count} examples for intent '{args['intent']}'"
        elif name == "count_category":
            count = tools.count_category(args['category'])
            return f"Count: {count} examples for category '{args['category']}'"
        elif name == "get_all_intents":
            intents = tools.get_all_intents()
            return f"Available intents:\n- " + "\n- ".join(intents)
        elif name == "get_top_categories":
            top = tools.get_top_categories(args["n"])
            return f"Top {args['n']} categories:\n- " + "\n- ".join(top)
        elif name == "sum":
            return f"Sum: {tools.sum_values(args['a'], args['b'])}"
        elif name == "finish":
            return tools.finish()
        elif name == "show_examples":
            examples = tools.show_examples(
                args['n'], 
                args.get('category'), 
                args.get('intent')
            )
            return format_examples(examples)
        elif name == "get_intent_distribution":
            intent_counts = tools.get_intent_distribution()
            return "Intent Distribution:\n" + "\n".join([f"- {intent}: {count}" for intent, count in intent_counts.items()])
        elif name == "summarize":
            return tools.summarize(args['user_request'])
        return f"Tool '{name}' not implemented"
    except Exception as e:
        return f"Tool error: {str(e)}"

# --- Tool Schemas ---
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "select_semantic_intent",
            "description": "Find matching intent names",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent_names": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["intent_names"]
            }
        }   
    },
    {
        "type": "function",
        "function": {
            "name": "select_semantic_category",
            "description": "Find matching category names",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_names": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["category_names"]
            }
        }   
    },
    {
        "type": "function",
        "function": {
            "name": "count_intent",
            "description": "Count examples by intent",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"}
                },
                "required": ["intent"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "get_all_categories",
        "description": "List all unique categories in the dataset",
        "parameters": {"type": "object", "properties": {}}  # No parameters needed
    }
},
    {
        "type": "function",
        "function": {
            "name": "count_category",
            "description": "Count examples by category",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"}
                },
                "required": ["category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "show_examples",
            "description": "Show dataset examples",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "category": {"type": "string"},
                    "intent": {"type": "string"}
                },
                "required": ["n"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Generate summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_request": {"type": "string"}
                },
                "required": ["user_request"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "get_all_intents",
            "description": "List all unique intents in the dataset",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_categories",
            "description": "Return the top N most frequent categories",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"}
                },
                "required": ["n"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sum",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_intent_distribution",
            "description": "Return a count of examples for each intent in the dataset",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call this tool if the user asks a question that is outside the dataset’s domain (customer service) ",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# --- OpenAI Call ---
def ask_openai_with_tools(user_query: str, tools: CustomerServiceTools) -> str:
    messages = [
        {"role": "system", "content": ("You are a customer support data analyst. Use tools to answer questions. ""For questions about available categories or intents, use get_all_categories or get_all_intents."
    "Only respond to questions that relate directly to customer service topics in the provided dataset. "
    "Do not answer questions about public figures, people, unrelated facts, or any information not represented in the dataset. "
    "If the question is unrelated or cannot be answered using the tools or data, call the 'finish' tool."
    "You may use multiple tools in sequence if needed. "
        "Examples:\n"
    "- If the user says 'How many refund requests did we get?', call:\n"
    "   1. select_semantic_intent([\"get_refund\"])\n"
    "   2. count()\n"
    "- If the user says 'Show examples of complaint', call:\n"
    "   1. select_semantic_intent([\"complaint\"])\n"
    "   2. show_examples(n=3)\n\n"
    "If the user says Category ACCOUNT, assume they are referring to the category label in the dataset, not any specific intent related to accounts."
    "Always use get_all_intents or get_all_categories to check valid values if you're unsure."
)},
        {"role": "user", "content": user_query}
    ]

    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto"
        )
        

        message = response.choices[0].message
        print("User query:", user_query)
        print("Tool calls:", message.tool_calls)
        if message.tool_calls:
            tool_responses = []
            for tool_call in message.tool_calls:
                tool_responses.append(run_tool(tool_call, tools))
            return "\n\n".join(tool_responses)
        return message.content or "No response generated"
    except Exception as e:
        return f"OpenAI call failed: {str(e)}"

# --- Streamlit UI ---
df = load_dataset_to_df()
tools = CustomerServiceTools(df)

st.title("📊 Customer Support Chatbot")
st.write(f"Dataset stats: {len(df)} rows, {df['intent'].nunique()} intents, {df['category'].nunique()}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt = st.chat_input("Ask about the dataset...")
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)

if prompt:
    st.session_state.chat_history.append(("user", prompt))
    with st.spinner("Analyzing..."):
        reply = ask_openai_with_tools(prompt, tools)
    st.session_state.chat_history.append(("assistant", reply))
    st.chat_message("assistant").markdown(reply)
