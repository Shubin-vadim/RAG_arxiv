from llama_index.core import PromptTemplate

"""
Module-level documentation.

This module provides prompt templates for generating prompts for the RAG (Retrieval-Augmented Generation) system.

Attributes:
    user_template (PromptTemplate): Template for generating prompts for user queries.
    refine_template (PromptTemplate): Template for refining existing answers based on new context.

Note:
    These prompt templates are used to generate prompts for the RAG system. They follow specific rules and formats to ensure effective generation of responses.
"""


system_prompt_str = """
You are an expert Q&A system that is trusted around the world.
Always answer the query using the provided context information, and not prior knowledge.
Some rules to follow:
1. Never directly reference the given context in your answer.
2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
"""

user_prompt_str = """
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer:
"""

refine_prompt_str = """
The original query is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
{context_msg}
------------
Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.
Refined Answer:
"""

user_template = PromptTemplate(user_prompt_str)
refine_template = PromptTemplate(refine_prompt_str)
