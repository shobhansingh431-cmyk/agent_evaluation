from deepeval.test_case import LLMTestCase


## Defining AI-focused test cases across the expected score range.
test_cases = [
    # Expected score: high / near 1.0
    LLMTestCase(
        input="What is retrieval-augmented generation?",
        actual_output="""
        Retrieval-augmented generation, or RAG, is a technique where an LLM
        retrieves relevant external documents or knowledge and uses that context
        to generate a more accurate, grounded answer.
        """,
        expected_output="""
        Retrieval-augmented generation is a technique where a language model
        retrieves relevant external information and uses it as context to produce
        more accurate and grounded responses.
        """,
    ),
    # Expected score: good / around 0.8
    LLMTestCase(
        input="What is an embedding model used for in AI?",
        actual_output="""
        An embedding model converts text into vectors so similar meanings are
        close together and can be searched or compared.
        """,
        expected_output="""
        An embedding model converts data such as text into numerical vectors
        that capture semantic meaning, enabling similarity search, clustering,
        retrieval, and comparison.
        """,
    ),
    # Expected score: medium / around 0.6
    LLMTestCase(
        input="What is fine-tuning an AI model?",
        actual_output="""
        Fine-tuning means training an existing model more on your own examples.
        """,
        expected_output="""
        Fine-tuning adapts a pretrained AI model by continuing training on a
        task-specific or domain-specific dataset so the model performs better
        for that use case.
        """,
    ),
    # Expected score: low / around 0.2 to 0.4
    LLMTestCase(
        input="What is hallucination in large language models?",
        actual_output="""
        Hallucination is when the model gives a long answer.
        """,
        expected_output="""
        Hallucination is when a language model generates information that sounds
        plausible but is false, unsupported, or not grounded in the provided
        context.
        """,
    ),
    # Expected score: zero / near 0.0
    LLMTestCase(
        input="What is prompt engineering?",
        actual_output="""
        Prompt engineering is a method for cooling computer hardware with liquid.
        """,
        expected_output="""
        Prompt engineering is the practice of designing and refining prompts or
        instructions to guide an AI model toward better, safer, or more useful
        responses.
        """,
    ),
]
