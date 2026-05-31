from deepeval.test_case import LLMTestCase
## Defining Test Cases
test_cases = [

LLMTestCase(
    input="What is Kubernetes?",

    actual_output="""
    Kubernetes is a container orchestration platform.
    """,

    expected_output="""
    Kubernetes is an open-source container orchestration
    platform used to automate deployment, scaling,
    and management of containers.
    """
),

LLMTestCase(
    input="What is Docker?",

    actual_output="""
    Docker is used for containers.
    """,

    expected_output="""
    Docker is a platform used to build, package,
    and run applications inside containers.
    """
),

LLMTestCase(
    input="What is Terraform?",

    actual_output="""
    Terraform is Infrastructure as Code.
    """,

    expected_output="""
    Terraform is an Infrastructure as Code tool
    used to provision and manage cloud resources.
    """
    )
]
