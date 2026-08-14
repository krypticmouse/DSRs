"""LangChain reference point: prompt template -> fake chat model -> string parser.

Note this pipeline does structurally LESS than DSPy/DSRs — no typed field
protocol, no output coercion, no constraint checking. Included as a floor for
'minimal Python chain overhead', not as a like-for-like harness.
"""

import time

t_import = time.perf_counter()
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
import langchain_core  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - t_import

CONTEXT = (
    "France is a country in Western Europe. Its capital and largest city is Paris, "
    "known for the Eiffel Tower and the Louvre."
)

if __name__ == "__main__":
    print(f"langchain-core {langchain_core.__version__}  (import: {IMPORT_SECONDS:.2f}s)")

    model = FakeListChatModel(responses=["Paris is the capital of France."])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question using the context. Be concise and accurate."),
            ("user", "Question: {question}\nContext: {context}"),
        ]
    )
    chain = prompt | model | StrOutputParser()

    warmup, iters = 100, 2000
    for i in range(warmup):
        chain.invoke({"question": f"warm {i}", "context": CONTEXT})

    t0 = time.perf_counter()
    for i in range(iters):
        chain.invoke({"question": f"What is the capital of France? {i}", "context": CONTEXT})
    elapsed = time.perf_counter() - t0
    print(f"chain.invoke (prompt|fake|parser)        {elapsed / iters * 1e6:10.1f} us/op")
