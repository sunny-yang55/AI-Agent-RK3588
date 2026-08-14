"""
LLM Adapter Test
"""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT))
from tools.llm.adapter import LLMAdapter


def main():

    print("=" * 50)
    print("AI-Agent LLM Test")
    print("=" * 50)


    llm =LLMAdapter()


    print("Provider:")
    print(llm.provider)


    print("\nModel:")
    print(llm.model)


    print("\nSending request...\n")


    answer = llm.chat(
        "请介绍一下你自己"
    )


    print("Response:")
    print(answer)



if __name__ == "__main__":
    main()