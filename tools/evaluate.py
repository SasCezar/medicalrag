import asyncio

from dotenv import load_dotenv

load_dotenv()


def main():
    from medrag.application.evaluation.evaluate import run_evaluation

    result = asyncio.run(run_evaluation())

    print(result)


if __name__ == "__main__":
    main()
