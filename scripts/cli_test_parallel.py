import asyncio
import time

import httpx

in_progress = 0


async def request():
    global in_progress
    async with httpx.AsyncClient(timeout=60) as client:
        in_progress += 1
        print(f"{int(time.time())} start: {in_progress}")
        async with client.stream(
            "POST",
            "https://ai.dev.fcio.net/openai/v1/chat/completions",
            # "http://127.0.0.1:23211/openai/v1/chat/completions",
            # "http://127.0.0.1:23211/openai/v1/test",
            headers={"Authorization": "Bearer ..."},
            json={
                "model": "gpt-oss:120b",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the nature of humanity?",
                    }
                ],
                "stream": True,
            },
        ) as result:
            if result.status_code != 200:
                print(f"Error: {result.status_code}")
                return
            else:
                async for _ in result.aiter_text():
                    # print(chunk)
                    pass
    in_progress -= 1
    print(f"stop: {in_progress}")


async def main():
    tasks: list[asyncio.Task[None]] = []
    for _ in range(1000):
        tasks.append(asyncio.create_task(request()))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
