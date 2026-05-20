#!/usr/bin/env python3
"""
Test script to verify the client disconnect chain.

Tests that when a client disconnects mid-stream, the cancellation
propagates through the chain and frees resources at each layer.

Usage:
    # Test against ike00 inference server directly:
    python3 test_disconnect_chain.py --host ike00.fcio.net --port 8000 --jump 185.105.255.34

    # Test against local dev instance:
    python3 test_disconnect_chain.py --host localhost --port 8000

    # Test through the gateway on services42:
    python3 test_disconnect_chain.py --host services42.fcio.net --port 23211 --gateway --token <token>
"""

import argparse
import asyncio
import sys
import time

import httpx
import httpx_sse


async def test_inference_disconnect(
    host: str, port: int, model: str = "gpt-oss:20b"
):
    """Test disconnect against the inference server directly."""
    url = f"http://{host}:{port}/models/{model}/proxy/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Write a very long essay about the history of computing. Go into extreme detail.",
            }
        ],
        "max_tokens": 2000,
        "stream": True,
    }

    print(f"POST {url}")
    print("Waiting for response...")

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                print(
                    f"ERROR: status {response.status_code}: {body.decode()[:500]}"
                )
                return False

            print(f"Status: {response.status_code}, reading chunks...")
            chunk_count = 0
            start = time.monotonic()

            async for _ in httpx_sse.EventSource(response).aiter_sse():
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(
                        f"  received {chunk_count} chunks in {time.monotonic() - start:.1f}s"
                    )

                # Disconnect after 5 chunks
                if chunk_count >= 5:
                    elapsed = time.monotonic() - start
                    print(
                        f"\n⛔ Disconnecting after {chunk_count} chunks ({elapsed:.2f}s)"
                    )
                    # Close the response without draining — simulates client disconnect
                    await response.aclose()
                    break
            else:
                # Normal completion (no break)
                elapsed = time.monotonic() - start
                print(
                    f"\n✅ Stream completed normally: {chunk_count} chunks in {elapsed:.2f}s"
                )
                return True

    elapsed = time.monotonic() - start
    print(f"\n⚠️  Client disconnected. Total time: {elapsed:.2f}s")
    print(
        "  If the server properly cancelled, the request should have ended quickly."
    )
    print("  If it continued generating, expect ~10-30s for the full response.")
    print()
    print(
        "  Check inference server logs for how long the request actually ran:"
    )
    print("    grep 'proxy' /var/log/skvaider/inference.log | tail -5")
    return True


async def test_gateway_disconnect(
    host: str, port: int, token: str, model: str = "gpt-oss:20b"
):
    """Test disconnect through the gateway (full chain: client → gateway → inference → vLLM)."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Write a very long essay about the history of computing. Go into extreme detail.",
            }
        ],
        "max_tokens": 2000,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {token}"}

    print(f"POST {url} (via gateway)")
    print("Waiting for response...")

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST", url, json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                print(
                    f"ERROR: status {response.status_code}: {body.decode()[:500]}"
                )
                return False

            print(f"Status: {response.status_code}, reading chunks...")
            chunk_count = 0
            start = time.monotonic()

            async for _ in httpx_sse.EventSource(response).aiter_sse():
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(
                        f"  received {chunk_count} chunks in {time.monotonic() - start:.1f}s"
                    )

                if chunk_count >= 5:
                    elapsed = time.monotonic() - start
                    print(
                        f"\n⛔ Disconnecting after {chunk_count} chunks ({elapsed:.2f}s)"
                    )
                    await response.aclose()
                    break
            else:
                elapsed = time.monotonic() - start
                print(
                    f"\n✅ Stream completed normally: {chunk_count} chunks in {elapsed:.2f}s"
                )
                return True

    elapsed = time.monotonic() - start
    print(f"\n⚠️  Client disconnected. Total time: {elapsed:.2f}s")
    print("  Check gateway + inference logs to verify cancellation propagated.")
    return True


async def test_nonstream_disconnect(
    host: str, port: int, model: str = "gpt-oss:20b"
):
    """Test disconnect on a non-streaming request (should be fast as server hasn't started yet)."""
    url = f"http://{host}:{port}/models/{model}/proxy/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Write a very long essay about the history of computing. Go into extreme detail.",
            }
        ],
        "max_tokens": 2000,
        "stream": False,
    }

    print(f"POST {url} (non-streaming)")
    print("Waiting for response...")

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Start the request as a task so we can cancel it
            task = asyncio.create_task(client.post(url, json=payload))
            # Cancel after 0.5s (before server finishes generating)
            await asyncio.sleep(0.5)
            print("\n⛔ Cancelling request after 0.5s")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    except Exception as e:
        print(f"  Exception: {type(e).__name__}: {e}")

    elapsed = time.monotonic() - start
    print(f"\n  Total time: {elapsed:.2f}s")
    if elapsed < 2:
        print("  ✅ Request cancelled quickly")
    else:
        print(
            f"  ⚠️  Request took {elapsed:.1f}s — may not have been cancelled promptly"
        )
    return True


async def main():
    parser = argparse.ArgumentParser(description="Test client disconnect chain")
    parser.add_argument(
        "--host", default="localhost", help="Host to connect to"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to connect to"
    )
    parser.add_argument(
        "--model", default="gpt-oss:20b", help="Model to test with"
    )
    parser.add_argument(
        "--gateway",
        action="store_true",
        help="Test through gateway (requires --token)",
    )
    parser.add_argument(
        "--token", default="", help="Auth token for gateway mode"
    )
    parser.add_argument(
        "--jump", default="", help="SSH jump host (for SSH tunneling)"
    )
    args = parser.parse_args()

    if args.gateway and not args.token:
        print("ERROR: --gateway requires --token")
        sys.exit(1)

    print(f"Target: {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print()

    if args.gateway:
        await test_gateway_disconnect(
            args.host, args.port, args.token, args.model
        )
    else:
        # Test 1: Streaming disconnect
        print("=" * 60)
        print("TEST 1: Streaming disconnect")
        print("=" * 60)
        await test_inference_disconnect(args.host, args.port, args.model)

        print()

        # Test 2: Non-streaming disconnect
        print("=" * 60)
        print("TEST 2: Non-streaming disconnect")
        print("=" * 60)
        await test_nonstream_disconnect(args.host, args.port, args.model)


if __name__ == "__main__":
    asyncio.run(main())
