import asyncio
import copy
import datetime
import hashlib
import hmac
import json
import ssl
import time
import uuid
from asyncio import CancelledError
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable

import rfc8785
import structlog.stdlib
from websockets import ClientConnection
from websockets.asyncio.client import connect

from aramaki.collection import Collection, ReplicationManager
from aramaki.db import DBSessionManager

log = structlog.stdlib.get_logger()


class MessageReplaySet:
    ids: set[str]
    ages: deque[
        tuple[float, str]
    ]  # monotonic increasing timestamp from left to right

    TIMEOUT = 60 * 60 + 5 * 60  # 1h 5m

    def __init__(self):
        self.ids = set()
        self.ages = deque()
        self.last_expire = time.time()

    def check(self, id: str):
        if self.last_expire < time.time() - 60:
            self.expire()
        if id in self.ids:
            raise KeyError(f"ID already seen: {id}")

    def mark(self, id):
        self.ids.add(id)
        self.ages.append((time.time(), id))

    def expire(self):
        cutoff = time.time() - self.TIMEOUT
        while self.ages:
            t, id = self.ages.popleft()
            if t > cutoff:
                self.ages.appendleft((t, id))
                break
            self.ids.remove(id)


class Manager:
    collections: dict[type, ReplicationManager]

    def __init__(
        self, principal, application, url, secret, state_directory: Path
    ):
        self.websocket: ClientConnection | None = None
        self.known_messages = MessageReplaySet()
        self.principal = principal
        self.application = application
        self.url = url
        self.secret = secret
        self.state_directory = state_directory

        self.subscriptions = {}
        self.callbacks = {}
        self.tasks: set[asyncio.Task] = set()
        self.websocket_ready = asyncio.Event()

        self.collections = {}

        if not self.state_directory.exists():
            self.state_directory.mkdir()
        self.db = DBSessionManager(state_directory)
        # This is blocking on purpose, to ensure we have a clean DB when everything starts up.
        self.db.upgrade()

        self.tasks.add(asyncio.create_task(self.run()))

    def register_message_handler(
        self,
        type_: str,
        callback: Callable[[dict], Awaitable[Any]],
        **scope: str,
    ):
        self.subscriptions[type_] = scope
        self.callbacks[type_] = callback

    def register_collection(
        self, cls_: type["Collection"]
    ) -> ReplicationManager:
        """Activate a collection and provide a factory that can be used with `svcs`."""
        assert cls_ not in self.collections
        self.collections[cls_] = manager = ReplicationManager(self, cls_)
        return manager

    async def run(self):
        loop = asyncio.get_running_loop()
        log.info("start-manager")
        while True:
            try:
                log.info("directory-connection", status="connecting")
                # TODO: testing
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                async with connect(self.url, ssl=ssl_context) as websocket:
                    self.websocket = websocket
                    log.info("directory-connection", status="connected")

                    log.info("Sending subscription.")
                    subscription = {
                        "@type": "aramaki.subscription",
                        "@application": self.application,
                        "matches": [
                            {
                                "@type": type_,
                                "scope": scope,
                            }
                            for type_, scope in self.subscriptions.items()
                        ],
                    }
                    await websocket.send(self.prepare_message(subscription))
                    self.websocket_ready.set()

                    # XXX error handling, e.g. if authentication has failed
                    # -> wait for a response

                    log.info("Waiting for messages ...")
                    async for message in websocket:
                        loop.create_task(self.process(message))
            except CancelledError:
                return
            except Exception:
                log.exception("unexpected-exception")
            finally:
                self.websocket_ready.clear()
                self.websocket = None
            # XXX exponential backoff / csmacd
            log.info("connection lost, backing off")
            await asyncio.sleep(5)

    def stop(self):
        for collection in self.collections.values():
            collection.stop()
        for task in self.tasks:
            task.cancel()

    async def process(self, message):
        message = json.loads(message)
        self.authenticate(message)
        if message["@type"] in self.callbacks:
            await self.callbacks[message["@type"]](message)

    def authenticate(self, message):
        """Authenticate whether this message has originated from the advertised
        principal.

        """
        advertised_principal = message["@principal"]
        assert advertised_principal == "@directory"

        expiry = datetime.datetime.fromisoformat(message["@expiry"])
        # Only accept messages that are not expired
        if expiry < datetime.datetime.now(datetime.UTC):
            raise Exception(
                f"message too old (expired at {message['@expiry']})"
            )

        self.known_messages.check(message["@id"])

        check_message = copy.deepcopy(message)
        advertised_signature = check_message["@signature"].pop("signature")

        signature = hmac.new(
            self.secret.encode("ascii"),
            rfc8785.dumps(check_message),
            hashlib.sha256,
        ).hexdigest()

        if signature != advertised_signature:
            raise Exception(
                f"signature mismatch {signature} != {advertised_signature}"
            )
        # Once the message is authenticated, store the ID.
        # This prevents a DOS attack enumerating IDs
        self.known_messages.mark(message["@id"])

    def sign_message(self, message):
        log.info(rfc8785.dumps(message))
        signature = hmac.new(
            self.secret.encode("ascii"), rfc8785.dumps(message), hashlib.sha256
        ).hexdigest()
        message["@signature"]["signature"] = signature

    def prepare_message(self, message):
        now = datetime.datetime.now().astimezone(datetime.UTC)
        message_template = {
            "@context": "https://flyingcircus.io/ns/aramaki",
            "@version": 1,
            "@signature": {"alg": "HS256"},
            "@principal": self.principal,
            "@application": self.application,
            "@issued": now.isoformat(),
            "@expiry": (now + datetime.timedelta(hours=1)).isoformat(),
            "@id": uuid.uuid4().hex,
        }
        message = copy.deepcopy(message)
        message.update(message_template)
        self.sign_message(message)
        return json.dumps(message)

    async def send_message(self, type: str, message: dict):
        # TODO: backoff, retry
        # TODO: testing
        # XXX: logging scrub contents to not leak secrets
        # XXX: add (persistent?) buffering
        await self.websocket_ready.wait()
        log.info(f"sending message {type}: {message}")
        await self.websocket.send(
            self.prepare_message(message | {"@type": type})
        )
