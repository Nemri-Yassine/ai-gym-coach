"""Shared Kafka helpers. Bootstrap comes from config; logs stay under C:/awc_tmp."""
from __future__ import annotations

import time

from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

from awc.config import KAFKA_BOOTSTRAP, KAFKA_TOPICS


def broker_up(bootstrap=KAFKA_BOOTSTRAP, timeout_s=5.0) -> bool:
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap,
            client_id="awc-probe",
            request_timeout_ms=int(timeout_s * 1000),
        )
        admin.list_topics()
        admin.close()
        return True
    except Exception:  # noqa: BLE001 — probe only
        return False


def ensure_topics(bootstrap=KAFKA_BOOTSTRAP, topics=KAFKA_TOPICS):
    admin = KafkaAdminClient(bootstrap_servers=bootstrap, client_id="awc-admin")
    try:
        existing = set(admin.list_topics())
        missing = [
            NewTopic(name=name, num_partitions=1, replication_factor=1)
            for name in topics if name not in existing
        ]
        if missing:
            try:
                admin.create_topics(missing, validate_only=False)
            except TopicAlreadyExistsError:
                pass
    finally:
        admin.close()


def make_producer(bootstrap=KAFKA_BOOTSTRAP) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: v if isinstance(v, (bytes, bytearray)) else str(v).encode("utf-8"),
        key_serializer=lambda v: None if v is None else str(v).encode("utf-8"),
        acks=1,
        linger_ms=5,
        retries=3,
    )


def make_consumer(group_id, topics, bootstrap=KAFKA_BOOTSTRAP, from_beginning=False) -> KafkaConsumer:
    return KafkaConsumer(
        *topics,
        bootstrap_servers=bootstrap,
        group_id=group_id,
        auto_offset_reset="earliest" if from_beginning else "latest",
        enable_auto_commit=True,
        consumer_timeout_ms=1000,
        value_deserializer=lambda b: b.decode("utf-8"),
        key_deserializer=lambda b: None if b is None else b.decode("utf-8"),
    )


def wait_for_broker(bootstrap=KAFKA_BOOTSTRAP, timeout_s=60.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if broker_up(bootstrap):
            return True
        time.sleep(1.0)
    return False


def publish(producer, topic, message, key=None):
    """Validate and send. message is a schema dataclass with to_json()."""
    from awc.streaming.schemas import validate

    validate(message)
    future = producer.send(topic, value=message.to_json(), key=key or message.session)
    return future
