CREATE TABLE IF NOT EXISTS collection_replication_status(collection TEXT PRIMARY KEY, partition TEXT PRIMARY KEY, record_id TEXT PRIMARY KEY, `version INTEGER, data BLOB)
