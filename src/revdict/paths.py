from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "rev_dictionary"
INDEX_DIR = CACHE_DIR / "index"
RAW_WIKTIONARY_PATH = CACHE_DIR / "raw-wiktextract-data.jsonl.gz"
RAW_NGRAM_FICTION_PATH = CACHE_DIR / "raw-ngram-fiction-1grams.jsonl.gz"
RAW_NGRAM_FICTION_TOTALCOUNTS_PATH = CACHE_DIR / "raw-ngram-fiction-totalcounts.txt"
DAEMON_SOCKET_PATH = CACHE_DIR / "daemon.sock"
DAEMON_PID_PATH = CACHE_DIR / "daemon.pid"
DAEMON_LOG_PATH = CACHE_DIR / "daemon.log"
QUERY_HISTORY_PATH = CACHE_DIR / "query_history"
