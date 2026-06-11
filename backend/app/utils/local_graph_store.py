"""
SQLite图谱存储
替代JSON文件存储，将图谱数据（节点、边、情节）存储在单个SQLite文件中。

存储结构:
    {storage_dir}/
      graphs.db          - 唯一SQLite数据库文件（包含所有图谱）
      graphs.db-wal      - SQLite WAL日志（自动管理）
      graphs.db-shm      - SQLite共享内存（自动管理）
      mirofish_xxx/      - JSON备份目录（迁移后保留，只读）

向后兼容:
    - 与JSON版LocalGraphStore保持完全相同的API
    - 数据自动迁移：首次运行时检测JSON文件并导入SQLite
    - JSON文件保留为备份，不删除
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .logger import get_logger

logger = get_logger('mirofish.local_graph_store')

# Chemin du fichier SQLite dans le répertoire de stockage
DB_FILENAME = "graphs.db"

# Verrou global pour SQLite (thread-safe via WAL + check_same_thread=False)
_local = threading.local()


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Récupère une connexion SQLite par thread (pool implicite via threading.local)."""
    if not hasattr(_local, 'conn') or _local.conn is None:
        _local.conn = _create_connection(db_path)
    # Vérifie que la connexion pointe toujours sur le bon fichier
        # (cas où storage_dir change entre les instances)
    return _local.conn


def _create_connection(db_path: str) -> sqlite3.Connection:
    """Crée une nouvelle connexion SQLite avec les bons paramètres."""
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")          # Write-Ahead Logging = lectures sans blocage
    conn.execute("PRAGMA synchronous=NORMAL")         # Équilibre vitesse/sécurité
    conn.execute("PRAGMA cache_size=-8000")           # Cache de ~8 Mo
    conn.execute("PRAGMA temp_store=MEMORY")          # Temp en RAM = plus rapide
    conn.execute("PRAGMA foreign_keys=ON")            # Intégrité référentielle
    conn.row_factory = sqlite3.Row                     # Retourne des dict-like
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Crée les tables si elles n'existent pas."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS graphs (
            graph_id    TEXT PRIMARY KEY,
            name        TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL,
            ontology    TEXT,
            metadata_json TEXT
        );

        CREATE TABLE IF NOT EXISTS nodes (
            graph_id    TEXT NOT NULL,
            uuid        TEXT NOT NULL,
            name        TEXT NOT NULL DEFAULT '',
            labels      TEXT NOT NULL DEFAULT '[]',
            summary     TEXT NOT NULL DEFAULT '',
            attributes  TEXT NOT NULL DEFAULT '{}',
            created_at  TEXT NOT NULL,
            PRIMARY KEY (graph_id, uuid)
        );

        CREATE TABLE IF NOT EXISTS edges (
            graph_id         TEXT NOT NULL,
            uuid             TEXT NOT NULL,
            name             TEXT NOT NULL DEFAULT '',
            fact             TEXT NOT NULL DEFAULT '',
            source_node_uuid TEXT NOT NULL,
            target_node_uuid TEXT NOT NULL,
            created_at       TEXT,
            valid_at         TEXT,
            invalid_at       TEXT,
            expired_at       TEXT,
            attributes       TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (graph_id, uuid)
        );

        CREATE TABLE IF NOT EXISTS episodes (
            graph_id   TEXT NOT NULL,
            uuid       TEXT NOT NULL,
            text       TEXT NOT NULL,
            created_at TEXT NOT NULL,
            processed  INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (graph_id, uuid)
        );

        CREATE INDEX IF NOT EXISTS idx_nodes_graph_id   ON nodes(graph_id);
        CREATE INDEX IF NOT EXISTS idx_nodes_name       ON nodes(name);
        CREATE INDEX IF NOT EXISTS idx_nodes_name_lower ON nodes(LOWER(name));
        CREATE INDEX IF NOT EXISTS idx_edges_graph_id   ON edges(graph_id);
        CREATE INDEX IF NOT EXISTS idx_edges_source     ON edges(source_node_uuid);
        CREATE INDEX IF NOT EXISTS idx_edges_target     ON edges(target_node_uuid);
        CREATE INDEX IF NOT EXISTS idx_episodes_graph_id ON episodes(graph_id);
    """)
    conn.commit()


# ── Utilitaires de sérialisation ─────────────────────────────────────────────

def _json_loads(value: Optional[str]) -> Any:
    """Désérialise une chaîne JSON en objet Python."""
    if value is None or value == '':
        return {} if value is not None else None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {} if isinstance(value, str) else value


def _json_dumps(value: Any) -> str:
    """Sérialise un objet Python en chaîne JSON."""
    if value is None:
        return 'null'
    return json.dumps(value, ensure_ascii=False)


def _row_to_node(row: sqlite3.Row) -> Dict[str, Any]:
    """Convertit une ligne SQLite en dict nœud compatible JSON."""
    return {
        "uuid": row["uuid"],
        "name": row["name"],
        "labels": _json_loads(row["labels"]),
        "summary": row["summary"],
        "attributes": _json_loads(row["attributes"]),
        "created_at": row["created_at"],
    }


def _row_to_edge(row: sqlite3.Row) -> Dict[str, Any]:
    """Convertit une ligne SQLite en dict arête compatible JSON."""
    return {
        "uuid": row["uuid"],
        "name": row["name"],
        "fact": row["fact"],
        "source_node_uuid": row["source_node_uuid"],
        "target_node_uuid": row["target_node_uuid"],
        "created_at": row["created_at"],
        "valid_at": row["valid_at"],
        "invalid_at": row["invalid_at"],
        "expired_at": row["expired_at"],
        "attributes": _json_loads(row["attributes"]),
    }


def _migrate_from_json(conn: sqlite3.Connection, storage_dir: str) -> bool:
    """
    Migre les données depuis les fichiers JSON vers SQLite.
    Retourne True si une migration a eu lieu.

    Les fichiers JSON ne sont pas supprimés après migration.
    """
    # Vérifie s'il y a des répertoires de graphes JSON
    graph_dirs = []
    if not os.path.exists(storage_dir):
        return False

    for entry in os.listdir(storage_dir):
        dir_path = os.path.join(storage_dir, entry)
        meta_path = os.path.join(dir_path, "metadata.json")
        if os.path.isdir(dir_path) and os.path.exists(meta_path):
            graph_dirs.append(entry)

    if not graph_dirs:
        return False

    logger.info(f"Migration JSON → SQLite : {len(graph_dirs)} graphe(s) trouvé(s)")

    migrated_count = 0
    for graph_id in sorted(graph_dirs):
        try:
            _migrate_one_graph(conn, storage_dir, graph_id)
            migrated_count += 1
            logger.info(f"  ✓ {graph_id} migré")
        except Exception as e:
            logger.error(f"  ✗ {graph_id} échec : {e}")

    conn.commit()
    logger.info(f"Migration terminée : {migrated_count}/{len(graph_dirs)} graphes migrés")
    return migrated_count > 0


def _migrate_one_graph(conn: sqlite3.Connection, storage_dir: str, graph_id: str) -> None:
    """Migre un seul graphe JSON vers SQLite."""
    base = os.path.join(storage_dir, graph_id)

    # metadata
    meta_path = os.path.join(base, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        conn.execute(
            "INSERT OR IGNORE INTO graphs (graph_id, name, description, created_at, ontology, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                graph_id,
                meta.get("name", ""),
                meta.get("description", ""),
                meta.get("created_at", datetime.now().isoformat()),
                _json_dumps(meta.get("ontology")),
                _json_dumps(meta),
            )
        )

    # nodes
    nodes_path = os.path.join(base, "nodes.json")
    if os.path.exists(nodes_path):
        with open(nodes_path, 'r', encoding='utf-8') as f:
            nodes = json.load(f)
        for node in nodes:
            conn.execute(
                "INSERT OR IGNORE INTO nodes (graph_id, uuid, name, labels, summary, attributes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    graph_id,
                    node.get("uuid", uuid.uuid4().hex),
                    node.get("name", ""),
                    _json_dumps(node.get("labels", [])),
                    node.get("summary", ""),
                    _json_dumps(node.get("attributes", {})),
                    node.get("created_at", datetime.now().isoformat()),
                )
            )

    # edges
    edges_path = os.path.join(base, "edges.json")
    if os.path.exists(edges_path):
        with open(edges_path, 'r', encoding='utf-8') as f:
            edges = json.load(f)
        for edge in edges:
            conn.execute(
                "INSERT OR IGNORE INTO edges (graph_id, uuid, name, fact, source_node_uuid, target_node_uuid, created_at, valid_at, invalid_at, expired_at, attributes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    graph_id,
                    edge.get("uuid", uuid.uuid4().hex),
                    edge.get("name", ""),
                    edge.get("fact", ""),
                    edge.get("source_node_uuid", ""),
                    edge.get("target_node_uuid", ""),
                    edge.get("created_at"),
                    edge.get("valid_at"),
                    edge.get("invalid_at"),
                    edge.get("expired_at"),
                    _json_dumps(edge.get("attributes", {})),
                )
            )

    # episodes
    episodes_path = os.path.join(base, "episodes.jsonl")
    if os.path.exists(episodes_path):
        with open(episodes_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                conn.execute(
                    "INSERT OR IGNORE INTO episodes (graph_id, uuid, text, created_at, processed) VALUES (?, ?, ?, ?, ?)",
                    (
                        graph_id,
                        ep.get("uuid", uuid.uuid4().hex),
                        ep.get("text", ""),
                        ep.get("created_at", datetime.now().isoformat()),
                        1 if ep.get("processed", True) else 0,
                    )
                )


# ═══════════════════════════════════════════════════════════════════════════════
# LocalGraphStore — API publique (identique à la version JSON)
# ═══════════════════════════════════════════════════════════════════════════════

class LocalGraphStore:
    """SQLite图谱存储

    API 100% compatible avec l'ancienne version JSON.
    Les données sont stockées dans un fichier SQLite unique ``graphs.db``
    dans le répertoire ``storage_dir``.
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        self.db_path = os.path.join(storage_dir, DB_FILENAME)
        db_exists = os.path.exists(self.db_path)

        # Connexion unique pour cette instance
        self.conn = _create_connection(self.db_path)
        _init_schema(self.conn)

        # Migration auto si SQLite tout juste créé et JSON existants
        if not db_exists:
            migrated = _migrate_from_json(self.conn, storage_dir)
            if migrated:
                logger.info("Migration JSON→SQLite effectuée. Fichiers JSON conservés.")
            else:
                logger.info("Nouvelle base SQLite créée (aucune donnée JSON trouvée).")

    # ── Nettoyage ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Ferme la connexion SQLite proprement."""
        try:
            self.conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    # ── 图谱生命周期 ──────────────────────────────────────────────────────────

    def create_graph(self, graph_id: str, name: str, description: str = "") -> None:
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT OR IGNORE INTO graphs (graph_id, name, description, created_at) VALUES (?, ?, ?, ?)",
            (graph_id, name, description, now),
        )
        self.conn.commit()
        logger.info(f"图谱已创建: {graph_id}")

    def delete_graph(self, graph_id: str) -> None:
        self.conn.execute("DELETE FROM episodes WHERE graph_id = ?", (graph_id,))
        self.conn.execute("DELETE FROM edges WHERE graph_id = ?", (graph_id,))
        self.conn.execute("DELETE FROM nodes WHERE graph_id = ?", (graph_id,))
        self.conn.execute("DELETE FROM graphs WHERE graph_id = ?", (graph_id,))
        self.conn.commit()
        logger.info(f"图谱已删除: {graph_id}")

    def graph_exists(self, graph_id: str) -> bool:
        cursor = self.conn.execute(
            "SELECT 1 FROM graphs WHERE graph_id = ?", (graph_id,)
        )
        return cursor.fetchone() is not None

    # ── 本体 ──────────────────────────────────────────────────────────────────

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE graphs SET ontology = ? WHERE graph_id = ?",
            (_json_dumps(ontology), graph_id),
        )
        self.conn.commit()

    def get_ontology(self, graph_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT ontology FROM graphs WHERE graph_id = ?", (graph_id,)
        )
        row = cursor.fetchone()
        if row is None or row["ontology"] is None:
            return None
        return _json_loads(row["ontology"])

    def get_metadata(self, graph_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM graphs WHERE graph_id = ?", (graph_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        meta = _json_loads(row["metadata_json"]) if row["metadata_json"] else {}
        # Complète avec les champs actuels si metadata_json est partiel
        meta.setdefault("graph_id", row["graph_id"])
        meta.setdefault("name", row["name"])
        meta.setdefault("description", row["description"])
        meta.setdefault("created_at", row["created_at"])
        meta.setdefault("ontology", _json_loads(row["ontology"]))
        return meta

    # ── 情节（Episode）────────────────────────────────────────────────────────

    def add_episode(self, graph_id: str, text: str) -> str:
        """Ajoute un épisode, retourne son UUID."""
        episode_id = uuid.uuid4().hex
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO episodes (graph_id, uuid, text, created_at, processed) VALUES (?, ?, ?, ?, 1)",
            (graph_id, episode_id, text, now),
        )
        self.conn.commit()
        return episode_id

    def add_episodes_batch(self, graph_id: str, texts: List[str]) -> List[str]:
        ids = []
        now = datetime.now().isoformat()
        for text in texts:
            episode_id = uuid.uuid4().hex
            self.conn.execute(
                "INSERT INTO episodes (graph_id, uuid, text, created_at, processed) VALUES (?, ?, ?, ?, 1)",
                (graph_id, episode_id, text, now),
            )
            ids.append(episode_id)
        self.conn.commit()
        return ids

    def episode_is_processed(self, graph_id: str, episode_uuid: str) -> bool:
        """En SQLite comme en JSON, les épisodes sont toujours traités immédiatement."""
        _ = graph_id, episode_uuid  # juste pour la compatibilité d'appel
        return True

    # ── 节点 ──────────────────────────────────────────────────────────────────

    def get_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM nodes WHERE graph_id = ?", (graph_id,)
        )
        return [_row_to_node(row) for row in cursor.fetchall()]

    def get_node(self, graph_id: str, node_uuid: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM nodes WHERE graph_id = ? AND uuid = ?",
            (graph_id, node_uuid),
        )
        row = cursor.fetchone()
        return _row_to_node(row) if row else None

    def upsert_node(
        self,
        graph_id: str,
        name: str,
        labels: Optional[List[str]] = None,
        summary: str = "",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insère ou met à jour un nœud par nom (case-insensitive). Retourne son UUID."""
        labels = labels or ["Entity"]
        attributes = attributes or {}

        # Cherche un nœud existant avec le même nom
        cursor = self.conn.execute(
            "SELECT uuid, labels, summary, attributes FROM nodes WHERE graph_id = ? AND LOWER(name) = LOWER(?)",
            (graph_id, name),
        )
        existing = cursor.fetchone()

        if existing:
            node_uuid = existing["uuid"]
            # Fusion des labels
            existing_labels = set(_json_loads(existing["labels"]))
            existing_labels.update(labels)
            merged_labels = list(existing_labels)
            # Résumé : garde l'ancien si le nouveau est vide
            merged_summary = summary if summary else (existing["summary"] or "")
            # Attributs : merge
            existing_attrs = _json_loads(existing["attributes"])
            if attributes:
                existing_attrs.update(attributes)
            self.conn.execute(
                "UPDATE nodes SET labels = ?, summary = ?, attributes = ? WHERE graph_id = ? AND uuid = ?",
                (_json_dumps(merged_labels), merged_summary, _json_dumps(existing_attrs),
                 graph_id, node_uuid),
            )
            self.conn.commit()
            return node_uuid

        # Création d'un nouveau nœud
        node_uuid = uuid.uuid4().hex
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO nodes (graph_id, uuid, name, labels, summary, attributes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (graph_id, node_uuid, name, _json_dumps(labels), summary, _json_dumps(attributes), now),
        )
        self.conn.commit()
        return node_uuid

    # ── 边 ───────────────────────────────────────────────────────────────────

    def get_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM edges WHERE graph_id = ?", (graph_id,)
        )
        return [_row_to_edge(row) for row in cursor.fetchall()]

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT * FROM edges WHERE graph_id = ? AND (source_node_uuid = ? OR target_node_uuid = ?)",
            (graph_id, node_uuid, node_uuid),
        )
        return [_row_to_edge(row) for row in cursor.fetchall()]

    def add_edge(self, graph_id: str, edge: Dict[str, Any]) -> str:
        edge_uuid = edge.get("uuid") or uuid.uuid4().hex
        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT INTO edges
               (graph_id, uuid, name, fact, source_node_uuid, target_node_uuid,
                created_at, valid_at, invalid_at, expired_at, attributes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                graph_id,
                edge_uuid,
                edge.get("name", ""),
                edge.get("fact", ""),
                edge.get("source_node_uuid", ""),
                edge.get("target_node_uuid", ""),
                edge.get("created_at", now),
                edge.get("valid_at"),
                edge.get("invalid_at"),
                edge.get("expired_at"),
                _json_dumps(edge.get("attributes", {})),
            ),
        )
        self.conn.commit()
        return edge_uuid

    def add_fact_edge(
        self,
        graph_id: str,
        source_uuid: str,
        target_uuid: str,
        name: str,
        fact: str,
    ) -> str:
        return self.add_edge(graph_id, {
            "name": name,
            "fact": fact,
            "source_node_uuid": source_uuid,
            "target_node_uuid": target_uuid,
        })

    # ── 搜索 ─────────────────────────────────────────────────────────────────

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> Dict[str, Any]:
        """
        Recherche par mots-clés dans les nœuds et arêtes du graphe.
        Compatible avec l'ancienne implémentation JSON.
        """
        query_lower = query.lower().strip()
        keywords = [
            w.strip()
            for w in query_lower.replace(',', ' ').replace('，', ' ').split()
            if len(w.strip()) > 1
        ]

        result_edges: List[Dict] = []
        result_nodes: List[Dict] = []
        facts: List[str] = []

        # ── Recherche dans les arêtes ──
        if scope in ("edges", "both"):
            pattern = f"%{query_lower}%"
            cursor = self.conn.execute(
                """SELECT * FROM edges
                   WHERE graph_id = ?
                     AND (LOWER(fact) LIKE ? OR LOWER(name) LIKE ?)
                   ORDER BY rowid""",
                (graph_id, pattern, pattern),
            )
            all_edges = [_row_to_edge(row) for row in cursor.fetchall()]

            # Score puis tri
            def edge_score(e: Dict) -> int:
                s = 0
                tl = (e.get("fact", "") + " " + e.get("name", "")).lower()
                if query_lower in tl:
                    s += 100
                for kw in keywords:
                    if kw in tl:
                        s += 10
                return s

            scored = sorted(
                [(edge_score(e), e) for e in all_edges if edge_score(e) > 0],
                key=lambda x: x[0], reverse=True,
            )
            for _, edge in scored[:limit]:
                result_edges.append(edge)
                if edge.get("fact"):
                    facts.append(edge["fact"])

        # ── Recherche dans les nœuds ──
        if scope in ("nodes", "both"):
            pattern = f"%{query_lower}%"
            cursor = self.conn.execute(
                """SELECT * FROM nodes
                   WHERE graph_id = ?
                     AND (LOWER(name) LIKE ? OR LOWER(summary) LIKE ?)
                   ORDER BY rowid""",
                (graph_id, pattern, pattern),
            )
            all_nodes = [_row_to_node(row) for row in cursor.fetchall()]

            def node_score(n: Dict) -> int:
                s = 0
                tl = (n.get("name", "") + " " + n.get("summary", "")).lower()
                if query_lower in tl:
                    s += 100
                for kw in keywords:
                    if kw in tl:
                        s += 10
                return s

            scored = sorted(
                [(node_score(n), n) for n in all_nodes if node_score(n) > 0],
                key=lambda x: x[0], reverse=True,
            )
            for _, node in scored[:limit]:
                result_nodes.append(node)
                if node.get("summary"):
                    facts.append(f"[{node['name']}]: {node['summary']}")

        return {"facts": facts, "edges": result_edges, "nodes": result_nodes}
