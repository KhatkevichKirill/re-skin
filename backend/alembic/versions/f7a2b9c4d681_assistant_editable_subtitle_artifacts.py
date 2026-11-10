"""Permit private clean-picture and immutable editable subtitle artifacts.

Revision ID: f7a2b9c4d681
Revises: e6c1d8f9a403
"""
from alembic import op

revision = "f7a2b9c4d681"
down_revision = "e6c1d8f9a403"
branch_labels = None
depends_on = None


def _constraint(table, name, expression):
    connection = op.get_bind()
    triggers = []
    if connection.dialect.name == "sqlite":
        triggers = list(connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?", (table,),
        ).scalars())
    with op.batch_alter_table(table) as batch:
        batch.drop_constraint(name, type_="check")
        batch.create_check_constraint(name, expression)
    for sql in triggers:
        connection.exec_driver_sql(sql)


def _rewrite(changes):
    connection = op.get_bind()
    if connection.dialect.name != "sqlite":
        for change in changes:
            _constraint(*change)
        return
    # SQLite's batch table rebuild temporarily drops a referenced parent.
    # Defer checks within one real transaction, then validate the restored
    # graph before resetting the transient DROP-table violation counters.
    # foreign_keys stays enabled throughout; invalid existing data fails.
    driver = connection.connection.driver_connection
    if not driver.in_transaction:
        connection.exec_driver_sql("BEGIN")
    previous = connection.exec_driver_sql("PRAGMA defer_foreign_keys").scalar()
    connection.exec_driver_sql("PRAGMA defer_foreign_keys=ON")
    for change in changes:
        _constraint(*change)
    if connection.exec_driver_sql("PRAGMA foreign_key_check").first() is not None:
        raise RuntimeError("assistant migration would leave invalid foreign-key references")
    connection.exec_driver_sql("PRAGMA defer_foreign_keys=OFF")
    if previous:
        connection.exec_driver_sql("PRAGMA defer_foreign_keys=ON")


def upgrade():
    _rewrite([
        ("assistant_artifacts", "ck_assistant_artifact_media", "media_kind IN ('video','audio','image','subtitle')"),
        ("assistant_version_artifacts", "ck_assistant_version_artifact_role", "role IN ('full','preview','input','clean','captions') AND slot >= 0"),
    ])


def downgrade():
    # Refuse to discard referenced subtitle history. Downgrade works only before
    # any new artifact kinds/roles have been committed.
    connection = op.get_bind()
    if (connection.exec_driver_sql("SELECT COUNT(*) FROM assistant_artifacts WHERE media_kind='subtitle'").scalar()
            or connection.exec_driver_sql("SELECT COUNT(*) FROM assistant_version_artifacts WHERE role IN ('clean','captions')").scalar()):
        raise RuntimeError("assistant subtitle history exists; refusing lossy downgrade")
    _rewrite([
        ("assistant_version_artifacts", "ck_assistant_version_artifact_role", "role IN ('full','preview','input') AND slot >= 0"),
        ("assistant_artifacts", "ck_assistant_artifact_media", "media_kind IN ('video','audio','image')"),
    ])
