import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_MODULE_PATH = PROJECT_ROOT / "Inventory_db .py"
SCHEMA_PATH = PROJECT_ROOT / "schema.sql"


def _load_database_module():
    spec = importlib.util.spec_from_file_location(
        "inventory_database_for_test",
        DATABASE_MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clear_candidates_removes_only_candidates_for_the_finished_task(
    tmp_path,
    monkeypatch,
):
    database = _load_database_module()
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "procurement.db"))
    database.init_db(str(SCHEMA_PATH))

    with database.get_conn() as connection:
        item_ids = []
        for index in range(2):
            cursor = connection.execute(
                """INSERT INTO items
                   (sku, name, current_stock, reorder_threshold)
                   VALUES (?, ?, 0, 10)""",
                (f"SKU-{index}", f"Item {index}"),
            )
            item_ids.append(cursor.lastrowid)
        task_ids = []
        for item_id in item_ids:
            cursor = connection.execute(
                """INSERT INTO procurement_tasks (item_id, target_quantity)
                   VALUES (?, 10)""",
                (item_id,),
            )
            task_ids.append(cursor.lastrowid)

    for task_id in task_ids:
        database.log_candidate_supplier(
            task_id,
            supplier_name=f"Supplier for task {task_id}",
            estimated_price=1.0,
        )

    database.clear_candidates(task_ids[0])

    assert database.get_candidates(task_ids[0]) == []
    assert len(database.get_candidates(task_ids[1])) == 1
