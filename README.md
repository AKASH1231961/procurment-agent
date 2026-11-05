# Agentic Procurement Negotiation System

A Python proof of concept for inventory monitoring and supplier procurement. A
LangChain inventory agent checks for low-stock items, searches for suppliers,
and—when an authorized maximum price is provided—negotiates with a seller agent.
A local Flask supervisor dashboard can launch the agent and display its output,
negotiation history, procurement status, and performance metrics.

## What it does

1. Checks inventory against reorder thresholds and opens procurement tasks.
2. Searches Google Shopping for wholesale suppliers and prices.
3. If an authorized maximum unit price is supplied, selects the lowest-priced
   candidate and runs a bounded buyer/seller negotiation.
4. Logs negotiation messages, outcomes, feedback, and token usage.
5. Shows inventory, conversations, and evaluation metrics in the supervisor UI.

The buyer will not negotiate without an explicitly supplied maximum price.
When running the agent, omitted terms default to 50 units and an opening offer
of 90% of the maximum price.

## Requirements

- Python 3.10 or later
- API credentials for Groq and SerpApi
- Local inventory database and schema (not included in this repository)
- Optional negotiation advice corpus (not included in this repository)

## Setup

Run commands from the project directory:

```bash
cd procurement-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Configure API keys

Create a local `.env` file in the project directory, or export the same
variables in your shell:

```dotenv
GROQ_API_KEY=your_groq_api_key
SERPAPI_KEY=your_serpapi_key
```

Never commit `.env` or paste API keys into source code. The `.gitignore`
excludes environment files.

### Provide local data files

The repository intentionally does not contain environment files, corpus data,
SQL schemas, or database files. They must be provided separately:

- Put the procurement schema in `schema.sql` in the project directory. It must
  define the `items`, `procurement_tasks`, and `candidate_suppliers` tables
  expected by `Inventory_db .py`.
- Put the negotiation corpus at
  `corpus/negotiation-strategy-sft-100k.jsonl` if tactical retrieval advice is
  needed. Retrieval is used for large price gaps or stalled negotiations.
- The runtime databases are local: `procurement.db` for inventory and
  `negotiation_activity.db` for live session activity. Do not copy production
  databases into source control.

Initialize the inventory database after supplying `schema.sql`:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path

path = Path("Inventory_db .py")
spec = importlib.util.spec_from_file_location("inventory_database", path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {path}")
database = importlib.util.module_from_spec(spec)
spec.loader.exec_module(database)
database.init_db("schema.sql")
print("Initialized procurement.db")
PY
```

Add inventory through the database helper or your own data-loading process.
For example, the helper's `add_item` function accepts a SKU, name, current
stock, reorder threshold, optional category, unit, and reorder quantity.

> **Fresh-clone note:** without the separately supplied schema and inventory
> database, the UI can start, but the inventory tools cannot perform a complete
> procurement run. Without the corpus, a negotiation that triggers tactical
> retrieval will report a missing corpus file.

## Run the supervisor dashboard

```bash
python supervisor_dashboard.py
```

Open local host . Enter the authorized
maximum unit price and click **Run agent**. The dashboard launches the fixed
agent entry point and streams its standard output and errors to the page. It
also refreshes inventory, tasks, conversations, and metrics periodically.

The development server binds to localhost and is intended for local use, not
direct public or production deployment.

## Run the inventory agent from a terminal

Run this in the project directory with the environment activated:

```bash
python inventory_langchain_agent.py --max-price 25
```

Optional arguments:

```bash
python inventory_langchain_agent.py \
  --max-price 25 \
  --initial-offer 20 \
  --quantity 50
```

`--initial-offer` cannot exceed `--max-price`. The maximum price is required
for negotiation; without it, the agent only checks inventory and sources
suppliers.

## Supervisor dashboard metrics

Metrics are calculated from the local procurement database and
`negotiation_logs.md`. A missing value (`—`) means the required observations
are not available, rather than a measured zero.

### Negotiation outcomes

- **Agreement rate:** agreed negotiations divided by completed negotiations.
  Completed means `agreed` or `rejected`; active or incomplete sessions are
  excluded. The dashboard also breaks this down by product and supplier.
- **Savings vs. budget:** average `(authorized maximum − agreed price) ÷
  authorized maximum` for agreements with the required prices. This compares
  against the buyer's limit; it is not proof of market savings.
- **Total cost avoided:** sums `(authorized maximum − agreed price) ×
  quantity` for eligible agreements. This is also relative to the buyer's
  limit, not a verified counterfactual cost.
- **Savings vs. seller's first ask:** average `(seller's first counter −
  agreed price) ÷ seller's first counter`, for agreed negotiations where the
  seller countered before the deal.
- **Price vs. logged benchmark:** average `(benchmark − agreed price) ÷
  benchmark` for agreements with a logged supplier quote. Positive means the
  final price is below the quote; negative means it is above. Search prices
  may be estimates and should be checked for matching product, unit, and
  currency.
- **Budget compliance:** agreements at or below the authorized maximum divided
  by agreements for which both values are available.

### Procurement status

- **Task completion:** settled tasks divided by closed tasks. Closed tasks have
  a closing timestamp and a `settled`, `failed`, or `cancelled` status.
- **Active tasks:** current tasks with `open`, `sourcing`, or `negotiating`
  status.
- **Inventory shortages:** items whose current stock is at or below the
  reorder threshold.
- **Live negotiations:** sessions marked in progress.

### Efficiency and model usage

- **Average negotiation time:** average elapsed time from the first to last
  logged message for completed sessions with usable timestamps.
- **Seller reply time:** average and median time from a buyer message to the
  seller's next response.
- **Buyer decision time:** average time from a seller response to the next
  buyer message. It can include retrieval and model-call time.
- **Token usage:** total logged input, output, and combined model tokens. This
  is not an API-cost estimate.

### Retrieval and repeated-session trends

- **Tactical retrieval effectiveness:** compares next-reply seller acceptance,
  negotiation-gap closure, and session agreement rates for sessions/rounds
  with and without retrieved advice. Retrieval is triggered by difficult
  negotiation signals, so this is descriptive, not a controlled experiment.
- **Learning over repeated sessions:** compares savings-vs-budget values from
  the first to latest completed session for the same product. Results can be
  affected by changes in supplier, quantity, or budget; they do not prove that
  coaching caused an improvement.
- **Savings by applied strategy:** groups measured savings by the prior-session
  coaching strategy recorded in the log.

## Tests

Run the test suite from the project root:

```bash
python -m pytest -q
```

## Project files

- `inventory_langchain_agent.py` — inventory-monitoring agent and CLI.
- `Inventory_db .py` — SQLite inventory and procurement-task helpers.
- `buyer_negotiation.py` — buyer strategy, negotiation loop, and transcript log.
- `seller_agent.py` — seller negotiation behavior.
- `negotiation_activity.py` — live session activity storage for the dashboard.
- `negotiation_metrics.py` — deal-quality, latency, learning, and retrieval
  metric calculations.
- `supervisor_dashboard.py` — local Flask dashboard and agent-run endpoints.
- `templates/`, `static/` — dashboard UI and styling.
- `tests/` — unit and integration tests.

## Data and privacy

`.env`, corpus files, SQL schemas, SQLite databases, negotiation logs, Python
virtual environments, and generated caches are excluded from Git. Keep API
credentials and company procurement data local and provide approved data
through your organization's secure process.
