"""The access model: who may read which chunks.

Every stored document carries three authorization attributes, copied onto each of its chunks:

- tenant_id     the tenant that owns it. A user only ever sees their own tenant's documents.
- access_level  public < internal < restricted < confidential.
- department    which department owns it.

A user's role sets the highest access level they may read. Levels up to `internal` are open to
the whole tenant; `restricted` and `confidential` documents additionally require the user to
belong to the document's department (admins belong to every department). A document or chunk
that is missing an attribute is treated as unreadable: the model fails closed.

`AccessScope` is built only from a validated token, never from request parameters, and is turned
into a SQL predicate that is ANDed into the retrieval query. Restricted chunks are therefore
never fetched, so they cannot be ranked, reranked, scored by the evidence gate, given to a
generator, or written to a trace. Caller-supplied metadata filters are a separate, additional
condition: they can narrow results but can never widen an access scope.
"""

from dataclasses import dataclass
from typing import Any

ACCESS_LEVELS = ("public", "internal", "restricted", "confidential")
ACCESS_RANK = {level: rank for rank, level in enumerate(ACCESS_LEVELS)}
OPEN_RANK = ACCESS_RANK["internal"]  # up to here: any user of the tenant
MISSING_RANK = 99  # an unlabelled chunk is above every user's clearance

ROLES = ("employee", "underwriter", "compliance", "admin")
ROLE_MAX_LEVEL = {
    "employee": "internal",
    "underwriter": "restricted",
    "compliance": "restricted",
    "admin": "confidential",
}

DEFAULT_TENANT = "default"
DEFAULT_DEPARTMENT = "general"
DEFAULT_ACCESS_LEVEL = "internal"

# The rank of a chunk, read defensively from JSON metadata: anything that is not a small
# integer counts as MISSING_RANK.
_CHUNK_RANK = (
    "CASE WHEN {m}->>'access_rank' ~ '^[0-9]{{1,2}}$' "
    f"THEN ({{m}}->>'access_rank')::int ELSE {MISSING_RANK} END"
)


@dataclass(frozen=True)
class AccessScope:
    tenant_id: str
    max_rank: int
    departments: frozenset[str]
    all_departments: bool
    roles: tuple[str, ...]

    @property
    def max_level(self) -> str:
        return ACCESS_LEVELS[self.max_rank]

    def _params(self) -> dict[str, Any]:
        return {
            "acc_tenant": self.tenant_id,
            "acc_rank": self.max_rank,
            "acc_open": OPEN_RANK,
            "acc_all": self.all_departments,
            "acc_depts": sorted(self.departments),
        }

    def chunk_clause(self, alias: str = "") -> tuple[str, dict[str, Any]]:
        """A SQL predicate (and its bound parameters) over `chunks.metadata`."""
        meta = f"{alias}metadata"
        rank = _CHUNK_RANK.format(m=meta)
        sql = (
            f"({meta}->>'tenant_id' = %(acc_tenant)s"
            f" AND {rank} <= %(acc_rank)s"
            f" AND ({rank} <= %(acc_open)s OR %(acc_all)s"
            f" OR {meta}->>'department' = ANY(%(acc_depts)s::text[])))"
        )
        return sql, self._params()

    def document_clause(self, alias: str = "") -> tuple[str, dict[str, Any]]:
        """The same rule over the `documents` columns."""
        rank = (
            f"CASE {alias}access_level WHEN 'public' THEN 0 WHEN 'internal' THEN 1"
            f" WHEN 'restricted' THEN 2 WHEN 'confidential' THEN 3 ELSE {MISSING_RANK} END"
        )
        sql = (
            f"({alias}tenant_id = %(acc_tenant)s"
            f" AND {rank} <= %(acc_rank)s"
            f" AND ({rank} <= %(acc_open)s OR %(acc_all)s"
            f" OR {alias}department = ANY(%(acc_depts)s::text[])))"
        )
        return sql, self._params()

    def permits(self, tenant_id: str, access_level: str, department: str) -> bool:
        """The same rule in Python, for tests and for clarity."""
        rank = ACCESS_RANK.get(access_level, MISSING_RANK)
        return (
            tenant_id == self.tenant_id
            and rank <= self.max_rank
            and (rank <= OPEN_RANK or self.all_departments or department in self.departments)
        )


def scope_for(
    roles: list[str] | tuple[str, ...], tenant_id: str, departments: list[str] | tuple[str, ...]
) -> AccessScope | None:
    """The scope a user's roles grant, or None if none of their roles is recognised.

    Unknown roles are ignored rather than trusted: a token cannot invent a role.
    """
    known = tuple(dict.fromkeys(r for r in roles if r in ROLE_MAX_LEVEL))
    if not known:
        return None
    return AccessScope(
        tenant_id=tenant_id,
        max_rank=max(ACCESS_RANK[ROLE_MAX_LEVEL[r]] for r in known),
        departments=frozenset(d.strip().lower() for d in departments if d.strip()),
        all_departments="admin" in known,
        roles=known,
    )
